"""Qwen3.8-27B OpenAI-compatible server with the V8 GPU + ANE prefill (16 Nov 2026). The release server of the
anemll-forge-v02 checkout (scripts/qwen38_server.py: Core AI target + DFlash2 drafter, prompt-cache snapshots,
decoding unchanged) with Engine.prefill replaced: when a request's new part is long (>= --v8-min tokens) it runs on
the V8 engine (v8_engine.py: every FFN on the ANE, every mixer on the GPU) to the prompt's end, the last 128-row pair
padded, the final logits from V8's own head. The GPU's attention writes and reads the ANE's INT8 KV cache (KV8; with
--v8-k8v16 1 the GPU also keeps an fp16 copy of V for its PV, the KV8 core's cost at 64K+ back). Snapshots (system /
turn) are taken during the run at the pair end at or below each cut (DeltaNet states captured per layer as that pair
completes, the next pair gated); the drafter's context gets the tap features V8 captures. A continuation (a snapshot
or the state after decoding) runs on V8 too: the bridge feeds prefill blocks up to the next 64- or 128-aligned point
(the one with fewer V8 pairs; 64 only when the span crosses no KV segment boundary), the DeltaNet states go to the GPU
(sync, on the GPU), the KV is shared. Everything else (and any V8 failure) falls back to the bridge's own prefill.
    PY v8_server.py [--v8-min 256] [--v8-kv8 1] [--v8-k8v16 1] [--v8-attproj-from P] <qwen38_server.py arguments>
(the qwen38_server.py arguments as `forge.py serve --dry-run` prints them; its environment too; v8_server.sh).
env V8_ALIGN=64|128 (continuation start alignment), V8_TAIL_MIN (192: a shorter rest after a snapshot -> bridge)."""
import argparse, os, sys, time
from pathlib import Path

V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
PIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(V02 / "scripts"))
sys.path.insert(0, str(PIPE))
import numpy as np  # noqa: E402

import qwen38_server as S  # noqa: E402

A2 = 128   # V8 granularity: a pair of 64-token blocks
V8_ALIGN = int(os.environ.get("V8_ALIGN", "64"))   # V8 starts on this alignment (64: unless the span crosses a KV segment)
V8_TAIL_MIN = int(os.environ.get("V8_TAIL_MIN", "192"))   # a shorter rest after an aligned snapshot goes to the bridge


class V8Engine(S.Engine):
    def __init__(self, a, v):
        super().__init__(a)
        from v8_engine import V8Prefill
        self.v8min, self.v8err = v.v8_min, 0
        t = time.time()
        self.v8 = V8Prefill(self.model, Path(a.model_dir), v.v8_gpu_manifest, PIPE / "engine", dylib=v.v8_dylib,
                            streams=v.v8_streams, kv8=bool(v.v8_kv8), k8v16=bool(v.v8_k8v16), attproj_from=v.v8_attproj_from,
                            lmax=max(self.model.ladder), log=S.log)
        S.log(f"v8: prefill engine ready in {time.time() - t:.0f}s (V8 for new spans >= {self.v8min} tokens)")

    def _feed_blocks(self, ids_, ingest):
        """ids_ (> T tokens) through the bridge as prefill blocks only, so nothing is left pending (its feed ends a
        rest of <= T rows with a T-token call + accept, which leaves pending rows and keeps V8 from taking over).
        Pieces of tp() rows; a rest in (tp, tp + T] is split in two halves (each > T)."""
        T, logits, k = self.model.T, None, 0
        assert len(ids_) > T, len(ids_)
        while k < len(ids_):
            r, tp = len(ids_) - k, self.model.tp() or 64
            take = r if r <= tp else (r // 2 if r <= tp + T else tp)
            logits = self._feed(ids_[k:k + take], ingest)
            k += take
        return logits

    def _v8_segments(self, ids, i, end, cuts, ingest, done):
        """V8 over [i, end) in one run (the last pair padded); bridge snapshots on the aligned cuts in between, taken
        without stopping. Returns (position, logits)."""
        sn = {}
        for cut, key in sorted(cuts):
            a_cut = i + (cut - i) // A2 * A2 if cut >= i else cut   # pair ends of a run from i
            if i < a_cut < end:
                sn.setdefault(a_cut, []).append((key, cut))
            elif a_cut == i and i > 0 and key not in self.snaps:   # the state is there already
                self._snapshot(key, ids[:a_cut]); done.add(key)

        def on_snap(pos, snap):
            for key, cut in sn[pos]:
                self.snaps[key] = {"ids": list(ids[:pos]), **snap}
                if self.drafter is not None:
                    self.snaps[key]["draft"] = (self.drafter.slot_pos.copy(), list(self.drafter.pending))
                done.add(key)
                S.log(f"v8: snapshot '{key}' at {pos} (cut {cut}), taken during the run")
        st = self.v8.run(ids[i:end], i, ingest, snap_at=sorted(sn), on_snap=on_snap)
        self.fed.extend(ids[i:end])
        S.log(f"v8: {st['tokens']} tok ({st['padded']} pad) in {st['seconds']:.1f}s ({st['tok_s']:.0f} tok/s) + "
              f"{1e3 * st['post_s']:.0f} ms handover / drafter context | setup {1e3 * st['setup_s']:.0f} ms "
              f"(V fill {1e3 * st['fill_s']:.0f})")
        return end, st["logits"]

    def _v8_start(self, i, end):
        """Where V8 can take over from the bridge at position i (feeding > T rows to get there, a prefill block):
        i itself when aligned and nothing is pending, else the next V8_ALIGN point (128 if that run would cross a KV
        segment boundary)."""
        if not self.model.pending and self.v8.can_start(i, end - i):
            return i
        # candidates: the next 64- / 128-aligned point. A bridge prefill block costs about one V8 pair whatever its rows
        # (both ~0.2-0.6 s, growing with the context): fewest blocks + pairs, then fewer pairs
        tp = self.model.tp() or 64
        cand = [a_i for al in sorted({V8_ALIGN, A2}) for a_i in [(i + 9 + al - 1) // al * al] if self.v8.can_start(a_i, end - a_i)]
        if not cand:
            return (i + 9 + A2 - 1) // A2 * A2
        def cost(a_i):
            r, pairs = a_i - i, -(-(end - a_i) // A2)
            blocks = r // tp + (0 if r % tp == 0 else 1 if r % tp > self.model.T else 1)   # _feed_blocks' pieces
            return (blocks + pairs, pairs)
        return min(cand, key=cost)

    def prefill(self, ids):
        """The release server's prefill (same cache / snapshot / drafter semantics) with the V8 path for long spans."""
        n = len(self.fed)
        if n and not (len(ids) > n and ids[:n] == self.fed):
            j = next((i for i, (a_, b_) in enumerate(zip(ids, self.fed)) if a_ != b_), min(len(ids), n))
            show = lambda t: self.tok.decode(t[max(0, j - 12):j + 24], skip_special_tokens=False)  # noqa: E731
            print(f"[cache] model state {n} tok, prompt {len(ids)} tok, first difference at {j}:\n"
                  f"   state : {show(self.fed)!r}\n   prompt: {show(ids)!r}", flush=True)
        best = max((s_ for s_ in self.snaps.values()
                    if len(ids) > len(s_["ids"]) and ids[:len(s_["ids"])] == s_["ids"]),
                   key=lambda s_: len(s_["ids"]), default=None)
        cold = False
        if n and len(ids) > n and ids[:n] == self.fed and (best is None or n >= len(best["ids"])):
            reused, how = n, "continue (model state is a prefix of the prompt)"
        elif best is not None:
            key = next(k for k, v in self.snaps.items() if v is best)
            self._restore(best)
            reused, how = len(self.fed), f"restore '{key}' snapshot"
        else:
            self.model.reset()
            if self.drafter is not None:
                self.drafter.reset()
            self.fed, reused, how, cold = [], 0, "cold (no reusable state)", True
        S.log(f"request: prompt {len(ids)} tok | cache: {how}, reused {reused} | prefilling {len(ids) - reused} tok ...")
        t_pre = time.perf_counter()
        turn_cut = max((i for i, t in enumerate(ids) if t == self.im_start), default=0)
        h = self.user_hdr
        sys_cut = next((i for i in range(len(ids) - len(h) + 1) if ids[i:i + len(h)] == h), 0)
        cuts = []
        if sys_cut > reused and ("system" not in self.snaps or self.snaps["system"]["ids"] != ids[:sys_cut]):
            cuts.append((sys_cut, "system"))
        if turn_cut > reused:
            cuts.append((turn_cut, "turn"))
        end = len(ids)   # V8 runs to the end (the last pair padded); snapshots on aligned cuts
        use_v8 = self.v8 is not None and end - reused >= self.v8min
        v8_i = 0 if cold else (self._v8_start(reused, end) if use_v8 else reused)
        ingest = None
        if self.drafter is not None:   # the rows a draft or snapshot can see, for exact and for aligned cuts
            ends = [c for c, _ in cuts] + [v8_i + (c - v8_i) // A2 * A2 for c, _ in cuts if c >= v8_i] + [len(ids)]
            ingest = S.DraftIngestWindows(self.drafter, ends) if os.environ.get("DRAFT_INGEST", "window") != "all" else \
                self.drafter.add_context
        logits, i, done = None, reused, set()
        if use_v8:
            try:
                if not cold:   # continue from the bridge's state: aligned and nothing pending, else feed up to there
                    if v8_i > i:
                        a_i = v8_i   # > T rows to the next aligned point: prefill blocks only
                        if a_i < end:
                            tb = time.perf_counter()
                            logits = self._feed_blocks(ids[i:a_i], ingest); self.fed.extend(ids[i:a_i])
                            S.log(f"v8: bridge {a_i - i} rows to {a_i} in {1e3 * (time.perf_counter() - tb):.0f} ms")
                            i = a_i
                            if self.model.pending:
                                S.log(f"v8: {self.model.pending} rows still pending at {i}: the bridge prefills the rest")
                    if not self.model.pending and self.v8.can_start(i, end - i):
                        tb = time.perf_counter()
                        self.v8.sync_from_bridge()
                        S.log(f"v8: sync at {i} in {1e3 * (time.perf_counter() - tb):.0f} ms")
                if not self.model.pending and self.v8.can_start(i, end - i) and end - i >= self.v8min:
                    i, lg = self._v8_segments(ids, i, end, cuts, ingest, done)
                    if lg is not None:
                        logits = lg
            except Exception as e:  # noqa: BLE001  the bridge's state stays at self.fed: continue there
                self.v8err += 1
                S.log(f"v8: failed ({type(e).__name__}: {str(e)[:160]}); the bridge continues from {len(self.fed)}")
                i = len(self.fed)
                self.v8.gpu_pos = None
        progress = S.PrefillProgress(len(ids) - i, i)
        for cut, key in sorted(cuts) + [(len(ids), None)]:
            if key in done:
                continue
            while cut > i:
                j = min(cut, i + progress.slice) if progress.slice else cut
                logits = self._feed(ids[i:j], ingest)
                self.fed.extend(ids[i:j])
                i = j
                progress.update(i)
            if key and i == cut:   # only at the exact position (a cut V8 already passed keeps its aligned snapshot or none)
                self._snapshot(key, ids[:cut])
        dt = time.perf_counter() - t_pre
        self.prefill_stats = {"tokens": len(ids) - reused, "seconds": dt,
                              "tokens_per_second": (len(ids) - reused) / max(dt, 1e-9),
                              "cached_tokens": reused,
                              "logits_finite": bool(np.isfinite(logits).all()) if logits is not None else None}
        S.log(f"prefill: {len(ids) - reused} tok in {dt:.1f}s ({(len(ids) - reused) / max(dt, 1e-9):.0f} tok/s) | decoding ...")
        return logits, reused


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--v8-min", type=int, default=256, help="V8 for new spans of at least this many tokens")
    p.add_argument("--v8-kv8", type=int, default=1, help="1: the GPU's attention on the ANE's INT8 KV cache (KV8)")
    p.add_argument("--v8-attproj-from", type=int, default=None, help="attention projections on the ANE from this position")
    p.add_argument("--v8-k8v16", type=int, default=1, help="1 (with --v8-kv8 1): the GPU keeps an fp16 copy of V for its "
                   "PV (+32 KB / token GPU memory, the KV8 core's 5-6%% at 64K+ back)")
    p.add_argument("--v8-streams", type=int, default=8)
    p.add_argument("--v8-dylib", default="libgpu_v8_srv2.dylib")
    p.add_argument("--v8-gpu-manifest", type=Path, default=PIPE / "w_shared2r/manifest_best.json")
    v, rest = p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    sys.stdout, sys.stderr = S.SurvivingStream(sys.stdout), S.SurvivingStream(sys.stderr)
    import coreai_compile_guide as G
    G.watch_parent()
    a = S.parse()
    S.startup_banner(a)
    sys.path.insert(0, "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4")
    import hashlib, json
    import deploy_journal as DJ   # every Core AI load is journaled (target build + drafter)
    man = json.loads((Path(a.model_dir) / "manifest.json").read_text())
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": "server", "mode": "v8_server"}, "v8srv-%d-%d" % (os.getpid(), int(time.time())))
    dev = DJ.m6_device()
    pkg = {"path": str(a.model_dir), "label": f"V8 server (port {a.port}): target build + drafter",
           "files": [c["file"] for c in man["chunks"]] + [man["head"]["file"]] + ([a.draft] if a.draft else [])}
    aid = J.attempt(dev, pkg, functions=["all"], request={"compute": "ane"})
    t0 = time.time()
    try:
        engine = V8Engine(a, v)
    except Exception as e:  # noqa: BLE001
        J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
    J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
    server = S.ThreadingHTTPServer((a.host, a.port), S.make_handler(engine))
    print(f"[{S.MODEL_ID}] serving OpenAI API on http://{a.host}:{a.port}/v1 (V8 prefill, "
          f"{'speculative ON' if a.draft else 'speculative OFF'}, ctx {a.ctx})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
