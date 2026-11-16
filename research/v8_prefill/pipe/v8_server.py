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
env V8_ALIGN=64|128 (continuation start alignment), V8_REST_MIN (128: V8 after an alignment only for a rest
of at least this), V8_TAIL_T (8: a last pair with this many real rows or fewer -> one T-row bridge call), V8_DEC_SNAP (256:
decode snapshot interval, 0 off), V8_DEC_KEEP (3), V8_TEXT_MATCH (1: keep the state's own tokens where the prompt's text
equals the state's), V8_FILL_CHECK (1: check the GPU's fp16 V copy against the ANE cache before every warm run)."""
import argparse, bisect, codecs, itertools, os, sys, time
from pathlib import Path

V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
PIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(V02 / "scripts"))
sys.path.insert(0, str(PIPE))
import numpy as np  # noqa: E402

import qwen38_server as S  # noqa: E402

A2 = 128   # V8 granularity: a pair of 64-token blocks
V8_ALIGN = int(os.environ.get("V8_ALIGN", "64"))   # V8 starts on this alignment (64: unless the span crosses a KV segment)
V8_REST_MIN = int(os.environ.get("V8_REST_MIN", "128"))   # V8 after an alignment only for a rest of at least this
V8_TAIL_T = int(os.environ.get("V8_TAIL_T", "8"))   # a last pair with <= this many real rows: one T-row bridge call instead
DEC_SNAP = int(os.environ.get("V8_DEC_SNAP", "256"))   # a snapshot every this many decoded tokens (0: off)
DEC_KEEP = int(os.environ.get("V8_DEC_KEEP", "3"))     # decode snapshots kept (the latest; about 97 MB each)
TEXT_MATCH = os.environ.get("V8_TEXT_MATCH", "1") == "1"   # reuse the state's own tokens where the prompt's text equals it


def _bytes_to_unicode():
    """GPT-2 byte-level BPE: byte -> printable character (the tokenizer's token strings)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


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
        # token bytes (byte-level BPE): the text match and the decode's UTF-8 hold-back
        self._bd = {c: b for b, c in _bytes_to_unicode().items()}
        self._added = {i: t.content for i, t in self.tok.added_tokens_decoder.items()}
        self._tbc = {}
        self._dec_last, self._dec_slot, self._dec_n, self._dec_ms = 0, -1, 0, 0.0
        probe = "<|im_start|>user\nprobe: café ⇔ 你好 \U0001f600 <tool_call>\n</tool_call><|im_end|>\n"
        pid = self.tok.encode(probe, add_special_tokens=False)
        self.text_match = TEXT_MATCH and b"".join(self._tb(t) for t in pid) == probe.encode() and \
            self.tok.decode(pid, skip_special_tokens=False) == probe
        S.log(f"v8: text-level prompt match {'on' if self.text_match else 'OFF (token bytes do not round-trip)'}; "
              f"decode snapshots every {DEC_SNAP} tokens (last {DEC_KEEP}); V8 rest >= {V8_REST_MIN}, "
              f"tail <= {V8_TAIL_T} rows on the bridge")

    # ---- token bytes ----
    def _tb(self, t):
        b = self._tbc.get(t)
        if b is None:
            s_ = self._added.get(t)
            if s_ is not None:
                b = s_.encode()
            else:
                ts = self.tok.convert_ids_to_tokens(int(t))
                try:
                    b = bytes(self._bd[c] for c in ts)
                except KeyError:
                    b = self.tok.decode([int(t)], skip_special_tokens=False).encode()
            self._tbc[t] = b
        return b

    def _utf8_open(self, ids):
        """The token bytes of ids end inside a UTF-8 character (its bytes not all generated yet)."""
        d = codecs.getincrementaldecoder("utf-8")("replace")
        d.decode(b"".join(self._tb(t) for t in ids[-4:]), final=False)
        return bool(d.getstate()[0])

    # ---- prompt: the state's own tokens where the text is the same ----
    def render(self, messages, tools, kw):
        """The release rendering; when the model state is not a token prefix of it but the prompt's text starts with
        the state's text for more tokens (the client sent back the model's own output, re-tokenized differently),
        the state's tokens up to there + the rest of the text tokenized."""
        ids = super().render(messages, tools, kw)
        f, n = self.fed, len(self.fed)
        if not self.text_match or not n or (len(ids) > n and ids[:n] == f):
            return ids
        t0 = time.perf_counter()
        j = next((i for i, (a_, b_) in enumerate(zip(ids, f)) if a_ != b_), min(len(ids), n))
        try:
            pb = b"".join(self._tb(t) for t in ids)
            parts = [self._tb(t) for t in f]
            fb = b"".join(parts)
            mf, mp = memoryview(fb), memoryview(pb)
            lo, hi = 0, min(len(fb), len(pb))
            while lo < hi:   # the longest common byte prefix
                mid = (lo + hi + 1) // 2
                if mf[:mid] == mp[:mid]:
                    lo = mid
                else:
                    hi = mid - 1
            off = list(itertools.accumulate(len(p_) for p_ in parts))
            k = bisect.bisect_right(off, lo)            # state tokens wholly inside the common text
            while k > 0 and off[k - 1] < len(pb) and (pb[off[k - 1]] & 0xC0) == 0x80:
                k -= 1                                   # the rest must start on a character
            if k <= j:
                return ids
            s0 = off[k - 1]
            new = list(f[:k]) + self.tok.encode(pb[s0:].decode("utf-8"), add_special_tokens=False)
            if b"".join(self._tb(t) for t in new) != pb or \
                    [t for t in new if t in self._added] != [t for t in ids if t in self._added]:
                S.log(f"[cache] text match rejected (bytes or special tokens differ); token match {j} of {n}")
                return ids
        except Exception as e:  # noqa: BLE001  the release rendering stands
            S.log(f"[cache] text match failed ({type(e).__name__}: {str(e)[:100]})")
            return ids
        S.log(f"[cache] text match: the state's tokens kept for {k} of {n} (token match stopped at {j}); prompt "
              f"{len(ids)} -> {len(new)} tok ({1e3 * (time.perf_counter() - t0):.0f} ms)")
        return new

    # ---- decode: UTF-8 hold-back, decode snapshots, snapshot validity ----
    def generate(self, *args, **kwargs):
        orig = self.tok.decode

        def dec(ids, *a_, **k_):   # a character split across tokens is not streamed until its last byte is in
            s_ = orig(ids, *a_, **k_)
            if s_.endswith("�") and len(ids) and self._utf8_open(ids):
                s_ = s_[:-1]
            return s_
        self.tok.decode = dec
        self._dec_n, self._dec_ms = 0, 0.0
        try:
            return super().generate(*args, **kwargs)
        finally:
            try:
                del self.tok.decode
            except AttributeError:
                pass
            self._prune()
            if self._dec_n:
                S.log(f"decode snapshots: {self._dec_n} taken ({self._dec_ms:.0f} ms); kept: {', '.join(sorted(self.snaps))}")

    def draft_cycle(self, *args, **kwargs):
        new, committed = super().draft_cycle(*args, **kwargs)
        if DEC_SNAP > 0:
            pos = len(self.fed) + len(committed)
            if pos - self._dec_last >= DEC_SNAP and self.model.pos == pos:
                t = time.perf_counter()
                self._dec_slot = (self._dec_slot + 1) % DEC_KEEP
                self._snapshot(f"dec{self._dec_slot}", self.fed + list(committed))
                self._dec_last = pos
                self._dec_n += 1
                self._dec_ms += 1e3 * (time.perf_counter() - t)
        return new, committed

    def _prune(self):
        """Snapshots whose KV rows were rewritten since (not a prefix of the tokens now in the cache: another
        conversation, a summary request, a different continuation) are dropped: restoring one would pair its
        DeltaNet state with another prompt's KV rows."""
        f = self.fed
        bad = [k for k, s_ in self.snaps.items() if not (len(s_["ids"]) <= len(f) and f[:len(s_["ids"])] == s_["ids"])]
        for k in bad:
            del self.snaps[k]
        if bad:
            S.log(f"snapshots dropped (their KV rows were rewritten): {', '.join(sorted(bad))}")

    # ---- prefill ----
    def _feed_blocks(self, ids_, ingest):
        """ids_ (>= 1 token) through the bridge as prefill blocks only (1 .. tp() rows each, every row committed), so
        nothing is left pending and V8 can take over. A block costs the same whatever its rows."""
        m, logits, k = self.model, None, 0
        on_f = ingest if ingest is not None else (self.drafter.add_context if self.drafter is not None else None)
        while k < len(ids_):
            r = len(ids_) - k
            if not m.fit(m.pos + min(r, m.tp() or 64)):
                raise ValueError(f"{m.pos + r} positions exceed the largest context")
            if not m.has_prefill():   # no prefill entry here: the release feed (may leave rows pending)
                logits = self._feed(ids_[k:], ingest)
                break
            take = min(r, m.tp())
            p0 = m.pos
            logits = m.prefill_block(ids_[k:k + take])
            if on_f is not None and getattr(on_f, "wants", lambda a_, n_: True)(p0, take):
                on_f(m.features_prefill(take), np.arange(p0, p0 + take))
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
              f"(V fill {1e3 * st['fill_s']:.0f} ms, {st.get('fill_rows', 0)} rows)")
        return end, st["logits"]

    def _v8_pairs(self, rest):
        """(V8 pairs, bridge T-row calls) for a V8 run over `rest` tokens: a last pair with <= V8_TAIL_T real rows
        goes to one T-row call of the bridge instead (when the run keeps at least one pair)."""
        q, rem = divmod(rest, A2)
        if 0 < rem <= V8_TAIL_T and q >= 1:
            return q, 1
        return q + (1 if rem else 0), 0

    def _v8_start(self, i, end):
        """Where V8 takes over from the bridge at position i: i itself when aligned and nothing is pending, else one of
        the next 64- / 128-aligned points (a 64 start only when the run crosses no KV segment boundary), fed by prefill
        blocks of 1 .. tp() rows. A bridge block costs about one V8 pair (both ~0.2-0.5 s, growing with the context),
        a T-row call about a third: the fewest blocks + pairs, then the fewest pairs."""
        if not self.model.pending and self.v8.can_start(i, end - i):
            return i
        tp = self.model.tp() or 64
        n64 = (i // 64 + 1) * 64
        cand = {a_ for a_ in (n64, n64 + 64, (i // A2 + 1) * A2) if a_ % V8_ALIGN == 0 or a_ % A2 == 0}
        cand = [a_ for a_ in sorted(cand) if a_ < end and self.v8.can_start(a_, end - a_)]
        if not cand:
            return (i // A2 + 1) * A2

        def cost(a_):
            pairs, tcalls = self._v8_pairs(end - a_)
            return (-(-(a_ - i) // tp) + pairs + 0.35 * tcalls, pairs)
        return min(cand, key=cost)

    def prefill(self, ids):
        """The release server's prefill (cache / snapshot / drafter semantics) with the V8 path for long spans, plus:
        'gen' snapshot at the prompt's end (exact; the bridge path no longer splits the prompt at the last header),
        decode snapshots, alignment blocks of any 1..tp() rows, V8 decided on the rest after the alignment, a
        nearly empty last pair on the bridge, and snapshots whose KV rows were rewritten dropped."""
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
        if self.v8 is not None:
            self.v8.rewind(reused)   # the KV rows from here on are rewritten: the GPU's V copy is valid below at most
        S.log(f"request: prompt {len(ids)} tok | cache: {how}, reused {reused} | prefilling {len(ids) - reused} tok ...")
        t_pre = time.perf_counter()
        turn_cut = max((i for i, t in enumerate(ids) if t == self.im_start), default=0)
        h = self.user_hdr
        sys_cut = next((i for i in range(len(ids) - len(h) + 1) if ids[i:i + len(h)] == h), 0)
        cuts = []
        if sys_cut > reused and ("system" not in self.snaps or self.snaps["system"]["ids"] != ids[:sys_cut]):
            cuts.append((sys_cut, "system"))
        if turn_cut > reused:   # only inside a V8 run (free there); 'gen' at the end covers it otherwise
            cuts.append((turn_cut, "turn"))
        end = len(ids)
        use_v8 = self.v8 is not None and end - reused >= self.v8min
        v8_i = 0 if cold else (self._v8_start(reused, end) if use_v8 else reused)
        if use_v8 and not cold and end - v8_i < V8_REST_MIN:
            S.log(f"v8: {end - v8_i} tokens after the alignment at {v8_i} (< {V8_REST_MIN}): the bridge prefills all")
            use_v8, v8_i = False, reused
        v8_end = end
        if use_v8:
            pairs, tcalls = self._v8_pairs(end - v8_i)
            if tcalls:
                v8_end = v8_i + pairs * A2
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
                        tb = time.perf_counter()
                        logits = self._feed_blocks(ids[i:v8_i], ingest); self.fed.extend(ids[i:v8_i])
                        S.log(f"v8: bridge {v8_i - i} rows to {v8_i} in {1e3 * (time.perf_counter() - tb):.0f} ms")
                        i = v8_i
                        if self.model.pending:
                            S.log(f"v8: {self.model.pending} rows still pending at {i}: the bridge prefills the rest")
                    if not self.model.pending and self.v8.can_start(i, v8_end - i):
                        tb = time.perf_counter()
                        self.v8.sync_from_bridge()
                        S.log(f"v8: sync at {i} in {1e3 * (time.perf_counter() - tb):.0f} ms")
                if not self.model.pending and v8_end > i and self.v8.can_start(i, v8_end - i):
                    i, lg = self._v8_segments(ids, i, v8_end, cuts, ingest, done)
                    if lg is not None:
                        logits = lg
                    if v8_end < end:
                        S.log(f"v8: the last {end - v8_end} rows on the bridge (one T-row call, not a padded pair)")
            except Exception as e:  # noqa: BLE001  the bridge's state stays at self.fed: continue there
                self.v8err += 1
                S.log(f"v8: failed ({type(e).__name__}: {str(e)[:160]}); the bridge continues from {len(self.fed)}")
                i = len(self.fed)
                self.v8.gpu_pos = None
        progress = S.PrefillProgress(len(ids) - i, i)
        for cut, key in sorted(c for c in cuts if c[1] != "turn") + [(len(ids), None)]:
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
        self._snapshot("gen", ids)   # the prompt's end, exact: the next request extends it (or a decode snapshot)
        self._dec_last = len(ids)
        self._prune()
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
    p.add_argument("--v8-dylib", default="libgpu_v8_srv3.dylib")
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
