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
import argparse, bisect, codecs, inspect, itertools, os, re, sys, time
from pathlib import Path

V02 = Path(os.environ.get("V8_V02", "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02"))   # the release tree
PIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(V02 / "scripts"))
sys.path.insert(0, str(PIPE))
import numpy as np  # noqa: E402

import qwen38_server as S  # noqa: E402
import qwen38_kv_cache as KC  # noqa: E402
# the ported head-free prefill blocks (gregcmartin/anemll-forge-speedup 8410a33): logits only where they are read
FEED_LOGITS_KW = "logits" in inspect.signature(S.Engine._feed).parameters


def _no_logits(fn):
    """{'logits': False} when fn takes it (the ported head-free blocks), else {}."""
    try:
        return {"logits": False} if "logits" in inspect.signature(fn).parameters else {}
    except (TypeError, ValueError):
        return {}

A2 = 128   # V8 granularity: a pair of 64-token blocks
V8_ALIGN = int(os.environ.get("V8_ALIGN", "64"))   # V8 starts on this alignment (64: unless the span crosses a KV segment)
V8_REST_MIN = int(os.environ.get("V8_REST_MIN", "128"))   # V8 after an alignment only for a rest of at least this
V8_TAIL_T = int(os.environ.get("V8_TAIL_T", "8"))   # a last pair with <= this many real rows: one T-row bridge call instead
DEC_SNAP = int(os.environ.get("V8_DEC_SNAP", "256"))   # a snapshot every this many decoded tokens (0: off)
DEC_KEEP = int(os.environ.get("V8_DEC_KEEP", "3"))     # decode snapshots kept (the latest; about 97 MB each)
TEXT_MATCH = os.environ.get("V8_TEXT_MATCH", "1") == "1"   # reuse the state's own tokens where the prompt's text equals it
# 21 Nov 2026 (fresh vs used server): the 'system' snapshot keeps the drafter's ring only as slot positions; a restore
# keeps every slot whose position still matches, so after another prompt (e.g. a cold request) rewrote those positions
# the drafter drafts from that prompt's rows (or, after a reset, from none). DRAFT_RING_SNAP: off (default, unchanged),
# log (copy the snapshot's visible ring slots; at restore log kept / other content / hidden), fix (log + write them back).
RING_SNAP = os.environ.get("DRAFT_RING_SNAP", "off")
RING_KEYS = ("system",)   # the only snapshot another prompt's rows can leave in place ('sysk' is its copy; _prune drops the rest)
# 21 Nov 2026 research switch (chat:2 leftover): COLD_SCRUB=kv makes a cold start also put every KV row back in its fresh
# state (K / V int8 0, scales KV_UNUSED_SCALE), as a new process has it. Off (empty) by default: the served behavior.
COLD_SCRUB = set(x for x in os.environ.get("COLD_SCRUB", "").split(",") if x)


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
    # ---- drafter ring in the 'system' snapshot (DRAFT_RING_SNAP) ----
    def _ring_save(self, key):
        d, sn = self.drafter, self.snaps.get(key)
        if RING_SNAP == "off" or d is None or key not in RING_KEYS or sn is None or "draft" not in sn:
            return
        sl = np.nonzero(sn["draft"][0] >= 0)[0]
        sn["draft_ring"] = (sl, [(k_[:, sl].copy(), v_[:, sl].copy()) for k_, v_ in d.ring])

    def _snapshot(self, key, ids):
        super()._snapshot(key, ids)
        self._ring_save(key)

    def _ring_restore(self, snap, cur):
        d, rs = self.drafter, snap.get("draft_ring")
        if d is None or rs is None or cur is None:
            return
        sl, kv = rs
        sp = snap["draft"][0]
        if not len(sl):   # every row of the snapshot was still pending (a system cut of <= R tokens): nothing to write
            return
        kept = cur[sl] == sp[sl]
        other = np.zeros(len(sl), bool)
        for (k_, v_), (ks, vs) in zip(d.ring, kv):
            for x_, y_ in ((k_, ks), (v_, vs)):   # axis 1 = slot (as _commit indexes the ring)
                other |= np.moveaxis(x_[:, sl] != y_, 1, -1).reshape(-1, len(sl)).any(0)
        other &= kept
        if RING_SNAP == "fix":
            for (k_, v_), (ks, vs) in zip(d.ring, kv):
                k_[:, sl] = ks
                v_[:, sl] = vs
            d.slot_pos[sl] = sp[sl]
        S.log(f"drafter ring ({RING_SNAP}): snapshot slots {len(sl)}, kept by position {int(kept.sum())} "
              f"({int(other.sum())} hold another prompt's rows), hidden {int((~kept).sum())}"
              f"{' -> all written back' if RING_SNAP == 'fix' else ''}")

    def __init__(self, a, v):
        super().__init__(a)
        from v8_engine import V8Prefill
        self.v8min, self.v8err = v.v8_min, 0
        t = time.time()
        self.v8 = V8Prefill(self.model, Path(a.model_dir), v.v8_gpu_manifest, PIPE / "engine", dylib=v.v8_dylib,
                            streams=v.v8_streams, kv8=bool(v.v8_kv8), k8v16=bool(v.v8_k8v16), attproj_from=v.v8_attproj_from,
                            lmax=max(self.model.ladder), log=S.log)
        S.log(f"v8: prefill engine ready in {time.time() - t:.0f}s (V8 for new spans >= {self.v8min} tokens)")
        self.dec8 = None
        if v.v8_dec8:   # the decode-verify split (an 8-layer dec8 build): ANE parts around a GPU attention
            from dec_split import DecSplit
            self.dec8 = DecSplit(self.model, Path(a.model_dir), log=S.log).install()
        # token bytes (byte-level BPE): the text match and the decode's UTF-8 hold-back
        self._bd = {c: b for b, c in _bytes_to_unicode().items()}
        self._added = {i: t.content for i, t in self.tok.added_tokens_decoder.items()}
        self._tbc = {}
        self._dec_last, self._dec_slot, self._dec_n, self._dec_ms = 0, -1, 0, 0.0
        self._tm_memo = None
        self._ab_init(os.environ.get("V8_AB", ""))
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
        ids0 = ids   # the release tokens (the memo maps a prefix of these)
        memo = self._tm_memo
        if memo is not None:   # a stretch where the release tokens and the state's cover the same text (last match)
            c_, s_ = memo
            if len(ids) >= len(c_) and n >= len(s_) and ids[:len(c_)] == c_ and f[:len(s_)] == s_:
                ids = s_ + ids[len(c_):]       # the same text, the state's tokens for that stretch
                if len(ids) > n and ids[:n] == f:
                    S.log(f"[cache] text match (memo, {len(s_)} state tokens): the state's tokens kept for {n} of {n}; "
                          f"prompt -> {len(ids)} tok ({1e3 * (time.perf_counter() - t0):.1f} ms)")
                    return ids
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
            # the memo: the longest release-token / state-token prefixes ending on the same byte (<= s0)
            offc = list(itertools.accumulate(len(self._tb(t)) for t in ids0))
            com = sorted(set(off[:k]) & set(offc))
            val = com[bisect.bisect_right(com, s0) - 1] if com and com[0] <= s0 else 0
            if val:
                kk, jj = bisect.bisect_left(off, val) + 1, bisect.bisect_left(offc, val) + 1
                if [t for t in f[:kk] if t in self._added] == [t for t in ids0[:jj] if t in self._added]:
                    self._tm_memo = (list(ids0[:jj]), list(f[:kk]))
        except Exception as e:  # noqa: BLE001  the release rendering stands
            S.log(f"[cache] text match failed ({type(e).__name__}: {str(e)[:100]})")
            return ids
        S.log(f"[cache] text match: the state's tokens kept for {k} of {n} (token match stopped at {j}); prompt "
              f"{len(ids)} -> {len(new)} tok ({1e3 * (time.perf_counter() - t0):.0f} ms)")
        return new

    # ---- decode A/B (V8_AB=g1l,g3l,g3e): request i takes arm i % len; g<ms> = the draft gap, l / e = the drafter's
    #      context written lazily (before the next draft: the ported default) or eagerly (right after the verify) ----
    def _ab_init(self, spec):
        self._ab = [a_ for a_ in spec.split(",") if a_]
        assert all(re.fullmatch(r"g\d+(\.\d+)?[le]", a_) for a_ in self._ab), self._ab
        self._ab_i, self._ab_eager = 0, False
        if self._ab and self.drafter is not None and hasattr(self.drafter, "flush"):
            add0 = self.drafter.add_context

            def add_context(feats, positions, flush=True):
                return add0(feats, positions, flush=flush or self._ab_eager)
            self.drafter.add_context = add_context
        if self._ab:
            S.log(f"[ab] decode A/B over requests: {', '.join(self._ab)}")

    def _ab_next(self):
        if not self._ab:
            return None
        arm = self._ab[self._ab_i % len(self._ab)]
        self._ab_i += 1
        S.DRAFT_GAP, self._ab_eager = float(arm[1:-1]) / 1e3, arm[-1] == "e"
        S.log(f"[ab] arm {arm}: draft gap {arm[1:-1]} ms, drafter context {'eager' if self._ab_eager else 'lazy'}")
        return arm

    # ---- decode: UTF-8 hold-back, decode snapshots, snapshot validity ----
    def generate(self, *args, **kwargs):
        # the release server with the ported IncrementalDetokenizer holds an incomplete character back itself (a
        # decode that drops the trailing U+FFFD would break its hold-back); else a decode that does
        patch = not hasattr(S, "IncrementalDetokenizer")
        if patch:
            orig = self.tok.decode

            def dec(ids, *a_, **k_):   # a character split across tokens is not streamed until its last byte is in
                s_ = orig(ids, *a_, **k_)
                if s_.endswith("�") and len(ids) and self._utf8_open(ids):
                    s_ = s_[:-1]
                return s_
            self.tok.decode = dec
        self._dec_n, self._dec_ms = 0, 0.0
        self._dprof_arm()
        self._ab_next()
        try:
            return super().generate(*args, **kwargs)
        finally:
            if patch:
                try:
                    del self.tok.decode
                except AttributeError:
                    pass
            self._prune()
            if self.dec8 is not None and self.dec8.stats["steps"]:
                st = self.dec8.stats
                S.log(f"dec8: {st['steps']} split verify steps, {st['wall_ms'] / st['steps']:.1f} ms each "
                      f"(GPU attention {st['gpu_ms'] / st['steps']:.1f} ms)")
                st.update(steps=0, wall_ms=0.0, gpu_ms=0.0)
            if self._dec_n:
                S.log(f"decode snapshots: {self._dec_n} taken ({self._dec_ms:.0f} ms); kept: {', '.join(sorted(self.snaps))}")
            self._dprof_log()

    # ---- DRAFT_PROF=1 (19 Nov 2026, timing only): the drafter's ANE calls per propose (body = the draft entry,
    #      head = the target's head on the drafter's 7 rows, ctx = ctx64 context calls), logged per request ----
    def _dprof_arm(self):
        d = self.drafter
        if not os.environ.get("DRAFT_PROF") or d is None or getattr(self, "_dprof", None) is not None:
            if getattr(self, "_dprof", None) is not None:
                for k in self._dprof:
                    self._dprof[k] = [0, 0.0]
            return
        self._dprof = {"body": [0, 0.0], "head": [0, 0.0], "ctx": [0, 0.0], "propose": [0, 0.0]}

        class Timed:
            def __init__(s_, inner, key):
                s_.inner, s_.key = inner, key

            def run(s_, *a_, **k_):
                t = time.perf_counter()
                r = s_.inner.run(*a_, **k_)
                e = self._dprof[s_.key]; e[0] += 1; e[1] += time.perf_counter() - t
                return r
        for attr, key in (("draft_plan", "body"), ("head_plan", "head"), ("ctx_plan", "ctx")):
            if hasattr(d, attr):
                setattr(d, attr, Timed(getattr(d, attr), key))
        p0 = d.propose
        tl = os.environ.get("DRAFT_TREE_LOG")   # tree-drafting study: per propose the candidates, their unary scores and hp
        self._dtree = [] if tl else None

        def propose(*a_, **k_):
            t = time.perf_counter()
            r = p0(*a_, **k_)
            e = self._dprof["propose"]; e[0] += 1; e[1] += time.perf_counter() - t
            if self._dtree is not None and r[0]:
                info = r[1]
                f = info._make_logits
                cl = dict(zip(f.__code__.co_freevars, (c.cell_contents for c in f.__closure__)))
                raw, rr = cl["raw"], cl.get("r")
                cand = info["cand"].numpy().astype(np.int32)
                un = np.take_along_axis(raw, cand, 1).astype(np.float32) * (1.0 if rr is None else rr[:, None])
                self._dtree.append({"anchor": int(a_[0]), "p0": int(a_[1]), "path": list(r[0]), "cand": cand,
                                    "unary": un.astype(np.float32), "hp": info["hp"].numpy().astype(np.float16)})
            return r
        d.propose = propose
        S.log("drafter: DRAFT_PROF timing armed (draft_plan / head_plan / ctx_plan / propose)")

    def _dprof_log(self):
        dp = getattr(self, "_dprof", None)
        if getattr(self, "_dtree", None):
            od = Path(os.environ["DRAFT_TREE_LOG"]); od.mkdir(parents=True, exist_ok=True)
            fn = od / f"req_{int(time.time() * 1e3)}.npz"
            tr = self._dtree
            np.savez(fn, anchor=np.array([x["anchor"] for x in tr]), p0=np.array([x["p0"] for x in tr]),
                     path=np.array([x["path"] for x in tr], np.int32), cand=np.stack([x["cand"] for x in tr]),
                     unary=np.stack([x["unary"] for x in tr]), hp=np.stack([x["hp"] for x in tr]),
                     cycles=np.array(self._dcyc, np.int64).reshape(-1, 3), tokens=np.array(self._dtok, np.int64))
            S.log(f"drafter tree log: {len(tr)} proposes -> {fn}")
            self._dtree.clear()
        if getattr(self, "_dcyc", None) is not None:
            self._dcyc, self._dtok = [], []
        if not dp or not dp["propose"][0]:
            return
        n = dp["propose"][0]
        S.log("drafter prof: " + f"{n} proposes | " + " ".join(f"{k} {1e3 * v[1] / max(1, v[0]):.2f} ms x {v[0] / n:.2f}" for k, v in dp.items())
              + f" | propose host {1e3 * (dp['propose'][1] - dp['body'][1] - dp['head'][1]) / n:.2f} ms")

    def draft_cycle(self, *args, **kwargs):
        p_ = self.model.pos
        new, committed = super().draft_cycle(*args, **kwargs)
        if getattr(self, "_dtree", None) is not None:   # (start position, accepted drafts, bonus token) + the committed stream
            if getattr(self, "_dcyc", None) is None:
                self._dcyc, self._dtok = [], []
            self._dcyc.append((p_, len(new) - 1, int(new[-1]) if new else -1))
            self._dtok += [int(t) for t in committed] + ([] if not new else [])
        return self._dc_tail(new, committed)

    def _dc_tail(self, new, committed):
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
        bad = [k for k, s_ in self.snaps.items() if "kv" not in s_   # 'sysk' carries its own KV rows
               and not (len(s_["ids"]) <= len(f) and f[:len(s_["ids"])] == s_["ids"])]
        for k in bad:
            del self.snaps[k]
        if bad:
            S.log(f"snapshots dropped (their KV rows were rewritten): {', '.join(sorted(bad))}")

    # ---- the 'system' snapshot across a prompt that rewrites its KV rows (a summary request) ----
    def _cold_scrub(self):
        """COLD_SCRUB=kv: every KV row of every attention layer back to a fresh process's state (research)."""
        if "kv" not in COLD_SCRUB or not hasattr(self.model, "kv"):
            return
        t, nb, unused = time.perf_counter(), 0, float(os.environ.get("KV_UNUSED_SCALE", "1"))
        for d in self.model.kv:
            for name, val in d.items():
                mm = re.fullmatch(r"(k|v|ks|vs)(\d+)_(\d+)", name)
                if mm:
                    val[1][:] = unused if mm[1] in ("ks", "vs") else 0
                    nb += val[1].nbytes
        S.log(f"cold scrub (COLD_SCRUB=kv): KV rows reset to the fresh state ({nb / 2 ** 20:.0f} MB, "
              f"{1e3 * (time.perf_counter() - t):.0f} ms)")

    def _kv_rows(self, n, put=None):
        """KV rows [0, n) of every attention layer's segments: a copy {(chunk, name): array} (put None), or written
        back from one. {} without a segmented KV history (nothing is kept then)."""
        m = self.model
        if not getattr(m, "kv_seg", 0) or not hasattr(m, "kv"):
            return {}
        starts = KC.seg_starts(m.kv_seg, m.cap(m.ctx))
        out = {}
        for ci, d in enumerate(m.kv):
            for name, val in d.items():
                mm = re.fullmatch(r"(k|v|ks|vs)(\d+)_(\d+)", name)
                if not mm or int(mm[3]) + 1 >= len(starts) or starts[int(mm[3])] >= n:
                    continue
                s0, s1 = starts[int(mm[3])], starts[int(mm[3]) + 1]
                a_ = val[1]
                ax = [i for i, L in enumerate(a_.shape) if L == s1 - s0]
                assert len(ax) == 1, (name, a_.shape, s1 - s0)
                sl = tuple(slice(0, min(n, s1) - s0) if i == ax[0] else slice(None) for i in range(a_.ndim))
                if put is None:
                    out[(ci, name)] = np.array(a_[sl])
                else:
                    a_[sl] = put[(ci, name)]
        return out

    def _save_sysk(self, ids):
        """Before a prompt that does not start with the 'system' snapshot's tokens rewrites the KV rows under it (Pi's
        summary request, another conversation), keep it as 'sysk' with a copy of those rows (about 33 KB per token):
        the conversation's next prompt restores it instead of a cold prefill. One is kept; its restore frees the copy."""
        s_ = self.snaps.get("system")
        if s_ is None or "kv" in s_:
            return
        ns = len(s_["ids"])
        if len(ids) >= ns and ids[:ns] == s_["ids"]:
            return
        k_ = self.snaps.get("sysk")
        if k_ is not None and k_["ids"] == s_["ids"]:
            return
        t = time.perf_counter()
        kv = self._kv_rows(ns)
        if not kv:
            return
        self.snaps["sysk"] = {**s_, "kv": kv}
        mb_ = sum(a_.nbytes for a_ in kv.values()) / 2 ** 20
        S.log(f"snapshot 'system' ({ns} tok) kept as 'sysk' with its KV rows ({mb_:.0f} MB, "
              f"{1e3 * (time.perf_counter() - t):.0f} ms): this prompt rewrites them")

    def _restore_cold(self, snap, what, e):
        """A restore that failed part way: drop that snapshot and start cold, so nothing is left with one prompt's
        tokens and state over another prompt's KV rows (prefill then reuses 0 tokens). 21 Nov review."""
        S.log(f"snapshot {what} not restored ({type(e).__name__}: {str(e)[:120]}): cold start")
        for k_ in [k_ for k_, v_ in self.snaps.items() if v_ is snap]:
            del self.snaps[k_]
        self.model.reset()
        if self.drafter is not None:
            self.drafter.reset()
        self.fed = []
        if self.v8 is not None:
            self.v8.rewind(0)
        self._cold_scrub()

    def _restore(self, snap):
        cur = self.drafter.slot_pos.copy() if RING_SNAP != "off" and self.drafter is not None else None
        kv = snap.get("kv")
        if kv is None:
            try:
                super()._restore(snap)
                self._ring_restore(snap, cur)
            except Exception as e:  # noqa: BLE001
                self._restore_cold(snap, "'system' / 'turn' / 'gen'", e)
            return
        n_ = len(snap["ids"])
        try:   # its rows come back from the copy, whatever the cache kept: first a context entry that holds them (a
            # summary request since may have shrunk the cache below them)
            if not self.model.fit(n_):
                raise ValueError(f"{n_} rows exceed the largest context")
            self.model.hi = max(self.model.hi, n_)
            super()._restore(snap)
            self._ring_restore(snap, cur)
            t = time.perf_counter()
            self._kv_rows(n_, put=kv)   # the base restore above already set tokens / state: these rows must follow
        except Exception as e:  # noqa: BLE001  start cold (prefill then reuses 0 tokens)
            self._restore_cold(snap, "'sysk'", e)
            return
        if self.v8 is not None:
            self.v8.rewind(0)   # the GPU's fp16 V copy of those rows is another prompt's
        self.snaps.pop("sysk", None)
        self.snaps["system"] = {k_: v_ for k_, v_ in snap.items() if k_ != "kv"}
        S.log(f"snapshot 'sysk': KV rows [0, {len(snap['ids'])}) written back in {1e3 * (time.perf_counter() - t):.0f} ms")

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
                logits = self._feed(ids_[k:], ingest, **({"logits": False} if FEED_LOGITS_KW else {}))
                break
            take = min(r, m.tp())
            p0 = m.pos
            logits = m.prefill_block(ids_[k:k + take], **_no_logits(m.prefill_block))
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
                    self._ring_save(key)
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
        # 21 Nov 2026: the key before _save_sysk, which replaces 'sysk' with another conversation's 'system' (two
        # conversations alternating: the lookup then found no key, StopIteration). Restoring 'sysk' rewrites those rows
        # with its own copy and becomes 'system', so the other conversation's 'system' is not kept then.
        key = next((k for k, v in self.snaps.items() if v is best), None)
        if key != "sysk":
            self._save_sysk(ids)   # before anything rewrites the KV rows under the 'system' snapshot
        cold = False
        if n and len(ids) > n and ids[:n] == self.fed and (best is None or n >= len(best["ids"])):
            reused, how = n, "continue (model state is a prefix of the prompt)"
        elif best is not None:
            self._restore(best)
            reused, how = len(self.fed), f"restore '{key}' snapshot"
        else:
            self.model.reset()
            if self.drafter is not None:
                self.drafter.reset()
            self.fed, reused, how, cold = [], 0, "cold (no reusable state)", True
            self._cold_scrub()
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
                logits = self._feed(ids[i:j], ingest, **({"logits": j == len(ids)} if FEED_LOGITS_KW else {}))
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
        pld = os.environ.get("PREFILL_LOGITS_DIR")   # deploy checks (19 Nov 2026): the prompt's last-token logits, one .npy per request
        if pld and logits is not None:
            Path(pld).mkdir(parents=True, exist_ok=True)
            fn = Path(pld) / f"prefill_{len(ids)}_{int(time.time() * 1e3)}.npy"
            np.save(fn, np.asarray(logits, np.float32)); S.log(f"prefill logits -> {fn}")
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
    p.add_argument("--v8-dec8", type=int, default=0, help="1: decode verify split (an 8-layer dec8 build, "
                   "v8/build_dec8_chunk.py): ANE parts around a GPU attention where it is faster (env V8_DEC8)")
    v, rest = p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    print(f"[{time.strftime('%H:%M:%S')}] drafter ring snapshot: {RING_SNAP} (DRAFT_RING_SNAP; 'system' keys {RING_KEYS})"
      f"{' | cold scrub: ' + ','.join(sorted(COLD_SCRUB)) if COLD_SCRUB else ''}", flush=True)
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
    if v.v8_dec8:
        from dec_split import patch_runtime
        patch_runtime()   # one Model per 8-layer package, the entries through the manifest's aliases
    try:
        engine = V8Engine(a, v)
    except Exception as e:  # noqa: BLE001
        J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
    J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
    if os.environ.get("TREE") == "1":   # tree verify (20 Nov 2026, tree_verify.py): a dec8t build's t8 / tdva / tdvba entries
        import tree_verify
        tree_verify.install(engine, S.log)
    server = S.ThreadingHTTPServer((a.host, a.port), S.make_handler(engine))
    print(f"[{S.MODEL_ID}] serving OpenAI API on http://{a.host}:{a.port}/v1 (V8 prefill, "
          f"{'speculative ON' if a.draft else 'speculative OFF'}, ctx {a.ctx})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
