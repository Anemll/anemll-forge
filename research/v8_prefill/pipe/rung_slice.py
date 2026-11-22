"""Sliced decode entries for the V8 server (22 Nov 2026, research/v8_prefill/results/22nov, NEXT.md item 1 step B).
RESEARCH ONLY: installed by tree_verify.install when RUNG_SLICE is set; unset, nothing here runs (the served path).
On a build with v8s_<w>k / t8s_<w>k entries (v8/build_dec8t_chunk.py TREE_SLICED, v8/build_dec8ts_all.sh), a verify
call whose rows all lie below a sliced width w runs the sliced entry instead of the ladder entry it slices:
  chain   mb.call (non-split, <= 8 rows)   v8_<ctx>k -> v8s_<w>k
  tree    Tree.call (non-split)            t8_<ctx>k -> t8s_<w>k
when the runtime's current entry mb.ctx is the one w binds (24K: 32K; 40K / 48K / 56K: 64K) and p0 + 8 <= w; the
smallest such w. A sliced entry binds the runtime's KV segment buffers of that entry (same names, same shapes) and its
own (1, w) mask, filled exactly as the runtime fills its mask (_common: -1e4, then 0 below p0). The 22 Nov probe
(results/22nov/rung_probe): bit-exact against the bigger entry with keys >= w masked off.
Env: RUNG_SLICE=24,48 (widths in K, each in the build), RUNG_SLICE_CHECK=1 (parity: every sliced call also runs the
bigger entry on the same inputs, records max |logit diff|, KL(big || sliced) and argmax agreement per row, and keeps the
bigger entry's outputs, i.e. the served path's).
Per request one log line: calls per entry, and in check mode the parity stats."""
import gc, hashlib, inspect, os

import numpy as np

WIDTHS = sorted(int(x) * 1024 for x in os.environ.get("RUNG_SLICE", "").split(",") if x)
CHECK = os.environ.get("RUNG_SLICE_CHECK", "0") == "1"
CALL_SHA = "c9994f2c35817642"   # CoreAIQwenBridge.call as replicated in Slicer.chain_call (22 Nov 2026)


def _sha_call(mb):
    src = inspect.getsource(type(mb).call)
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def _kl_stats(lb, ls):
    """per row: max |lb - ls|, KL(softmax lb || softmax ls), argmax equal"""
    a, b = np.asarray(lb, np.float64), np.asarray(ls, np.float64)
    pa = a - a.max(1, keepdims=True); pa = pa - np.log(np.exp(pa).sum(1, keepdims=True))
    pb = b - b.max(1, keepdims=True); pb = pb - np.log(np.exp(pb).sum(1, keepdims=True))
    kl = (np.exp(pa) * (pa - pb)).sum(1)
    return float(np.abs(a - b).max()), float(kl.max()), int((a.argmax(1) == b.argmax(1)).sum()), a.shape[0]


class Slicer:
    def __init__(self, eng, tree, log=print):
        mb = eng.model
        self.mb, self.tree, self.log, self.split = mb, tree, log, getattr(eng, "dec8", None)
        assert WIDTHS, "RUNG_SLICE is empty"
        sha = _sha_call(mb)
        assert sha == CALL_SHA, f"runtime call changed ({sha} != {CALL_SHA}): update Slicer.chain_call first"
        self.bind = {}
        for w in WIDTHS:
            self.bind[w] = min(c for c in mb.ladder if c > w)
            for kind in ("v8s", "t8s"):
                nm = f"{kind}_{w // 1024}k"
                missing = [i for i, ch in enumerate(mb.chunks) if nm not in ch["fns"]]
                assert not missing, f"{nm} missing in chunks {missing}: not a sliced build (v8/build_dec8ts_all.sh)"
        self.masks = {}
        for w in WIDTHS:   # one (1, w) mask per width, shared by every chunk's v8s / t8s entries (layouts checked)
            b = mb.chunks[0]["fns"][f"t8s_{w // 1024}k"].buffer("input", "mask")
            assert tuple(b.shape) == (1, w), (w, b.shape)
            for ch in mb.chunks:
                for kind in ("v8s", "t8s"):
                    f = ch["fns"][f"{kind}_{w // 1024}k"]
                    assert list(f.spec("input", "mask")["strides"]) == list(b.strides), (kind, w)
                    for n in ("anc", "ctap") if kind == "t8s" else ():
                        assert list(f.spec("input", n)["strides"]) == list(mb.inp[n][0].strides), (kind, w, n)
            self.masks[w] = (b, b.np)
        self.req = {}
        self.chk = [0, 0.0, 0.0, 0, 0]   # calls, max |diff|, max KL, argmax equal rows, rows
        self.log(f"slice: BANNER sliced decode entries {[w // 1024 for w in WIDTHS]}K (binding the "
                 f"{sorted({b // 1024 for b in self.bind.values()})}K entries' KV) | chain v8s + tree t8s | check "
                 f"{'on (bigger entry kept: served outputs)' if CHECK else 'off'} | runtime call sha {sha} | build {mb.root}")

    def pick(self, p0):
        mb = self.mb
        if self.split is not None and self.split.use(p0):
            return None
        for w in WIDTHS:
            if self.bind[w] == mb.ctx and p0 + mb.T <= w:
                return w
        return None

    def _count(self, e):
        self.req[e] = self.req.get(e, 0) + 1

    def _plan_sliced(self, w, entry, fill):
        """fill the inputs (they write mb.mask: the sliced one) and get the plan bound to the sliced mask"""
        mb = self.mb
        full = mb.mask
        mb.mask = self.masks[w]
        try:
            fill()
            return mb._plan(entry)
        finally:
            mb.mask = full

    def _check(self, big, p0, n, which):
        """parity: run the bigger entry on the same inputs (its own full mask), compare, keep its outputs"""
        mb = self.mb
        ls = np.array(mb.logits[:n])
        m = mb.mask[1]
        m[:] = -1e4
        m[0, :p0] = 0
        mb._plan(big)[which].run()
        lb = np.array(mb.logits[:n])
        d, kl, eq, rows = _kl_stats(lb, ls)
        c = self.chk
        c[0] += 1; c[1] = max(c[1], d); c[2] = max(c[2], kl); c[3] += eq; c[4] += rows
        return big

    def chain_call(self, ids, logits=True):
        mb = self.mb
        T, n, p0 = mb.T, len(ids), mb.pos
        w = self.pick(p0) if 0 < n <= T and mb.fit(p0 + n) else None
        if w is None:
            self._count(f"v8_{mb.ctx // 1024}k")
            return self.prev_call(ids, logits)
        entry = f"v8s_{w // 1024}k"
        d = mb.inp

        def fill():
            xw = d["x"][1]
            xw[:] = 0
            xw[0, :, 0, :n] = mb.emb[ids].T
            mb._common(d, p0, n, T)
        self._plan_sliced(w, entry, fill)[2 if logits else 0].run()
        last = entry
        if CHECK and logits:
            last = self._check(f"v8_{mb.ctx // 1024}k", p0, n, 2)
        mb._flip()
        mb._last = last
        mb.stats["calls"] += 1
        mb._n, mb._prefill = n, False
        self._count(entry)
        return np.array(mb.logits[:n]) if logits else None

    def tree_call(self, tok, depth, anc, ctap, lo, hi):
        mb, t = self.mb, self.tree
        p0 = mb.pos
        w = self.pick(p0) if mb.fit(p0 + 8) else None
        if w is None:
            self._count(f"t8_{mb.ctx // 1024}k")
            return self.prev_tree_call(tok, depth, anc, ctap, lo, hi)
        entry = f"t8s_{w // 1024}k"
        self._plan_sliced(w, entry, lambda: t._inputs(tok, depth, anc, ctap))[2].run()
        last = entry
        if CHECK:
            last = self._check(f"t8_{mb.ctx // 1024}k", p0, 8, 2)
        mb._flip()
        mb._last = last
        mb.stats["calls"] += 1
        mb._n, mb._prefill = 8, False
        self._count(entry)
        return np.array(mb.logits[:8]), False

    def release(self, ctx):
        """after the runtime releases a left entry: the sliced entries binding it too (they hold its KV buffers)"""
        self.prev_release(ctx)
        names = [f"{k}_{w // 1024}k" for w, b in self.bind.items() if b == ctx for k in ("v8s", "t8s")]
        if not names:
            return
        mb = self.mb
        left = [(ch, e) for ch in mb.chunks for e in names]
        for ch, e in left:
            del ch["fns"][e]
        gc.collect()
        for ch, e in left:
            ch["fns"][e] = ch["model"].function(ch["names"][e])

    def request_end(self):
        try:
            if self.req:
                s = ", ".join(f"{e} {n}" for e, n in sorted(self.req.items()))
                c = self.chk
                ck = (f" | check: {c[0]} sliced calls, max |logit diff| {c[1]:.3g}, max KL {c[2]:.3g}, argmax equal "
                      f"{c[3]}/{c[4]} rows" if CHECK else "")
                self.log(f"slice: verify calls {s}{ck}")
        finally:
            self.req = {}
            self.chk = [0, 0.0, 0.0, 0, 0]
            self.prev_request_end()


def install(eng, tree, log=print):
    s = Slicer(eng, tree, log)
    mb = eng.model
    s.prev_call, mb.call = mb.call, s.chain_call
    s.prev_tree_call, tree.call = tree.call, s.tree_call
    s.prev_release, mb._release_entry = mb._release_entry, s.release
    s.prev_request_end, tree.request_end = tree.request_end, s.request_end
    eng.slicer = s
    return s
