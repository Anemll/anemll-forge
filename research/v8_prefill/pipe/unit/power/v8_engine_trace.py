"""TRACE COPY (18 Nov 2026, pipe/unit/power/v8_engine_trace.py) of pipe/v8_engine.py for offline analysis only: identical
schedule, plus self.trace = per GPU job (layer, block, CPU submit, GPU start / end, CPU wait done) and per ANE call
(unit, layer, block, CPU start / end), times in time.perf_counter() seconds (GPU times converted from the Metal host
clock with an offset measured at start). The server keeps using pipe/v8_engine.py.
V8 prefill engine for the server (16 Nov 2026): the V8 schedule of v8_prefill.py (pair mode over a shared build:
every FFN on the ANE through the build's ffn2 entries, every mixer on the GPU engine, 8 streams of 64-token blocks
as 4 pairs) as an object over an already loaded CoreAIQwenBridge (the server's model; one copy of the weights).
  run(ids, p0, ingest)   positions [p0, p0 + len(ids)): len and p0 multiples of 128; continues the GPU state (call
                         sync_from_bridge first when the bridge moved on its own). KV8: the GPU writes / reads the
                         bridge's own INT8 KV, else the KV rows are handed over; every DeltaNet state is handed over at
                         the end (bridge pos = p0 + len, pending 0). The drafter's tap features (residual stream after
                         the manifest's tap layers) of the positions `ingest` wants are captured from the FFN outputs
                         and passed to ingest(feats [n, taps x 5120], positions), in position order.
  sync_from_bridge()     the bridge's DeltaNet states -> the GPU (bridge pending must be 0; KV8 only)
Attention projections on the ANE (attproj: the build's ffnq2 / offn2 entries) for blocks at positions >= attproj_from.
"""
import ctypes as C_, json, os, threading, time
from pathlib import Path

import numpy as np

NL, TP = 64, 64


class V8Prefill:
    def __init__(self, mb, build, gpu_manifest, engine_dir, dylib="libgpu_v8_srv.dylib", streams=8, kv8=True, k8v16=False,
                 attproj_from=None, lmax=98304, gpu_inflight=2, log=print):
        import coreai_bridge as B
        import qwen38_kv_cache as KC
        self.mb, self.B, self.KC, self.log, self.kv8, self.inflight = mb, B, KC, log, kv8, gpu_inflight
        self.man = json.loads((Path(build) / "manifest.json").read_text())
        self.taps = list(self.man["taps"])
        gl = json.loads(Path(gpu_manifest).read_text())["layers"]
        self.att = [L["i"] for L in gl if L["kind"] != "gdn"]
        self.gdn = [L["i"] for L in gl if L["kind"] == "gdn"]
        fns, fns2, fq, fo = {}, {}, {}, {}
        for ch in mb.chunks:
            for name, f in ch["fns"].items():
                for pre, d in (("ffn2_L", fns2), ("ffnq2_L", fq), ("offn2_L", fo), ("ffn_L", fns)):
                    if name.startswith(pre) and name[len(pre):].isdigit():
                        d[int(name[len(pre):])] = f
                        break
        assert sorted(fns2) == list(range(NL)) and sorted(fns) == list(range(NL)), "a shared build with ffn / ffn2 entries"
        assert streams % 2 == 0
        self.NS = streams
        f0 = fns[0]
        self.xa = [f0.buffer("input", "x") for _ in range(streams)]    # GPU out -> ANE in
        self.xb = [f0.buffer("output", "y") for _ in range(streams)]   # ANE out -> GPU in
        assert list(self.xa[0].strides) == [5120 * TP, TP, TP, 1] and list(self.xb[0].strides) == [5120 * TP, TP, TP, 1]
        for nm in ("xa", "xb"):
            assert list(fns2[0].buffer("input", nm).strides) == list(self.xa[0].strides), nm
        self.UNITS = [(2 * p, 2 * p + 1) for p in range(streams // 2)]
        self.uplans = [[B.Plan([fns2[i].bind({"xa": self.xa[u[0]], "xb": self.xa[u[1]]}, {"ya": self.xb[u[0]], "yb": self.xb[u[1]]})])
                        for i in range(NL)] for u in self.UNITS]
        # ---- GPU engine (in-process) ----
        if kv8:
            os.environ["KV8"] = "1"
        self.k8v16 = bool(kv8 and k8v16)   # keys INT8 from the ANE cache, the values also fp16 on the GPU (PV at the fp16 rate)
        if self.k8v16:
            os.environ["K8V16"] = "1"
        os.environ.update({"V8_MANIFEST": str(gpu_manifest), "V8_KERNELS": str(Path(engine_dir) / "kernels.metal"),
                           "NT": str(2 * TP), "LMAX": str(lmax)})
        g = C_.CDLL(str(Path(engine_dir) / dylib))
        g.v8_init.restype = C_.c_int32
        g.v8_wrap.restype, g.v8_wrap.argtypes = C_.c_int32, [C_.c_void_p, C_.c_size_t]
        g.v8_wait2.restype, g.v8_wait2.argtypes = C_.c_double, [C_.c_int32, C_.POINTER(C_.c_double)]
        g.v8_submit_pair.restype, g.v8_submit_pair.argtypes = C_.c_int32, [C_.c_int32] * 6
        g.v8_submit_pair_nv.restype, g.v8_submit_pair_nv.argtypes = C_.c_int32, [C_.c_int32] * 7
        g.v8_submit_attcore.restype, g.v8_submit_attcore.argtypes = C_.c_int32, [C_.c_int32, C_.POINTER(C_.c_int32), C_.c_int32]
        g.v8_kv.restype, g.v8_kv.argtypes = C_.c_int32, [C_.c_int32, C_.c_int32, C_.c_int32, C_.c_void_p, C_.c_void_p]
        g.v8_state.restype, g.v8_state.argtypes = C_.c_int32, [C_.c_int32, C_.c_void_p, C_.c_void_p]
        g.v8_set_state.restype, g.v8_set_state.argtypes = C_.c_int32, [C_.c_int32, C_.c_void_p, C_.c_void_p]
        g.v8_kv8_seg.restype, g.v8_kv8_seg.argtypes = C_.c_int32, [C_.c_int32] * 12
        g.v8_reset.restype = None
        if hasattr(g, "v8_kv8_v16_fill"):
            g.v8_kv8_v16_fill.restype, g.v8_kv8_v16_fill.argtypes = C_.c_double, [C_.c_int32]
        self.fill_from = hasattr(g, "v8_kv8_v16_fill_from")   # patch_fillfrom.py: only the rows the GPU copy lacks
        if self.fill_from:
            g.v8_kv8_v16_fill_from.restype, g.v8_kv8_v16_fill_from.argtypes = C_.c_double, [C_.c_int32, C_.c_int32]
            g.v8_v16_ptr.restype, g.v8_v16_ptr.argtypes = C_.c_void_p, [C_.c_int32]
        self.v16_ok = 0   # K8V16: the GPU's fp16 V copy matches the ANE cache for rows [0, v16_ok)
        self.lmax, self.fill_check = lmax, os.environ.get("V8_FILL_CHECK", "0") == "1"
        g.v8_state_out.restype = C_.c_double
        g.v8_state_out.argtypes = [C_.c_int32] + [C_.POINTER(C_.c_int32)] * 4 + [C_.c_int32, C_.c_int32]
        g.v8_state_in.restype, g.v8_state_in.argtypes = C_.c_double, [C_.c_int32] + [C_.POINTER(C_.c_int32)] * 3
        self.g = g
        log(f"v8: GPU engine {dylib}, {g.v8_init()} layers, {streams} streams, KV {'INT8 (ANE cache, KV8)' if kv8 else 'fp16 + handover'}")
        self.wa = [g.v8_wrap(x.address, x.nbytes) for x in self.xa]
        self.wb = [g.v8_wrap(x.address, x.nbytes) for x in self.xb]
        assert min(self.wa + self.wb) >= 0
        self.APJ, self.AF = set(), attproj_from
        if attproj_from is not None:   # attention layers' q / k / v / o on the ANE, fused into the neighbouring FFN entries
            APJ = {La for La in fo if La - 1 >= 0}
            assert APJ and all(La - 1 in fq for La in APJ), (sorted(fq), sorted(fo))
            La0 = min(APJ)
            self.qplans, self.oplans, self.ahs = [], [], []
            for u in self.UNITS:
                qo = {n_: fq[La0 - 1].buffer("output", n_) for n_ in ("ya", "qa", "ka", "va", "yb", "qb", "kb", "vb")}
                oi = {n_: fo[La0].buffer("input", n_) for n_ in ("oa", "ob")}
                self.qplans.append({La - 1: B.Plan([fq[La - 1].bind({"xa": self.xa[u[0]], "xb": self.xa[u[1]]}, qo)]) for La in APJ})
                self.oplans.append({La: B.Plan([fo[La].bind({"oa": oi["oa"], "ob": oi["ob"], "xa": qo["ya"], "xb": qo["yb"]},
                                                            {"ya": self.xb[u[0]], "yb": self.xb[u[1]]})]) for La in APJ})
                hs = [g.v8_wrap(qo[n_].address, qo[n_].nbytes) for n_ in ("qa", "qb", "ka", "kb", "va", "vb")] + \
                     [g.v8_wrap(oi[n_].address, oi[n_].nbytes) for n_ in ("oa", "ob")]
                assert min(hs) >= 0
                self.ahs.append((C_.c_int32 * 8)(*hs))
            self.APJ = APJ
            log(f"v8: attention projections on the ANE from position {attproj_from}")
        self.seg_ctx = None
        self.gpu_pos = 0
        h = mb.head   # the build's head (h8), for the logits after a span's last token
        self.hx = h.buffer("input", "x")
        self.hout = {n_: h.buffer("output", n_) for n_ in h.output_names}
        self.hplan = B.Plan([h.bind({"x": self.hx}, self.hout)])
        self.cbuf, self.rbuf = np.empty((3, 10240), np.float16), np.empty((48, 128, 128), np.float16)
        # the bridge's DeltaNet state buffers (both sets), wrapped once: the handover / sync run on the GPU
        self.sth = None
        try:
            exp = {"conv": [10240, 1], "rec": [16384, 128, 1], "pend": [25 * 128, 128, 1]}
            sth = []
            for par in (0, 1):
                hs = {"conv": [], "rec": [], "pend": []}
                for L in self.gdn:
                    ci, j = self._where(L)
                    for nm in ("conv", "rec", "pend"):
                        b_ = mb.chunks[ci]["stb"][par][f"{nm}{j}"]
                        assert list(b_.strides) == exp[nm], (nm, list(b_.shape), list(b_.strides))
                        h_ = g.v8_wrap(b_.address, b_.nbytes); assert h_ >= 0
                        hs[nm].append(h_)
                sth.append({nm: (C_.c_int32 * len(v))(*v) for nm, v in hs.items()})
            self.sth = sth
            self.glay = (C_.c_int32 * len(self.gdn))(*self.gdn)
            self.nconv, self.npend = 11 * 10240, 48 * 25 * 128
            log(f"v8: DeltaNet state handover on the GPU ({len(self.gdn)} layers x 2 sets wrapped)")
        except Exception as e:  # noqa: BLE001
            log(f"v8: host state handover ({type(e).__name__}: {e})")

    # ---- the bridge <-> GPU state ----
    def _where(self, i):
        ci = i // 4
        return ci, i - self.man["chunks"][ci]["layers"][0]

    def _register_segments(self):
        """KV8: every attention layer's INT8 KV segments of the bridge's current context -> the GPU engine."""
        mb = self.mb
        starts = self.KC.seg_starts(mb.kv_seg, mb.cap(mb.ctx))
        # the buffers themselves (a reset shrinks the cache and a later fit binds NEW segments, possibly at the same ctx)
        sig = tuple(mb.kv[self._where(Lk)[0]][f"k{self._where(Lk)[1]}_{gi}"][0].address for Lk in self.att for gi in range(len(starts) - 1))
        if self.seg_ctx == sig:
            return
        n = 0
        for Lk in self.att:
            ci, j = self._where(Lk)
            for gi in range(len(starts) - 1):
                bk, kn = mb.kv[ci][f"k{j}_{gi}"]; bv, vn = mb.kv[ci][f"v{j}_{gi}"]
                bks, ksn = mb.kv[ci][f"ks{j}_{gi}"]; bvs, vsn = mb.kv[ci][f"vs{j}_{gi}"]
                rows = starts[gi + 1] - starts[gi]
                assert kn.shape == (4, 256, rows) and vn.shape == (4, rows, 256) and ksn.shape == (4, rows)
                hs = [self.g.v8_wrap(b_.address, b_.nbytes) for b_ in (bk, bv, bks, bvs)]
                assert min(hs) >= 0
                assert self.g.v8_kv8_seg(Lk, starts[gi], rows, *hs, kn.strides[0], kn.strides[1], vn.strides[0], vn.strides[1],
                                         ksn.strides[0] // 2) > 0
                n += 1
        self.seg_ctx = sig
        self.log(f"v8: {n} ANE KV segments registered (ctx {mb.ctx})")

    def can_start(self, p0, n):
        """A run [p0, p0 + n) can start at p0: a 64-aligned p0 (KV8; a start at 64 mod 128 shifts every 128-row call by
        64, so its span must not cross an ANE KV segment boundary: a call's rows stay in one segment)."""
        if p0 % (2 * TP) == 0:
            return True
        if p0 % TP or not self.kv8:
            return False
        n_pad = -(-n // (2 * TP)) * 2 * TP
        bounds = self.KC.seg_starts(self.mb.kv_seg, self.mb.cap(self.mb.ladder[-1]))[1:]
        return not any(p0 < b < p0 + n_pad for b in bounds)

    def rewind(self, pos):
        """The bridge's KV rows from pos on are about to be rewritten (a restore to pos, or a cold start at 0): the
        GPU's fp16 V copy is valid below pos at most."""
        self.v16_ok = min(self.v16_ok, pos)

    def _check_v16(self, upto, f0):
        """V8_FILL_CHECK=1: the GPU's fp16 V copy against the ANE's INT8 V for rows [0, upto): every row of the last
        8192 and of [f0 - 512, upto), a sample below. Error in units of the row's INT8 scale: 0 for converted rows,
        <= 0.5 (+ fp16 rounding) for the GPU's own fp16 rows, far more for a stale row. Returns the max."""
        mb = self.mb
        starts = self.KC.seg_starts(mb.kv_seg, mb.cap(mb.ctx))
        rng = np.random.default_rng(upto)
        rows = set(range(max(0, upto - 8192), upto)) | set(range(max(0, f0 - 512), upto))
        rows |= set(int(r) for r in rng.integers(0, max(1, upto), 2048) if r < upto)
        rows = np.array(sorted(rows))
        worst = 0.0
        for Lk in self.att:
            ci, j = self._where(Lk)
            p = self.g.v8_v16_ptr(Lk)
            assert p, f"no fp16 V copy for layer {Lk}"
            vc = np.ctypeslib.as_array((C_.c_uint16 * (4 * self.lmax * 256)).from_address(p)).view(np.float16).reshape(4, self.lmax, 256)
            for gi in range(len(starts) - 1):
                sel = rows[(rows >= starts[gi]) & (rows < starts[gi + 1])]
                if not len(sel):
                    continue
                v8 = mb.kv[ci][f"v{j}_{gi}"][1][:, sel - starts[gi], :].astype(np.float32)
                vs = mb.kv[ci][f"vs{j}_{gi}"][1][:, sel - starts[gi]].astype(np.float32)
                e = np.abs(vc[:, sel, :].astype(np.float32) - v8 * vs[:, :, None]) / np.maximum(vs[:, :, None], 1e-8)
                worst = max(worst, float(e.max()))
        return worst

    def sync_from_bridge(self):
        """The bridge's DeltaNet states -> the GPU, so run() continues from the bridge's position (KV8: the KV is shared)."""
        mb = self.mb
        assert self.kv8 and mb.pending == 0, "sync needs KV8 and no pending rows"
        if self.sth is not None:
            st = self.sth[mb.par]
            assert self.g.v8_state_in(len(self.gdn), self.glay, st["conv"], st["rec"]) >= 0
            self.gpu_pos = mb.pos
            return
        for L in self.gdn:
            ci, j = self._where(L)
            cur = mb.chunks[ci]["stw"][mb.par]
            conv = np.ascontiguousarray(cur[f"conv{j}"][:3], np.float16)
            rec = np.ascontiguousarray(cur[f"rec{j}"], np.float16).reshape(48, 128, 128)
            assert self.g.v8_set_state(L, conv.ctypes.data, rec.ctypes.data) == 0
        self.gpu_pos = mb.pos

    def _handover(self, p0, n):
        mb = self.mb
        gpu_states = self.sth is not None
        if gpu_states:
            st = self.sth[mb.par]
            assert self.g.v8_state_out(len(self.gdn), self.glay, st["conv"], st["rec"], st["pend"], self.nconv, self.npend) >= 0
        for L in range(NL):
            ci, j = self._where(L)
            if L in self.att:
                if self.kv8:
                    continue
                kb, vb = np.empty((4, n, 256), np.float16), np.empty((4, n, 256), np.float16)
                assert self.g.v8_kv(L, p0, n, kb.ctypes.data, vb.ctypes.data) == 0
                self.KC.append_rows_seg(mb.kv[ci], {f"k{j}_new": kb, f"v{j}_new": vb}, [j], p0, n, mb.kv_cache_dtype,
                                        mb.kv_seg, mb.keys_t)
            elif not gpu_states:
                assert self.g.v8_state(L, self.cbuf.ctypes.data, self.rbuf.ctypes.data) == 0
                cur = mb.chunks[ci]["stw"][mb.par]
                cur[f"conv{j}"][:] = 0; cur[f"conv{j}"][:3] = self.cbuf
                cur[f"rec{j}"][:] = self.rbuf
                cur[f"pend{j}"][:] = 0
        mb.pos, mb.pending, mb.hi = p0 + n, 0, max(mb.hi, p0 + n)

    # ---- the prefill ----
    def _snap_dict(self, pos, states):
        """A bridge snapshot (CoreAIQwenBridge.snapshot's format) of position pos from captured GPU states."""
        mb = self.mb
        out = []
        for ci, ch in enumerate(mb.chunks):
            d = {}
            for nm, w in ch["curw"].items():
                j = int(nm[4:]) if nm.startswith(("conv", "pend")) else int(nm[3:])
                L = ch["layers"][0] + j
                a_ = np.zeros(w.shape, w.dtype)
                if nm.startswith("conv"):
                    a_[:3] = states[L][0]
                elif nm.startswith("rec"):
                    a_[:] = states[L][1].reshape(w.shape)
                d[nm] = a_
            out.append(d)
        return {"pos": pos, "pending": 0, "ctx": mb.ctx, "states": out}

    def _dumpj(self, units, L, cur):
        """V8_DUMP (debug, 19 Nov 2026): every GPU job's outputs (the ANE input buffers it wrote), by (layer, block)."""
        if getattr(self, "dump", None) is not None:
            for u_ in units:
                for s in u_:
                    self.dump[f"{L}_{cur[s]}"] = np.array(self.xa[s].np[0, :, 0, :], np.float16).copy()

    def run(self, ids, p0, ingest=None, decode_room=64, snap_at=(), on_snap=None):
        """Positions [p0, p0 + len(ids)): p0 a multiple of 128; a length that is not one is padded to the next pair
        (pad rows: the DeltaNet state and conv history stop at the last real token, v8_submit_pair_nv). Returns the
        stats and the logits after the last token. snap_at: aligned positions inside the span where a bridge snapshot
        is taken without stopping (each DeltaNet layer's state is copied as that layer's pair ending there completes,
        before its next pair runs); on_snap(pos, snapshot) is called in position order, after the features before pos
        went into ingest (so a drafter state taken there matches)."""
        mb, g = self.mb, self.g
        n = len(ids)
        assert n > 0 and self.can_start(p0, n), (p0, n)
        t_in = time.perf_counter()
        n_pad = -(-n // (2 * TP)) * 2 * TP
        nv_last = n - (n_pad - 2 * TP)          # real rows in the last pair (1 .. 128)
        ids = list(ids) + [ids[-1]] * (n_pad - n)
        if p0 == 0:   # a cold prefill: fresh GPU state
            g.v8_reset(); self.gpu_pos = 0
        assert self.gpu_pos == p0, f"GPU state at {self.gpu_pos}, run from {p0}: sync_from_bridge first"
        assert mb.fit(p0 + n_pad + decode_room), "context too small"
        if self.kv8:
            self._register_segments()
        t_f = time.perf_counter()
        f0 = 0
        if self.k8v16 and p0 > 0:   # the GPU's fp16 V copy of the rows behind the run, from the ANE's INT8 cache
            if self.fill_from:      # only the rows written since the copy was last valid (decode, bridge blocks)
                f0 = min(self.v16_ok, p0) // TP * TP
                assert g.v8_kv8_v16_fill_from(f0, p0) >= 0, "K8V16 fill failed"
            else:
                assert g.v8_kv8_v16_fill(p0) >= 0, "K8V16 fill failed"
            self.v16_ok = p0
        t_f = time.perf_counter() - t_f
        if self.k8v16 and p0 > 0 and self.fill_check and self.fill_from:
            tc = time.perf_counter()
            e = self._check_v16(p0, f0)
            self.log(f"v8: fill check rows [0, {p0}) (filled from {f0}): max error {e:.3f} x scale "
                     f"({'OK' if e < 0.75 else 'STALE ROWS'}, {1e3 * (time.perf_counter() - tc):.0f} ms)")
            assert e < 0.75, f"K8V16 fill check failed: {e:.3f}"
        emb = mb.emb
        blocks = [ids[i:i + TP] for i in range(0, n_pad, TP)]
        nb = len(blocks)
        kl, cl = (n - 1) // TP, (n - 1) % TP   # the last real token: block, column
        last = {}
        snap_blk = {}   # first block after a snapshot position -> position
        for sp in snap_at:
            assert (sp - p0) % (2 * TP) == 0 and p0 < sp < p0 + n, (sp, p0, n)
            snap_blk[(sp - p0) // TP] = sp
        caps = {sp: {} for sp in snap_at}   # position -> {layer: (conv [3][10240], rec [48][128][128])}
        gdn_set = set(self.gdn)
        NS = min(self.NS, nb)
        UNITS = self.UNITS[:NS // 2]
        cv = threading.Condition()
        gcount = [0] * NL
        cur, nxt = list(range(NS)), [None] * NS
        aq = [[] for _ in UNITS]
        err, left = [], [nb]
        gq, ninfl = [], [0]
        feats = {}   # block -> {tap layer: [64, 5120] fp16}
        want = (lambda a_, n_: True) if ingest is None else getattr(ingest, "wants", lambda a_, n_: True)
        APJ, AF = self.APJ, self.AF

        def start(s, k):
            cur[s] = k
            self.xb[s].np[0, :, 0, :] = emb[blocks[k]].T
            nxt[s] = 0
        for s in range(NS):
            start(s, s)

        def stop(e):
            err.append(e)
            with cv:
                left[0] = 0; cv.notify_all()

        def submitter():
            try:
                while True:
                    with cv:
                        while True:
                            if left[0] == 0:
                                gq.append(None); cv.notify_all(); return
                            ready = [u for u in UNITS if all(cur[s] is not None and nxt[s] is not None for s in u)
                                     and len({nxt[s] for s in u}) == 1 and gcount[nxt[u[0]]] == cur[u[0]]
                                     and not (cur[u[0]] in snap_blk and nxt[u[0]] in gdn_set
                                              and nxt[u[0]] not in caps[snap_blk[cur[u[0]]]])]
                            if ready and ninfl[0] < self.inflight:
                                u = min(ready, key=lambda u_: (cur[u_[0]], nxt[u_[0]])); L, k = nxt[u[0]], cur[u[0]]
                                for s in u:
                                    nxt[s] = None
                                gcount[L] = k + len(u); ninfl[0] += 1
                                break
                            cv.wait()
                    pos = p0 + k * TP
                    if getattr(self, "dump", None) is not None:   # V8_DUMP: the job's inputs (the ANE outputs it reads)
                        for s_ in u:
                            self.dump[f"in{L}_{cur[s_]}"] = np.array(self.xb[s_].np[0, :, 0, :], np.float16).copy()
                    t_sub = time.perf_counter()
                    if L in APJ and pos >= AF:
                        t = g.v8_submit_attcore(L, self.ahs[UNITS.index(u)], pos)
                    elif k == nb - 2 and nv_last < 2 * TP:   # the last pair holds padding: stop the DeltaNet state at nv
                        t = g.v8_submit_pair_nv(L, self.wb[u[0]], self.wb[u[1]], self.wa[u[0]], self.wa[u[1]], pos, nv_last)
                    else:
                        t = g.v8_submit_pair(L, self.wb[u[0]], self.wb[u[1]], self.wa[u[0]], self.wa[u[1]], pos)
                    if t < 0:
                        raise RuntimeError(f"v8_submit {L} failed {t}")
                    with cv:
                        gq.append((t, u, L, k, t_sub)); cv.notify_all()
            except Exception as e:  # noqa: BLE001
                stop(e)

        def waiter():
            ts = (C_.c_double * 2)()
            try:
                while True:
                    with cv:
                        while not gq:
                            cv.wait()
                        j = gq.pop(0)
                    if j is None:
                        return
                    t, u, L, kk, t_sub = j
                    if g.v8_wait2(t, ts) < 0:
                        raise RuntimeError(f"v8_wait {L} failed")
                    self._dumpj((u,), L, cur)
                    self.trace.append(("gpu", L, kk, t_sub, ts[0], ts[1], time.perf_counter()))
                    if L in gdn_set and kk + 2 in snap_blk:   # this layer's state at a snapshot position: copy it now
                        cb_, rb_ = np.empty((3, 10240), np.float16), np.empty((48, 128, 128), np.float16)
                        assert g.v8_state(L, cb_.ctypes.data, rb_.ctypes.data) == 0
                        with cv:
                            caps[snap_blk[kk + 2]][L] = (cb_, rb_)
                    with cv:
                        ninfl[0] -= 1; aq[UNITS.index(u)].append(L); cv.notify_all()
            except Exception as e:  # noqa: BLE001
                stop(e)

        def ane(ui):
            u = UNITS[ui]
            try:
                while True:
                    with cv:
                        while not aq[ui] and left[0] > 0:
                            cv.wait()
                        if left[0] == 0 and not aq[ui]:
                            return
                        L = aq[ui].pop(0)
                    k = cur[u[0]]
                    pl = (self.qplans[ui].get(L) or self.oplans[ui].get(L)) if APJ and p0 + k * TP >= AF else None
                    ta = time.perf_counter()
                    if self.ane_lock is not None:   # V8_ANE_LOCK (debug): one ANE call per layer function at a time
                        with self.ane_lock[L]:
                            (pl or self.uplans[ui][L]).run()
                    else:
                        (pl or self.uplans[ui][L]).run()
                    self.trace.append(("ane", L, k, ui, ta, time.perf_counter()))
                    if L in self.taps:   # the residual stream after layer L: the drafter's tap feature
                        for s in u:
                            if cur[s] * TP < n and want(p0 + cur[s] * TP, TP):
                                feats.setdefault(cur[s], {})[L] = np.array(self.xb[s].np[0, :, 0, :], np.float16).T.copy()
                    if L == NL - 1:
                        for s in u:
                            if cur[s] == kl:
                                last["h"] = np.array(self.xb[s].np[0, :, 0, cl], np.float16)
                    if L + 1 < NL:
                        with cv:
                            for s in u:
                                nxt[s] = L + 1
                            cv.notify_all()
                        continue
                    with cv:
                        for s in u:
                            left[0] -= 1
                            if cur[s] + NS < nb:
                                start(s, cur[s] + NS)
                            else:
                                cur[s] = None
                        cv.notify_all()
            except Exception as e:  # noqa: BLE001
                stop(e)
        th = [threading.Thread(target=submitter, daemon=True), threading.Thread(target=waiter, daemon=True)] + \
             [threading.Thread(target=ane, args=(ui,), daemon=True) for ui in range(len(UNITS))]
        self.trace = []
        self.dump = {} if os.environ.get("V8_DUMP") else None
        self.ane_lock = [threading.Lock() for _ in range(NL)] if os.environ.get("V8_ANE_LOCK") else None
        t0 = time.perf_counter()
        self.trace_t0 = t0
        for x in th:
            x.start()
        for x in th:
            x.join()
        if err:
            self.gpu_pos = None
            raise err[0]
        t1 = time.perf_counter()
        if self.dump is not None:
            np.savez(os.environ["V8_DUMP"] + f"_{n}.npz", **self.dump)
        self._handover(p0, n)
        self.gpu_pos = p0 + n
        self.v16_ok = p0 + n   # the run's own fp16 V rows (its pad rows beyond p0 + n are not valid)
        pend_snaps = sorted(snap_at)
        for k in range((n + TP - 1) // TP):
            while pend_snaps and pend_snaps[0] <= p0 + k * TP:   # every feature before the snapshot position is in
                sp = pend_snaps.pop(0)
                if on_snap is not None:
                    on_snap(sp, self._snap_dict(sp, caps[sp]))
            if ingest is not None and k in feats and all(L in feats[k] for L in self.taps):
                m = min(TP, n - k * TP)   # real rows of the block
                ingest(np.concatenate([feats[k][L][:m] for L in self.taps], axis=1), np.arange(p0 + k * TP, p0 + k * TP + m))
        hx = self.hx.np; hx[:] = 0; hx[0, :, 0, 0] = last["h"]
        self.hplan.run()
        logits = np.array(self.hout["logits"].np[0], np.float32)
        return {"tokens": n, "padded": n_pad - n, "seconds": t1 - t0, "post_s": time.perf_counter() - t1,
                "setup_s": t0 - t_in, "fill_s": t_f, "fill_rows": (p0 - f0) if self.k8v16 and p0 > 0 else 0,
                "tok_s": n / (t1 - t0), "logits": logits}
