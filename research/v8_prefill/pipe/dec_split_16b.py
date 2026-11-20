"""Decode-verify split for the V8 server (17 Nov 2026) on the 8-layer build (v8/build_dec8_chunk.py, server manifest
v8/dec8_srv_manifest.py). Per 8-layer package: ANE dva (group 0 up to its attention projections) -> GPU attention
(engine/dec_kernels.metal: INT8 q, the 8 new K / V rows into the ANE's own KV segments, flash decoding, x sigmoid(gate),
o into the next call's input) -> ANE dvba (group 0's o projection + FFN, then group 1 up to its projections) -> GPU
attention -> ANE dvb (group 1's o projection + FFN). The ANE programs bind no KV: the same ones at every context. The
KV cache, DeltaNet state sets and per-call inputs are the runtime's own (CoreAIQwenBridge), so a step may run either
path; the GPU writes all 8 rows of the block at p0 .. p0 + 7 (bit-exact with the host's quantize), so accept() after a
split step only moves the position.
  patch_runtime()              before the model is built: load each package once (two logical chunks share it) and
                               map canonical entry names through the manifest's "alias"
  DecSplit(mb, build).install() after: mb.call / features / accept use the split when use(p0) says so
Env: V8_DEC8 (auto: split in the 96K entry, and in the 64K entry up to V8_DEC8_MID keys; on: always; off), V8_DEC8_MID
(0: none; the runtime check measured the 64K entry at -0.7% at 33K keys and +1.8% at 49K, the 96K entry -7.4% at 66K,
-4.5% at 82K, -2.8% at 95K).
GPU clock keep-alive (18 Nov 2026, engine/dec_gpu_clk.swift, results/18nov/gpu_clock): DEC_DYLIB (libdec_gpu_clk.dylib =
libdec_gpu.dylib + the keep-alive; libdec_gpu.dylib = without), DEC_KA_HOLD_MS (60: every split step holds the GPU clock
up to 60 ms past its last call; 0 = off), DEC_KA_CTL (a file; "off" in it turns the keep-alive off at run time, anything
else or no file = DEC_KA_HOLD_MS; checked once per split step)."""
import ctypes as C_, hashlib, json, os, time
from pathlib import Path

import numpy as np

ENG = Path(__file__).resolve().parent / "engine"
MODE = os.environ.get("V8_DEC8", "auto")
MID = int(os.environ.get("V8_DEC8_MID", "0"))
DEC_DYLIB = os.environ.get("DEC_DYLIB", "libdec_gpu_clk.dylib")
KA_HOLD = float(os.environ.setdefault("DEC_KA_HOLD_MS", "60"))   # read by libdec_gpu_clk at its first call
KA_CTL = os.environ.get("DEC_KA_CTL", "/Volumes/SSD4TB/anemll-forge-research/contention/ctl/ka")
# research (queue 16b, 19 Nov 2026; unset = the served behaviour): DEC_METAL (kernels file in engine/, default
# dec_kernels.metal), EXPECT_DEC_ATTN / EXPECT_DEC_METAL / EXPECT_DEC_DYLIB (the attention kernel name; sha256 prefixes:
# the server refuses to start on a mismatch), DEC_LOGITS_DIR (each split step's accepted rows' logits, fp16, one file
# per start position, first request only)
DEC_METAL = os.environ.get("DEC_METAL", "dec_kernels.metal")
LOGITS_DIR = os.environ.get("DEC_LOGITS_DIR")


def _mapped(name):
    lib = C_.CDLL(None)
    lib._dyld_image_count.restype = C_.c_uint32
    lib._dyld_get_image_name.restype, lib._dyld_get_image_name.argtypes = C_.c_char_p, [C_.c_uint32]
    return [p for p in (lib._dyld_get_image_name(i).decode() for i in range(lib._dyld_image_count())) if p.endswith("/" + name)]


def patch_runtime():
    """Two logical 4-layer chunks per 8-layer package: one bridge Model per package, the entries through "alias".
    Applied when the release server imports qwen38_coreai_model (an import hook: the module reads COREAI_DIR, MODEL,
    EXPORT_DIR at import, which the server sets just before), or right away when it is already imported."""
    import importlib.abc, importlib.machinery, sys
    if "qwen38_coreai_model" in sys.modules:
        return _patch(sys.modules["qwen38_coreai_model"])

    class Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != "qwen38_coreai_model":
                return None
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            run = spec.loader.exec_module

            def exec_module(module):
                run(module)
                _patch(module)
            spec.loader.exec_module = exec_module
            return spec
    sys.meta_path.insert(0, Hook())


def _patch(QM):
    orig_entries, orig_load = QM.cache_entries, QM.CoreAIQwenBridge._load
    models = {}

    def cache_entries(man, ch, mode):
        return dict(ch["alias"]) if "alias" in ch else orig_entries(man, ch, mode)

    def _load(self, file, entries, compiled=None):
        if file not in models:
            models[file] = orig_load(self, file, [], compiled)[0]
        m = models[file]
        return m, {e: m.function(e) for e in entries}
    QM.cache_entries, QM.CoreAIQwenBridge._load = cache_entries, _load


class DecSplit:
    def __init__(self, mb, build, log=print):
        self.mb, self.log = mb, log
        self.B = B = mb.B
        man = json.loads((Path(build) / "manifest.json").read_text())
        ch = man["chunks"]
        assert len(ch) == len(mb.chunks) and all("dec8" in c for c in ch), "not a dec8 server manifest"
        self.nf = len(ch) // 2
        g = C_.CDLL(str(ENG / DEC_DYLIB))
        g.dec_init.restype, g.dec_init.argtypes = C_.c_int32, [C_.c_char_p]
        g.dec_wrap.restype, g.dec_wrap.argtypes = C_.c_int32, [C_.c_void_p, C_.c_size_t]
        g.dec_seg.restype, g.dec_seg.argtypes = C_.c_int32, [C_.c_int32] * 7
        g.dec_clear.restype, g.dec_clear.argtypes = C_.c_int32, [C_.c_int32]
        g.dec_run.restype, g.dec_run.argtypes = C_.c_double, [C_.c_int32, C_.POINTER(C_.c_int32), C_.POINTER(C_.c_uint32), C_.c_int32]
        assert g.dec_init(str(ENG / DEC_METAL).encode()) > 0, "dec_init failed"
        self.g = g
        self.ka, self._ka_m = hasattr(g, "dec_ka_set_hold"), 0
        if self.ka:
            g.dec_ka_set_hold.restype, g.dec_ka_set_hold.argtypes = C_.c_double, [C_.c_double]
            g.dec_ka_get_hold.restype, g.dec_ka_get_hold.argtypes = C_.c_double, []
            g.dec_ka_info.restype, g.dec_ka_info.argtypes = None, [C_.POINTER(C_.c_double)]
        mp = _mapped(DEC_DYLIB)
        km = ENG / DEC_METAL
        if hasattr(g, "dec_variant"):
            g.dec_variant.restype, g.dec_variant.argtypes = C_.c_int32, [C_.c_char_p, C_.c_size_t]
            self._vb = C_.create_string_buffer(256); g.dec_variant(self._vb, 256); variant = self._vb.value.decode()
        else:
            self._vb, variant = None, f"attn={os.environ.get('DEC_ATTN', 'dec_attn_r4')} (served dylib: no dec_variant)"
        dsha = hashlib.sha256(Path(mp[0]).read_bytes()).hexdigest() if mp else "-"
        msha = hashlib.sha256(km.read_bytes()).hexdigest()
        log(f"dec8: BANNER decode GPU dylib {mp[0] if mp else 'NOT MAPPED'} sha256 {dsha} | kernels {km} sha256 {msha} | "
            f"{variant} | V8_DEC8={MODE}")
        att = variant.split()[0].split("=", 1)[1]
        for k, have in (("EXPECT_DEC_ATTN", att), ("EXPECT_DEC_METAL", msha), ("EXPECT_DEC_DYLIB", dsha)):
            want = os.environ.get(k)
            if want and not (have == want if k == "EXPECT_DEC_ATTN" else have.startswith(want)):
                raise SystemExit(f"dec8: ABORT {k}={want} but loaded {have}")
        self._lg, self._last_p0 = None, None
        log(f"dec8: decode GPU engine {mp[0] if mp else 'NOT MAPPED: ' + DEC_DYLIB} sha256 "
            f"{hashlib.sha256(Path(mp[0]).read_bytes()).hexdigest()[:16] if mp else '-'}, kernels {km.name} "
            f"{hashlib.sha256(km.read_bytes()).hexdigest()[:16]}; keep-alive "
            f"{(f'hold {g.dec_ka_get_hold():g} ms (DEC_KA_HOLD_MS)' if g.dec_ka_get_hold() > 0 else 'off (DEC_KA_HOLD_MS 0)') if self.ka else 'not in this dylib'}"
            f", switch {KA_CTL}")
        d = mb.inp
        self.files = []
        for j in range(self.nf):
            c0, c1 = mb.chunks[2 * j], mb.chunks[2 * j + 1]
            assert c0["model"] is c1["model"], "the two groups of a package must share one Model (patch_runtime)"
            m = c0["model"]
            fa, fba, fb = m.function("dva"), m.function("dvba"), m.function("dvb")
            F = {"fa": fa, "fba": fba, "fb": fb, "c0": c0, "c1": c1, "taps0": c0["taps"], "taps1": c1["taps"],
                 "yprev": ch[2 * j]["dec8"]["y_prev_tap"], "last0": c0["last"], "last1": c1["last"]}
            F["ao"] = {n: fa.buffer("output", n) for n in fa.output_names if not n.endswith("_out")}
            F["bao"] = {n: fba.buffer("output", n) for n in fba.output_names if not n.endswith("_out")}
            F["bo"] = {"y": fb.buffer("output", "y")}
            F["o_ba"], F["o_b"] = fba.buffer("input", "o"), fb.buffer("input", "o")
            self.files.append(F)
        # shared per-call inputs: the runtime's (written by mb._common); their layout must be the split entries' too
        fa0 = self.files[0]["fa"]
        for n in ("cos", "sin", "conv_sel", "commit", "commit_last", "x"):
            assert list(d[n][0].strides) == list(fa0.spec("input", n)["strides"]), f"input {n}: layout differs"
        for F in self.files:
            for n in ("q", "g", "kt", "vt"):
                assert list(F["ao"][n].strides) == list(F["bao"][n].strides), n
            assert list(F["o_ba"].strides) == list(F["o_b"].strides)
            sp = lambda f, n: list(f.spec("input", n)["strides"])  # noqa: E731
            assert list(F["ao"]["xr"].strides) == sp(F["fba"], "xr_prev") and list(F["bao"]["xr"].strides) == sp(F["fb"], "x")
            assert list(F["bo"]["y"].strides) == sp(fa0, "x") == list(mb.head.buffer("input", "x").strides), "y -> x layout"
        a0, o0 = self.files[0]["ao"], self.files[0]["o_ba"]
        self.st = (C_.c_uint32 * 9)(a0["q"].strides[0], a0["q"].strides[1], a0["kt"].strides[0], a0["kt"].strides[1],
                                    a0["vt"].strides[0], a0["vt"].strides[1], a0["g"].strides[0], a0["g"].strides[1], o0.strides[1])
        w = lambda b_: g.dec_wrap(b_.address, b_.nbytes)  # noqa: E731
        for F in self.files:
            F["Ha"] = (C_.c_int32 * 5)(*[w(F["ao"][n]) for n in ("q", "g", "kt", "vt")], w(F["o_ba"]))
            F["Hba"] = (C_.c_int32 * 5)(*[w(F["bao"][n]) for n in ("q", "g", "kt", "vt")], w(F["o_b"]))
            assert min(F["Ha"]) >= 0 and min(F["Hba"]) >= 0
        self._kvobjs = None   # the KV buffers registered (held: a reallocated buffer cannot reuse an id)
        self._plans = {}
        self.stats = {"steps": 0, "ane_ms": 0.0, "gpu_ms": 0.0, "wall_ms": 0.0}
        log(f"dec8: split verify ready ({self.nf} packages: dva -> GPU -> dvba -> GPU -> dvb), mode {MODE}, "
            f"64K entry up to {MID} keys")

    # ---- KV segments -> GPU (again whenever the runtime's segment buffers change) ----
    def _sync_kv(self):
        mb, g = self.mb, self.g
        objs = [v[0] for kv in mb.kv for v in kv.values()]
        if self._kvobjs is not None and len(objs) == len(self._kvobjs) and all(a is b for a, b in zip(objs, self._kvobjs)):
            return
        mode = mb.kv_cache_dtype
        assert mode == "kv8" and mb.keys_t, "the GPU kernels read INT8 K (dim-major) / INT8 V"
        for i, ch in enumerate(mb.chunks):
            j = ch["att_j"][0]
            kv, start, s = mb.kv[i], 0, 0
            g.dec_clear(i)
            while f"k{j}_{s}" in kv:
                kb, ksb, vb, vsb = (kv[f"{n}{j}_{s}"][0] for n in ("k", "ks", "v", "vs"))
                rows = kb.shape[2]
                assert list(kb.strides) == [256 * rows, rows, 1] and list(vb.strides) == [rows * 256, 256, 1], "KV layout"
                assert list(ksb.strides) == [rows, 1] and list(vsb.strides) == [rows, 1], "KV scale layout"
                h4 = [g.dec_wrap(b_.address, b_.nbytes) for b_ in (kb, ksb, vb, vsb)]
                assert min(h4) >= 0 and g.dec_seg(i, start, rows, *h4) > 0, f"dec_seg chunk {i} segment {s}"
                start += rows; s += 1
        self._kvobjs = objs

    def _plan(self, par):
        if par in self._plans:
            return self._plans[par]
        mb, B = self.mb, self.B
        d = mb.inp
        shared = {n: d[n][0] for n in ("cos", "sin", "conv_sel", "commit", "commit_last")}
        x, plans = d["x"][0], []
        for F in self.files:
            c0, c1 = F["c0"], F["c1"]
            s0, n0 = c0["stb"][par], c0["stb"][1 - par]
            s1, n1 = c1["stb"][par], c1["stb"][1 - par]
            pa = B.Plan([F["fa"].bind({"x": x, **shared, **s0}, {**F["ao"], **{f"{k}_out": v for k, v in n0.items()}})])
            pba = B.Plan([F["fba"].bind({"o": F["o_ba"], "xr_prev": F["ao"]["xr"], **shared, **s1},
                                        {**F["bao"], **{f"{k}_out": v for k, v in n1.items()}})])
            pb = B.Plan([F["fb"].bind({"o": F["o_b"], "x": F["bao"]["xr"]}, F["bo"])])
            plans.append((pa, pba, pb))
            x = F["bo"]["y"]
        head = B.Plan([mb.head.bind({"x": x}, mb.hout)])
        self._plans[par] = (plans, head)
        return self._plans[par]

    def use(self, p0):
        if MODE == "off":
            return False
        if MODE == "on":
            return True
        ctx = self.mb.ctx
        return ctx >= 98304 or (ctx == 65536 and p0 + 8 <= MID)

    # ---- the runtime's call / features / accept ----
    def install(self):
        mb, orig_call, orig_feat, orig_acc = self.mb, self.mb.call, self.mb._features, self.mb.accept

        def call(ids, logits=True):
            T, n, p0 = mb.T, len(ids), mb.pos
            if not mb.fit(p0 + n):
                raise ValueError(f"{p0 + n} positions exceed the largest context {mb.cap(mb.ladder[-1])}")
            if not self.use(p0) or p0 + T > mb.cap(mb.ctx):
                return orig_call(ids, logits)
            return self.call(ids, logits)

        def features(n):
            return self.features(n) if mb._last == "dec8" else orig_feat(n)

        def accept(k):
            if mb._last != "dec8":
                return orig_acc(k)
            assert 0 <= k <= mb._n
            if self._lg is not None:   # research: DEC_LOGITS_DIR
                p0_, ids_, lg_ = self._lg; self._lg = None
                fn = Path(LOGITS_DIR) / f"step_{p0_:07d}.npz"
                if not fn.exists():
                    np.savez(fn, p0=p0_, ids=np.array(ids_, np.int64), k=k, logits=np.asarray(lg_[:max(k, 1)], np.float16))
            mb.pending, mb.pos = k, mb.pos + k      # the GPU wrote the block's K / V rows during the step
            mb.hi = max(mb.hi, mb.pos)
        mb.call, mb._features, mb.accept = call, features, accept
        return self

    def _ka_switch(self):
        try:
            m = os.stat(KA_CTL).st_mtime_ns
        except OSError:
            m = None
        if m == self._ka_m:
            return
        self._ka_m = m
        off = False
        if m is not None:
            try:
                off = Path(KA_CTL).read_text().strip().lower() == "off"
            except OSError:
                pass
        want = 0.0 if off else KA_HOLD
        if want != self.g.dec_ka_get_hold():
            self.g.dec_ka_set_hold(want)
            self.log(f"dec8: keep-alive {'off' if want <= 0 else f'on, hold {want:g} ms'} ({KA_CTL})")

    def ka_info(self):
        """[command buffers, GPU busy s, s held] since load, and the current hold (ms); None without the keep-alive."""
        if not self.ka:
            return None
        o = (C_.c_double * 3)(); self.g.dec_ka_info(o)
        return list(o) + [self.g.dec_ka_get_hold()]

    def call(self, ids, logits=True):
        mb, g = self.mb, self.g
        T, n, p0 = mb.T, len(ids), mb.pos
        t0 = time.perf_counter()
        if self.ka:
            self._ka_switch()
        d = mb.inp
        xw = d["x"][1]
        xw[:] = 0
        xw[0, :, 0, :n] = mb.emb[ids].T
        mb._common(d, p0, n, T)
        if self._vb is not None and not (self._last_p0 is not None and self._last_p0 <= p0 <= self._last_p0 + 8):
            self.log(f"dec8: variant at p0 {p0}: {self.variant()}")   # research: the e4m3 copy's counters so far
        self._last_p0 = p0
        self._sync_kv()
        plans, head = self._plan(mb.par)
        gpu = 0.0
        for j, (pa, pba, pb) in enumerate(plans):
            F = self.files[j]
            pa.run()
            r = g.dec_run(2 * j, F["Ha"], self.st, p0)
            assert r >= 0, f"GPU attention, chunk {2 * j}: {r}"
            pba.run()
            r2 = g.dec_run(2 * j + 1, F["Hba"], self.st, p0)
            assert r2 >= 0, f"GPU attention, chunk {2 * j + 1}: {r2}"
            pb.run()
            gpu += r + r2
        if logits:
            head.run()
        mb._flip()
        mb._last = "dec8"
        mb.stats["calls"] += 1
        mb._n, mb._prefill = n, False
        s = self.stats
        s["steps"] += 1; s["gpu_ms"] += gpu; s["wall_ms"] += 1e3 * (time.perf_counter() - t0)
        out = np.array(mb.logits[:n]) if logits else None
        if LOGITS_DIR and out is not None:
            Path(LOGITS_DIR).mkdir(parents=True, exist_ok=True); self._lg = (p0, list(ids), out)
        return out

    def variant(self):
        if self._vb is None:
            return None
        self.g.dec_variant(self._vb, 256); return self._vb.value.decode()

    def features(self, n):
        taps = {}
        for F in self.files:
            for l in F["taps0"]:
                taps[l] = F["ao"][f"tap{l}"].np
            for l in F["taps1"]:
                taps[l] = F["bao"][f"tap{l}"].np
            if F["yprev"]:
                taps[F["last0"]] = F["bao"]["y_prev"].np
            if F["last1"] in self.mb.taps:
                taps[F["last1"]] = F["bo"]["y"].np
        return np.concatenate([taps[l][0, :, 0, :n].T for l in self.mb.taps], axis=1)
