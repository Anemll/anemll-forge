"""Decode-verify split, step-level timing (16 Nov 2026) on probe 2 (v8/build_dec_split_probe2.py: chunk 0's v8_16k..96k,
dva0, dvb0, dvba (fused B + next A), dvb1). A 16-chunk verify step at N keys, three ways (all ANE calls through the Swift
bridge, GPU = engine/libdec_gpu.dylib on a synthetic INT8 KV cache in the 96K ladder):
  ref      16 calls of chunk 0's verify entry for N's context entry, in ONE bridge run (how the server runs a step);
  unfused  16 x (dva0 -> GPU -> dvb0);
  fused    dva0 -> 15 x (GPU -> dvba) -> GPU -> dvb1.
The numbers are timing only (the loops feed chunk 0 / 1 entries to themselves). Every model load is journaled.
    BUILDPY dec_chain_time.py PROBE2_DIR"""
import ctypes as C_, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
for p_ in (V02 / "scripts", V02 / "coreai" / "swift_bridge", Path("/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4")):
    sys.path.insert(0, str(p_))
import ane_compile_mode, deploy_journal as DJ  # noqa: E402
ane_compile_mode.apply(log=lambda m: None)
import coreai_bridge as B  # noqa: E402

ENG = Path(__file__).resolve().parent.parent / "engine"
SEG = [8192, 8192, 16384, 32768, 32768]
KEYS = [int(x) for x in os.environ.get("KEYS", "8192,16384,32768,49152,65536,81920,98304").split(",")]
REPS = int(os.environ.get("REPS", "12"))
NCH = 16


def entry_for(n):
    for c in (16384, 32768, 65536, 98304):
        if n <= c:
            return c
    raise ValueError(n)


def main():
    probe = Path(sys.argv[1])
    info = json.loads(next(probe.glob("*.aimodel.info.json")).read_text())
    path = probe / info["file"]
    t0 = time.time()
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a", "out": str(probe),
                    "mode": "dec_chain_time"}, f"decchain-{os.getpid()}-{int(t0)}")
    dev = DJ.m6_device(); pkg = {"path": str(path), "main_hash": (path / "main.hash").read_bytes().hex(), "label": "decode split probe 2 (chunks L00-07), bridge"}
    aid = J.attempt(dev, pkg, functions=info["entries"], request={"compute": "ane", "compile_mode": os.environ.get("MPSGRAPH_ANE_BONDED_COMPILE_MODE"), "via": "coreai_bridge"})
    try:
        m = B.Model(path, compute="ane")
        F = {n: m.function(n) for n in info["entries"]}
    except Exception as e:  # noqa: BLE001
        J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
    J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    g = C_.CDLL(str(ENG / "libdec_gpu.dylib"))
    g.dec_init.restype, g.dec_init.argtypes = C_.c_int32, [C_.c_char_p]
    g.dec_wrap.restype, g.dec_wrap.argtypes = C_.c_int32, [C_.c_void_p, C_.c_size_t]
    g.dec_seg.restype, g.dec_seg.argtypes = C_.c_int32, [C_.c_int32] * 7
    g.dec_run.restype, g.dec_run.argtypes = C_.c_double, [C_.c_int32, C_.POINTER(C_.c_int32), C_.POINTER(C_.c_uint32), C_.c_int32]
    assert g.dec_init(str(ENG / "dec_kernels.metal").encode()) > 0

    def bufs(f, kind):
        return {n: f.buffer(kind, n) for n in (f.input_names if kind == "input" else f.output_names)}
    # reference plans: 16 x the chunk's verify entry in one run
    ref = {}
    for c in (16384, 32768, 65536, 98304):
        f = F[f"v8_{c // 1024}k"]; ri, ro = bufs(f, "input"), bufs(f, "output")
        ri["mask"].np[...] = 0
        b_ = f.bind(ri, ro)
        ref[c] = (B.Plan([b_] * NCH), B.Plan([b_]))
    fa, fb0, fba, fb1 = F["dva0"], F["dvb0"], F["dvba"], F["dvb1"]
    ai, ao = bufs(fa, "input"), bufs(fa, "output")
    o_buf = fba.buffer("input", "o")
    assert list(o_buf.strides) == list(fb0.spec("input", "o")["strides"]) == list(fb1.spec("input", "o")["strides"])
    sA, sB = bufs(fba, "output"), bufs(fba, "output")
    bai = {n: fba.buffer("input", n) for n in fba.input_names if n not in ("o", "xr_prev")}
    plan_a = B.Plan([fa.bind(ai, ao)])
    plan_b0 = B.Plan([fb0.bind({"o": o_buf, "x": ao["xr"]}, {"y": fb0.buffer("output", "y")})])
    ba_first = B.Plan([fba.bind({"o": o_buf, "xr_prev": ao["xr"], **bai}, sA)])
    ba_odd = B.Plan([fba.bind({"o": o_buf, "xr_prev": sA["xr"], **bai}, sB)])    # reads A, writes B
    ba_even = B.Plan([fba.bind({"o": o_buf, "xr_prev": sB["xr"], **bai}, sA)])   # reads B, writes A
    plan_b1A = B.Plan([fb1.bind({"o": o_buf, "x": sA["xr"]}, {"y": fb1.buffer("output", "y")})])
    plan_b1B = B.Plan([fb1.bind({"o": o_buf, "x": sB["xr"]}, {"y": fb1.buffer("output", "y")})])
    for s_ in (sA, sB):
        for n in ("q", "g", "kt", "vt"):
            assert list(s_[n].strides) == list(ao[n].strides), n
    st = (C_.c_uint32 * 9)(ao["q"].strides[0], ao["q"].strides[1], ao["kt"].strides[0], ao["kt"].strides[1],
                           ao["vt"].strides[0], ao["vt"].strides[1], ao["g"].strides[0], ao["g"].strides[1], o_buf.strides[1])
    oh = g.dec_wrap(o_buf.address, o_buf.nbytes)
    H = {}
    for k, s_ in (("a", ao), ("A", sA), ("B", sB)):
        H[k] = (C_.c_int32 * 5)(*[g.dec_wrap(s_[n].address, s_[n].nbytes) for n in ("q", "g", "kt", "vt")], oh)
        assert min(H[k]) >= 0
    r = np.random.default_rng(3)
    keep, start = [], 0
    for L in SEG:
        bk, bv = B.Buffer((4, 256, L), "int8"), B.Buffer((4, L, 256), "int8")
        bks, bvs = B.Buffer((4, L), "float16"), B.Buffer((4, L), "float16")
        bk.np[...] = r.integers(-127, 128, bk.shape, dtype=np.int8); bv.np[...] = r.integers(-127, 128, bv.shape, dtype=np.int8)
        bks.np[...] = 0.02; bvs.np[...] = 0.01
        keep += [bk, bv, bks, bvs]
        h4 = [g.dec_wrap(b_.address, b_.nbytes) for b_ in (bk, bks, bv, bvs)]
        assert min(h4) >= 0 and g.dec_seg(99, start, L, *h4) > 0
        start += L
    ai["x"].np[...] = (r.standard_normal(ai["x"].shape) * 0.02).astype(np.float16); ai["cos"].np[...] = 1
    bai["cos"].np[...] = 1

    def step_unfused(p0):
        for _ in range(NCH):
            plan_a.run(); g.dec_run(99, H["a"], st, p0); plan_b0.run()

    def step_fused(p0):
        plan_a.run(); g.dec_run(99, H["a"], st, p0); ba_first.run(); cur = "A"
        for k in range(1, NCH - 1):
            g.dec_run(99, H[cur], st, p0)
            (ba_odd if cur == "A" else ba_even).run(); cur = "B" if cur == "A" else "A"
        g.dec_run(99, H[cur], st, p0); (plan_b1A if cur == "A" else plan_b1B).run()

    def tm(fn, reps=REPS):
        ts = []
        for _ in range(reps):
            t1 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t1)
        return 1e3 * float(np.median(ts))
    for _ in range(3):
        step_fused(8184); step_unfused(8184); ref[16384][0].run()
    rows = []
    per_call = {c: tm(ref[c][1].run, 30) for c in ref}
    print("ANE verify entry, one chunk (ms): " + ", ".join(f"v8_{c // 1024}k {v:.2f}" for c, v in per_call.items()), flush=True)
    a_ms, b0_ms, ba_ms = tm(plan_a.run, 30), tm(plan_b0.run, 30), tm(ba_first.run, 30)
    print(f"split entries alone (ms): dva0 {a_ms:.2f}, dvb0 {b0_ms:.2f}, dvba {ba_ms:.2f}", flush=True)
    for n in KEYS:
        p0, c = n - 8, entry_for(n)
        t_ref = tm(ref[c][0].run); t_un = tm(lambda: step_unfused(p0)); t_fu = tm(lambda: step_fused(p0))
        row = {"keys": n, "entry": c, "ms_ref_step": t_ref, "ms_unfused_step": t_un, "ms_fused_step": t_fu}
        rows.append(row)
        print(f"keys {n:6d} (entry v8_{c // 1024}k): 16-chunk step ref {t_ref:7.2f} ms | split unfused {t_un:7.2f} ({100 * (t_un / t_ref - 1):+.1f}%) "
              f"| fused {t_fu:7.2f} ({100 * (t_fu / t_ref - 1):+.1f}%)", flush=True)
    (probe / "dec_chain_time.json").write_text(json.dumps({"per_call": per_call, "dva0": a_ms, "dvb0": b0_ms, "dvba": ba_ms, "rows": rows}, indent=1))


main()
