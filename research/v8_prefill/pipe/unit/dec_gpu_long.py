"""Decode-verify split, timing at long contexts (16 Nov 2026): the probe chunk's dva / dvb on the ANE (Swift bridge) with
the GPU attention (engine/libdec_gpu.dylib) in between, reading a synthetic INT8 KV cache in the 96K build's segment
ladder (KV_SEG 8192, 8192, 16384, 32768, 32768; IOSurface buffers, random codes), so the GPU runs in the real interleaved
regime (an ANE call before and after every GPU call) at 8K-96K keys. Per position: wall of the chain A -> GPU -> B, its
parts, and the GPU time as standalone back-to-back calls. Every model load is journaled.
    BUILDPY dec_gpu_long.py PROBE_DIR"""
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
POS = [int(x) for x in os.environ.get("POS", "8184,16376,32760,49144,65528,81912,98296").split(",")]
REPS = int(os.environ.get("REPS", "40"))


def main():
    probe = Path(sys.argv[1])
    info = json.loads(next(probe.glob("*.aimodel.info.json")).read_text())
    path = probe / info["file"]
    t0 = time.time()
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a", "out": str(probe),
                    "mode": "dec_gpu_long"}, f"decgpulong-{os.getpid()}-{int(t0)}")
    dev = DJ.m6_device(); pkg = {"path": str(path), "main_hash": (path / "main.hash").read_bytes().hex(), "label": "decode split probe (chunk L00-03), bridge"}
    aid = J.attempt(dev, pkg, functions=["dva", "dvb", "v8_16k"], request={"compute": "ane", "compile_mode": os.environ.get("MPSGRAPH_ANE_BONDED_COMPILE_MODE"), "via": "coreai_bridge"})
    try:
        m = B.Model(path, compute="ane")
        fr, fa, fb = m.function("v8_16k"), m.function("dva"), m.function("dvb")
    except Exception as e:  # noqa: BLE001
        J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
    J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
    g = C_.CDLL(str(ENG / "libdec_gpu.dylib"))
    g.dec_init.restype, g.dec_init.argtypes = C_.c_int32, [C_.c_char_p]
    g.dec_wrap.restype, g.dec_wrap.argtypes = C_.c_int32, [C_.c_void_p, C_.c_size_t]
    g.dec_seg.restype, g.dec_seg.argtypes = C_.c_int32, [C_.c_int32] * 7
    g.dec_run.restype, g.dec_run.argtypes = C_.c_double, [C_.c_int32, C_.POINTER(C_.c_int32), C_.POINTER(C_.c_uint32), C_.c_int32]
    g.dec_run_spin.restype, g.dec_run_spin.argtypes = g.dec_run.restype, g.dec_run.argtypes
    g.dec_arm.restype, g.dec_arm.argtypes = C_.c_int32, g.dec_run.argtypes
    g.dec_fire.restype, g.dec_fire.argtypes = C_.c_int32, [C_.c_int32]
    g.dec_spin.restype, g.dec_spin.argtypes = C_.c_double, [C_.c_int32]
    g.dec_run_lat.restype, g.dec_run_lat.argtypes = C_.c_double, g.dec_run.argtypes + [C_.POINTER(C_.c_double)]
    LAT = (C_.c_double * 5)()
    g.dec_arm2.restype, g.dec_arm2.argtypes = C_.c_int32, g.dec_run.argtypes
    g.dec_fire2.restype, g.dec_fire2.argtypes = None, [C_.c_int32]
    g.dec_wait2.restype, g.dec_wait2.argtypes = C_.c_int32, [C_.c_int32, C_.c_int32]
    g.dec_kick_after.restype, g.dec_kick_after.argtypes = None, [C_.c_int32]
    assert g.dec_init(str(ENG / "dec_kernels.metal").encode()) > 0
    ri = {n: fr.buffer("input", n) for n in fr.input_names}; ro = {n: fr.buffer("output", n) for n in fr.output_names}
    ai = {n: fa.buffer("input", n) for n in fa.input_names}; ao = {n: fa.buffer("output", n) for n in fa.output_names}
    bi = {"o": fb.buffer("input", "o"), "x": ao["xr"]}; bo = {"y": fb.buffer("output", "y")}
    r = np.random.default_rng(3)
    ai["x"].np[...] = (r.standard_normal(ai["x"].shape) * 0.02).astype(np.float16)
    ai["cos"].np[...] = 1
    so = bi["o"].strides
    st = (C_.c_uint32 * 9)(ao["q"].strides[0], ao["q"].strides[1], ao["kt"].strides[0], ao["kt"].strides[1],
                           ao["vt"].strides[0], ao["vt"].strides[1], ao["g"].strides[0], ao["g"].strides[1], so[1])
    hs = (C_.c_int32 * 5)(*[g.dec_wrap(b_.address, b_.nbytes) for b_ in (ao["q"], ao["g"], ao["kt"], ao["vt"], bi["o"])])
    keep, start = [], 0
    for L in SEG:   # synthetic cache (layer id 99)
        bk, bv = B.Buffer((4, 256, L), "int8"), B.Buffer((4, L, 256), "int8")
        bks, bvs = B.Buffer((4, L), "float16"), B.Buffer((4, L), "float16")
        bk.np[...] = r.integers(-127, 128, bk.shape, dtype=np.int8); bv.np[...] = r.integers(-127, 128, bv.shape, dtype=np.int8)
        bks.np[...] = 0.02; bvs.np[...] = 0.01
        keep += [bk, bv, bks, bvs]
        h4 = [g.dec_wrap(b_.address, b_.nbytes) for b_ in (bk, bks, bv, bvs)]
        assert min(h4) >= 0 and g.dec_seg(99, start, L, *h4) > 0
        start += L
    plan_r, plan_a, plan_b = B.Plan([fr.bind(ri, ro)]), B.Plan([fa.bind(ai, ao)]), B.Plan([fb.bind(bi, bo)])
    for _ in range(5):
        plan_a.run(); g.dec_run(99, hs, st, 8184); plan_b.run()
    ref = []
    for _ in range(REPS):
        t1 = time.perf_counter(); plan_r.run(); ref.append(time.perf_counter() - t1)
    print(f"v8_16k verify entry alone: {1e3 * np.median(ref):.2f} ms (the ANE's attention at 16K inside)", flush=True)
    rep = []
    for P in POS:
        wa, wg, wb, wall, gpu = [], [], [], [], []
        for _ in range(REPS):
            t1 = time.perf_counter(); plan_a.run(); t2 = time.perf_counter()
            gpu.append(g.dec_run(99, hs, st, P)); t3 = time.perf_counter()
            plan_b.run(); t4 = time.perf_counter()
            wa.append(t2 - t1); wg.append(t3 - t2); wb.append(t4 - t3); wall.append(t4 - t1)
        lt = []   # latency breakdown in the interleaved regime: encode + commit, commit -> GPU start, GPU, GPU end -> wake
        for _ in range(REPS):
            plan_a.run(); g.dec_run_lat(99, hs, st, P, LAT); plan_b.run()
            lt.append([LAT[1] - LAT[0], LAT[2] - LAT[1], LAT[3] - LAT[2], LAT[4] - LAT[3]])
        lt = 1e3 * np.median(np.array(lt), 0)
        print(f"   latency (ms): encode+commit {lt[0]:.3f}, commit->GPU start {lt[1]:.3f}, GPU {lt[2]:.3f}, GPU end->wake {lt[3]:.3f}", flush=True)
        # (a GPU-side flag wait armed before A was tried 16 Nov: the GPU never sees the CPU's store -> removed)
        res = {0: (0, 0, None), 1: (0, 0, None)}
        # warm-up kick LEAD ms before part A's expected end (A's median wall from the plain loop above)
        a_us = 1e6 * float(np.median(wa))
        kick = {}
        for lead in (0.15, 0.25, 0.35):
            wl, gw, gt_ = [], [], []
            for _ in range(REPS):
                t1 = time.perf_counter(); g.dec_kick_after(int(a_us - 1e3 * lead)); plan_a.run(); t2 = time.perf_counter()
                gt_.append(g.dec_run_lat(99, hs, st, P, LAT)); t3 = time.perf_counter(); plan_b.run(); t4 = time.perf_counter()
                wl.append(t4 - t1); gw.append(t3 - t2)
            kick[lead] = (1e3 * float(np.median(wl)), 1e3 * float(np.median(gw)), float(np.median(gt_)))
        print(f"keys {P + 8:6d}: plain chain {1e3 * np.median(wall):.2f} ms = A {1e3 * np.median(wa):.2f} + GPU wall {1e3 * np.median(wg):.3f} "
              f"(GPU {np.median(gpu):.3f}) + B {1e3 * np.median(wb):.2f}", flush=True)
        print("   kick: " + "; ".join(f"{ld} ms ahead -> chain {v[0]:.2f} (GPU wall {v[1]:.3f}, GPU {v[2]:.3f})" for ld, v in kick.items()), flush=True)
        if os.environ.get("SPIN") != "1":
            rep.append({"P": P, "lat_ms": lt.tolist(), "kick": {str(k): v for k, v in kick.items()}, "plain": {k: 1e3 * float(np.median(x)) for k, x in
                        (("chain", wall), ("A", wa), ("gpu_wall", wg), ("B", wb))}, "gpu": float(np.median(gpu))}); continue
        ws, we = [], []   # spin wait; armed ahead + event fire + spin
        for _ in range(REPS):
            t1 = time.perf_counter(); plan_a.run(); g.dec_run_spin(99, hs, st, P); plan_b.run(); ws.append(time.perf_counter() - t1)
        for _ in range(REPS):
            t1 = time.perf_counter(); tk = g.dec_arm(99, hs, st, P); plan_a.run(); t2 = time.perf_counter()
            g.dec_fire(tk); assert g.dec_spin(tk) == 0; t3 = time.perf_counter(); plan_b.run(); we.append((time.perf_counter() - t1, t3 - t2))
        warm = [g.dec_run(99, hs, st, P) for _ in range(REPS)]
        row = {"P": P, "keys": P + 8, "ms_chain": 1e3 * float(np.median(wall)), "ms_A": 1e3 * float(np.median(wa)),
               "ms_gpu_wall": 1e3 * float(np.median(wg)), "ms_gpu": float(np.median(gpu)), "ms_B": 1e3 * float(np.median(wb)),
               "ms_gpu_warm": float(np.median(warm)), "ms_gpu_p90": float(np.percentile(gpu, 90)),
               "ms_chain_spin": 1e3 * float(np.median(ws)), "ms_chain_event": 1e3 * float(np.median([w[0] for w in we])),
               "ms_gpu_wall_event": 1e3 * float(np.median([w[1] for w in we]))}
        rep.append(row)
        print(f"keys {P + 8:6d}: chain {row['ms_chain']:.2f} ms = A {row['ms_A']:.2f} + GPU wall {row['ms_gpu_wall']:.3f} "
              f"(GPU {row['ms_gpu']:.3f}, p90 {row['ms_gpu_p90']:.3f}; back to back {row['ms_gpu_warm']:.3f}) + B {row['ms_B']:.2f}; "
              f"chain spin {row['ms_chain_spin']:.2f}, armed+event {row['ms_chain_event']:.2f} (GPU wall {row['ms_gpu_wall_event']:.3f})", flush=True)
    (probe / "dec_gpu_long.json").write_text(json.dumps({"ref_16k_ms": 1e3 * float(np.median(ref)), "rows": rep}, indent=1))


main()
