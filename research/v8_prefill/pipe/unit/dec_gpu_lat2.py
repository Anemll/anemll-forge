"""Decode-verify split, GPU start latency by gap kind (dec_gpu_lat2.py, from dec_gpu_long.py) (16 Nov 2026): the probe chunk's dva / dvb on the ANE (Swift bridge) with
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
                    "mode": "dec_gpu_lat2"}, f"decgpulat2-{os.getpid()}-{int(t0)}")
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
    g.dec_residency.restype = C_.c_int32
    g.dec_kick_info.restype, g.dec_kick_info.argtypes = None, [C_.POINTER(C_.c_double)]
    KI = (C_.c_double * 4)()
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
    plan_a, plan_b = B.Plan([fa.bind(ai, ao)]), B.Plan([fb.bind(bi, bo)])
    P = int(os.environ.get("P", "8184"))
    for _ in range(5):
        plan_a.run(); g.dec_run(99, hs, st, P); plan_b.run()
    a_s = float(np.median([(lambda t: (plan_a.run(), time.perf_counter() - t)[1])(time.perf_counter()) for _ in range(10)]))

    def spin(sec):
        t = time.perf_counter() + sec
        while time.perf_counter() < t:
            pass
    gaps = {"spin": lambda: spin(a_s), "sleep": lambda: time.sleep(a_s), "ane": plan_a.run}

    def measure(tag):
        for kind, gap in gaps.items():
            for kick in (False, True):
                ls, kk = [], []
                for _ in range(REPS):
                    if kick:
                        g.dec_kick_after(int(1e6 * a_s - int(os.environ.get("LEAD_US", "300"))))
                    gap(); g.dec_run_lat(99, hs, st, P, LAT)
                    ls.append([LAT[2] - LAT[1], LAT[3] - LAT[2], LAT[4] - LAT[3]])
                    if kick:
                        time.sleep(0.002); g.dec_kick_info(KI)   # relative to the real commit (ms): kick commit, kick GPU start, kick target
                        kk.append([KI[0] - LAT[1], KI[1] - LAT[1], KI[3] - LAT[1]])
                ls = 1e3 * np.median(np.array(ls), 0)
                if kk:
                    kq = 1e3 * np.median(np.array(kk), 0)
                    print(f"      kick vs the real commit (ms): target {kq[2]:+.3f}, committed {kq[0]:+.3f}, GPU start {kq[1]:+.3f}")
                print(f"{tag:12s} gap {kind:5s} ({1e3 * a_s:.2f} ms) kick {str(kick):5s}: commit->start {ls[0]:.3f}, GPU {ls[1]:.3f}, end->wake {ls[2]:.3f}", flush=True)
    measure("default")


main()
