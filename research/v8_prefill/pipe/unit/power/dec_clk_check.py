"""Integration check of the GPU clock keep-alive in engine/libdec_gpu_clk.dylib (dec_gpu_clk.swift, 18 Nov 2026), the
way the server would call it: ctypes, dec_run per attention layer, the server's gap pattern (CPU sleeps of 5 / 9 ms
between calls = its dvba / dvb + dva ANE calls, + a 14 ms draft before every 16th call = one verify step of 16 GPU
calls). Phases: off (the served behaviour) vs on (dec_ka_hold(HOLD) before each dec_run = DEC_KA_HOLD_MS auto mode),
at 64K and 95K keys, interleaved over ROUNDS rounds. Per call: dec_run's GPU ms and the wall ms around it; per verify
step: the sum over its 16 calls. Output o checksummed (the keep-alive must not change it). IOReport per phase.
    PY dec_clk_check.py OUT_DIR [ROUNDS=4] [SECS=4] [HOLD_MS=60]"""
import ctypes as C_, json, mmap, os, signal, subprocess, sys, time
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D))
import telemetry as T
ENG = D.parents[1] / "engine"
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
SECS = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
HOLD = float(sys.argv[4]) if len(sys.argv) > 4 else 60.0
SEG = [8192, 8192, 16384, 32768, 65536]
keep = []


def abuf(shape, dtype):
    n = int(np.prod(shape)) * np.dtype(dtype).itemsize; m = mmap.mmap(-1, (n + 16383) // 16384 * 16384); keep.append(m)
    return np.frombuffer(m, dtype, count=int(np.prod(shape))).reshape(shape)


os.environ.setdefault("DEC_KA_HOLD_MS", "0")   # explicit dec_ka_hold calls below (same effect as the auto hook)
g = C_.CDLL(str(ENG / "libdec_gpu_clk.dylib"))
g.dec_init.restype, g.dec_init.argtypes = C_.c_int32, [C_.c_char_p]
g.dec_wrap.restype, g.dec_wrap.argtypes = C_.c_int32, [C_.c_void_p, C_.c_size_t]
g.dec_seg.restype, g.dec_seg.argtypes = C_.c_int32, [C_.c_int32] * 7
g.dec_run.restype, g.dec_run.argtypes = C_.c_double, [C_.c_int32, C_.POINTER(C_.c_int32), C_.POINTER(C_.c_uint32), C_.c_int32]
g.dec_ka_hold.restype, g.dec_ka_hold.argtypes = C_.c_int32, [C_.c_double]
g.dec_ka_stop.restype, g.dec_ka_stop.argtypes = None, []
g.dec_ka_info.restype, g.dec_ka_info.argtypes = None, [C_.POINTER(C_.c_double)]
assert g.dec_init(str(ENG / "dec_kernels.metal").encode()) > 0
w = lambda a: g.dec_wrap(a.ctypes.data, a.nbytes)  # noqa: E731
r = np.random.default_rng(9); start = 0
for L in SEG:
    K, V = abuf((4, 256, L), np.int8), abuf((4, L, 256), np.int8); KS, VS = abuf((4, L), np.float16), abuf((4, L), np.float16)
    K[...] = r.integers(-127, 128, K.shape, dtype=np.int8); V[...] = r.integers(-127, 128, V.shape, dtype=np.int8)
    KS[...] = (0.01 + 0.02 * r.random(KS.shape)).astype(np.float16); VS[...] = (0.003 + 0.01 * r.random(VS.shape)).astype(np.float16)
    assert g.dec_seg(0, start, L, w(K), w(KS), w(V), w(VS)) > 0; start += L
q, gt = abuf((24, 8, 256), np.float16), abuf((24, 8, 256), np.float16); kt, vt = abuf((4, 8, 256), np.float16), abuf((4, 8, 256), np.float16)
q[...] = r.standard_normal(q.shape).astype(np.float16); o = abuf((6144 * 8,), np.float16)
kt[...] = (0.5 * r.standard_normal(kt.shape)).astype(np.float16); vt[...] = (0.5 * r.standard_normal(vt.shape)).astype(np.float16)
gt[...] = r.standard_normal(gt.shape).astype(np.float16)
H = (C_.c_int32 * 5)(w(q), w(gt), w(kt), w(vt), w(o)); st = (C_.c_uint32 * 9)(2048, 256, 2048, 256, 2048, 256, 2048, 256, 8)
for _ in range(3):
    g.dec_run(0, H, st, 65536)
io = subprocess.Popen([str(D / "iorep_sample"), "100", "0", str(out / "io.jsonl")])
time.sleep(1.5)
PAT = [5.0, 9.0]; DRAFT = 14.0


def phase(name, P, ka):
    calls, steps, sums = [], [], []
    t_end = time.time() + SECS; i = 0; step = [0.0, 0.0]
    t0 = time.time()
    while time.time() < t_end:
        gap = PAT[i % 2] + (DRAFT if i % 16 == 0 else 0.0)
        time.sleep(gap / 1e3)
        if ka:
            g.dec_ka_hold(HOLD)
        a = time.perf_counter(); ms = g.dec_run(0, H, st, P); b = time.perf_counter()
        assert ms > 0, ms
        calls.append((ms, 1e3 * (b - a))); step[0] += ms; step[1] += 1e3 * (b - a)
        sums.append(float(np.sum(o.astype(np.float32))))
        i += 1
        if i % 16 == 0:
            steps.append(tuple(step)); step = [0.0, 0.0]
    t1 = time.time()
    if ka:
        g.dec_ka_stop()
    c = np.array(calls[16:]); s = np.array(steps[1:])   # the first step: settling
    return {"name": name, "P": P, "ka": ka, "t0": t0 + 0.3, "t1": t1, "calls": len(c),
            "gpu_ms": float(np.median(c[:, 0])), "wall_ms": float(np.median(c[:, 1])),
            "step_gpu_ms": float(np.median(s[:, 0])) if len(s) else None, "step_wall_ms": float(np.median(s[:, 1])) if len(s) else None,
            "o_sum": sums[-1]}


phases = []
try:
    regs = [("64K off", 65536, False), ("64K on", 65536, True), ("95K off", 95232, False), ("95K on", 95232, True)]
    for rd in range(ROUNDS):
        for k in range(len(regs)):
            nm, P, ka = regs[(k + rd) % len(regs)]
            time.sleep(1.5)
            x = phase(nm, P, ka); x["round"] = rd; phases.append(x)
            print(f"r{rd} {nm}: GPU {x['gpu_ms']:.3f} wall {x['wall_ms']:.3f} ms per call; per 16-call step GPU {x['step_gpu_ms']:.1f} "
                  f"wall {x['step_wall_ms']:.1f} ms; o_sum {x['o_sum']:.4f}", flush=True)
finally:
    time.sleep(0.5); io.send_signal(signal.SIGINT); io.wait(10)
info = (C_.c_double * 3)(); g.dec_ka_info(info)
rows = T.io_rows(out / "io.jsonl")
for x in phases:
    s = T.summarize(rows, x["t0"], x["t1"])
    ctrl = {k: v for k, v in s.get("gpu_ctrl", {}).items() if k != "IDLE_OFF"}; tc = sum(ctrl.values()) or 1
    x["gpu_mhz"], x["gpu_w"], x["ctrl"] = s.get("gpu_mhz"), s.get("gpu_w_energy_model"), {k: round(100 * v / tc) for k, v in ctrl.items()}
summ = {}
for x in phases:
    summ.setdefault(x["name"], []).append(x)
res = {"phases": phases, "ka_info": list(info), "summary": {}}
for nm, xs in summ.items():
    f = lambda k: [round(float(np.median([x[k] for x in xs])), 3), round(min(x[k] for x in xs), 3), round(max(x[k] for x in xs), 3)]  # noqa: E731
    res["summary"][nm] = {"gpu_ms": f("gpu_ms"), "wall_ms": f("wall_ms"), "step_gpu_ms": f("step_gpu_ms"), "step_wall_ms": f("step_wall_ms"),
                          "gpu_mhz": float(np.median([x["gpu_mhz"] for x in xs])), "gpu_w": float(np.median([x["gpu_w"] for x in xs])),
                          "o_sums": sorted({round(x["o_sum"], 3) for x in xs})}
    print(nm, json.dumps(res["summary"][nm]))
print("keep-alive totals: command buffers %d, GPU busy %.1f s, held %.1f s" % tuple(info))
(out / "dec_clk_check.json").write_text(json.dumps(res, indent=1))
