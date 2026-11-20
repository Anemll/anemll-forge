"""Decode attention against the GPU clock (18 Nov 2026). The served decode kernel (engine/libdec_gpu.dylib, dec_attn_r4)
on one synthetic INT8 layer (dec_gpu_steady.py's setup: segments 8K+8K+16K+32K+64K, random codes) at P keys, by regime:
  steady   50 copies per command buffer (the GPU at its loaded clock)
  b2b      one call per command buffer, back to back
  gapN     one call after each N ms of CPU sleep (the server: an ANE call of about 5 ms between the GPU's attention calls)
  gapN_ka  gapN with a GPU keep-alive beside it (gpu_load alu at duty KA_DUTY, another process)
IOReport recording (iorep_sample, 100 ms) for the whole run: GPU MHz (active-weighted) and controller states per regime.
    PY dec_clock.py OUT_DIR [P,P,...] [GAP,GAP,...]"""
import ctypes as C_, json, mmap, os, signal, subprocess, sys, time
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D))
import telemetry as T
ENG = D.parents[1] / "engine"
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
PS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "65536,95232").split(",")]
GAPS = [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1,5").split(",")]
SECS = float(os.environ.get("SECS", "5")); KA_DUTY = os.environ.get("KA_DUTY", "0.1")
SEG = [8192, 8192, 16384, 32768, 65536]
keep = []


def abuf(shape, dtype):
    n = int(np.prod(shape)) * np.dtype(dtype).itemsize; m = mmap.mmap(-1, (n + 16383) // 16384 * 16384); keep.append(m)
    return np.frombuffer(m, dtype, count=int(np.prod(shape))).reshape(shape)


g = C_.CDLL(str(ENG / os.environ.get("DEC_DYLIB", "libdec_gpu.dylib")))
g.dec_init.restype, g.dec_init.argtypes = C_.c_int32, [C_.c_char_p]
g.dec_wrap.restype, g.dec_wrap.argtypes = C_.c_int32, [C_.c_void_p, C_.c_size_t]
g.dec_seg.restype, g.dec_seg.argtypes = C_.c_int32, [C_.c_int32] * 7
g.dec_run.restype, g.dec_run.argtypes = C_.c_double, [C_.c_int32, C_.POINTER(C_.c_int32), C_.POINTER(C_.c_uint32), C_.c_int32]
g.dec_bench.restype, g.dec_bench.argtypes = C_.c_double, [C_.c_int32, C_.POINTER(C_.c_int32), C_.POINTER(C_.c_uint32), C_.c_int32, C_.c_int32]
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
H = (C_.c_int32 * 5)(w(q), w(gt), w(kt), w(vt), w(o)); st = (C_.c_uint32 * 9)(2048, 256, 2048, 256, 2048, 256, 2048, 256, 8)
io = subprocess.Popen([str(D / "iorep_sample"), "100", "0", str(out / "io.jsonl")])
time.sleep(2)
phases = []


def phase(name, fn):
    t0 = time.time(); ms = fn(); t1 = time.time()
    phases.append({"name": name, "t0": t0, "t1": t1, "ms": ms})
    print(f"{name}: median {np.median(ms):.3f} ms per call ({len(ms)} calls)", flush=True)
    time.sleep(2)


try:
    for P in PS:
        for _ in range(3):
            g.dec_bench(0, H, st, P, 50)

        def steady():
            ms, t = [], time.time()
            while time.time() < t + SECS:
                ms.append(g.dec_bench(0, H, st, P, 50))
            return ms

        def b2b():
            ms, t = [], time.time()
            while time.time() < t + SECS:
                ms.append(g.dec_run(0, H, st, P))
            return ms

        def gapped(gap):
            def f():
                ms, t = [], time.time()
                while time.time() < t + SECS:
                    time.sleep(gap / 1e3); ms.append(g.dec_run(0, H, st, P))
                return ms
            return f

        phase(f"{P // 1024}K steady", steady)
        phase(f"{P // 1024}K b2b", b2b)
        for gap in GAPS:
            phase(f"{P // 1024}K gap{gap:g}", gapped(gap))
        for gap in GAPS:
            ka = subprocess.Popen([str(D / "gpu_load"), "alu", str(SECS + 30), "1024", "8", KA_DUTY], stdout=subprocess.DEVNULL)
            time.sleep(1.5)
            phase(f"{P // 1024}K gap{gap:g}_ka", gapped(gap))
            ka.terminate()   # SIGTERM: a background shell leaves SIGINT ignored in its children (gpu_load never exited)
            try:
                ka.wait(5)
            except subprocess.TimeoutExpired:
                ka.kill(); ka.wait(5)
finally:
    time.sleep(1); io.send_signal(signal.SIGINT); io.wait(10)
rows = T.io_rows(out / "io.jsonl")
res = []
for p in phases:
    s = T.summarize(rows, p["t0"] + 0.3, p["t1"])
    x = {"phase": p["name"], "ms_median": round(float(np.median(p["ms"])), 3), "calls": len(p["ms"]),
         "gpu_mhz": s.get("gpu_mhz"), "gpu_active_pct": s.get("gpu_active_pct"), "gpu_ctrl": s.get("gpu_ctrl"),
         "dram_total_gbs": s.get("dram_total_agents_gbs")}
    res.append(x); print(json.dumps(x), flush=True)
(out / "dec_clock.json").write_text(json.dumps(res, indent=1))
