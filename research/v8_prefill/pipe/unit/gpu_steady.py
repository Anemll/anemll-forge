"""Steady GPU load through the V8 engine: attention layer 3 at a fixed position P (KV rows below P not computed) and
DeltaNet layer 0, 128-row calls, for SECONDS; prints per-call GPU-time medians (Metal timestamps) per layer.
    MainPY gpu_steady.py SECONDS [P] [TAG]"""
import ctypes as C, mmap, os, sys, time, numpy as np
E = "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/engine"
secs = float(sys.argv[1]); P = int(sys.argv[2]) if len(sys.argv) > 2 else 61440; tag = sys.argv[3] if len(sys.argv) > 3 else ""
os.environ.update({"V8_MANIFEST": "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/w_shared2r/manifest.json",
                   "V8_KERNELS": E + "/kernels.metal", "NT": "128", "LMAX": str(P + 128), "KV8": "0"})
g = C.CDLL(E + "/libgpu_v8.dylib")
g.v8_init.restype = C.c_int32; g.v8_wrap.restype, g.v8_wrap.argtypes = C.c_int32, [C.c_void_p, C.c_size_t]
g.v8_mix.restype, g.v8_mix.argtypes = C.c_double, [C.c_int32] * 4
g.v8_init()
def mbuf(n):
    b = mmap.mmap(-1, n); return b, g.v8_wrap(C.addressof(C.c_char.from_buffer(b)), n)
n = 5120 * 128 * 2
bi, hi = mbuf(n); bo, ho = mbuf(n)
xi = np.frombuffer(bi, np.float16).reshape(5120, 128)
xi[:] = (np.random.default_rng(5).standard_normal((5120, 128)) * 0.5).astype(np.float16)
g.v8_reset()
for _ in range(3):
    g.v8_mix(3, hi, ho, P); g.v8_mix(0, hi, ho, P)
ta, tg = [], []
t_end = time.time() + secs
while time.time() < t_end:
    ta.append(g.v8_mix(3, hi, ho, P))
    tg.append(g.v8_mix(0, hi, ho, P))
print(f"GPU {tag:22s} attention@{P // 1024}K {np.median(ta):6.2f} ms   DeltaNet {np.median(tg):5.2f} ms   ({len(ta)} pairs of calls)", flush=True)
