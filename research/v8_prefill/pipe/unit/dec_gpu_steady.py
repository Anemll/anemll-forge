"""Decode attention kernel, steady-state timing (17 Nov 2026): one synthetic layer (KV segments 8K+8K+16K+32K+64K,
random INT8), P keys; dec_bench = ITERS copies of the work in one command buffer (no CPU gaps, the GPU at its loaded
clock) vs dec_run one at a time with a CPU wait each (as the earlier tests timed it). Variant via env (DEC_SM,
DEC_QGROUPS, DEC_CK) of the test dylib.   PY dec_gpu_steady.py [P]"""
import ctypes as C_, mmap, os, sys
from pathlib import Path
import numpy as np
ENG = Path(__file__).resolve().parents[1] / "engine"
SEG = [8192, 8192, 16384, 32768, 65536]
keep = []
def abuf(shape, dtype):
    n = int(np.prod(shape)) * np.dtype(dtype).itemsize; m = mmap.mmap(-1, (n + 16383) // 16384 * 16384); keep.append(m)
    return np.frombuffer(m, dtype, count=int(np.prod(shape))).reshape(shape)
g = C_.CDLL(str(ENG / os.environ.get("DEC_DYLIB", "libdec_gpu_t.dylib")))
g.dec_init.restype, g.dec_init.argtypes = C_.c_int32, [C_.c_char_p]
g.dec_wrap.restype, g.dec_wrap.argtypes = C_.c_int32, [C_.c_void_p, C_.c_size_t]
g.dec_seg.restype, g.dec_seg.argtypes = C_.c_int32, [C_.c_int32] * 7
g.dec_run.restype, g.dec_run.argtypes = C_.c_double, [C_.c_int32, C_.POINTER(C_.c_int32), C_.POINTER(C_.c_uint32), C_.c_int32]
g.dec_bench.restype, g.dec_bench.argtypes = C_.c_double, [C_.c_int32, C_.POINTER(C_.c_int32), C_.POINTER(C_.c_uint32), C_.c_int32, C_.c_int32]
assert g.dec_init(str(ENG / os.environ.get("DEC_METAL", "dec_kernels.metal")).encode()) > 0
w = lambda a: g.dec_wrap(a.ctypes.data, a.nbytes)
r = np.random.default_rng(9); start = 0
for L in SEG:
    K, V = abuf((4, 256, L), np.int8), abuf((4, L, 256), np.int8); KS, VS = abuf((4, L), np.float16), abuf((4, L), np.float16)
    K[...] = r.integers(-127, 128, K.shape, dtype=np.int8); V[...] = r.integers(-127, 128, V.shape, dtype=np.int8); KS[...] = (0.01 + 0.02 * r.random(KS.shape)).astype(np.float16); VS[...] = (0.003 + 0.01 * r.random(VS.shape)).astype(np.float16)
    assert g.dec_seg(0, start, L, w(K), w(KS), w(V), w(VS)) > 0; start += L
q, gt = abuf((24, 8, 256), np.float16), abuf((24, 8, 256), np.float16); kt, vt = abuf((4, 8, 256), np.float16), abuf((4, 8, 256), np.float16)
q[...] = r.standard_normal(q.shape).astype(np.float16); o = abuf((6144 * 8,), np.float16)
H = (C_.c_int32 * 5)(w(q), w(gt), w(kt), w(vt), w(o)); st = (C_.c_uint32 * 9)(2048, 256, 2048, 256, 2048, 256, 2048, 256, 8)
P = int(sys.argv[1]) if len(sys.argv) > 1 else 120000
it = int(os.environ.get("ITERS", "50"))
for _ in range(3): g.dec_bench(0, H, st, P, it)
b = sorted(g.dec_bench(0, H, st, P, it) for _ in range(5))[2]
one = sorted(g.dec_run(0, H, st, P) for _ in range(30))[15]
mb = (P + 8) * 4 * 512 / 1e6
ck = float(np.abs(o.astype(np.float32)).sum())
if os.environ.get("DEC_SAVE"): np.save(os.environ["DEC_SAVE"], o.astype(np.float32))
print(f"{os.environ.get('TAG', '')} [output checksum {ck:.1f}]: steady {b:.3f} ms ({mb / b:.0f} GB/s of unique K/V) | one at a time {one:.3f} ms ({mb / one:.0f} GB/s)")
