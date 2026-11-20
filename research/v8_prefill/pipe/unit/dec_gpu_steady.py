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
F8V = float(os.environ["DEC_F8V"]) if os.environ.get("DEC_F8V") else None   # queue 16: V segments = e4m3 copy (v x F8VS)
if os.environ.get("DEC_DYLIB", "").startswith("libdec_gpu_f8d"):   # queue 16b: that dylib keeps its own e4m3 copy
    F8V = None


def f2e4m3(x):   # as kernels_f8.metal f2e4m3: bias 7, max 448, round to nearest even, saturating
    x = np.ascontiguousarray(x, np.float32); sgn = (x.view(np.uint32) >> 24) & 0x80
    a = np.minimum(np.abs(x), np.float32(448.0)); b = a.view(np.uint32).astype(np.uint64)
    b = b + 0x7FFFF + ((b >> 20) & 1)
    r = np.minimum((((b >> 23).astype(np.int64) - 120) << 3) | ((b >> 20) & 7).astype(np.int64), 0x7E)
    r = np.where(a < 0.015625, np.rint(a * 512.0).astype(np.int64), r)
    return (r.astype(np.uint32) | sgn).astype(np.uint8)


r = np.random.default_rng(9); start = 0
for L in SEG:
    K, V = abuf((4, 256, L), np.int8), abuf((4, L, 256), np.int8); KS, VS = abuf((4, L), np.float16), abuf((4, L), np.float16)
    K[...] = r.integers(-127, 128, K.shape, dtype=np.int8); V[...] = r.integers(-127, 128, V.shape, dtype=np.int8); KS[...] = (0.01 + 0.02 * r.random(KS.shape)).astype(np.float16); VS[...] = (0.003 + 0.01 * r.random(VS.shape)).astype(np.float16)
    if os.environ.get("DEC_KMAJ") == "1":   # queue 16c: the same K values stored key-major [kv][rows][256]
        Kt = np.ascontiguousarray(K.transpose(0, 2, 1)); K = np.frombuffer(K.data, np.int8, count=K.size).reshape(4, L, 256); K[...] = Kt
    if F8V is not None:   # the same random INT8 cache, its fp16 values (code x vs) as e4m3 x F8VS (one byte, same layout)
        for h in range(4):
            V[h] = f2e4m3(V[h].astype(np.float32) * VS[h].astype(np.float32)[:, None] * F8V).view(np.int8)
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
import hashlib
_sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()[:16]
_dl, _mt = ENG / os.environ.get("DEC_DYLIB", "libdec_gpu_t.dylib"), ENG / os.environ.get("DEC_METAL", "dec_kernels.metal")
print(f"BANNER dec_gpu_steady P={P} | dylib {_dl.name} sha {_sha(_dl)} | metal {_mt.name} sha {_sha(_mt)} | DEC_ATTN={os.environ.get('DEC_ATTN', 'dec_attn_r4')} "
      f"DEC_SM={os.environ.get('DEC_SM', '0')} DEC_F8V={os.environ.get('DEC_F8V', '-')} ITERS={it}", flush=True)
if os.environ.get("DEC_REF"):   # relative error of this output against a saved one (the served kernel's)
    ref = np.load(os.environ["DEC_REF"]); oo = o.astype(np.float32)
    print(f"vs {Path(os.environ['DEC_REF']).name}: rel L2 {np.linalg.norm(oo - ref) / np.linalg.norm(ref):.3e} | max abs {np.abs(oo - ref).max():.3e} "
          f"| cos {float(oo @ ref / np.linalg.norm(oo) / np.linalg.norm(ref)):.6f}", flush=True)
if os.environ.get("DEC_SAVE"): np.save(os.environ["DEC_SAVE"], o.astype(np.float32))
print(f"{os.environ.get('TAG', '')} [output checksum {ck:.1f}]: steady {b:.3f} ms ({mb / b:.0f} GB/s of unique K/V) | one at a time {one:.3f} ms ({mb / one:.0f} GB/s)")
