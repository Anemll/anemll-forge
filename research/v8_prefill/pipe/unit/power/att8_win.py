"""Windowed attention load on the served attention path (18 Nov 2026): the served engine's environment with KV8 + K8V16
(fold_check.py's setup: random INT8 KV segments in the ANE's layout, the GPU's fp16 V copy), attention layer 3 pair
calls at position P for SECONDS through libgpu_v8_fold.dylib, plain or with the softmax folded (FORM: plain | 64 |
64s | 128). Every 0.5 s prints "<epoch> GPU att att <ms> gdn -" (median GPU ms per call, as gpu_mix_win.py).
    MainPY att8_win.py FORM SECONDS P [TAG]"""
import ctypes as C, mmap, os, sys, time
from pathlib import Path
import numpy as np

form, secs, P0 = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
tag = sys.argv[4] if len(sys.argv) > 4 else ""
PP = Path("/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe"); E = PP / "engine"
LMAX = (P0 + 128 + 8191) // 8192 * 8192
os.environ.update({"V8_MANIFEST": os.environ.get("MANIFEST", str(PP / "w_dec8/manifest_best.json")), "V8_KERNELS": str(E / "kernels.metal"), "NT": "128",
                   "LMAX": str(LMAX), "KV8": "1", "K8V16": "1", "QK8": "1", "TKV": "2048", "FUSE": "all", "I8": "1", "I8T": "1",
                   "I8TV": "_k256", "LUTC": "1", "I8A": "1", "I8KV": "1", "LRM": "1"})
g = C.CDLL(str(E / os.environ.get("DYLIB", "libgpu_v8_fold.dylib")))
g.v8_init.restype = C.c_int32
g.v8_wrap.restype, g.v8_wrap.argtypes = C.c_int32, [C.c_void_p, C.c_size_t]
g.v8_wait2.restype, g.v8_wait2.argtypes = C.c_double, [C.c_int32, C.POINTER(C.c_double)]
g.v8_submit_pair.restype, g.v8_submit_pair.argtypes = C.c_int32, [C.c_int32] * 6
g.v8_kv8_seg.restype, g.v8_kv8_seg.argtypes = C.c_int32, [C.c_int32] * 12
g.v8_kv8_v16_fill.restype, g.v8_kv8_v16_fill.argtypes = C.c_double, [C.c_int32]
g.v8_fold.restype, g.v8_fold.argtypes = C.c_int32, [C.c_int32, C.c_int32]
g.v8_init()
L = 3
rng = np.random.default_rng(0)
keep = []


def wrap(nbytes):
    m = mmap.mmap(-1, nbytes); keep.append(m)
    return m, g.v8_wrap(C.addressof(C.c_char.from_buffer(m)), nbytes)


start = 0
for rows in (8192, 8192, 16384, 32768, 32768):
    if start >= LMAX:
        break
    rows = min(rows, LMAX - start)
    k, hk = wrap(4 * 256 * rows); v, hv = wrap(4 * rows * 256); ks, hks = wrap(4 * rows * 2); vs, hvs = wrap(4 * rows * 2)
    np.frombuffer(k, np.int8)[:] = rng.integers(-127, 128, 4 * 256 * rows, dtype=np.int8)
    np.frombuffer(v, np.int8)[:] = rng.integers(-127, 128, 4 * 256 * rows, dtype=np.int8)
    np.frombuffer(ks, np.float16)[:] = rng.uniform(0.004, 0.02, 4 * rows).astype(np.float16)
    np.frombuffer(vs, np.float16)[:] = rng.uniform(0.004, 0.02, 4 * rows).astype(np.float16)
    assert g.v8_kv8_seg(L, start, rows, hk, hv, hks, hvs, 256 * rows, rows, rows * 256, 256, rows) > 0
    start += rows
g.v8_kv8_v16_fill(LMAX)
assert g.v8_fold({"plain": 0, "64": 1, "64s": 2, "128": 3}[form], 0) == 1
blk = [wrap(5120 * 64 * 2) for _ in range(4)]
for m, _ in blk[:2]:
    np.frombuffer(m, np.float16)[:] = (rng.standard_normal(5120 * 64) * 0.5).astype(np.float16)
ts = (C.c_double * 2)()


def call():
    t = g.v8_submit_pair(L, blk[0][1], blk[1][1], blk[2][1], blk[3][1], P0)
    assert t > 0, t
    return g.v8_wait2(t, ts)


for _ in range(3):
    call()
print(f"ready {time.time():.3f}", flush=True)
ta, wa = [], []
t_end = time.time() + secs; w0 = time.time()
while time.time() < t_end:
    x = call(); ta.append(x); wa.append(x)
    if time.time() - w0 >= 0.5:
        print(f"{time.time():.3f} GPU att att {np.median(wa):7.3f} gdn       -", flush=True)
        wa, w0 = [], time.time()
print(f"GPU {tag:22s} att8 {form} attention@{P0 // 1024}K {np.median(ta):6.2f} ms ({len(ta)} calls)", flush=True)
