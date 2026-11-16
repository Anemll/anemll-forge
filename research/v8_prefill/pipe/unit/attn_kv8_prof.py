"""Attention core, fp16 GPU KV (+ QK8) vs KV8 (the ANE's INT8 segment layout), GPU only (16 Nov 2026): one attention
layer's pair call at position P (P keys behind it), clean ms and (PROFILE=1) per-kernel ms. KV8 segments are host
buffers in the bridge's layout (k [4][256][rows] dim-major, v [4][rows][256], ks / vs [4][rows] fp16; sizes as the
v8_shared2 kv_seg list), random codes. Run once per mode (the engine reads KV8 / QK8 at load):
    KV8=0|1 QK8=1 MainPY attn_kv8_prof.py P [CALLS]       (env FUSE / I8 ... as the best config, DYLIB)"""
import ctypes as C, mmap, os, sys
from pathlib import Path
import numpy as np

P = Path("/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe"); E = P / "engine"
pos, calls = int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 12
LMAX = 98304
os.environ.update({"V8_MANIFEST": str(P / "w_shared2r/manifest_best.json"), "V8_KERNELS": str(E / "kernels.metal"),
                   "NT": "128", "LMAX": str(LMAX)})
g = C.CDLL(str(E / os.environ.get("DYLIB", "libgpu_v8_srv.dylib")))
g.v8_init.restype = C.c_int32
g.v8_wrap.restype, g.v8_wrap.argtypes = C.c_int32, [C.c_void_p, C.c_size_t]
g.v8_wait2.restype, g.v8_wait2.argtypes = C.c_double, [C.c_int32, C.POINTER(C.c_double)]
g.v8_submit_pair.restype, g.v8_submit_pair.argtypes = C.c_int32, [C.c_int32] * 6
g.v8_kv8_seg.restype, g.v8_kv8_seg.argtypes = C.c_int32, [C.c_int32] * 12
g.v8_prof_dump.restype = None
print("gpu layers", g.v8_init(), "KV8", os.environ.get("KV8", "0"), "QK8", os.environ.get("QK8", "0"), "K8V16", os.environ.get("K8V16", "0"))
L = 3
rng = np.random.default_rng(0)
keep = []
def wrap(nbytes):
    m = mmap.mmap(-1, nbytes); keep.append(m)
    return m, g.v8_wrap(C.addressof(C.c_char.from_buffer(m)), nbytes)
if os.environ.get("KV8") == "1":
    start = 0
    for rows in (8192, 8192, 16384, 32768, 32768):
        k, hk = wrap(4 * 256 * rows); v, hv = wrap(4 * rows * 256); ks, hks = wrap(4 * rows * 2); vs, hvs = wrap(4 * rows * 2)
        np.frombuffer(k, np.int8)[:] = rng.integers(-127, 128, 4 * 256 * rows, dtype=np.int8)
        np.frombuffer(v, np.int8)[:] = rng.integers(-127, 128, 4 * 256 * rows, dtype=np.int8)
        np.frombuffer(ks, np.float16)[:] = 0.01; np.frombuffer(vs, np.float16)[:] = 0.01
        assert g.v8_kv8_seg(L, start, rows, hk, hv, hks, hvs, 256 * rows, rows, rows * 256, 256, rows) > 0
        start += rows
    if os.environ.get("K8V16") == "1":   # the GPU's fp16 V copy of the rows behind the call, from the INT8 segments
        g.v8_kv8_v16_fill.restype, g.v8_kv8_v16_fill.argtypes = C.c_double, [C.c_int32]
        print("K8V16 fill %.1f ms (GPU)" % g.v8_kv8_v16_fill(pos))
blk = [wrap(5120 * 64 * 2) for _ in range(4)]
for m, _ in blk:
    np.frombuffer(m, np.float16)[:] = (rng.standard_normal(5120 * 64) * 0.5).astype(np.float16)
ts = (C.c_double * 2)(); ms = []
for i in range(calls):
    t = g.v8_submit_pair(L, blk[0][1], blk[1][1], blk[2][1], blk[3][1], pos)
    assert t > 0, t
    ms.append(g.v8_wait2(t, ts))
print("attention layer %d pair call at position %d: median %.3f ms (min %.3f)" % (L, pos, np.median(ms[2:]), np.min(ms[2:])), flush=True)
if os.environ.get("OUT_NPY"):   # the layer's output blocks (A, B) of the last call, for a variant comparison
    np.save(os.environ["OUT_NPY"], np.stack([np.frombuffer(blk[2][0], np.float16), np.frombuffer(blk[3][0], np.float16)]))
if os.environ.get("PROFILE"):
    g.v8_prof_dump()
