"""A8 pair call vs the baseline pair call, GPU only (no ANE): per-call ms (clean) and, with PROFILE=1, per-kernel ms.
    MainPY a8_prof.py LAYER CALLS      (env DYLIB, MAN, PROFILE, FUSE / I8 ...)"""
import ctypes as C, mmap, os, sys
from pathlib import Path
import numpy as np

P = Path("/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe")
E = P / "engine"
L, calls = int(sys.argv[1]), int(sys.argv[2])
os.environ.update({"V8_MANIFEST": os.environ.get("MAN", str(P / "w_a8/manifest_a8.json")), "V8_KERNELS": str(E / "kernels.metal"),
                   "NT": "128", "LMAX": "4096"})
g = C.CDLL(str(E / os.environ.get("DYLIB", "libgpu_v8_a8.dylib")))
g.v8_init.restype = C.c_int32
g.v8_wrap.restype, g.v8_wrap.argtypes = C.c_int32, [C.c_void_p, C.c_size_t]
g.v8_wait2.restype, g.v8_wait2.argtypes = C.c_double, [C.c_int32, C.POINTER(C.c_double)]
g.v8_submit_pair.restype, g.v8_submit_pair.argtypes = C.c_int32, [C.c_int32] * 6
g.v8_submit_a8.restype, g.v8_submit_a8.argtypes = C.c_int32, [C.c_int32] * 10
g.v8_a8_setup.restype, g.v8_a8_setup.argtypes = C.c_int32, [C.c_void_p, C.c_size_t, C.c_int32, C.c_int32, C.c_int32]
print("gpu layers", g.v8_init())
pal8 = np.fromfile(P / "w_a8/pal8.bin", np.int8)
assert g.v8_a8_setup(pal8.ctypes.data, pal8.nbytes, 5120, 32, 5120) == 0
def mb(n):
    b = mmap.mmap(-1, n); return b, g.v8_wrap(C.addressof(C.c_char.from_buffer(b)), n)
rng = np.random.default_rng(0)
blk = [mb(5120 * 64 * 2) for _ in range(4)]
for b, _ in blk:
    np.frombuffer(b, np.float16)[:] = (rng.standard_normal(5120 * 64) * 0.5).astype(np.float16)
yb, hy = mb(1024 * 5120 * 2); np.frombuffer(yb, np.float16)[:] = (rng.standard_normal(1024 * 5120) * 0.1).astype(np.float16)
xb, hx = mb(1024 * 5120); tb, ht = mb(1024 * 32 * 2)
ts = (C.c_double * 2)()
prof = bool(os.environ.get("PROFILE"))
if prof:
    g.v8_prof_dump.restype = None
for name, sub in (("baseline v8_submit_pair", lambda p0: g.v8_submit_pair(L, blk[0][1], blk[1][1], blk[2][1], blk[3][1], p0)),
                  ("a8 v8_submit_a8", lambda p0: g.v8_submit_a8(L, blk[2][1], blk[3][1], p0, hy, 0, hx, ht, 0, 3))):
    ms = []
    for i in range(calls):
        t = sub((i % 16) * 128)
        assert t > 0, t
        ms.append(g.v8_wait2(t, ts))
    print("%-26s median %.3f ms (min %.3f)" % (name, np.median(ms[2:]), np.min(ms[2:])), flush=True)
    if prof:
        g.v8_prof_dump()
