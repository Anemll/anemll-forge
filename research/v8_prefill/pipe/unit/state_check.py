"""v8_state / v8_set_state: round trip (ANE layout -> GPU -> ANE layout) and ms per layer.  MainPY state_check.py"""
import ctypes as C, os, time
from pathlib import Path
import numpy as np
P = Path("/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe"); E = P / "engine"
os.environ.update({"V8_MANIFEST": str(P / "w_shared2r/manifest_best.json"), "V8_KERNELS": str(E / "kernels.metal"), "NT": "128", "LMAX": "1024"})
g = C.CDLL(str(E / os.environ.get("DYLIB", "libgpu_v8_srv.dylib")))
g.v8_init.restype = C.c_int32
for f in (g.v8_state, g.v8_set_state):
    f.restype, f.argtypes = C.c_int32, [C.c_int32, C.c_void_p, C.c_void_p]
print("layers", g.v8_init())
rng = np.random.default_rng(1)
conv = rng.standard_normal((3, 10240)).astype(np.float16); rec = rng.standard_normal((48, 128, 128)).astype(np.float16)
c2, r2 = np.empty_like(conv), np.empty_like(rec)
t = time.perf_counter(); assert g.v8_set_state(4, conv.ctypes.data, rec.ctypes.data) == 0; t1 = time.perf_counter()
assert g.v8_state(4, c2.ctypes.data, r2.ctypes.data) == 0; t2 = time.perf_counter()
print("round trip conv %s rec %s | set %.2f ms get %.2f ms" % (np.array_equal(conv, c2), np.array_equal(rec, r2), (t1 - t) * 1e3, (t2 - t1) * 1e3))
ts = []
for L in (0, 1, 2, 4, 5, 6) * 8:
    t = time.perf_counter(); g.v8_state(L, c2.ctypes.data, r2.ctypes.data); ts.append(time.perf_counter() - t)
print("v8_state: %.2f ms per layer (48 layers %.0f ms)" % (np.median(ts) * 1e3, np.median(ts) * 48e3))
