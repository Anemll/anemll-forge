"""GPU mixer time per call vs NT (rows per call): layer L, positions 0.. (short context), median of calls 4..; argv NT L [ncalls]"""
import ctypes as C, mmap, os, sys, numpy as np
E = "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/engine"
nt, L = int(sys.argv[1]), int(sys.argv[2]); nc = int(sys.argv[3]) if len(sys.argv) > 3 else 16
os.environ.update({"V8_MANIFEST": "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/w_shared/manifest.json",
                   "V8_KERNELS": E + "/kernels.metal", "NT": str(nt), "LMAX": str(nt * nc + 64), "KV8": "0"})
g = C.CDLL(E + "/" + os.environ.get("DYLIB", "libgpu_v8_fa2.dylib"))
g.v8_init.restype = C.c_int32; g.v8_wrap.restype, g.v8_wrap.argtypes = C.c_int32, [C.c_void_p, C.c_size_t]
g.v8_mix.restype, g.v8_mix.argtypes = C.c_double, [C.c_int32] * 4
g.v8_init()
def mbuf(n):
    b = mmap.mmap(-1, n); return b, g.v8_wrap(C.addressof(C.c_char.from_buffer(b)), n)
n = 5120 * nt * 2
bi, hi = mbuf(n); bo, ho = mbuf(n)
xi = np.frombuffer(bi, np.float16).reshape(5120, nt); xo = np.frombuffer(bo, np.float16).reshape(5120, nt)
rng = np.random.default_rng(5)
g.v8_reset(); ts = []
for c in range(nc):
    xi[:] = (rng.standard_normal((5120, nt)) * 0.5).astype(np.float16)
    ts.append(g.v8_mix(L, hi, ho, c * nt))
t = float(np.median(ts[4:]))
if os.environ.get('OUT'): np.save(os.environ['OUT'], xo.astype(np.float32) - xi.astype(np.float32))
if os.environ.get("PROFILE"): g.v8_prof_dump()
print(f"NT {nt} layer {L}: {t:.2f} ms/call  {t / nt * 1000:.1f} us/token", flush=True)
