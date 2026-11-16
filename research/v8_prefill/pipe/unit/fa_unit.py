"""Attention layer 3 through the GPU engine (fp16 KV), NT=128: fill to NPOS tokens, time calls near checkpoints,
save the mixer outputs (x_out - x_in) of the calls at each checkpoint. argv: out.npz NPOS; env FA2 etc. passed through."""
import ctypes as C, mmap, os, sys, numpy as np
E = "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/engine"
out, npos = sys.argv[1], int(sys.argv[2])
os.environ.update({"V8_MANIFEST": "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/w_shared/manifest.json",
                   "V8_KERNELS": E + "/kernels.metal", "NT": "128", "LMAX": str(npos + 128), "KV8": "0"})
g = C.CDLL(E + "/" + os.environ.get("DYLIB", "libgpu_v8_fa2.dylib"))
g.v8_init.restype = C.c_int32; g.v8_wrap.restype, g.v8_wrap.argtypes = C.c_int32, [C.c_void_p, C.c_size_t]
g.v8_mix.restype, g.v8_mix.argtypes = C.c_double, [C.c_int32] * 4
g.v8_init()
def mbuf(n):
    b = mmap.mmap(-1, n); return b, g.v8_wrap(C.addressof(C.c_char.from_buffer(b)), n)
n = 5120 * 128 * 2
bi, hi = mbuf(n); bo, ho = mbuf(n)
xi = np.frombuffer(bi, np.float16).reshape(5120, 128); xo = np.frombuffer(bo, np.float16).reshape(5120, 128)
rng = np.random.default_rng(5)
g.v8_reset()
chk = {p for p in (0, 1024, 4096, 8192, 16384, 32768, 65536) if p < npos}
ys, ts = {}, {}
for c in range(npos // 128):
    p = c * 128
    xi[:] = (rng.standard_normal((5120, 128)) * 0.5).astype(np.float16)
    t = g.v8_mix(3, hi, ho, p)
    assert t > 0, t
    if p in chk or (p - 128) in chk or (p - 256) in chk:
        base = max(q for q in chk if q <= p)
        ts.setdefault(base, []).append(t)
    if p in chk:
        ys[p] = (xo.astype(np.float32) - xi.astype(np.float32)).copy()
if os.environ.get("PROFILE"): g.v8_prof_dump()
np.savez(out, **{f"y{p}": v for p, v in ys.items()}, **{f"t{p}": np.array(v) for p, v in ts.items()})
print(os.environ.get("FA2", "base"), " ".join(f"{p}:{np.median(v):.2f}" for p, v in sorted(ts.items())), flush=True)
