"""One mixer layer (DeltaNet, or attention at P0) alone on the V8 GPU engine: 128-row calls of one GDN layer, median GPU ms (Metal timestamps).
Env passed through to the engine: PROFILE=1 (per-kernel ms, each dispatch in its own command buffer), SKIP=a,b (timing
ablation), CONC, REC8, ... DYLIB picks the engine (default libgpu_v8_attcore.dylib).
    python gdn_prof.py CALLS [LAYER] [TAG]"""
import ctypes as C, mmap, os, sys, numpy as np
E = "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/engine"
calls = int(sys.argv[1]); layer = int(sys.argv[2]) if len(sys.argv) > 2 else 0; tag = sys.argv[3] if len(sys.argv) > 3 else ""
P0 = int(os.environ.get("P0", "0")); NTV = int(os.environ.get("NTV", "128"))   # NTV: rows per call   # attention layers: call position (keys 0 .. P0 + 127; rows below P0 not computed)
os.environ.update({"V8_MANIFEST": os.environ.get("MANIFEST", "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/w_shared2r/manifest.json"),
                   "V8_KERNELS": E + "/kernels.metal", "NT": str(NTV), "LMAX": str(P0 + 2 * NTV), "KV8": "0"})
g = C.CDLL(E + "/" + os.environ.get("DYLIB", "libgpu_v8_attcore.dylib"))
g.v8_init.restype = C.c_int32; g.v8_wrap.restype, g.v8_wrap.argtypes = C.c_int32, [C.c_void_p, C.c_size_t]
g.v8_mix.restype, g.v8_mix.argtypes = C.c_double, [C.c_int32] * 4
g.v8_init()
def mbuf(n):
    b = mmap.mmap(-1, n); return b, g.v8_wrap(C.addressof(C.c_char.from_buffer(b)), n)
n = 5120 * NTV * 2
bi, hi = mbuf(n); bo, ho = mbuf(n)
np.frombuffer(bi, np.float16)[:] = (np.random.default_rng(5).standard_normal(5120 * NTV) * 0.5).astype(np.float16)
if os.environ.get("SAVE"):   # SAVE=out.npy: outputs of 4 consecutive calls from a fresh state (compare variants)
    g.v8_reset(); outs = []
    for _ in range(4):
        g.v8_mix(layer, hi, ho, P0); outs.append(np.frombuffer(bo, np.float16).copy())
    np.save(os.environ["SAVE"], np.stack(outs))
g.v8_reset()
for _ in range(10):
    g.v8_mix(layer, hi, ho, P0)
prof = "PROFILE" in os.environ
if prof:
    print("(warm-up profile, discarded)"); g.v8_prof_dump()
t = np.array([g.v8_mix(layer, hi, ho, P0) for _ in range(calls)])
print(f"L{layer}@{P0} {tag:28s} median {np.median(t):6.3f} ms  p10 {np.percentile(t, 10):6.3f}  ({calls} calls)", flush=True)
if prof:
    g.v8_prof_dump()
