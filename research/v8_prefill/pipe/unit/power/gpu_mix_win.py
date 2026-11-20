"""Windowed GPU mixer load through the V8 engine (18 Nov 2026; gpu_steady.py with timestamps): MODE att (attention layer
3 at position P, KV rows below P not computed), gdn (DeltaNet layer 0), gdnall (all 48 DeltaNet layers in turn) or both (alternating, as gpu_steady.py), 128-row
calls for SECONDS; every 0.5 s prints "<epoch> GPU <mode> att <ms> gdn <ms>" (median GPU ms per call in the window, Metal
timestamps), then a final median line. DYLIB (default libgpu_v8.dylib, gpu_steady's engine) and the engine env (FUSE,
I8, I8T, I8TV, LUTC, I8A, ...: the INT8 DeltaNet path is DYLIB=libgpu_v8_fuse.dylib FUSE=all I8=1 I8T=1 I8TV=_k256
LUTC=1 I8A=1) pass through; MANIFEST defaults to w_shared2r (as gpu_steady.py).
    MainPY gpu_mix_win.py MODE SECONDS [P] [TAG]"""
import ctypes as C, mmap, os, sys, time, numpy as np
E = "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/engine"
mode, secs = sys.argv[1], float(sys.argv[2]); P = int(sys.argv[3]) if len(sys.argv) > 3 else 61440
tag = sys.argv[4] if len(sys.argv) > 4 else ""
os.environ.update({"V8_MANIFEST": os.environ.get("MANIFEST", "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/w_shared2r/manifest.json"),
                   "V8_KERNELS": E + "/kernels.metal", "NT": "128", "LMAX": str(P + 128), "KV8": os.environ.get("KV8", "0")})
g = C.CDLL(E + "/" + os.environ.get("DYLIB", "libgpu_v8.dylib"))
g.v8_init.restype = C.c_int32; g.v8_wrap.restype, g.v8_wrap.argtypes = C.c_int32, [C.c_void_p, C.c_size_t]
g.v8_mix.restype, g.v8_mix.argtypes = C.c_double, [C.c_int32] * 4
g.v8_init()
def mbuf(n):
    b = mmap.mmap(-1, n); return b, g.v8_wrap(C.addressof(C.c_char.from_buffer(b)), n)
n = 5120 * 128 * 2
bi, hi = mbuf(n); bo, ho = mbuf(n)
np.frombuffer(bi, np.float16).reshape(5120, 128)[:] = (np.random.default_rng(5).standard_normal((5120, 128)) * 0.5).astype(np.float16)
g.v8_reset()
for _ in range(3):
    g.v8_mix(3, hi, ho, P); g.v8_mix(0, hi, ho, P)
print(f"ready {time.time():.3f}", flush=True)
ta, tg, wa, wg = [], [], [], []
t_end = time.time() + secs; w0 = time.time()
while time.time() < t_end:
    if mode in ("att", "both"):
        x = g.v8_mix(3, hi, ho, P); ta.append(x); wa.append(x)
    if mode in ("gdn", "both"):
        x = g.v8_mix(0, hi, ho, P); tg.append(x); wg.append(x)
    if mode == "gdnall":   # every DeltaNet layer in turn (cold caches, as in the pipeline)
        for L in range(64):
            if L % 4 != 3:
                x = g.v8_mix(L, hi, ho, P); tg.append(x); wg.append(x)
    if time.time() - w0 >= 0.5:
        f = lambda v: f"{np.median(v):7.3f}" if v else "      -"  # noqa: E731
        print(f"{time.time():.3f} GPU {mode} att {f(wa)} gdn {f(wg)}", flush=True)
        wa, wg, w0 = [], [], time.time()
f = lambda v: f"{np.median(v):6.2f}" if v else "-"  # noqa: E731
print(f"GPU {tag:22s} {mode} attention@{P // 1024}K {f(ta)} ms  DeltaNet {f(tg)} ms  ({len(ta)} / {len(tg)} calls)", flush=True)
