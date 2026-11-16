"""ANE DeltaNet mixer entries (v8_mix) vs the GPU engine's mixer, same input and state, two consecutive 64-row blocks:
y, conv rows and the recurrent state (ANE layout); mix2 vs two mix calls; timing. Journaled.
    MainPY mix_probe.py PKG LAYER"""
import ctypes as C_, hashlib, mmap, os, sys, time
from pathlib import Path
import numpy as np
for p in ("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02/coreai/swift_bridge",
          "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02/scripts",
          "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4"):
    sys.path.insert(0, p)
import ane_compile_mode as SOC
import coreai_bridge as B
import deploy_journal as DJ
SOC.apply(strict=True)
path, Li = Path(sys.argv[1]), int(sys.argv[2])
E = "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/engine"
os.environ.update({"V8_MANIFEST": "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/manifest_all64.json",
                   "V8_KERNELS": E + "/kernels.metal", "NT": "64", "LMAX": "4096"})
g = C_.CDLL(E + "/libgpu_v8.dylib")
g.v8_init.restype = C_.c_int32; g.v8_wrap.restype, g.v8_wrap.argtypes = C_.c_int32, [C_.c_void_p, C_.c_size_t]
g.v8_mix.restype, g.v8_mix.argtypes = C_.c_double, [C_.c_int32] * 4
g.v8_state.restype, g.v8_state.argtypes = C_.c_int32, [C_.c_int32, C_.c_void_p, C_.c_void_p]
g.v8_init()
FN = [f"mix_L{Li}", f"mix2_L{Li}"]
J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                "out": "mix probe", "mode": "v8_mix_probe"}, f"mixp-{os.getpid()}-{int(time.time())}")
dev = DJ.m6_device()
pk = {"path": str(path), "label": "V8 DeltaNet mixer package (probe)", "files": [path.name], "main_hash": (path / "main.hash").read_bytes().hex()}
aid = J.attempt(dev, pk, functions=FN, request={"compute": "ane"})
t0 = time.time()
try:
    m = B.Model(path, "ane")
except Exception as e:  # noqa: BLE001
    J.result(dev, aid, pk, "failed", error=f"{type(e).__name__}: {e}"); raise
J.result(dev, aid, pk, "ok", model_load_s=time.time() - t0)
f1, f2 = m.function(FN[0]), m.function(FN[1])
rng = np.random.default_rng(1)
X = [(rng.standard_normal((5120, 64)) * 0.3).astype(np.float16) for _ in range(2)]
# GPU, two blocks from zero state
n = 5120 * 64 * 2
bufs = [mmap.mmap(-1, n) for _ in range(2)]; arr = [np.frombuffer(b, np.float16).reshape(5120, 64) for b in bufs]
h = [g.v8_wrap(C_.addressof(C_.c_char.from_buffer(b)), n) for b in bufs]
g.v8_reset()
yg, cg, rg = [], np.empty((3, 10240), np.float16), np.empty((48, 128, 128), np.float16)
for b in range(2):
    arr[0][:] = X[b]; g.v8_mix(Li, h[0], h[1], b * 64); yg.append(arr[1].astype(np.float32).copy())
assert g.v8_state(Li, cg.ctypes.data, rg.ctypes.data) == 0
# ANE, mix twice (state carried by the host) and mix2 once
io = {n_: f1.buffer("input", n_) for n_ in ("x", "conv", "rec")}; oo = {n_: f1.buffer("output", n_) for n_ in ("y", "conv_out", "rec_out")}
p1 = B.Plan([f1.bind(io, oo)])
io["conv"].np[:] = 0; io["rec"].np[:] = 0
ya = []
for b in range(2):
    io["x"].np[0, :, 0, :] = X[b]; p1.run(); ya.append(oo["y"].np[0, :, 0, :].astype(np.float32).copy())
    io["conv"].np[:] = oo["conv_out"].np; io["rec"].np[:] = oo["rec_out"].np
ca, ra = oo["conv_out"].np.astype(np.float32).copy(), oo["rec_out"].np.astype(np.float32).copy()
i2 = {n_: f2.buffer("input", n_) for n_ in ("xa", "xb", "conv", "rec")}; o2 = {n_: f2.buffer("output", n_) for n_ in ("ya", "yb", "conv_out", "rec_out")}
p2 = B.Plan([f2.bind(i2, o2)])
i2["xa"].np[0, :, 0, :] = X[0]; i2["xb"].np[0, :, 0, :] = X[1]; i2["conv"].np[:] = 0; i2["rec"].np[:] = 0
p2.run()
rel = lambda a, b: float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))  # noqa: E731
for b in range(2):
    dg, da = yg[b] - X[b], ya[b] - X[b]
    print(f"block {b}: mixer delta ANE vs GPU rel {rel(da, dg):.4f} (|delta| / |x| {np.linalg.norm(dg) / np.linalg.norm(X[b]):.3f})")
print(f"conv rows ANE vs GPU rel {rel(ca, cg.astype(np.float32)):.4f}; rec state rel {rel(ra, rg.astype(np.float32)):.4f}")
print(f"mix2 vs two mix calls: yb rel {rel(o2['yb'].np[0, :, 0, :].astype(np.float32), ya[1]):.2e}, rec rel {rel(o2['rec_out'].np.astype(np.float32), ra):.2e}")
for nm, p in (("mix (1 block)", p1), ("mix2 (2 blocks)", p2)):
    for _ in range(10):
        p.run()
    ts = []
    for _ in range(100):
        t = time.perf_counter(); p.run(); ts.append(time.perf_counter() - t)
    print(f"{nm}: {np.median(ts) * 1e3:.3f} ms per call")
