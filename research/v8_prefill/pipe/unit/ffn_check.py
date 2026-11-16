"""GPU FFN of one layer (chunked shared build, prep_ffn.py tables) vs the ANE's own ffn2_L<i> entry on the same input
(journaled Core AI load), fp16 and W8A8 (I8F) GPU paths in separate processes; prints GPU ms and the error of the
FFN delta (out - in) vs the ANE.    MainPY ffn_check.py LAYER CALLS [TAG]   (env: DYLIB, I8F, I8TV, FUSE, LUTC)"""
import ctypes as C, hashlib, mmap, os, sys, time
from pathlib import Path
import numpy as np
V02 = "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02"
for p in (V02 + "/coreai/swift_bridge", V02 + "/scripts",
          "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4"):
    sys.path.insert(0, p)
import coreai_bridge as CB
import deploy_journal as DJ
E = "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/engine"
layer, calls = int(sys.argv[1]), int(sys.argv[2]); tag = sys.argv[3] if len(sys.argv) > 3 else ""
os.environ.update({"V8_MANIFEST": "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/w_shared2r/manifest_ffn.json",
                   "V8_KERNELS": E + "/kernels.metal", "NT": "128", "LMAX": "256", "KV8": "0"})
g = C.CDLL(E + "/" + os.environ.get("DYLIB", "libgpu_v8_fuse_dev.dylib"))
g.v8_init.restype = C.c_int32; g.v8_wrap.restype, g.v8_wrap.argtypes = C.c_int32, [C.c_void_p, C.c_size_t]
g.v8_ffn.restype, g.v8_ffn.argtypes = C.c_double, [C.c_int32] * 3
g.v8_init()
def mbuf(n):
    b = mmap.mmap(-1, n); return b, g.v8_wrap(C.addressof(C.c_char.from_buffer(b)), n)
bi, hi = mbuf(5120 * 128 * 2); bo, ho = mbuf(5120 * 128 * 2)
x = np.frombuffer(bi, np.float16).reshape(5120, 128)
x[:] = (np.random.default_rng(7).standard_normal((5120, 128)) * 0.5).astype(np.float16)
g.v8_ffn(layer, hi, ho); gpu_out = np.frombuffer(bo, np.float16).reshape(5120, 128).astype(np.float32).copy()
t = np.array([g.v8_ffn(layer, hi, ho) for _ in range(calls)])
# the ANE's ffn2 entry of the same layer on the same input (blocks A = rows 0..63, B = 64..127)
build = Path("/Volumes/SSD4TB/anemll-forge-research/v8_shared2")
import json
ch = next(c for c in json.loads((build / "manifest.json").read_text())["chunks"] if c["layers"][0] <= layer <= c["layers"][1])
path, entry = build / ch["file"], f"ffn2_L{layer}"
J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                "out": "ffn check", "mode": "ffn_check"}, f"ffnchk-{os.getpid()}-{int(time.time())}")
dev = DJ.m6_device()
pk = {"path": str(path), "label": f"ffn check {path.name}:{entry}", "files": [path.name], "main_hash": (path / "main.hash").read_bytes().hex()}
aid = J.attempt(dev, pk, functions=[entry], request={"compute": "ane"})
try:
    m = CB.Model(path, "ane")
except Exception as e:  # noqa: BLE001
    J.result(dev, aid, pk, "failed", error=f"{type(e).__name__}: {e}"); raise
J.result(dev, aid, pk, "ok", model_load_s=0.0)
f = m.function(entry)
ins = {n: f.buffer("input", n) for n in f.input_names}; outs = {n: f.buffer("output", n) for n in f.output_names}
for n, h in (("xa", slice(0, 64)), ("xb", slice(64, 128))):
    ins[n].np[...] = x[:, h].reshape(ins[n].np.shape)
CB.Plan([f.bind(ins, outs)]).run()
ane_out = np.concatenate([outs["ya"].np.reshape(5120, 64), outs["yb"].np.reshape(5120, 64)], 1).astype(np.float32)
xf = x.astype(np.float32)
rel = np.linalg.norm(gpu_out - ane_out) / np.linalg.norm(ane_out - xf)
print(f"FFN L{layer} {tag:18s} GPU median {np.median(t):6.3f} ms  vs ANE ffn2: FFN-delta rel err {rel:.2e}  max abs {np.abs(gpu_out - ane_out).max():.3g}", flush=True)
