"""Time the cliff probe entries (sync, one binding per plan) and compare outputs. Journaled.  MainPY chunk_probe.py PKG"""
import hashlib, os, sys, time
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
path = Path(sys.argv[1])
FN = ["t64", "t64c2r", "t64a", "t64b", "t64d", "t64e", "t64f", "t64g"]
J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                "out": "chunk probe", "mode": "v8_chunk_probe"}, f"chunk-{os.getpid()}-{int(time.time())}")
dev = DJ.m6_device()
pk = {"path": str(path), "label": "V8 FFN cliff / chunk probe (layer 0)", "files": [path.name],
      "main_hash": (path / "main.hash").read_bytes().hex()}
aid = J.attempt(dev, pk, functions=FN, request={"compute": "ane"})
t0 = time.time()
try:
    m = B.Model(path, "ane")
except Exception as e:  # noqa: BLE001
    J.result(dev, aid, pk, "failed", error=f"{type(e).__name__}: {e}"); raise
J.result(dev, aid, pk, "ok", model_load_s=time.time() - t0)
print(f"load {time.time() - t0:.1f}s", flush=True)
X = (np.random.default_rng(0).standard_normal((1, 5120, 1, 128)) * 0.3).astype(np.float16)
outs = {}
for nm in FN:
    f = m.function(nm)
    x, y = f.buffer("input", "x"), f.buffer("output", "y")
    T = x.np.shape[-1]
    x.np[:] = X[..., :T]
    p = B.Plan([f.bind({"x": x}, {"y": y})])
    for _ in range(5):
        p.run()
    ts = []
    for _ in range(200):
        t = time.perf_counter(); p.run(); ts.append(time.perf_counter() - t)
    ms = float(np.median(ts)) * 1e3
    outs[nm] = y.np.astype(np.float32).copy()
    print(f"{nm}: {ms:.3f} ms (min {min(ts) * 1e3:.3f}) | {ms / T * 1e3:.1f} us/token | {ms / T * 64:.3f} ms per 64 rows", flush=True)
for nm in FN[1:]:
    d = np.abs(outs[nm] - outs["t64"]); print(f"{nm} vs t64: max|diff| {d.max():.4g} (rel {d.max() / np.abs(outs['t64']).max():.2e})")
