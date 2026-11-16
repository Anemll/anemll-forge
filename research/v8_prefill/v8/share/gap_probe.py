"""Per-prediction gap on the ANE: time ffn_L<i> (one 64-row block) and ffn2_L<i> (two blocks, one prediction) of one
v8_ffn2c package, sync, alone. Journaled.   MainPY gap_probe.py PKG LAYER"""
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
path, Li = Path(sys.argv[1]), int(sys.argv[2])
FN = [f"ffn_L{Li}", f"ffn2_L{Li}"]
J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                "out": "gap probe", "mode": "v8_gap_probe"}, f"gap-{os.getpid()}-{int(time.time())}")
dev = DJ.m6_device()
pk = {"path": str(path), "label": "V8 FFN-only package (gap probe)", "files": [path.name], "main_hash": (path / "main.hash").read_bytes().hex()}
aid = J.attempt(dev, pk, functions=FN, request={"compute": "ane"})
t0 = time.time()
try:
    m = B.Model(path, "ane")
except Exception as e:  # noqa: BLE001
    J.result(dev, aid, pk, "failed", error=f"{type(e).__name__}: {e}"); raise
J.result(dev, aid, pk, "ok", model_load_s=time.time() - t0)
rng = np.random.default_rng(0)
f1, f2 = m.function(FN[0]), m.function(FN[1])
x1, y1 = f1.buffer("input", "x"), f1.buffer("output", "y")
xa, xb, ya, yb = f2.buffer("input", "xa"), f2.buffer("input", "xb"), f2.buffer("output", "ya"), f2.buffer("output", "yb")
for b in (x1, xa, xb):
    b.np[:] = (rng.standard_normal(b.np.shape) * 0.3).astype(np.float16)
plans = {"1 block x1": B.Plan([f1.bind({"x": x1}, {"y": y1})]),
         "1 block x2 (2 predictions, one plan)": B.Plan([f1.bind({"x": x1}, {"y": y1}), f1.bind({"x": xa}, {"y": ya})]),
         "ffn2 (2 blocks, 1 prediction)": B.Plan([f2.bind({"xa": xa, "xb": xb}, {"ya": ya, "yb": yb})])}
res = {}
for rnd in range(3):
    for n, p in plans.items():
        for _ in range(10):
            p.run()
        ts = []
        for _ in range(200):
            t = time.perf_counter(); p.run(); ts.append(time.perf_counter() - t)
        res.setdefault(n, []).append(float(np.median(ts)) * 1e3)
for n, v in res.items():
    print(f"{n:40s} median ms per run: " + " ".join(f"{x:.3f}" for x in v))
t1, t11, t2 = (min(res[k]) for k in plans)
print(f"gap estimate: 2 x single - two-in-one-plan = {2 * t1 - t11:.3f} ms; two predictions - ffn2 = {t11 - t2:.3f} ms")
