"""ANE-only control for the trace: one FFN package, pair plans (two 64-row bindings) back to back, GPU idle; then
single-binding plans, then an 8-binding plan.   MainPY ane_alone.py PKG"""
import hashlib, os, sys, time, numpy as np
from pathlib import Path
sys.path.insert(0, "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02/coreai/swift_bridge")
sys.path.insert(0, "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02/scripts")
import ane_compile_mode as SOC
import coreai_bridge as B
sys.path.insert(0, "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4")
import deploy_journal as DJ
SOC.apply(strict=True)
J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                "out": "trace control", "mode": "v8_ane_alone_trace"}, f"v8a-{os.getpid()}-{int(time.time())}")
dev = DJ.m6_device(); path = Path(sys.argv[1])
pk = {"path": str(path), "label": "V8 FFN-only (ANE-alone trace control)", "files": [path.name],
      "main_hash": (path / "main.hash").read_bytes().hex()}
aid = J.attempt(dev, pk, functions=["ffn_L00"], request={"compute": "ane"})
t0 = time.time()
try:
    m = B.Model(path, "ane")
except Exception as e:  # noqa: BLE001
    J.result(dev, aid, pk, "failed", error=f"{type(e).__name__}: {e}"); raise
J.result(dev, aid, pk, "ok", model_load_s=time.time() - t0)
f = m.function("ffn_L00")
bs = []
for _ in range(8):
    x, y = f.buffer("input", "x"), f.buffer("output", "y")
    x.np[:] = (np.random.default_rng(0).standard_normal(x.np.shape) * 0.3).astype(np.float16)
    bs.append(f.bind({"x": x}, {"y": y}))
print("ready", flush=True)
time.sleep(3)  # the trace attaches here
for name, plan, n in (("pair", B.Plan(bs[:2]), 300), ("single", B.Plan(bs[:1]), 300), ("eight", B.Plan(bs), 80)):
    t = time.perf_counter()
    for _ in range(n):
        plan.run()
    dt = (time.perf_counter() - t) / n / len(plan.bindings)
    print(f"{name}: {dt * 1e3:.3f} ms per binding (wall)", flush=True)
    time.sleep(0.5)
