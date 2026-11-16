"""Loop one entry of a compiled package on a compute unit for SECONDS with random inputs (journaled); prints calls/s and
the median call time.   MainPY ane_load.py PKG ENTRY SECONDS [ane|gpu] [TAG]"""
import hashlib, os, sys, time
from pathlib import Path
import numpy as np
V02 = "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02"
for p in (V02 + "/coreai/swift_bridge", V02 + "/scripts",
          "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4"):
    sys.path.insert(0, p)
import coreai_bridge as CB
import deploy_journal as DJ
path, entry, secs = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3])
unit = sys.argv[4] if len(sys.argv) > 4 else "ane"; tag = sys.argv[5] if len(sys.argv) > 5 else ""
J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                "out": "ane load", "mode": "ane_load"}, f"aload-{os.getpid()}-{int(time.time())}")
dev = DJ.m6_device()
pk = {"path": str(path), "label": f"load loop {path.name}:{entry} on {unit}", "files": [path.name],
      "main_hash": (path / "main.hash").read_bytes().hex()}
aid = J.attempt(dev, pk, functions=[entry], request={"compute": unit})
t0 = time.time()
try:
    m = CB.Model(path, unit)
except Exception as e:  # noqa: BLE001
    J.result(dev, aid, pk, "failed", error=f"{type(e).__name__}: {e}"); raise
J.result(dev, aid, pk, "ok", model_load_s=time.time() - t0)
load_s = time.time() - t0
f = m.function(entry)
ins = {n: f.buffer("input", n) for n in f.input_names}
outs = {n: f.buffer("output", n) for n in f.output_names}
rng = np.random.default_rng(1)
for n, b in ins.items():
    if n not in ("conv", "rec"):
        b.np[...] = (rng.standard_normal(b.np.shape) * 0.3).astype(b.np.dtype)
pl = CB.Plan([f.bind(ins, outs)])
for _ in range(3):
    pl.run()
ts = []; t_end = time.time() + secs; t1 = time.time()
while time.time() < t_end:
    t = time.perf_counter(); pl.run(); ts.append((time.perf_counter() - t) * 1e3)
el = time.time() - t1
print(f"{unit.upper()} {tag:22s} {path.name}:{entry} median {np.median(ts):7.3f} ms  {len(ts) / el:7.1f} calls/s  busy {sum(ts) / 1e3 / el * 100:4.0f}%  (load {load_s:.1f}s)", flush=True)
