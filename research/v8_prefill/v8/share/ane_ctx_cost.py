"""ANE prefill cost vs context rung: the same 64-row blocks at the same positions (0 .. 64*NB) through each rung's
p64_<ctx>k entry (the bridge's ladder, resized up; reset keeps the rung). The difference between rungs is the cost of
the wider (masked) attention history; extrapolating down gives what 2K / 4K rungs would save. Journaled.
    MainPY ane_ctx_cost.py BUILD [NB]"""
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
V02 = "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02"
for p in (V02 + "/scripts", V02 + "/coreai", V02 + "/coreai/swift_bridge",
          "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4"):
    sys.path.insert(0, p)
import deploy_journal as DJ
from qwen38_coreai_model import CoreAIQwenBridge
build = Path(sys.argv[1]); NB = int(sys.argv[2]) if len(sys.argv) > 2 else 16
man = json.loads((build / "manifest.json").read_text())
J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                "out": "ane ctx cost", "mode": "ane_ctx_cost"}, f"actx-{os.getpid()}-{int(time.time())}")
dev = DJ.m6_device()
pkg = {"path": str(build), "label": "target entries (ctx rung cost probe)",
       "files": [c["file"] for c in man["chunks"]] + [man["head"]["file"]]}
aid = J.attempt(dev, pkg, functions=["p64", "h8"], request={"compute": "ane"})
t0 = time.time()
LADDER = [8192, 16384, 32768, 65536, 98304]
try:
    m = CoreAIQwenBridge(ctx=8192, ladder=LADDER, root=build, log=lambda s: None)
except Exception as e:  # noqa: BLE001
    J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
rng = np.random.default_rng(0)
ids = [int(x) for x in rng.integers(1000, 100000, NB * 64)]
res = {}
for C in LADDER:
    assert m.fit(C), C
    for rnd in range(2):   # round 0 warms the rung's entries up
        m.reset(shrink=False)
        assert m.ctx == C, (m.ctx, C)
        ts = []
        for b in range(NB):
            t = time.perf_counter(); m.prefill_block(ids[b * 64:(b + 1) * 64]); ts.append((time.perf_counter() - t) * 1e3)
    res[C] = ts
    print(f"rung {C // 1024:3d}K  per 64-row block: median {np.median(ts):7.2f} ms  min {min(ts):7.2f}  "
          f"=> {64 / np.median(ts) * 1e3:5.0f} tok/s", flush=True)
c = sorted(res); y = [np.median(res[k]) for k in c]
slope, icpt = np.polyfit(c, y, 1)
print(f"fit: block ms = {icpt:.2f} + {slope * 1024:.3f} per 1K of rung width (16 attention layers)")
for C in (2048, 4096, 8192):
    print(f"  predicted {C // 1024}K rung: {icpt + slope * C:.2f} ms/block => {64 / (icpt + slope * C) * 1e3:.0f} tok/s")
