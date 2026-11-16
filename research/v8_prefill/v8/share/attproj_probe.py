"""Time the attproj probe entries (ffn2_L2, ffnq2_L2, offn2_L3) on the ANE, sync, alone. Journaled.
    MainPY attproj_probe.py PKG"""
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
SPEC = {"ffn2_L2": (["xa", "xb"], ["ya", "yb"]),
        "ffnq2_L2": (["xa", "xb"], ["ya", "qa", "ka", "va", "yb", "qb", "kb", "vb"]),
        "offn2_L3": (["oa", "ob", "xa", "xb"], ["ya", "yb"])}
J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                "out": "attproj probe", "mode": "v8_attproj_probe"}, f"attp-{os.getpid()}-{int(time.time())}")
dev = DJ.m6_device()
pk = {"path": str(path), "label": "V8 attention-projection fusion probe", "files": [path.name], "main_hash": (path / "main.hash").read_bytes().hex()}
aid = J.attempt(dev, pk, functions=list(SPEC), request={"compute": "ane"})
t0 = time.time()
try:
    m = B.Model(path, "ane")
except Exception as e:  # noqa: BLE001
    J.result(dev, aid, pk, "failed", error=f"{type(e).__name__}: {e}"); raise
J.result(dev, aid, pk, "ok", model_load_s=time.time() - t0)
rng = np.random.default_rng(0)
plans = {}
for fn, (ins, outs) in SPEC.items():
    f = m.function(fn)
    ib = {n: f.buffer("input", n) for n in ins}; ob = {n: f.buffer("output", n) for n in outs}
    for b in ib.values():
        b.np[:] = (rng.standard_normal(b.np.shape) * 0.3).astype(np.float16)
    plans[fn] = (B.Plan([f.bind(ib, ob)]), ib, ob)
res = {}
for rnd in range(3):
    for n, (p, _, _) in plans.items():
        for _ in range(10):
            p.run()
        ts = []
        for _ in range(100):
            t = time.perf_counter(); p.run(); ts.append(time.perf_counter() - t)
        res.setdefault(n, []).append(float(np.median(ts)) * 1e3)
for n, v in res.items():
    print(f"{n:10s} median ms per pair: " + " ".join(f"{x:.3f}" for x in v))
b = min(res["ffn2_L2"])
print(f"extra on the ANE per pair: q/k/v projections {min(res['ffnq2_L2']) - b:.3f} ms, o projection {min(res['offn2_L3']) - b:.3f} ms")
