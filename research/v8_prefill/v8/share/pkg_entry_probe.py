"""Run one entry of an existing compiled package on the ANE a few times (zero inputs), journaled: which programs
dispatch and which hang.   MainPY pkg_entry_probe.py PKG ENTRY [N]"""
import hashlib, os, sys, time
from pathlib import Path
V02 = "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02"
for p in (V02 + "/coreai/swift_bridge", V02 + "/scripts",
          "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4"):
    sys.path.insert(0, p)
import coreai_bridge as CB
import deploy_journal as DJ
path, entry = Path(sys.argv[1]), sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 5
J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                "out": "pkg entry probe", "mode": "pkg_entry_probe"}, f"pep-{os.getpid()}-{int(time.time())}")
dev = DJ.m6_device()
pk = {"path": str(path), "label": f"entry probe {path.name}:{entry}", "files": [path.name],
      "main_hash": (path / "main.hash").read_bytes().hex()}
aid = J.attempt(dev, pk, functions=[entry], request={"compute": "ane"})
t0 = time.time()
try:
    m = CB.Model(path, "ane")
except Exception as e:  # noqa: BLE001
    J.result(dev, aid, pk, "failed", error=f"{type(e).__name__}: {e}"); raise
J.result(dev, aid, pk, "ok", model_load_s=time.time() - t0)
f = m.function(entry)
ins = {n: f.buffer("input", n) for n in f.input_names}
outs = {n: f.buffer("output", n) for n in f.output_names}
pl = CB.Plan([f.bind(ins, outs)])
print(f"{path.name}:{entry} loaded {time.time() - t0:.1f}s, running", flush=True)
ts = []
for _ in range(N):
    t = time.perf_counter(); pl.run(); ts.append((time.perf_counter() - t) * 1e3)
print(f"OK {path.name}:{entry} median {sorted(ts)[len(ts) // 2]:.2f} ms", flush=True)
