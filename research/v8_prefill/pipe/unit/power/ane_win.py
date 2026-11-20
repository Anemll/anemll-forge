"""Windowed ANE load (18 Nov 2026; v8/share/ane_load.py with timestamps): loops one entry of a compiled package on the
ANE for SECONDS with random inputs (journaled); every 0.5 s prints "<epoch> ANE <entry> <median ms> busy <pct>", then a
final median line. ANE_SLEEP_MS: sleep after every call (rate-controlled traffic for counter calibration).
    MainPY ane_win.py PKG ENTRY SECONDS [TAG]"""
import hashlib, os, sys, time
os.environ.setdefault("MPSGRAPH_ANE_BONDED_COMPILE_MODE", "2")   # as the server and contention.sh
from pathlib import Path
import numpy as np
V02 = "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02"
for p in (V02 + "/coreai/swift_bridge", V02 + "/scripts",
          "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4"):
    sys.path.insert(0, p)
import coreai_bridge as CB
import deploy_journal as DJ
path, entry, secs = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3]); tag = sys.argv[4] if len(sys.argv) > 4 else ""
J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                "out": "coupling load", "mode": "ane_win"}, f"anewin-{os.getpid()}-{int(time.time())}")
dev = DJ.m6_device()
pk = {"path": str(path), "label": f"coupling load {path.name}:{entry}", "files": [path.name],
      "main_hash": (path / "main.hash").read_bytes().hex() if (path / "main.hash").exists() else "UNKNOWN"}
aid = J.attempt(dev, pk, functions=[entry], request={"compute": "ane"})
t0 = time.time()
try:
    m = CB.Model(path, "ane")
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
print(f"ready {time.time():.3f} (load {load_s:.1f}s)", flush=True)
SLEEP = float(os.environ.get("ANE_SLEEP_MS", "0")) / 1e3
ts, w = [], []; t_end = time.time() + secs; t1 = w0 = time.time(); nwin = 0
while time.time() < t_end:
    t = time.perf_counter(); pl.run(); d = (time.perf_counter() - t) * 1e3; ts.append(d); w.append(d)
    if SLEEP:
        time.sleep(SLEEP)
    if time.time() - w0 >= 0.5:
        el = time.time() - w0
        print(f"{time.time():.3f} ANE {entry} {np.median(w):7.3f} busy {sum(w) / 1e3 / el * 100:4.0f} calls_per_s {len(w) / el:.1f}", flush=True)
        w, w0 = [], time.time()
el = time.time() - t1
print(f"ANE {tag:22s} {path.name}:{entry} median {np.median(ts):7.3f} ms  {len(ts) / el:7.1f} calls/s  (load {load_s:.1f}s)", flush=True)
