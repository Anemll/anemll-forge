"""ANE weight-streaming rate (17 Nov 2026): the split's three calls of one 8-layer dec8 package (dva -> dvba -> dvb,
all 8 layers' weights once) in a loop for SECONDS, dense random inputs; prints per 0.5 s window the median time of a
triple and its rate (package weight bytes / time) with a wall-clock timestamp. Journaled load.
    PY bw_ane.py PACKAGE_DIR SECONDS [DELAY]"""
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
sys.path[:0] = [str(V02 / "coreai" / "swift_bridge"), str(V02 / "scripts"), "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4"]
import ane_compile_mode, deploy_journal as DJ  # noqa: E402
ane_compile_mode.apply(log=lambda m: None)
import coreai_bridge as B  # noqa: E402
pkg_dir, secs = Path(sys.argv[1]), float(sys.argv[2])
delay = float(sys.argv[3]) if len(sys.argv) > 3 else 0
info = json.loads((pkg_dir.parent / (pkg_dir.name + ".info.json")).read_text())
J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a", "out": "bw", "mode": "bw_ane"},
               f"bwane-{os.getpid()}-{int(time.time())}")
dev = DJ.m6_device(); pkg = {"path": str(pkg_dir), "main_hash": (pkg_dir / "main.hash").read_bytes().hex(), "label": "dec8 package, ANE bandwidth probe"}
aid = J.attempt(dev, pkg, functions=["dva", "dvba", "dvb"], request={"compute": "ane", "via": "coreai_bridge"})
t0 = time.time()
try:
    m = B.Model(pkg_dir, compute="ane")
    F = {n: m.function(n) for n in ("dva", "dvba", "dvb")}
except Exception as e:  # noqa: BLE001
    J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
r = np.random.default_rng(1)
plans = []
for n, f in F.items():
    ins = {k: f.buffer("input", k) for k in f.input_names}
    for k, b in ins.items():
        if b.np.dtype == np.float16:
            b.np[...] = (r.standard_normal(b.shape) * (1.0 if k in ("cos", "sin") else 0.02)).astype(np.float16)
    plans.append(B.Plan([f.bind(ins, {k: f.buffer("output", k) for k in f.output_names})]))
for _ in range(5):
    for p in plans: p.run()
time.sleep(delay)
nbytes = info["mb"] * 1e6
t_end, win, ws = time.time() + secs, [], time.time()
while time.time() < t_end:
    t1 = time.perf_counter()
    for p in plans: p.run()
    win.append(time.perf_counter() - t1)
    now = time.time()
    if now - ws >= 0.5:
        md = float(np.median(win)); print(f"{now:.3f} ANE triple {1e3 * md:.2f} ms {nbytes / md / 1e9:.1f} GB/s", flush=True); win, ws = [], now
