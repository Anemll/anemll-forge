"""C2 (20 Nov 2026): a JetSpec++ adapter pass on the ANE, in a loop: PASS_CALLS (12) back-to-back calls of one LUT6 adapter
block entry (16 rows; each call streams the block's 280 MB of weights again, so a pass moves the adapter's 3.35 GB like
the 12 real blocks), logging every pass's wall ms with its start time until STOP_FILE exists or SECONDS elapse.
Journaled. Banner: package path + main.hash sha, entry, calls per pass.
    PY ane_pass_loop.py PACKAGE ENTRY OUT.jsonl [SECONDS]"""
import hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
sys.path[:0] = [str(V02 / "scripts"), str(V02 / "coreai" / "swift_bridge"),
                "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4",
                str(Path(__file__).resolve().parent)]
import coreai_bridge as B  # noqa: E402
import deploy_journal as DJ  # noqa: E402

pkg, entry, outp = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
secs = float(sys.argv[4]) if len(sys.argv) > 4 else 600
calls = int(os.environ.get("PASS_CALLS", "12"))
stop = Path(os.environ.get("STOP_FILE", "/tmp/claude-501/ane_loop.stop"))
src = (Path(__file__).resolve().parent / "tree_probe_time.py").read_text()
ns = {}
exec(src[src.index("def fill"):src.index("res = {")], {"np": np}, ns)
target = pkg.with_suffix(".aimodelc") if pkg.with_suffix(".aimodelc").exists() else pkg
J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a", "out": str(outp),
                "mode": "ane_pass_loop"}, f"anepassloop-{os.getpid()}-{int(time.time())}")
dev = DJ.m6_device()
pk = {"path": str(target), "label": "C2 ANE adapter-pass loop"}
aid = J.attempt(dev, pk, functions=[entry], request={"compute": "ane"})
model = B.Model(target, compute="ane")
J.result(dev, aid, pk, "ok", model_load_s=0)
fn = model.function(entry)
rng = np.random.default_rng(0)
anc, ctap = ns["chain_masks"](16)
ins = {n: ns["fill"](fn, n, 16, 16, None, anc, ctap, rng) for n in fn.input_names}
outs = {n: fn.buffer("output", n) for n in fn.output_names}
plan = B.Plan([fn.bind(ins, outs)])
for _ in range(5):
    plan.run()
sha = hashlib.sha256((pkg / "main.hash").read_bytes()).hexdigest()[:16] if (pkg / "main.hash").exists() else "-"
print(f"BANNER ane_pass_loop | {pkg} main.hash sha {sha} | entry {entry} | {calls} calls per pass | stop {stop}", flush=True)
t_end = time.time() + secs
with open(outp, "w") as f:
    while time.time() < t_end and not stop.exists():
        t0 = time.time(); p0 = time.perf_counter()
        for _ in range(calls):
            plan.run()
        f.write(json.dumps({"t": t0, "ms": 1e3 * (time.perf_counter() - p0)}) + "\n"); f.flush()
print("loop done", flush=True)
