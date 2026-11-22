"""ANE health check after the hop probe's 100 s cancel: load the scratch probe package (journaled), run mm then cp, no Metal."""
import hashlib, os, sys, time
from pathlib import Path
V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
sys.path[:0] = [str(V02 / "scripts"), str(V02 / "coreai" / "swift_bridge"), "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4"]
import ane_compile_mode; ane_compile_mode.apply(log=lambda m: None)
import coreai_bridge as B, deploy_journal as DJ
pkg = Path("/Volumes/SSD4TB/anemll-forge-research/hop_probe/hop_probe.aimodel")
J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a", "out": "-", "mode": "ane_health"}, f"anehealth-{os.getpid()}-{int(time.time())}")
dev = DJ.m6_device(); pk = {"path": str(pkg), "label": "hop probe health check", "main_hash": (pkg / "main.hash").read_bytes().hex()}
aid = J.attempt(dev, pk, functions=sys.argv[1:], request={"compute": "ane", "via": "coreai_bridge"})
m = B.Model(pkg, compute="ane"); J.result(dev, aid, pk, "ok", model_load_s=0)
for name in sys.argv[1:]:
    f = m.function(name)
    ins = {n: f.buffer("input", n) for n in f.input_names}; outs = {n: f.buffer("output", n) for n in f.output_names}
    p = B.Plan([f.bind(ins, outs)])
    t = time.time(); p.run(); print(f"{name}: ok {1e3 * (time.time() - t):.2f} ms", flush=True)
