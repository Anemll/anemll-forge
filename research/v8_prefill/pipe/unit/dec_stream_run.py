"""Journaled launcher of the Core AI ComputeStream test (unit/dec_stream/dec_stream): the Swift process loads probe 2 on
the ANE, so the load is journaled here (deploy_journal v7) around the subprocess.
    BUILDPY dec_stream_run.py PROBE2_DIR"""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02/scripts")
sys.path.insert(0, "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4")
import deploy_journal as DJ  # noqa: E402

here = Path(__file__).resolve().parent
probe = Path(sys.argv[1])
info = json.loads(next(probe.glob("*.aimodel.info.json")).read_text())
path = probe / info["file"]
exe = here / "dec_stream" / "dec_stream"
t0 = time.time()
J = DJ.Journal({"driver_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(), "plan_sha256": "n/a", "out": str(probe), "mode": "dec_stream"},
               f"decstream-{os.getpid()}-{int(t0)}")
dev = DJ.m6_device(); pkg = {"path": str(path), "main_hash": (path / "main.hash").read_bytes().hex(), "label": "decode split probe 2, Core AI ComputeStream (Swift)"}
aid = J.attempt(dev, pkg, functions=info["entries"], request={"compute": "ane", "compile_mode": os.environ.get("MPSGRAPH_ANE_BONDED_COMPILE_MODE"), "via": "dec_stream (Swift)"})
bridge = Path(os.environ.get("SYNC_BRIDGE", "/Volumes/SSD4TB/anemll-forge-research/v8_sync_bridge/sync_bridge.aimodel"))
J2pkg = {"path": str(bridge), "main_hash": (bridge / "main.hash").read_bytes().hex(), "label": "sync bridges (GPU), Core AI ComputeStream test"}
aid2 = J.attempt(dev, J2pkg, functions=["cpq", "cpo"], request={"compute": "gpu", "via": "dec_stream (Swift)"})
r = subprocess.run([str(exe), str(path), str(here.parent / "engine" / "dec_kernels.metal"), str(bridge)])
J.result(dev, aid2, J2pkg, "ok" if r.returncode == 0 else "failed", **({} if r.returncode == 0 else {"error": f"exit {r.returncode}"}))
J.result(dev, aid, pkg, "ok" if r.returncode == 0 else "failed", model_load_s=time.time() - t0, **({} if r.returncode == 0 else {"error": f"exit {r.returncode}"}))
sys.exit(r.returncode)
