"""ANE outputs of the 17 Nov LUT3 packages at 512 rows (entry r512, chain 4) against ref512.npz (18 Nov 2026): relative
error vs the served FFN on the host and vs each module's own host reference. Journaled. MainPY check512.py"""
import hashlib, os, sys, time
from pathlib import Path
import numpy as np
for p in ("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/scripts", "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/coreai",
          "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/coreai/swift_bridge",
          "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4"):
    sys.path.insert(0, p)
import ane_compile_mode
ane_compile_mode.apply(log=lambda m: None)
import coreai_bridge as CB
import deploy_journal as DJ
OUT = Path("/Volumes/SSD4TB/anemll-forge-research/a8probe/lut3_rows")
R = np.load(OUT / "ref512.npz"); x = R["x_512"]
rel = lambda y, r: float(np.linalg.norm(y - r) / np.linalg.norm(r))  # noqa: E731
J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                "out": str(OUT), "mode": "check512"}, "check512-%d-%d" % (os.getpid(), int(time.time())))
dev = DJ.m6_device()
for v in ("fp16", "i8gud"):
    dst = OUT / ("dec_%s_L36.aimodel" % v)
    pkg = {"path": str(dst), "label": "LUT3 512-row check %s" % v, "files": [dst.name],
           "main_hash": (dst / "main.hash").read_bytes().hex() if (dst / "main.hash").exists() else "UNKNOWN"}
    aid = J.attempt(dev, pkg, functions=["r512"], request={"compute": "ane"})
    t0 = time.time()
    try:
        m = CB.Model(dst, "ane")
    except Exception as e:  # noqa: BLE001
        J.result(dev, aid, pkg, "failed", error="%s: %s" % (type(e).__name__, e)); raise
    J.result(dev, aid, pkg, "ok", model_load_s=time.time() - t0)
    fn = m.function("r512")
    ib, ob = fn.buffer("input", "x"), fn.buffer("output", "y")
    ib.np[...] = x.reshape(ib.np.shape)
    CB.Plan([fn.bind({"x": ib}, {"y": ob})]).run()
    y = np.array(ob.np, np.float32).reshape(5120, 512)
    print("%-6s 512 rows x4: vs served host %.3e | vs own host %.3e | finite %s" % (
        v, rel(y, R["served_512"]), rel(y, R["%s_512" % v]), bool(np.isfinite(y).all())), flush=True)
    del m
