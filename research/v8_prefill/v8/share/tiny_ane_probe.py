"""Tiny fresh ANE program (one 1x1 conv 1024 -> 1024 + ReLU on 64 rows, random weights) to separate a stale-cache
problem from an ANE dispatch problem. build: writes OUT/tiny.aimodel (build venv). run: loads it on the ANE (journaled)
with whatever MPSGRAPH_ANE_BONDED_COMPILE_MODE the environment sets, runs 20 predictions, prints times.
    BUILDPY tiny_ane_probe.py build OUT_DIR      MainPY tiny_ane_probe.py run OUT_DIR"""
import hashlib, os, sys, time
from pathlib import Path
V02 = "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02"
cmd, out = sys.argv[1], Path(sys.argv[2])
if cmd == "build":
    import torch, torch.nn as nn
    sys.path[:0] = [V02 + "/coreai", V02 + "/scripts"]
    import qwen38_coreai_build as B

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__(); torch.manual_seed(0)
            self.c = nn.Conv2d(1024, 1024, 1, bias=False)
        def example(self):
            return (torch.randn(1, 1024, 1, 64, dtype=torch.float16),)
        def forward(self, x):
            return torch.relu(self.c(x))
    out.mkdir(parents=True, exist_ok=True)
    mb = B.save_program([("t", Tiny().eval().to(torch.float16), ["x"], ["y"])], out / "tiny.aimodel")
    print(f"built {out / 'tiny.aimodel'} {mb:.1f} MB")
elif cmd == "build_hot":  # compute-heavy, weight-light: one 2048x2048 conv (8 MB) applied 8 times on 512 rows
    import torch, torch.nn as nn
    sys.path[:0] = [V02 + "/coreai", V02 + "/scripts"]
    import qwen38_coreai_build as B

    class Hot(nn.Module):
        def __init__(self):
            super().__init__(); torch.manual_seed(0)
            self.c = nn.Conv2d(2048, 2048, 1, bias=False)
        def example(self):
            return (torch.randn(1, 2048, 1, 512, dtype=torch.float16),)
        def forward(self, x):
            for _ in range(8):
                x = torch.tanh(self.c(x))
            return x
    out.mkdir(parents=True, exist_ok=True)
    mb = B.save_program([("t", Hot().eval().to(torch.float16), ["x"], ["y"])], out / "hot.aimodel")
    print(f"built {out / 'hot.aimodel'} {mb:.1f} MB")
else:
    for p in (V02 + "/coreai/swift_bridge", V02 + "/scripts",
              "/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_eight_row_v4"):
        sys.path.insert(0, p)
    import numpy as np
    import coreai_bridge as CB
    import deploy_journal as DJ
    path = out / "tiny.aimodel"
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": "tiny ane probe", "mode": "tiny_ane_probe"}, f"tiny-{os.getpid()}-{int(time.time())}")
    dev = DJ.m6_device()
    pk = {"path": str(path), "label": f"tiny ANE probe (mode {os.environ.get('MPSGRAPH_ANE_BONDED_COMPILE_MODE')})",
          "files": [path.name], "main_hash": (path / "main.hash").read_bytes().hex()}
    aid = J.attempt(dev, pk, functions=["t"], request={"compute": "ane"})
    t0 = time.time()
    try:
        m = CB.Model(path, "ane")
    except Exception as e:  # noqa: BLE001
        J.result(dev, aid, pk, "failed", error=f"{type(e).__name__}: {e}"); raise
    J.result(dev, aid, pk, "ok", model_load_s=time.time() - t0)
    print(f"loaded in {time.time() - t0:.1f}s (mode {os.environ.get('MPSGRAPH_ANE_BONDED_COMPILE_MODE')})", flush=True)
    f = m.function("t"); x = f.buffer("input", "x"); y = f.buffer("output", "y")
    np.copyto(x.np, (np.random.randn(*x.shape) * 0.3).astype(np.float16)) if hasattr(x, "np") else None
    pl = CB.Plan([f.bind({"x": x}, {"y": y})])
    ts = []
    for _ in range(20):
        t = time.perf_counter(); pl.run(); ts.append((time.perf_counter() - t) * 1e3)
    print(f"OK 20 predictions, median {sorted(ts)[10]:.3f} ms", flush=True)
