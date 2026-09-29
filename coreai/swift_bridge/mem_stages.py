"""Where does a Core AI chunk's wired memory go? Loads one package through the bridge and prints system wired memory
and this process's IOSurface / footprint after each step (model load, function load, buffer allocation, first run,
second entry). Zero inputs.
    .venv/bin/python swift_bridge/mem_stages.py <chunk.aimodel> [entry1,entry2]"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coreai_bridge as B  # noqa: E402

PKG = Path(os.path.expanduser(sys.argv[1]))
ENTRIES = (sys.argv[2] if len(sys.argv) > 2 else "v8_24k,p64_24k").split(",")


def wired():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    return next(int(l.split()[-1].rstrip(".")) for l in out.splitlines() if "wired down" in l) * 16384 / 2 ** 30


def proc():
    """(footprint GB, IOSurface virtual GB, IOSurface resident GB) of this process from vmmap."""
    out = subprocess.run(["vmmap", "--summary", str(os.getpid())], capture_output=True, text=True).stdout
    fp = re.search(r"Physical footprint:\s+([\d.]+)([KMG])", out)
    scale = {"K": 1 / 2 ** 20, "M": 1 / 2 ** 10, "G": 1}

    def g(m):
        return float(m.group(1)) * scale[m.group(2)] if m else 0.0
    io = re.search(r"^IOSurface\s+([\d.]+)([KMG])\s+([\d.]+)([KMG])", out, re.M)
    return g(fp), (float(io.group(1)) * scale[io.group(2)] if io else 0.0), (float(io.group(3)) * scale[io.group(4)] if io else 0.0)


w0 = wired()


def report(step):
    f, iv, ir = proc()
    print(f"{step:34s} wired {wired() - w0:+6.2f} GB | process footprint {f:5.2f} GB | IOSurface virtual {iv:5.2f} "
          f"resident {ir:5.2f} GB", flush=True)


report("start")
model = B.Model(PKG)
report("AIModel loaded")
fns, bufs = [], []
for name in ENTRIES:
    fn = model.function(name)
    report(f"function {name} loaded")
    ins = {n: fn.buffer("input", n) for n in fn.input_names}
    outs = {n: fn.buffer("output", n) for n in fn.output_names}
    sts = {n: fn.buffer("state", n) for n in fn.state_names}
    nb = sum(b.np.nbytes for b in list(ins.values()) + list(outs.values()) + list(sts.values())) / 2 ** 30
    report(f"  buffers allocated ({nb:.2f} GB)")
    plan = B.Plan([fn.bind(ins, outs, sts)])
    t = time.time()
    plan.run()
    report(f"  first run ({time.time() - t:.1f}s)")
    for _ in range(5):
        plan.run()
    report("  after 5 runs")
    fns.append((fn, plan))
    bufs.append((ins, outs, sts))
print(f"package {PKG.name}: {sum(f.stat().st_size for f in PKG.rglob('*') if f.is_file()) / 2 ** 30:.2f} GB on disk")
