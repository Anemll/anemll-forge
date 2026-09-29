"""Prefill block size sweep on one Core AI chunk through the bridge: for each entry (v8 / p64 / p128 / p256 of a
package built with TPS=64,128,256) the wired memory it adds when loaded and first run, and its call time, projected to
the full model (16 chunks + ~5 ms head per call). Zero inputs (timing only).
    .venv/bin/python swift_bridge/tp_time.py [package.aimodel]
env: ENTRIES (default: every function in the package), N (calls per entry, default 20)"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coreai_bridge as B  # noqa: E402

PKG = Path(os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else
                              "~/Models/vq27b/coreai_tptest/mix25in_aw_cal_lr64mix/chunk_L00-03.aimodel"))
N = int(os.environ.get("N", "20"))


def wired():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    return next(int(l.split()[-1].rstrip(".")) for l in out.splitlines() if "wired down" in l) * 16384 / 2 ** 30


def rows(name):
    m = re.match(r"[vp](\d+)_", name)
    return int(m.group(1)) if m else 8


def main():
    w0 = wired()
    t = time.time()
    model = B.Model(PKG)
    print(f"{PKG.name}: model load {time.time() - t:.1f}s, wired {wired() - w0:+.2f} GB", flush=True)
    names = os.environ.get("ENTRIES", ",".join(model.function_names)).split(",")
    base, res = wired(), []
    for name in names:
        t = time.time()
        fn = model.function(name)
        ins = {n: fn.buffer("input", n) for n in fn.input_names}
        outs = {n: fn.buffer("output", n) for n in fn.output_names}
        sts = {n: fn.buffer("state", n) for n in fn.state_names}
        plan = B.Plan([fn.bind(ins, outs, sts)])
        plan.run()  # first run: specialization / ANE program load
        load = time.time() - t
        w = wired()
        for _ in range(3):
            plan.run()
        ts = []
        for _ in range(N):
            t1 = time.perf_counter()
            plan.run()
            ts.append(time.perf_counter() - t1)
        ms = float(np.median(ts)) * 1e3
        r = rows(name)
        full = 16 * ms + 5.0
        res.append((name, r, ms, w - base, load, r / full * 1e3))
        print(f"  {name:10s} rows {r:3d}: load+first run {load:5.1f}s, wired +{w - base:.2f} GB (cumulative) | "
              f"{ms:6.2f} ms/chunk, {r / ms * 1e3:6.0f} rows/s/chunk -> full model ~{full:.0f} ms/call, "
              f"~{r / full * 1e3:.0f} tok/s", flush=True)
    print(f"total wired added: {wired() - w0:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
