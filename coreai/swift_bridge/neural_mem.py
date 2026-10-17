"""Neural Engine memory of one Core AI chunk package, per entry point, on the Mac: loads the package through the
bridge (bonded compile mode 2, as the server and the iPhone app), loads each entry, runs it once (a program's scratch
is wired at its first call, sized by its largest entry), reads this process's footprint (the "neural" ledger, outside
phys_footprint on macOS), times 20 calls, and with --unload releases everything and reads it again.
    .venv/bin/python coreai/swift_bridge/neural_mem.py <chunk.aimodel> [--entries v8_24k,p32_24k] [--unload]
Zero inputs. Run with nothing else on the Neural Engine for clean timings (a busy server makes them noisy; the
memory figures are not affected). Use a package no other process has loaded: when a server holds the same package,
its program stays loaded and --unload still reads the weights (299.5 MB of 335.7 for a 4x64 chunk, 17 Oct).
Used for the prefill scratch measurements (Bonsai-1.5b: docs/research/IPHONE_LOW_MEMORY_2026-10-17.md)."""
from __future__ import annotations

import argparse
import gc
import os
import re
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MPSGRAPH_ANE_BONDED_COMPILE_MODE", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import coreai_bridge as B  # noqa: E402


def neural_mb() -> tuple[float, float]:
    """(neural, phys_footprint) of this process in MB, from footprint(1)."""
    out = subprocess.run(["footprint", "-p", str(os.getpid()), "-f", "bytes"], capture_output=True, text=True).stdout
    aux = dict(re.findall(r"^\s+(neural|phys_footprint):\s+(\d+) B", out, re.M))
    return int(aux.get("neural", 0)) / 1e6, int(aux.get("phys_footprint", 0)) / 1e6


def report(step: str) -> float:
    n, f = neural_mb()
    print(f"{step:30s} neural {n:8.1f} MB | footprint {f:7.1f} MB", flush=True)
    return n


def measure(package: Path, entries: str | None):
    """Load, call and time each entry; everything it holds goes away when it returns."""
    t = time.time()
    model = B.Model(package)
    names = model.function_names
    print(f"package {package.name}: functions {names}", flush=True)
    keep = []
    for e in (entries.split(",") if entries else names):
        fn = model.function(e)
        if not keep:
            print(f"  first function loaded in {time.time() - t:.1f} s (a first load compiles the package)", flush=True)
        report(f"function {e} loaded")
        ins = {n: fn.buffer("input", n) for n in fn.input_names}
        outs = {n: fn.buffer("output", n) for n in fn.output_names}
        sts = {n: fn.buffer("state", n) for n in fn.state_names}
        plan = B.Plan([fn.bind(ins, outs, sts)])
        plan.run()
        report(f"  {e} first call")
        ts = []
        for _ in range(20):
            t0 = time.time()
            plan.run()
            ts.append(time.time() - t0)
        ts.sort()
        print(f"  {e}: median {ts[10] * 1e3:.2f} ms, min {ts[0] * 1e3:.2f} ms per call", flush=True)
        keep.append((fn, plan, ins, outs, sts))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("package", type=Path)
    ap.add_argument("--entries", help="comma-separated entry points (default: all)")
    ap.add_argument("--unload", action="store_true", help="release the package at the end and read the memory again")
    a = ap.parse_args()
    report("start")
    measure(a.package, a.entries)
    if a.unload:   # the memory goes a little after the model does: poll until it stops changing (3 s at most)
        gc.collect()
        t, n0 = time.time(), neural_mb()[0]
        while time.time() - t < 3:
            time.sleep(0.25)
            n1 = neural_mb()[0]
            if abs(n1 - n0) < 1:
                break
            n0 = n1
        report(f"unloaded ({time.time() - t:.2f} s)")


if __name__ == "__main__":
    main()
