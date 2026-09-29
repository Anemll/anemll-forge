"""Decode speed of a v2 build: per-token ms from scratch and after a prefill block, with memory stats.
    CTX=8192 [V2_NO_PREFILL=1] python qwen38_v2_speed.py"""
import os
import subprocess
import time

import numpy as np

import qwen38_ane_model as M


def mem():
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout.splitlines()[-1]
    rss = int(subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout) / 2**20
    return f"{out.strip()}; process RSS {rss:.1f} GB"


def steps(m, n):
    ts = []
    for i in range(n):
        t0 = time.perf_counter()
        m.step(2000 + i)
        ts.append(1e3 * (time.perf_counter() - t0))
        if ts[-1] > 2000:
            print("slow step (>2 s): stopping", flush=True)
            break
    return ts


m = M.AneQwen2()
print("loaded;", mem(), flush=True)
m.reset()
ts = steps(m, 12)
print("decode from scratch ms:", " ".join(f"{t:.0f}" for t in ts), f"-> {1e3 / np.median(ts):.2f} tok/s", flush=True)
if m.chunks[0]["pre"]:
    t0 = time.perf_counter()
    m.feed(list(range(1000, 1064)))
    print(f"prefill 64: {1e3 * (time.perf_counter() - t0):.0f} ms;", mem(), flush=True)
    ts = steps(m, 12)
    print("decode after prefill ms:", " ".join(f"{t:.0f}" for t in ts), f"-> {1e3 / np.median(ts):.2f} tok/s", flush=True)
print("end;", mem(), flush=True)
