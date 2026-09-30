"""Does a long run of Core AI calls leak output IOSurfaces? One real chunk (v8_16k of chunk L00-03 of the Core AI
target), 300 calls per variant; wired memory and call time at the start / end of each variant:
    plain     outputs fed back as inputs, nothing else
    numpy     + .numpy() (DLPack view) of two outputs per call, dropped immediately
    bufcopy   + a copy of every state output into a persistent IOSurface through the buffer protocol
    .venv/bin/python coreai_leak_probe.py"""
from __future__ import annotations

# Use only the helpers shipped in this repository.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import asyncio
import ctypes
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from coreai.runtime import AIModel, NDArray
from coreai.runtime._ndarray import StorageKind

from coreai_bench_helpers import specialization_for

ROOT = Path.home() / "Models/vq27b/coreai_ane6/mix25_aw_cal_lr64mix"
IOS = StorageKind.IO_SURFACE
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300


def wired():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(out.split("page size of ")[1].split()[0])
    return next(int(l.split()[-1].rstrip(".")) for l in out.splitlines() if "wired down" in l) * page / 2**30


def view(nd):
    mv = memoryview(nd._tensor)  # noqa: SLF001
    ptr = np.frombuffer(mv, dtype=np.uint8).__array_interface__["data"][0]
    return np.ctypeslib.as_array((ctypes.c_uint8 * mv.nbytes).from_address(ptr))


async def main():
    m = await AIModel.load(ROOT / "chunk_L00-03.aimodel", specialization_options=specialization_for("ane"))
    f = m.load_function("v8_16k")
    d = f.desc
    shapes = {n: tuple(d.input_descriptor(n).shape) for n in d.input_names}
    rng = np.random.default_rng(0)
    base = {}
    for n, s in shapes.items():
        a = np.zeros(s, np.float16)
        if n == "x":
            a = (rng.standard_normal(s) * 0.1).astype(np.float16)
        base[n] = NDArray(a, IOS)
    states = [n for n in base if n.endswith(("0", "1", "2")) and n[:-1] in ("conv", "rec", "pend")]
    import os
    variants = os.environ.get("VARIANTS", "plain,numpy,bufcopy").split(",")
    every = int(os.environ.get("GC_EVERY", "0"))
    if os.environ.get("GC_FREEZE") == "1":  # move everything alive now out of the collector's reach
        import torch  # noqa: F401  (the real runtime has torch / transformers imported)
        gc.collect()
        gc.freeze()
    for variant in variants:
        cur = dict(base)
        pers = {n: NDArray(np.zeros(shapes[n], np.float16), IOS) for n in states}
        pv = {n: view(pers[n]) for n in states}
        gc.collect()
        w0, ts = wired(), []
        for i in range(N):
            t = time.perf_counter()
            out = await f(inputs=cur)
            if variant == "plain":
                cur = {**cur, **{n: out[f"{n}_out"] for n in states}}
            elif variant == "numpy":
                cur = {**cur, **{n: out[f"{n}_out"] for n in states}}
                _ = out["y"].numpy().sum(), out["k3_new"].numpy().sum()
            else:
                for n in states:
                    pv[n][:] = view(out[f"{n}_out"])
                cur = {**cur, **pers}
            del out
            if every and i % every == every - 1:
                gc.collect(int(os.environ.get("GC_GEN", "2")))
            ts.append(1e3 * (time.perf_counter() - t))
        gc.collect()
        print(f"{variant:8s}: {N} calls, first 20 median {np.median(ts[:20]):.2f} ms, last 20 median "
              f"{np.median(ts[-20:]):.2f} ms, wired {w0:.2f} -> {wired():.2f} GB", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
