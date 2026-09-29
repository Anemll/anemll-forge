"""One decoder chunk through the Python Core AI binding (reference) and through the bridge, same seeded inputs, then a
long bridge run.
    .venv/bin/python test_chunk_parity.py ref      # Python binding: writes /tmp-free npz next to this file
    .venv/bin/python test_chunk_parity.py bridge   # bridge: parity vs the npz, then N calls (stability, timing)
env: CHUNK (dir/chunk_L04-07.aimodel), ENTRIES (v8_16k,p64_16k), N (5000), NP (500 prefill calls)"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CHUNK = Path(os.path.expanduser(os.environ.get(
    "CHUNK", "~/Models/vq27b/coreai_ane7i/mix25in_aw_cal_lr64mix/chunk_L04-07.aimodel")))
ENTRIES = os.environ.get("ENTRIES", "v8_16k,p64_16k").split(",")
REF = HERE / f"ref_{CHUNK.stem}.npz"


def inputs_for(names_shapes, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for n, s in names_shapes:
        if n == "mask":
            a = np.full(s, -1e4, np.float32)
            a[0, : s[1] // 3] = 0
        elif n in ("conv_sel", "conv_sel_out"):
            a = np.zeros(s, np.float32)
            a[np.arange(3), 2 + np.arange(3)] = 1
        elif n in ("commit", "commit_last", "valid"):
            a = np.zeros(s, np.float32)
            a.reshape(-1)[:2] = 1
        elif n in ("cos", "sin"):
            a = (np.cos if n == "cos" else np.sin)(rng.uniform(0, 6.3, s))
        elif n.startswith("rec") or n.startswith("pend"):
            a = rng.standard_normal(s) * 0.01
        else:
            a = rng.standard_normal(s) * 0.1
        out[n] = a.astype(np.float16)
    return out


def wired():
    o = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    pg = int(o.split("page size of ")[1].split()[0])
    return next(int(l.split()[-1].rstrip(".")) for l in o.splitlines() if "wired down" in l) * pg / 2**30


def footprint():
    o = subprocess.run(["footprint", "-p", str(os.getpid())], capture_output=True, text=True).stdout
    for l in o.splitlines():
        if "phys_footprint:" in l:
            v, u = l.split()[-2:]
            return float(v) / (1024 if u.startswith("K") else 1 if u.startswith("M") else 1 / 1024) / 1024
    return float("nan")


def ref():
    from coreai.runtime import AIModel, NDArray
    from coreai.runtime._ndarray import StorageKind
    sys.path.insert(0, str(HERE.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ane-vector-lut/coreai
    from coreai_util import specialization_for

    async def go():
        t = time.time()
        m = await AIModel.load(CHUNK, specialization_options=specialization_for("ane"))
        print(f"ref: loaded {CHUNK.name} in {time.time() - t:.1f} s", flush=True)
        save = {}
        for e, entry in enumerate(ENTRIES):
            f = m.load_function(entry)
            d = f.desc
            ins = inputs_for([(n, tuple(d.input_descriptor(n).shape)) for n in d.input_names], 100 + e)
            out = await f(inputs={n: NDArray(a, StorageKind.IO_SURFACE) for n, a in ins.items()})
            for n, a in ins.items():
                save[f"{entry}/in/{n}"] = a
            for n, v in out.items():
                save[f"{entry}/out/{n}"] = v.numpy()
            print(f"ref: {entry} {len(ins)} inputs -> {len(out)} outputs", flush=True)
        np.savez(REF, **save)
    asyncio.run(go())


def bridge():
    sys.path.insert(0, str(HERE))
    import coreai_bridge as B
    ref = np.load(REF)
    t = time.time()
    m = B.Model(CHUNK)
    print(f"bridge: loaded {CHUNK.name} in {time.time() - t:.1f} s; functions {m.function_names}", flush=True)
    fns = {e: m.function(e) for e in ENTRIES}
    plans, outs_all = {}, {}
    for entry, f in fns.items():
        ins = {n: f.buffer("input", n) for n in f.input_names}
        for n, b in ins.items():
            b.np[...] = ref[f"{entry}/in/{n}"]
        outs = {n: f.buffer("output", n) for n in f.output_names}
        for o in outs.values():
            o.np[...] = np.nan
        plans[entry], outs_all[entry] = B.Plan([f.bind(ins, outs)]), outs
        plans[entry].run()
        worst = 0.0
        for n, o in outs.items():
            r, g = ref[f"{entry}/out/{n}"].astype(np.float32), o.np.astype(np.float32)
            nan = int(np.isnan(g).sum())
            d = float(np.max(np.abs(r - g))) if not nan else float("nan")
            worst = max(worst, d) if not nan else float("nan")
            print(f"  {entry} {n:10s} {str(o.shape):18s} max|ref-bridge| {d:.3g}  exact {np.array_equal(r, g)}"
                  f"{f'  UNWRITTEN {nan}' if nan else ''}")
        print(f"parity {entry}: worst max abs diff {worst:.3g}", flush=True)
    N, NP = int(os.environ.get("N", "5000")), int(os.environ.get("NP", "500"))
    for entry, n_calls in ((ENTRIES[0], N), (ENTRIES[-1], NP)):
        pl = plans[entry]
        w0, f0, ts = wired(), footprint(), []
        y0 = outs_all[entry]["y"].np.copy()
        every = max(1, n_calls // 10)
        print(f"{entry}: {n_calls} calls, wired {w0:.2f} GB, footprint {f0:.3f} GB", flush=True)
        for i in range(1, n_calls + 1):
            t0 = time.perf_counter()
            pl.run()
            ts.append(1e3 * (time.perf_counter() - t0))
            if i % every == 0:
                w, fp = wired(), footprint()
                print(f"  {i:5d}: median {np.median(ts[-every:]):.2f} ms, p90 {np.percentile(ts[-every:], 90):.2f} ms"
                      f" | wired {w:.2f} GB ({w - w0:+.2f}) | footprint {fp:.3f} GB ({fp - f0:+.3f})", flush=True)
        same = np.array_equal(y0, outs_all[entry]["y"].np)
        print(f"{entry}: done, median {np.median(ts):.2f} ms, first-100 {np.median(ts[:100]):.2f}, last-100 "
              f"{np.median(ts[-100:]):.2f}; y after {n_calls} calls identical to call 1: {same}", flush=True)


if __name__ == "__main__":
    {"ref": ref, "bridge": bridge}[sys.argv[1]]()
