"""Core AI: the ported gated-attention layer 3 (+ its MLP) of chunk L00-03 with entry points v8_2k / v8_8k / v8_16k
(T=8 verify, KV history as read-only inputs of the entry's length, mask (1, ctx), k/v rows of the block out) sharing
one weight copy. Wired memory as the entry points are loaded, parity vs the torch port per entry (KV / mask from the
Core ML reference calls, zero-padded to 16K), call ms per entry.
    .venv/bin/python coreai_attn_entries.py export|ane"""
from __future__ import annotations

# Use only the helpers shipped in this repository.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import asyncio
import gc
import os
import sys
import time

import numpy as np
import torch

os.environ["LAYERS"] = "3"
import coreai_chunk_port as C  # noqa: E402

CTXS = [2048, 8192, 16384]
NAME = "L3_attn_2k_8k_16k"


def inputs(ctx: int, call: int):
    """Chunk inputs for the attention-only port (names x, cos, sin, mask, k0, v0) from the Core ML reference."""
    src = 2048 if ctx == 2048 else 8192
    R = np.load(C.DATA / f"chunk_L00-03_ref_ctx{src}.npz")
    d = {n: R[f"c{call}/in/{n}"] for n in ("x", "cos", "sin")}
    mask, k3, v3 = R[f"c{call}/in/mask"], R[f"c{call}/in/k3"], R[f"c{call}/in/v3"]
    if ctx > src:
        pad = ctx - src
        mask = np.concatenate([mask, np.full((1, pad), -1e4, np.float16)], 1)
        k3 = np.concatenate([k3, np.zeros((k3.shape[0], pad, k3.shape[2]), np.float16)], 1)
        v3 = np.concatenate([v3, np.zeros((v3.shape[0], pad, v3.shape[2]), np.float16)], 1)
    d.update(mask=mask, k0=k3, v0=v3)
    return d


def full_inputs(ctx: int, call: int, names):
    d = inputs(ctx, call)
    d.setdefault("conv_sel", np.zeros((3, C.T + 3), np.float16))
    d.setdefault("commit", np.zeros((1, C.P, 1), np.float16))
    d.setdefault("commit_last", np.zeros((1, C.P, 1), np.float16))
    return [d[n] for n in names], {}


def export():
    C.CTXS[:] = CTXS
    C.NAME = NAME
    C.ref_io = full_inputs
    C.mode_export()


async def ane():
    from coreai.runtime import AIModel, NDArray
    from coreai_bench_helpers import specialization_for
    path = C.ROOT / f"{NAME}.aimodel"
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
    gc.collect()
    w0, t0 = C.wired_gb(), time.time()
    model = await AIModel.load(path, specialization_options=specialization_for("ane"))
    print(f"[{NAME}] {size:.0f} MB on disk; load {time.time() - t0:.0f}s +{C.wired_gb() - w0:.2f} GB; {C.placement(t0)}", flush=True)
    fns = {}
    for ctx in CTXS:
        fns[ctx] = model.load_function(f"v8_{ctx // 1024}k")
        print(f"   load_function v8_{ctx // 1024}k: wired +{C.wired_gb() - w0:.2f} GB", flush=True)
    W = C.load_weights()
    for ctx in CTXS:
        fn = fns[ctx]
        tm = C.Chunk(W, ctx).eval().float()
        names = tm.input_names()
        rep = []
        for call in range(3):
            d = dict(zip(names, full_inputs(ctx, call, names)[0]))
            out = await fn(inputs={n: NDArray(d[n].astype(np.float16)) for n in names if n in fn.desc.input_names})
            with torch.no_grad():
                t = dict(zip(tm.output_names(), [x.numpy() for x in tm(*[torch.from_numpy(a.astype(np.float32))
                                                                           for a in full_inputs(ctx, call, names)[0]])]))
            cs = []
            for k in ("y", "k0_new", "v0_new"):
                a, b = out[k].numpy().astype(np.float64).ravel(), t[k].astype(np.float64).ravel()
                cs.append(f"{k} {a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30):.5f}/{np.linalg.norm(a) / np.linalg.norm(b):.3f}")
            rep.append(f"c{call}: " + " ".join(cs))
        d = dict(zip(names, full_inputs(ctx, 1, names)[0]))
        feed = {n: NDArray(d[n].astype(np.float16)) for n in names if n in fn.desc.input_names}
        ts = []
        for _ in range(20):
            t1 = time.perf_counter()
            await fn(inputs=feed)
            ts.append(1e3 * (time.perf_counter() - t1))
        print(f"   v8_{ctx // 1024}k: {np.median(ts):.2f} ms (p10 {np.percentile(ts, 10):.2f}); wired +{C.wired_gb() - w0:.2f} GB; "
              f"parity vs torch (cos/|ratio|) " + " | ".join(rep), flush=True)
        del tm
        gc.collect()


if __name__ == "__main__":
    export() if sys.argv[1] == "export" else asyncio.run(ane())
