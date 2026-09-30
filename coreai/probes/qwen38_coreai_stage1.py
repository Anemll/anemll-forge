"""Stage 1 of the Core AI target port, on real chunks (qwen38_coreai_build.py):
    entries : chunk L00-03 with v8 entries at KV 2K / 8K / 16K / 32K / 64K + p64 at 2K: disk use (asset + compile cache),
              wired memory per load_function, placement, call time per entry, and prefill consistency on the ANE
              (one p64 call vs 8 chained v8 calls committing 8 rows each: y rows and k / v rows)
    size8   : one 8-layer chunk (L04-11) vs two 4-layer chunks (L04-07 + L08-11), v8 at 16K: call time and wired
    .venv/bin/python qwen38_coreai_stage1.py entries|size8"""
from __future__ import annotations

import asyncio
import gc
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # coreai (builder, coreai_util)
import qwen38_coreai_build as B
from coreai.runtime import AIModel, NDArray
from coreai.runtime._ndarray import StorageKind
from coreai_util import specialization_for

CACHE = Path.home() / "Library/Caches/coreai-cache"
P, cdim, nv, dk, dv, nkv, hd, hid, rot = B.P, B.cdim, B.nv, B.dk, B.dv, B.nkv, B.hd, B.hid, B.rot
f16 = np.float16


def wired_gb() -> float:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(out.split("page size of ")[1].split()[0])
    return next(int(l.split()[-1].rstrip(".")) for l in out.splitlines() if "wired down" in l) * page / 2**30


def free_gb() -> float:
    return shutil.disk_usage(Path.home()).free / 2**30


def placement(since: float) -> str:
    mans = [p for p in CACHE.glob("*/*/*/*/model.aimodelx/**/manifest.plist") if p.stat().st_mtime >= since]
    if not mans:
        return "cached"
    text = max(mans, key=lambda p: p.stat().st_mtime).read_bytes()
    return ("fully on ANE" if b"mps.fullyPlacedOnANE" in text else "NOT fully on ANE") + f", {text.count(b'_ANE_region_')} region refs"


def cosv(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


class Inputs:
    """Host-side inputs of one chunk (lazy-commit schedule, like qwen38_ane_model.AneQwen3.call / prefill_block)."""

    def __init__(self, gdn_j, att_j, ctx, T=8):
        ctx = B.kv_len(ctx, T)
        self.gdn_j, self.att_j, self.ctx = gdn_j, att_j, ctx
        self.st = {}
        for j in gdn_j:
            self.st |= {f"conv{j}": np.zeros((P + 3, cdim), f16), f"rec{j}": np.zeros((nv, dk, dv), f16),
                        f"pend{j}": np.zeros((nv, 3 * P + 1, dv), f16)}
        self.kv = {f"{s}{j}": np.zeros((nkv, ctx, hd), f16) for j in att_j for s in ("k", "v")}
        inv = 1.0 / B.CFG["rope_parameters"]["rope_theta"] ** (np.arange(0, rot, 2) / rot)
        self.inv, self.pos, self.pending = inv, 0, 0

    def common(self, x, T, n):
        p0, k = self.pos, self.pending
        pos = np.minimum(np.arange(p0, p0 + T), p0 + n - 1)
        ang = np.concatenate([np.outer(pos, self.inv)] * 2, axis=1)
        sel = np.zeros((3, P + 3), f16)
        sel[np.arange(3), k + np.arange(3)] = 1
        com, last = np.zeros((1, P, 1), f16), np.zeros((1, P, 1), f16)
        com[0, :k] = 1
        if k:
            last[0, k - 1] = 1
        xin = np.zeros((1, hid, 1, T), f16)
        xin[0, :, 0, :n] = x.T
        mask = np.where(np.arange(self.ctx)[None, :] < p0, 0, -1e4).astype(f16)
        return {"x": xin, "cos": np.cos(ang).astype(f16), "sin": np.sin(ang).astype(f16), "mask": mask,
                "conv_sel": sel, "commit": com, "commit_last": last, **self.st, **self.kv}

    def feed_verify(self, x):
        return self.common(x, 8, len(x))

    def feed_prefill(self, x, T=64):
        d = self.common(x, T, len(x))
        n = len(x)
        so = np.zeros((3, T + 3), f16)
        so[np.arange(3), n + np.arange(3)] = 1
        valid = np.zeros((1, T, 1), f16)
        valid[0, :n, 0] = 1
        return d | {"conv_sel_out": so, "valid": valid}

    def accept(self, out, k, prefill=False):
        for n_ in list(self.st):
            self.st[n_] = np.asarray(out[f"{n_}_out"], f16).copy()
        for n_ in self.kv:
            self.kv[n_][:, self.pos:self.pos + k] = np.asarray(out[f"{n_}_new"], f16)[:, :k]
        self.pos += k
        self.pending = 0 if prefill else k


async def call(fn, feed):
    out = await fn(inputs={n: NDArray(v, StorageKind.IO_SURFACE) for n, v in feed.items()})
    return {k: v.numpy() for k, v in out.items()}


async def timed(fn, feed, n=15):
    nd = {k: NDArray(v, StorageKind.IO_SURFACE) for k, v in feed.items()}
    await fn(inputs=nd)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        await fn(inputs=nd)
        ts.append(1e3 * (time.perf_counter() - t))
    return float(np.median(ts)), float(np.percentile(ts, 10))


def embeddings(n, seed=0):
    ck = B.M.Checkpoint()
    e = ck.get("model.language_model.embed_tokens.weight")
    ids = np.random.default_rng(seed).integers(1000, 100000, n)
    import torch
    return e[torch.from_numpy(ids)].to(torch.float16).numpy()


async def entries():
    ctxs, pctxs = [2048, 8192, 16384, 32768, 65536], [2048]
    path = B.OUT / "chunk_L00-03.aimodel"
    f0 = free_gb()
    if os.environ.get("REBUILD") == "1":
        shutil.rmtree(path, ignore_errors=True)
    if not path.exists():
        ck = B.M.Checkpoint()
        t = time.time()
        info = B.build_chunk(ck, [0, 1, 2, 3], ctxs, pctxs)
        print(f"[build] {time.time() - t:.0f}s, {info}; disk free {f0:.1f} -> {free_gb():.1f} GiB", flush=True)
        del ck
        gc.collect()
    gdn_j, att_j = [0, 1, 2], [3]
    f1, w0, t0 = free_gb(), wired_gb(), time.time()
    model = await AIModel.load(path, specialization_options=specialization_for("ane"))
    print(f"[load] {time.time() - t0:.0f}s, wired +{wired_gb() - w0:.2f} GB, disk free {f1:.1f} -> {free_gb():.1f} GiB; "
          f"{placement(t0)}", flush=True)
    fns = {}
    for name in [f"v8_{c // 1024}k" for c in ctxs] + [f"p64_{c // 1024}k" for c in pctxs]:
        t = time.time()
        fns[name] = model.load_function(name)
        print(f"   load_function {name}: {time.time() - t:.1f}s, wired +{wired_gb() - w0:.2f} GB", flush=True)
    x = embeddings(72)
    # timing per entry at a mid position
    for ctx in ctxs:
        inp = Inputs(gdn_j, att_j, ctx)
        inp.pos, inp.pending = min(ctx // 2, inp.ctx - 8), 3
        ms, p10 = await timed(fns[f"v8_{ctx // 1024}k"], inp.feed_verify(x[:8]))
        print(f"[time] v8_{ctx // 1024}k: {ms:.2f} ms (p10 {p10:.2f}), wired +{wired_gb() - w0:.2f} GB", flush=True)
    inp = Inputs(gdn_j, att_j, 2048)
    ms, p10 = await timed(fns["p64_2k"], inp.feed_prefill(x[:64]))
    print(f"[time] p64_2k: {ms:.2f} ms (p10 {p10:.2f}) = {64 / ms * 1e3:.0f} tok/s per chunk; 8 x v8_2k would be "
          f"{8 * (await timed(fns['v8_2k'], inp.feed_verify(x[:8])))[0]:.1f} ms", flush=True)
    # prefill consistency: one p64 call vs 8 chained v8 calls (all rows committed), cold start at position 0
    a = Inputs(gdn_j, att_j, 2048)
    out_a = await call(fns["p64_2k"], a.feed_prefill(x[:64]))
    a.accept(out_a, 64, prefill=True)
    b = Inputs(gdn_j, att_j, 2048)
    ys, ks = [], []
    for c in range(8):
        out_b = await call(fns["v8_2k"], b.feed_verify(x[8 * c:8 * c + 8]))
        ys.append(out_b["y"][0, :, 0, :])
        ks.append(out_b["k3_new"])
        b.accept(out_b, 8)
    yb = np.concatenate(ys, 1)
    ya = out_a["y"][0, :, 0, :]
    per = [cosv(ya[:, 8 * c:8 * c + 8], yb[:, 8 * c:8 * c + 8]) for c in range(8)]
    print(f"[prefill vs 8 x verify] y cos per block {[round(v, 5) for v in per]}; k rows cos "
          f"{cosv(out_a['k3_new'], np.concatenate(ks, 1)):.5f}; y norm ratio {np.linalg.norm(ya) / np.linalg.norm(yb):.4f}",
          flush=True)
    # then one more v8 call on both (A: committed state from prefill; B: 8 pending rows committed now)
    out_a2 = await call(fns["v8_2k"], a.feed_verify(x[64:72]))
    out_b2 = await call(fns["v8_2k"], b.feed_verify(x[64:72]))
    print(f"[next verify call after 64 tokens] y cos {cosv(out_a2['y'], out_b2['y']):.5f}, rec0 cos "
          f"{cosv(out_a2['rec0_out'], out_b2['rec0_out']):.5f}, rec2 cos {cosv(out_a2['rec2_out'], out_b2['rec2_out']):.5f}",
          flush=True)
    print(f"[end] wired +{wired_gb() - w0:.2f} GB, disk free {free_gb():.1f} GiB", flush=True)


async def size8():
    ck = B.M.Checkpoint()
    specs = {"chunk_L04-07": [4, 5, 6, 7], "chunk_L08-11": [8, 9, 10, 11], "chunk_L04-11": list(range(4, 12))}
    for name, layers in specs.items():
        if not (B.OUT / f"{name}_s8.aimodel").exists():
            B.build_chunk(ck, layers, [16384], [], name=f"{name}_s8")
    res = {}
    for name, layers in specs.items():
        gdn_j = [j for j, l in enumerate(layers) if B.CFG["layer_types"][l] == "linear_attention"]
        att_j = [j for j in range(len(layers)) if j not in gdn_j]
        w0, t0 = wired_gb(), time.time()
        model = await AIModel.load(B.OUT / f"{name}_s8.aimodel", specialization_options=specialization_for("ane"))
        fn = model.load_function("v8_16k")
        inp = Inputs(gdn_j, att_j, 16384)
        inp.pos, inp.pending = 8192, 3
        ms, p10 = await timed(fn, inp.feed_verify(embeddings(8)), n=20)
        res[name] = ms
        print(f"[{name}] load {time.time() - t0:.0f}s, wired +{wired_gb() - w0:.2f} GB, {placement(t0)}; v8_16k "
              f"{ms:.2f} ms (p10 {p10:.2f})", flush=True)
        del fn, model
        gc.collect()
    print(f"[size8] 2 x 4 layers {res['chunk_L04-07'] + res['chunk_L08-11']:.2f} ms vs 8 layers {res['chunk_L04-11']:.2f} ms",
          flush=True)


async def inplace():
    """Real chunk on the ANE: KV input rows rewritten in place (writable view of the NDArray's storage) between calls
    vs a fresh NDArray of the same contents - does the ANE read the in-place writes?"""
    sys.path.insert(0, str(B.SCRIPTS))
    from qwen38_coreai_model import writable
    model = await AIModel.load(B.OUT / "chunk_L00-03.aimodel", specialization_options=specialization_for("ane"))
    fn = model.load_function("v8_2k")
    inp = Inputs([0, 1, 2], [3], 2048)
    inp.pos, inp.pending = 512, 3
    x = embeddings(8)
    r = np.random.default_rng(2)
    feed = inp.feed_verify(x)
    kv_nd = {n: NDArray(np.zeros((nkv, 2048, hd), f16), StorageKind.IO_SURFACE) for n in ("k3", "v3")}
    kv_w = {n: writable(v) for n, v in kv_nd.items()}
    worst = 1.0
    for it in range(4):
        for n in kv_w:
            kv_w[n][:, :512] = (r.standard_normal((nkv, 512, hd)) * 0.5).astype(f16)     # in place
        base = {k: NDArray(v, StorageKind.IO_SURFACE) for k, v in feed.items() if k not in ("k3", "v3")}
        o1 = await fn(inputs={**base, **kv_nd})
        o2 = await fn(inputs={**base, **{n: NDArray(kv_w[n].copy(), StorageKind.IO_SURFACE) for n in kv_w}})
        c = cosv(o1["y"].numpy(), o2["y"].numpy())
        worst = min(worst, c)
        print(f"[inplace] rewrite {it}: y (in-place KV) vs y (fresh NDArray) cos {c:.6f}, max|d| "
              f"{float(np.abs(o1['y'].numpy().astype(np.float32) - o2['y'].numpy().astype(np.float32)).max()):.3g}", flush=True)
    print(f"[inplace] worst cos {worst:.6f} -> {'ANE reads in-place KV writes' if worst > 0.9999 else 'STALE / MISMATCH'}",
          flush=True)


if __name__ == "__main__":
    asyncio.run({"entries": entries, "size8": size8, "inplace": inplace}[sys.argv[1]]())
