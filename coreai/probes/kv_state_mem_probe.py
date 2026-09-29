"""Does passing the KV history as a Core AI STATE (read-only in the graph, AFM style) avoid the ~3x-KV internal
buffers that a KV INPUT costs? One attention layer shaped like Qwen3.8's (24 query heads over 4 KV heads, head dim
256, 8 rows, 24K history) plus two fp16 1x1 convs so it lands on the ANE. Two packages that differ only in how K / V
arrive; for each: placement, wired memory after load / load_function / first call, and call time.
    .venv/bin/python kv_state_mem_probe.py [input|state ...]"""
from __future__ import annotations

import asyncio
import gc
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import coreai_torch
from coreai.runtime import AIModel, NDArray
from coreai.runtime._ndarray import StorageKind
from coreai_opt.casting import cast_to_16_bit_precision

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ane-vector-lut/coreai (builder, coreai_util)
from coreai_util import specialization_for

import os
HID, NH, NKV, D, T = 5120, 24, 4, 256, 8
CTX = int(os.environ.get("CTX", "16384"))
GRP = NH // NKV
ROOT = Path(__file__).resolve().parent / "artifacts_kv_state_mem"
CACHE = Path.home() / "Library/Caches/coreai-cache"


def wired_gb():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    return next(int(l.split()[-1].rstrip(".")) for l in out.splitlines() if "wired down" in l) * 16384 / 2 ** 30


class Attn(nn.Module):
    def __init__(self, state: bool) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.state = state
        self.q = nn.Conv2d(HID, NH * D, 1, bias=False)
        self.o = nn.Conv2d(NH * D, HID, 1, bias=False)
        if state:  # state buffers written in the graph by a whole-buffer blend (AFM "extend" style)
            self.kp = nn.Conv2d(HID, NKV * D, 1, bias=False)
            self.vp = nn.Conv2d(HID, NKV * D, 1, bias=False)
            self.register_buffer("k", torch.zeros(NKV, CTX, D))
            self.register_buffer("v", torch.zeros(NKV, CTX, D))

    def attend(self, x, k, v):
        q = self.q(x).reshape(NKV, GRP, D, T).permute(0, 1, 3, 2).reshape(NKV, GRP * T, D)
        p = torch.softmax((q @ k.transpose(1, 2)) * D ** -0.5, -1)
        o = (p @ v).reshape(NKV, GRP, T, D).permute(0, 1, 3, 2).reshape(1, NH * D, 1, T)
        return self.o(o)

    def forward(self, x, a=None, b=None):
        if not self.state:  # a, b = K, V history inputs (read-only, as in our chunks)
            return self.attend(x, a, b)
        keep, onehot = a, b  # keep (1, CTX, 1): 0 at the written rows; onehot (1, CTX, T): row t -> its position
        kn = self.kp(x).reshape(NKV, D, T).transpose(1, 2)            # (NKV, T, D)
        vn = self.vp(x).reshape(NKV, D, T).transpose(1, 2)
        k2 = self.k * keep + onehot @ kn
        v2 = self.v * keep + onehot @ vn
        self.k.copy_(k2)
        self.v.copy_(v2)
        return self.attend(x, k2, v2)


def build(kind: str) -> Path:
    out = ROOT / f"{kind}_{CTX // 1024}k.aimodel"
    if out.exists():
        return out
    m = Attn(kind == "state").eval().to(torch.float16)
    x = torch.randn(1, HID, 1, T, dtype=torch.float16)
    if kind == "state":
        keep = torch.ones(1, CTX, 1, dtype=torch.float16)
        onehot = torch.zeros(1, CTX, T, dtype=torch.float16)
        ex, names, states = (x, keep, onehot), ["x", "keep", "onehot"], ["k", "v"]
    else:
        kv = torch.randn(NKV, CTX, D, dtype=torch.float16)
        ex, names, states = (x, kv, kv.clone()), ["x", "k", "v"], []
    ep = torch.export.export(m, ex, strict=False).run_decompositions(coreai_torch.get_decomp_table())
    cast_to_16_bit_precision(ep)
    conv = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    conv.add_exported_program(ep, input_names=names, output_names=["y"], state_names=states or None,
                              entrypoint_name=f"a{CTX // 1024}k")
    prog = conv.to_coreai()
    prog.optimize()
    ROOT.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out, ignore_errors=True)
    prog.save_asset(out)
    return out


def placement(since: float) -> str:
    mans = [p for p in CACHE.glob("*/*/*/*/model.aimodelx/**/manifest.plist") if p.stat().st_mtime >= since]
    if not mans:
        return "cached"
    text = max(mans, key=lambda p: p.stat().st_mtime).read_bytes()
    return ("fully on ANE" if b"mps.fullyPlacedOnANE" in text else "NOT fully on ANE") + \
        f", GPU activity {'no' if b'mps.noGPUActivity' in text else 'yes'}"


async def run(kind: str) -> None:
    path = build(kind)
    gc.collect()
    w0, t0 = wired_gb(), time.time()
    model = await AIModel.load(path, specialization_options=specialization_for("ane"))
    w1 = wired_gb()
    fn = model.load_function(f"a{CTX // 1024}k")
    w2 = wired_gb()
    x = NDArray(np.random.default_rng(0).standard_normal((1, HID, 1, T)).astype(np.float16), StorageKind.IO_SURFACE)
    kvsz = NKV * CTX * D
    k = NDArray(np.zeros((NKV, CTX, D), np.float16), StorageKind.IO_SURFACE)
    v = NDArray(np.zeros((NKV, CTX, D), np.float16), StorageKind.IO_SURFACE)
    w3 = wired_gb()
    keep = NDArray(np.ones((1, CTX, 1), np.float16), StorageKind.IO_SURFACE)
    oh = np.zeros((1, CTX, T), np.float16); oh[0, np.arange(T), np.arange(T)] = 1
    onehot = NDArray(oh, StorageKind.IO_SURFACE)
    call = (lambda: fn(inputs={"x": x, "keep": keep, "onehot": onehot}, state={"k": k, "v": v})) if kind == "state" else \
        (lambda: fn(inputs={"x": x, "k": k, "v": v}))
    await call()
    w4 = wired_gb()
    ts = []
    for _ in range(30):
        t1 = time.perf_counter()
        await call()
        ts.append(time.perf_counter() - t1)
    w5 = wired_gb()
    print(f"[{kind:5s}] {placement(t0)} | K+V {2 * kvsz * 2 / 2 ** 30:.3f} GB | wired: load {w1 - w0:+.2f}, "
          f"load_function {w2 - w0:+.2f}, buffers {w3 - w0:+.2f}, first call {w4 - w0:+.2f}, after 30 {w5 - w0:+.2f} GB "
          f"| {np.median(ts) * 1e3:.2f} ms/call", flush=True)


if __name__ == "__main__":
    for kind in (sys.argv[1:] or ["input", "state"]):
        asyncio.run(run(kind))
