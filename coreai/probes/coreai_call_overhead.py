"""Core AI per-call overhead on the ANE, twin of ane-vector-lut/scripts/qwen38_call_overhead.py (Core ML): the same
1x1 conv + add, optionally with a large KV-like input (4, S, 256) of which only 8 values are read (per-call cost of
passing a big buffer: mapped or copied?), timed alone and as a chain of 16 dependent calls.
    .venv/bin/python coreai_call_overhead.py"""
from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import coreai_torch
from coreai.runtime import AIModel, NDArray
from coreai.runtime._ndarray import StorageKind
from coreai_opt.casting import cast_to_16_bit_precision

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ane-vector-lut/coreai (builder, coreai_util)
from coreai_util import specialization_for

ROOT = Path(__file__).resolve().parent / "artifacts_call_overhead"
CACHE = Path.home() / "Library/Caches/coreai-cache"
CASES = [((1, 64, 1, 8), 0), ((1, 512, 1, 8), 0), ((1, 512, 1, 8), 8192), ((1, 512, 1, 8), 65536)]


class Tiny(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c, c, 1, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.from_numpy(
                (np.random.default_rng(0).standard_normal((c, c, 1, 1)) * c**-0.5).astype(np.float32)))

    def forward(self, x):
        return self.conv(x) + 1.0


class TinyKV(Tiny):
    def forward(self, x, kvin):
        return self.conv(x) + kvin[0:1, 0:1, 0:8].reshape(1, 1, 1, 8) + 1.0


def build(shape, kv) -> Path:
    tag = "x".join(map(str, shape)) + (f"_kv{kv}" if kv else "")
    out = ROOT / f"conv_{tag}.aimodel"
    if out.exists():
        return out
    model = (TinyKV if kv else Tiny)(shape[1]).eval().to(torch.float16)
    ex = (torch.randn(*shape, dtype=torch.float16),) + ((torch.randn(4, kv, 256, dtype=torch.float16),) if kv else ())
    ep = torch.export.export(model, ex, strict=False).run_decompositions(coreai_torch.get_decomp_table())
    cast_to_16_bit_precision(ep)
    conv = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    conv.add_exported_program(ep, input_names=["x"] + (["kvin"] if kv else []), output_names=["y"])
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
    return "ANE" if b"mps.fullyPlacedOnANE" in text else ("ANE+GPU" if b"_ANE_region_" in text else "GPU")


async def run(shape, kv, backing=StorageKind.BYTES) -> None:
    t0 = time.time()
    model = await AIModel.load(build(shape, kv), specialization_options=specialization_for("ane"))
    fn = model.load_function("main")
    names = list(fn.desc.input_names)
    xname = next(n for n in names if n.startswith("x"))
    x = NDArray(np.ones(shape, np.float16), backing)
    feed = {xname: x}
    if kv:
        feed[next(n for n in names if n != xname)] = NDArray(np.zeros((4, kv, 256), np.float16), backing)
    for _ in range(20):
        await fn(inputs=feed)
    ts = []
    for _ in range(300):
        t = time.perf_counter()
        await fn(inputs=feed)
        ts.append(1e3 * (time.perf_counter() - t))
    chain = []
    for _ in range(50):
        t = time.perf_counter()
        cur = dict(feed)
        for _ in range(16):
            out = await fn(inputs=cur)
            cur[xname] = list(out.values())[0] if isinstance(out, dict) else out
        chain.append(1e3 * (time.perf_counter() - t))
    mb_io = (np.prod(shape) + kv * 4 * 256) * 2 / 2**20
    print(f"{str(shape) + (f' + kv {kv}' if kv else ''):26s} {backing.value:10s} ({mb_io:7.2f} MB in) on {placement(t0)}: single call median "
          f"{np.median(ts):.3f} ms (p10 {np.percentile(ts, 10):.3f}, p90 {np.percentile(ts, 90):.3f}); chain of 16: "
          f"{np.median(chain):.2f} ms = {np.median(chain) / 16:.3f} ms/call", flush=True)


def main() -> None:
    for shape, kv in CASES:
        for backing in (StorageKind.BYTES, StorageKind.IO_SURFACE):
            asyncio.run(run(shape, kv, backing))


if __name__ == "__main__":
    main()
