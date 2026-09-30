"""Host cost of a Core AI call with a real chunk's I/O shapes but trivial compute: 25 inputs / 12 outputs shaped like
chunk L00-03 of the Qwen3.8 target (conv (11, 10240) x3, rec (48, 128, 128) x3, pend (48, 25, 128) x3, k / v new
(4, 8, 256), y (1, 5120, 1, 8)), each output = its input * 0.5 (+ a 1x1 conv on y so the program lands on the ANE).
If this costs most of the ~5 ms per chunk Core AI is slower than Core ML, the gap is I/O handling, not the ANE program.
    .venv/bin/python coreai_output_cost.py"""
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # coreai (builder, coreai_util)
from coreai_util import specialization_for

OUT = Path(__file__).resolve().parent / "artifacts_output_cost" / "io12.aimodel"
SHAPES = {"x": (1, 5120, 1, 8), "conv0": (11, 10240), "conv1": (11, 10240), "conv2": (11, 10240),
          "rec0": (48, 128, 128), "rec1": (48, 128, 128), "rec2": (48, 128, 128),
          "pend0": (48, 25, 128), "pend1": (48, 25, 128), "pend2": (48, 25, 128),
          "k3": (4, 8, 256), "v3": (4, 8, 256)}
NAMES = list(SHAPES)


class IO12(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(5120, 5120, 1, bias=False)
        nn.init.normal_(self.conv.weight, std=5120 ** -0.5)

    def forward(self, x, conv0, conv1, conv2, rec0, rec1, rec2, pend0, pend1, pend2, k3, v3):
        y = self.conv(x)
        return (y, conv0 * 0.5, conv1 * 0.5, conv2 * 0.5, rec0 * 0.5, rec1 * 0.5, rec2 * 0.5,
                pend0 * 0.5, pend1 * 0.5, pend2 * 0.5, k3 * 0.5, v3 * 0.5)


def build() -> Path:
    if OUT.exists():
        return OUT
    ex = tuple(torch.randn(*SHAPES[n], dtype=torch.float16) for n in NAMES)
    ep = torch.export.export(IO12().eval().to(torch.float16), ex, strict=False).run_decompositions(
        coreai_torch.get_decomp_table())
    cast_to_16_bit_precision(ep)
    conv = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    conv.add_exported_program(ep, input_names=NAMES, output_names=[f"{n}_out" for n in NAMES])
    prog = conv.to_coreai()
    prog.optimize()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(OUT, ignore_errors=True)
    prog.save_asset(OUT)
    return OUT


async def main() -> None:
    m = await AIModel.load(build(), specialization_options=specialization_for("ane"))
    f = m.load_function("main")
    ins = {n: NDArray(np.random.default_rng(0).standard_normal(SHAPES[n]).astype(np.float16) * 0.1,
                      StorageKind.IO_SURFACE) for n in NAMES}
    for _ in range(10):
        out = await f(inputs=ins)
    ts = []
    for _ in range(100):
        t = time.perf_counter()
        out = await f(inputs=ins)
        ts.append(1e3 * (time.perf_counter() - t))
    chain = []
    for _ in range(20):  # feed outputs back as inputs, as the runtime does
        cur = dict(ins)
        t = time.perf_counter()
        for _ in range(16):
            out = await f(inputs=cur)
            cur = {n: out[f"{n}_out"] for n in NAMES}
        chain.append(1e3 * (time.perf_counter() - t) / 16)
    # persistent IOSurface inputs, outputs copied into them (host memcpy through the storages' buffer protocol)
    import ctypes

    def view(nd):
        mv = memoryview(nd._tensor)  # noqa: SLF001
        ptr = np.frombuffer(mv, dtype=np.uint8).__array_interface__["data"][0]
        return np.ctypeslib.as_array((ctypes.c_uint8 * mv.nbytes).from_address(ptr))
    pers = {n: NDArray(np.zeros(SHAPES[n], np.float16), StorageKind.IO_SURFACE) for n in NAMES}
    pv = {n: view(pers[n]) for n in NAMES}
    for n in NAMES:
        pv[n][:] = view(ins[n])
    out = await f(inputs=pers)
    print("output storage kinds:", {n: str(getattr(out[f"{n}_out"], "storage_kind", "?")) for n in NAMES[:2]}, flush=True)
    chain2 = []
    for _ in range(20):
        t = time.perf_counter()
        for _ in range(16):
            out = await f(inputs=pers)
            for n in NAMES:
                pv[n][:] = view(out[f"{n}_out"])
        chain2.append(1e3 * (time.perf_counter() - t) / 16)
    print(f"chained via persistent IOSurface inputs + memcpy: {np.median(chain2):.2f} ms per call", flush=True)
    mb = sum(np.prod(s) for s in SHAPES.values()) * 2 / 2**20
    print(f"12 inputs / 12 outputs ({mb:.1f} MB each way), trivial compute: single call median {np.median(ts):.2f} ms "
          f"(p90 {np.percentile(ts, 90):.2f}); chained (outputs -> inputs) {np.median(chain):.2f} ms per call", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
