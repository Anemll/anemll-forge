"""KV cache shared by two Core AI programs: the ANE reader takes the max-length buffer (NKV, SMAX, D) as a plain INPUT and
attends over a static window [:, :ctx] (entry points per ctx); a small writer program owns the SAME NDArray as its
STATE and writes T rows at `pos` with a slice assignment (slice_update; may run on the GPU). Checks: reader placement
(fully on ANE?), writer placement / time, whether the reader sees the writer's rows (no host copy), reader time
per window with the max-length input.
    .venv/bin/python coreai_kv_writer_probe.py"""
from __future__ import annotations

# Use only the helpers shipped in this repository.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

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

from coreai_bench_helpers import specialization_for

C, D, NKV, T, SMAX, NL = 512, 256, 4, 8, 65536, 16
CTXS = (2048, 16384, 65536)
ROOT = Path(__file__).resolve().parent / "artifacts_kv_writer"
CACHE = Path.home() / "Library/Caches/coreai-cache"


class Reader(nn.Module):
    def __init__(self, ctx: int) -> None:
        super().__init__()
        self.ctx = ctx
        torch.manual_seed(0)
        self.conv = nn.Conv2d(C, C, 1, bias=False)

    def forward(self, x, kv, mask):                      # kv (NKV, SMAX, D) input; window [:, :ctx]
        y = self.conv(x)
        rows = y[0, :D, 0, :].transpose(0, 1)
        win = kv[:, : self.ctx]
        o = torch.softmax(rows.unsqueeze(0) @ win.transpose(1, 2) * D ** -0.5 + mask, -1) @ win
        return y, o


class Writer(nn.Module):
    """Writes the T new rows of NL layers' K and V caches at pos (one call per token for all layers)."""

    def __init__(self) -> None:
        super().__init__()
        for i in range(NL):
            self.register_buffer(f"k{i}", torch.zeros(NKV, SMAX, D))
            self.register_buffer(f"v{i}", torch.zeros(NKV, SMAX, D))

    def forward(self, pos, *rows):                       # rows: k0_new, v0_new, ... (NKV, T, D)
        p = pos.item()
        torch._check(p >= 0)
        torch._check(p <= SMAX - T)
        for i in range(NL):
            getattr(self, f"k{i}")[:, p:p + T] = rows[2 * i]
            getattr(self, f"v{i}")[:, p:p + T] = rows[2 * i + 1]
        return pos + 0


def save(name, entries, state_names=None) -> Path:
    out = ROOT / f"{name}.aimodel"
    if out.exists():
        return out
    conv = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    for ename, m, ex, ins, outs in entries:
        ep = torch.export.export(m, ex, strict=False).run_decompositions(coreai_torch.get_decomp_table())
        cast_to_16_bit_precision(ep)
        conv.add_exported_program(ep, input_names=ins, output_names=outs, state_names=state_names,
                                  entrypoint_name=ename)
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
        f", {text.count(b'_ANE_region_')} ANE region refs, GPU activity {'no' if b'mps.noGPUActivity' in text else 'yes'}"


async def main() -> None:
    f16 = torch.float16
    rpath = save("reader", [(f"r{c // 1024}k", Reader(c).eval().to(f16),
                             (torch.randn(1, C, 1, T, dtype=f16), torch.randn(NKV, SMAX, D, dtype=f16), torch.zeros(1, c, dtype=f16)),
                             ["x", "kv", "mask"], ["y", "o"]) for c in CTXS])
    wex = (torch.tensor([16], dtype=torch.int32),) + tuple(torch.randn(NKV, T, D, dtype=f16) for _ in range(2 * NL))
    wpath = save("writer", [("w", Writer().eval().to(f16), wex, ["pos"] + [f"{s}{i}_new" for i in range(NL) for s in "kv"],
                             ["pos_out"])], state_names=[f"{s}{i}" for i in range(NL) for s in "kv"])
    t0 = time.time()
    reader = await AIModel.load(rpath, specialization_options=specialization_for("ane"))
    print(f"reader load {time.time() - t0:.0f}s; {placement(t0)}", flush=True)
    t0 = time.time()
    writer = await AIModel.load(wpath, specialization_options=specialization_for("ane"))
    print(f"writer load {time.time() - t0:.0f}s; {placement(t0)}", flush=True)
    rf = {c: reader.load_function(f"r{c // 1024}k") for c in CTXS}
    wf = writer.load_function("w")
    r = np.random.default_rng(1)
    kv = {f"{s}{i}": NDArray(np.zeros((NKV, SMAX, D), np.float16), StorageKind.IO_SURFACE) for i in range(NL) for s in "kv"}
    rows = {f"{s}{i}_new": NDArray((r.standard_normal((NKV, T, D)) * 0.5).astype(np.float16), StorageKind.IO_SURFACE)
            for i in range(NL) for s in "kv"}
    pos = 1000
    await wf(inputs={"pos": NDArray(np.array([pos], np.int32)), **rows}, state=kv)
    k0 = kv["k0"].numpy()
    print(f"writer: rows at {pos} match: {float(np.abs(k0[:, pos:pos + T] - rows['k0_new'].numpy()).max()):.3g}; "
          f"nonzero rows {int((np.abs(k0).sum((0, 2)) > 0).sum())}", flush=True)
    # reader sees the written rows? attention of x's rows over a window whose only nonzero rows are the written ones
    x = NDArray((r.standard_normal((1, C, 1, T)) * 0.1).astype(np.float16), StorageKind.IO_SURFACE)
    for c in CTXS:
        mask = NDArray(np.where(np.arange(c)[None, :] >= pos, 0, -1e4).astype(np.float16), StorageKind.IO_SURFACE)
        out = await rf[c](inputs={"x": x, "kv": kv["k0"], "mask": mask})
        o = out["o"].numpy()
        y = out["y"].numpy()[0, :D, 0, :].T.astype(np.float32)
        win = k0[:, :c].astype(np.float32)
        s = y[None] @ win.transpose(0, 2, 1) * D ** -0.5 + np.where(np.arange(c) >= pos, 0, -1e4)[None, None]
        s = np.exp(s - s.max(-1, keepdims=True))
        ref = (s / s.sum(-1, keepdims=True)) @ win
        cos = float((o.astype(np.float64).ravel() @ ref.ravel()) / (np.linalg.norm(o) * np.linalg.norm(ref) + 1e-30))
        feed = {"x": x, "kv": kv["k0"], "mask": mask}
        ts = []
        for _ in range(20):
            t = time.perf_counter()
            await rf[c](inputs=feed)
            ts.append(1e3 * (time.perf_counter() - t))
        print(f"reader r{c // 1024}k: sees writer rows cos {cos:.5f}; {np.median(ts):.3f} ms (p10 {np.percentile(ts, 10):.3f})",
              flush=True)
    ts = []
    for i in range(20):
        t = time.perf_counter()
        await wf(inputs={"pos": NDArray(np.array([pos + 8 * i], np.int32)), **rows}, state=kv)
        ts.append(1e3 * (time.perf_counter() - t))
    print(f"writer (16 layers K+V, 8 rows): {np.median(ts):.3f} ms (p10 {np.percentile(ts, 10):.3f})", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
