"""KV cache as a Core AI state written IN the graph with a slice assignment at a runtime position (aten.slice_scatter ->
coreai.slice_update, the op AFM uses) instead of index_copy_ (-> scatter_nd, which sent the graph to the GPU):
one max-length state kv (NKV, SMAX, D); entry points s2k / s16k / s64k attend over windows [:, :ctx]; each call writes
its T rows at `pos` (int32 input). Checks: placement (fully on ANE?), rows land at pos and persist, earlier rows
untouched, call time per window.
    .venv/bin/python coreai_kv_slice_probe.py"""
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

C, D, NKV, T, SMAX = 512, 256, 4, 8, 65536
CTXS = (2048, 16384, 65536)
ROOT = Path(__file__).resolve().parent / "artifacts_kv_slice"
CACHE = Path.home() / "Library/Caches/coreai-cache"


class SliceKV(nn.Module):
    def __init__(self, ctx: int) -> None:
        super().__init__()
        self.ctx = ctx
        torch.manual_seed(0)
        self.conv = nn.Conv2d(C, C, 1, bias=False)
        self.register_buffer("kv", torch.zeros(NKV, SMAX, D))

    def forward(self, x, pos, mask):                     # x (1, C, 1, T), pos (1,) int32, mask (1, ctx)
        y = self.conv(x)
        rows = y[0, :D, 0, :].transpose(0, 1)            # (T, D)
        p = pos.item()
        torch._check(p >= 0)
        torch._check(p <= SMAX - T)
        self.kv[:, p:p + T] = rows.unsqueeze(0).expand(NKV, T, D)
        win = self.kv[:, : self.ctx]
        o = torch.softmax(rows.unsqueeze(0) @ win.transpose(1, 2) * D ** -0.5 + mask, -1) @ win
        return y, o


def build() -> Path:
    out = ROOT / "slice_kv.aimodel"
    if out.exists():
        return out
    conv = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    for ctx in CTXS:
        m = SliceKV(ctx).eval().to(torch.float16)
        ex = (torch.randn(1, C, 1, T, dtype=torch.float16), torch.tensor([16], dtype=torch.int32),
              torch.zeros(1, ctx, dtype=torch.float16))
        ep = torch.export.export(m, ex, strict=False).run_decompositions(coreai_torch.get_decomp_table())
        cast_to_16_bit_precision(ep)
        conv.add_exported_program(ep, input_names=["x", "pos", "mask"], output_names=["y", "o"], state_names=["kv"],
                                  entrypoint_name=f"s{ctx // 1024}k")
    prog = conv.to_coreai()
    prog.optimize()
    ir = str(prog)
    ROOT.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out, ignore_errors=True)
    prog.save_asset(out)
    print(f"IR ops: slice_update {ir.count('slice_update')}, scatter {ir.count('scatter')}", flush=True)
    return out


def placement(since: float) -> str:
    mans = [p for p in CACHE.glob("*/*/*/*/model.aimodelx/**/manifest.plist") if p.stat().st_mtime >= since]
    if not mans:
        return "cached"
    text = max(mans, key=lambda p: p.stat().st_mtime).read_bytes()
    return ("fully on ANE" if b"mps.fullyPlacedOnANE" in text else "NOT fully on ANE") + \
        f", {text.count(b'_ANE_region_')} ANE region refs, GPU activity {'no' if b'mps.noGPUActivity' in text else 'yes'}"


async def main() -> None:
    path = build()
    t0 = time.time()
    model = await AIModel.load(path, specialization_options=specialization_for("ane"))
    print(f"load {time.time() - t0:.0f}s; {placement(t0)}", flush=True)
    fns = {ctx: model.load_function(f"s{ctx // 1024}k") for ctx in CTXS}
    r = np.random.default_rng(1)
    state = NDArray(np.zeros((NKV, SMAX, D), np.float16), StorageKind.IO_SURFACE)
    xs = [(r.standard_normal((1, C, 1, T)) * 0.1).astype(np.float16) for _ in range(3)]
    ok = True
    for i, (ctx, pos) in enumerate(((2048, 100), (16384, 108), (65536, 60000))):
        fn = fns[ctx]
        mask = np.where(np.arange(ctx)[None, :] < pos + T, 0, -1e4).astype(np.float16)
        out = await fn(inputs={"x": NDArray(xs[i], StorageKind.IO_SURFACE), "pos": NDArray(np.array([pos], np.int32)),
                               "mask": NDArray(mask, StorageKind.IO_SURFACE)}, state={"kv": state})
        kv = state.numpy()
        y = out["y"].numpy()[0, :D, 0, :].T.astype(np.float32)
        err = float(np.abs(kv[:, pos:pos + T].astype(np.float32) - y[None]).max())
        nz = sorted({int(p) for p in np.nonzero(np.abs(kv).sum((0, 2)) > 0)[0]})
        print(f"call {i} (s{ctx // 1024}k, pos {pos}): max|kv[pos:pos+T]-rows| {err:.3g}; nonzero rows {nz[:3]}..{nz[-3:]} "
              f"({len(nz)} rows)", flush=True)
        ok &= err < 1e-2
    for ctx in CTXS:
        feed = {"x": NDArray(xs[0], StorageKind.IO_SURFACE), "pos": NDArray(np.array([200], np.int32)),
                "mask": NDArray(np.zeros((1, ctx), np.float16), StorageKind.IO_SURFACE)}
        await fns[ctx](inputs=feed, state={"kv": state})
        ts = []
        for _ in range(20):
            t = time.perf_counter()
            await fns[ctx](inputs=feed, state={"kv": state})
            ts.append(1e3 * (time.perf_counter() - t))
        print(f"s{ctx // 1024}k: {np.median(ts):.3f} ms (p10 {np.percentile(ts, 10):.3f})", flush=True)
    print("RESULT:", "slice_update KV writes correct" if ok else "MISMATCH", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
