"""Do Core AI entry points of one program share weight memory on the ANE? (Core ML multifunction models do not:
every loaded function wires its own weight copy; Apple's AFM ships all 17 context / row functions in one ANE
binary.) One conv chain (S x Conv2d(C, C), dense FP16) exported twice from the same module - "t8" (8 rows) and
"t64" (64 rows) - into one .aimodel; wired memory after load, each load_function and each first call, plus a
single-entry baseline and the per-call time.
    .venv/bin/python coreai_entry_share.py"""
from __future__ import annotations

# Use only the helpers shipped in this repository.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import asyncio
import gc
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import coreai_torch
from coreai.runtime import AIModel, NDArray
from coreai_opt.casting import cast_to_16_bit_precision

from coreai_bench_helpers import to_numpy
from coreai_bench_helpers import specialization_for

C, S = 4096, 16
ROOT = Path(__file__).resolve().parent / "artifacts_entry_share"
CACHE = Path.home() / "Library/Caches/coreai-cache"


def wired_gb() -> float:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(out.split("page size of ")[1].split()[0])
    return next(int(l.split()[-1].rstrip(".")) for l in out.splitlines() if "wired down" in l) * page / 2**30


class AttnChain(nn.Module):
    """Chain + attention of the output over K / V inputs of length S (varies per entry point: context ladder)."""
    def __init__(self) -> None:
        super().__init__()
        self.chain = Chain()

    def forward(self, x, k, v, mask):                     # x (1, C, 1, T); k, v (S, 256); mask (1, S)
        x = self.chain(x)
        q = x[0, :256, 0, :].transpose(0, 1)               # (T, 256)
        o = torch.softmax(q @ k.transpose(0, 1) * 0.0625 + mask, -1) @ v
        return x, o


class Chain(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.convs = nn.ModuleList(nn.Conv2d(C, C, 1, bias=False) for _ in range(S))
        for c in self.convs:
            nn.init.normal_(c.weight, std=C**-0.5)

    def forward(self, x):
        for c in self.convs:
            x = c(x)
        return x


def export(model: nn.Module, rows: int):
    ex = torch.randn(1, C, 1, rows, dtype=torch.float16)
    ep = torch.export.export(model, (ex,), strict=False).run_decompositions(coreai_torch.get_decomp_table())
    cast_to_16_bit_precision(ep)
    return ep


def build(name: str, entries: dict[str, int]) -> Path:
    out = ROOT / f"{name}.aimodel"
    if out.exists():
        return out
    model = Chain().eval().to(torch.float16)
    conv = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    for entry, rows in entries.items():
        conv.add_exported_program(export(model, rows), input_names=["x"], output_names=["y"], entrypoint_name=entry)
    prog = conv.to_coreai()
    prog.optimize()
    ROOT.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out, ignore_errors=True)
    prog.save_asset(out)
    return out


def size_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


async def run(path: Path, entries: dict[str, int]) -> None:
    gc.collect()
    w0 = wired_gb()
    t = time.time()
    model = await AIModel.load(path, specialization_options=specialization_for("ane"))
    log = [f"load {time.time() - t:.1f}s +{wired_gb() - w0:.2f} GB"]
    fns = {}
    for entry in entries:
        fns[entry] = model.load_function(entry)
        log.append(f"load_function {entry} +{wired_gb() - w0:.2f} GB")
    for entry, rows in entries.items():
        fn = fns[entry]
        name = list(fn.desc.input_names)[0]
        x = NDArray(np.random.default_rng(1).standard_normal((1, C, 1, rows)).astype(np.float16) * 0.1)
        out = await fn(inputs={name: x})
        ts = []
        for _ in range(20):
            t = time.perf_counter()
            out = await fn(inputs={name: x})
            ts.append(1e3 * (time.perf_counter() - t))
        y = to_numpy(list(out.values())[0] if isinstance(out, dict) else out)
        log.append(f"{entry} call {np.median(ts):.2f} ms (finite {np.isfinite(y).all()}) +{wired_gb() - w0:.2f} GB")
    print(f"[{path.stem}] ({size_mb(path):.0f} MB on disk) " + " | ".join(log), flush=True)
    del fns, model
    gc.collect()
    time.sleep(1)
    print(f"   released: +{wired_gb() - w0:.2f} GB", flush=True)


def placement() -> str:
    mans = sorted(CACHE.glob("*/*/*/*/model.aimodelx/**/manifest.plist"), key=lambda p: p.stat().st_mtime)
    if not mans:
        return "no cache manifest"
    text = mans[-1].read_bytes()
    return ", ".join(f for f in ("ANE_region", "mps.fullyPlacedOnANE", "mps.noGPUActivity") if f.encode() in text) or "GPU"


def build_attn(name: str, entries: dict[str, tuple[int, int]], lut: bool) -> Path:
    out = ROOT / f"{name}.aimodel"
    if out.exists():
        return out
    model = AttnChain().eval().to(torch.float16)
    conv = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    for entry, (rows, ctx) in entries.items():
        ex = (torch.randn(1, C, 1, rows, dtype=torch.float16), torch.randn(ctx, 256, dtype=torch.float16),
              torch.randn(ctx, 256, dtype=torch.float16), torch.zeros(1, ctx, dtype=torch.float16))
        ep = torch.export.export(model, ex, strict=False).run_decompositions(coreai_torch.get_decomp_table())
        cast_to_16_bit_precision(ep)
        conv.add_exported_program(ep, input_names=["x", "k", "v", "mask"], output_names=["y", "o"],
                                  entrypoint_name=entry)
    prog = conv.to_coreai()
    prog.optimize()
    if lut:  # vector 2 x 16 (2 bits / weight), per-tensor, like the target's MLP
        from coreai_opt.coreai_utils.common import CompressionGranularity
        from coreai_opt.coreai_utils.passes import weight_palettization
        from coreai_opt.coreai_utils.passes.weight_palettization import palettize_weights
        # coreai-opt 0.2.1 reads conv2d `.groups`, which the OpView lacks (as in bench_vector_lut.py)
        weight_palettization._is_cluster_dim_valid = (
            lambda op, cluster_dim, channel_axis: list(op.result.type.shape)[channel_axis] % cluster_dim == 0)
        prog = palettize_weights(prog, lut_dtype=None, n_bits=4, granularity=CompressionGranularity.PER_TENSOR, cluster_dim=2,
                                 weight_num_threshold=1024, enable_fast_kmeans_mode=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out, ignore_errors=True)
    prog.save_asset(out)
    return out


async def run_attn(path: Path, entries: dict[str, tuple[int, int]]) -> None:
    gc.collect()
    w0 = wired_gb()
    model = await AIModel.load(path, specialization_options=specialization_for("ane"))
    log, fns = [f"load +{wired_gb() - w0:.2f} GB"], {}
    for entry in entries:
        fns[entry] = model.load_function(entry)
        log.append(f"fn {entry} +{wired_gb() - w0:.2f} GB")
    r = np.random.default_rng(1)
    for entry, (rows, ctx) in entries.items():
        fn = fns[entry]
        names = list(fn.desc.input_names)
        vals = {"x": r.standard_normal((1, C, 1, rows)) * 0.1, "k": r.standard_normal((ctx, 256)),
                "v": r.standard_normal((ctx, 256)), "mask": np.zeros((1, ctx))}
        feed = {n: NDArray(vals[n].astype(np.float16)) for n in names}
        await fn(inputs=feed)
        ts = []
        for _ in range(20):
            t = time.perf_counter()
            await fn(inputs=feed)
            ts.append(1e3 * (time.perf_counter() - t))
        log.append(f"{entry} {np.median(ts):.2f} ms +{wired_gb() - w0:.2f} GB")
    print(f"[{path.stem}] ({size_mb(path):.0f} MB) " + " | ".join(log), flush=True)
    del fns, model
    gc.collect()
    time.sleep(1)


def main() -> None:
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "ladder":
        ent = {"s2k": (8, 2048), "s8k": (8, 8192), "s16k": (8, 16384), "p64_s2k": (64, 2048)}
        for lut in (False, True):
            p = build_attn("ladder_" + ("v2n4" if lut else "fp16"), ent, lut)
            asyncio.run(run_attn(p, ent))
            print(f"   placement: {placement()}", flush=True)
        return
    print(f"chain {S} x Conv2d({C}, {C}) fp16 = {S * C * C * 2 / 1e6:.0f} MB weights", flush=True)
    one = {"t8": 8}
    two = {"t8": 8, "t64": 64}
    p1, p2 = build("single_t8", one), build("dual_t8_t64", two)
    asyncio.run(run(p1, one))
    print(f"   placement: {placement()}", flush=True)
    asyncio.run(run(p2, two))
    print(f"   placement: {placement()}", flush=True)


if __name__ == "__main__":
    main()
