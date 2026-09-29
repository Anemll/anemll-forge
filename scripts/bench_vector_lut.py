#!/usr/bin/env python3
"""ANE vector palettization check: does a LUT entry of several weights run natively?

Scalar palettization stores one weight per LUT entry; vector palettization
(coreai-opt ``cluster_dim > 1``) stores ``cluster_dim`` consecutive output-channel
weights per entry, so one index fetches a whole vector. The ANE compiler has a
vector-palette path (``palette_vector_size``, "only supported at Cout for ANE").

Workload: S sequential 1x1 nn.Conv2d(C, C) on a (1, C, H, W) input with small
H*W, so the run is weight-bandwidth bound. Every layer's weights are generated
from a random LUT (unit-variance outputs), so compression is exact (UNIQUE mode,
no k-means). If the ANE decompresses on the fly, time tracks compressed
bits/weight; if the LUT is expanded before running, time matches dense FP16.

Variants (bits/weight = n_bits / cluster_dim; LUT values = 2**n_bits * cluster_dim):
  dense   FP16 weights                        16 bits/weight
  s8/s4/s2  scalar, 256/16/4-entry LUT        8 / 4 / 2
  v2n4    vector of 2,  16 entries (32 values)   2
  v4n4    vector of 4,  16 entries (64 values)   1
  v4n6    vector of 4,  64 entries (256 values)  1.5
  v2n6    vector of 2,  64 entries (128 values)  3
  v8n4    vector of 8,  16 entries (128 values)  0.5
  v8n2    vector of 8,   4 entries (32 values)   0.25
  v16n4   vector of 16, 16 entries (256 values)  0.25
  v2n8    vector of 2, 256 entries (512 values)  4   (over the ANE limit)

The ANE compiler takes vector LUTs up to vector size 16 and 256 LUT values
(entries x vector size); larger ones fail ANE validation and run elsewhere.

Checks per variant: IR has coreai.lut_to_dense with the expected vector size,
placement from the compile cache manifest (mps.fullyPlacedOnANE /
mps.noGPUActivity), output cosine vs an FP32 torch reference, timing.

Usage:
  uv run python bench_vector_lut.py                    # all variants
  uv run python bench_vector_lut.py dense s4 v2 v4
Do NOT set USE_LOCAL_COREAI.
"""

from __future__ import annotations

import argparse
import asyncio
import plistlib
import re
import statistics
import time
from pathlib import Path

import coreai_torch
import numpy as np
import torch
import torch.nn as nn
from coreai.runtime import AIModel, NDArray
from coreai_opt.casting import cast_to_16_bit_precision
from coreai_opt.coreai_utils.common import CompressionGranularity, DType
from coreai_opt.coreai_utils.passes import weight_palettization
from coreai_opt.coreai_utils._utils.palettize_utils import LutParams
from coreai_opt.coreai_utils.passes.weight_palettization import palettize_weights

from bench_sparsity import to_numpy
from bench_stacked import SEED, remove_path, specialization_for

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts_vector_lut"
CACHE = Path.home() / "Library/Caches/coreai-cache"

VARIANTS = {  # name: (cluster_dim, n_bits); None = dense FP16
    "dense": None,
    "s8": (1, 8),
    "s4": (1, 4),
    "s2": (1, 2),
    "v2n4": (2, 4),
    "v4n4": (4, 4),
    "v4n6": (4, 6),
    "v2n6": (2, 6),
    "v8n4": (8, 4),
    "v8n2": (8, 2),
    "v16n4": (16, 4),
    "v2n8": (2, 8),  # 512 LUT values: over the ANE's 256-value limit
}

# Exact LUT-generated weights: hand the pass the LUT and indices each layer was
# built from (keyed by the FP16 weight bytes); fall back to UNIQUE mode.
_orig_blockwise_compress = weight_palettization._blockwise_compress
KNOWN_LUTS: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}


def _unique_blockwise_compress(original_data, mode, *args, **kwargs):
    known = KNOWN_LUTS.get(np.ascontiguousarray(original_data, np.float16).tobytes())
    block_sizes = args[1] if len(args) > 1 else kwargs.get("block_sizes")
    per_tensor = not block_sizes or all(b in (0, n) for b, n in zip(block_sizes, original_data.shape))
    if known is not None and per_tensor:
        lut, idx = known
        cluster_dim = lut.shape[1]
        extra = (1,) * (original_data.ndim - 2)
        return LutParams(
            indices=idx.reshape(*idx.shape, *extra).astype(np.uint8),
            lut=lut.reshape(*(1,) * original_data.ndim, *lut.shape),
            vector_axis=0 if cluster_dim > 1 else None,
        )
    return _orig_blockwise_compress(original_data, "UNIQUE", *args, **kwargs)


weight_palettization._blockwise_compress = _unique_blockwise_compress


def _groups1_cluster_dim_valid(op, cluster_dim: int, channel_axis: int) -> bool:
    # coreai-opt 0.2.1's check reads conv2d `.groups`, which the OpView lacks
    # (AttributeError). All convs here are groups=1, so the shape test suffices.
    shape = list(op.result.type.shape)
    return shape[channel_axis] % cluster_dim == 0


weight_palettization._is_cluster_dim_valid = _groups1_cluster_dim_valid


class ConvChain(nn.Module):
    """S layers of 1x1 Conv2d(C, C), or nn.Linear(C, C) with op="linear"."""

    def __init__(self, c: int, stack: int, op: str = "conv") -> None:
        super().__init__()
        self.op = op
        make = (lambda: nn.Conv2d(c, c, 1, bias=False)) if op == "conv" else (
            lambda: nn.Linear(c, c, bias=False))
        self.convs = nn.ModuleList(make() for _ in range(stack))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x)
        return x


def lut_weight(lut: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """(C_out, C_in) weight with w[o*cd + j, i] = lut[idx[o, i], j]."""
    n_out, c_in = idx.shape
    return lut[idx].transpose(0, 2, 1).reshape(n_out * lut.shape[1], c_in)


def input_shape(c: int, hw: int, op: str) -> tuple[int, ...]:
    return (1, c, hw, hw) if op == "conv" else (hw * hw, c)


def build_model(
    c: int, hw: int, stack: int, cfg: tuple[int, int] | None, op: str = "conv"
) -> ConvChain:
    """Chain whose every layer is generated from one random LUT.

    Each LUT is rescaled (a scalar, so the LUT structure stays exact) so that
    the benchmark input keeps unit RMS through every layer; unscaled random
    chains drift far enough over 32 layers to overflow FP16.
    """
    torch.manual_seed(SEED)
    model = ConvChain(c, stack, op).eval()
    rng = np.random.default_rng(SEED)
    cluster_dim, n_bits = cfg or (1, 8)  # dense: 256-level weights, stored FP16
    torch.manual_seed(SEED)
    x = torch.randn(*input_shape(c, hw, op)).numpy().astype(np.float64)
    x = x.reshape(c, -1) if op == "conv" else x.T
    with torch.no_grad():
        for conv in model.convs:
            lut = rng.standard_normal((1 << n_bits, cluster_dim)) * c**-0.5
            idx = rng.integers(0, 1 << n_bits, size=(c // cluster_dim, c))
            y = lut_weight(lut, idx) @ x
            lut = (lut / np.sqrt((y**2).mean())).astype(np.float16)
            w = lut_weight(lut, idx).astype(np.float32)
            KNOWN_LUTS[w.astype(np.float16).tobytes()] = (lut, idx)
            conv.weight.copy_(torch.from_numpy(w).view_as(conv.weight))
            x = w.astype(np.float64) @ x
    return model


def export(
    name: str, c: int, hw: int, stack: int, force: bool, op: str = "conv", group_size: int = 0,
    lut_dtype: str = "fp16",
) -> tuple[Path, ConvChain, str]:
    cfg = VARIANTS[name]
    tag = ("" if op == "conv" else f"_{op}") + (f"_G{group_size}" if group_size else "")
    tag += "" if lut_dtype == "fp16" else f"_{lut_dtype}"
    out = ARTIFACTS / f"{name}_C{c}_HW{hw}_S{stack}{tag}.aimodel"
    ir_path = out.with_suffix(".mlir")
    model = build_model(c, hw, stack, cfg, op)
    if out.exists() and ir_path.exists() and not force:
        return out, model, ir_path.read_text()
    example = torch.randn(*input_shape(c, hw, op))
    exported = torch.export.export(
        model.to(torch.float16), (example.to(torch.float16),), strict=False
    ).run_decompositions(coreai_torch.get_decomp_table())
    model.float()
    cast_to_16_bit_precision(exported)
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.add_exported_program(exported, input_names=["x"], output_names=["y"])
    program = converter.to_coreai()
    program.optimize()
    if cfg is not None:
        cluster_dim, n_bits = cfg
        program = palettize_weights(
            program,
            lut_dtype={"fp16": None, "int8": DType.INT8, "fp8": DType.FP8_E4M3FN}[lut_dtype],
            n_bits=n_bits,
            granularity=(CompressionGranularity.PER_GROUPED_CHANNEL if group_size
                         else CompressionGranularity.PER_TENSOR),
            group_size=group_size or 32,
            cluster_dim=cluster_dim,
            weight_num_threshold=1024,
            enable_fast_kmeans_mode=False,
        )
    ir = str(program)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    remove_path(out)
    program.save_asset(out)
    ir_path.write_text(ir)
    return out, model, ir


def ir_summary(ir: str) -> str:
    luts = re.findall(r"coreai\.lut_to_dense[^\n]*", ir)
    if not luts:
        return "no lut_to_dense (dense weights)"
    shapes = sorted(set(re.findall(r"tensor<(\d+x\d+x\d+x\d+x\d+x?f16)>", " ".join(luts))))
    return f"{len(luts)}x lut_to_dense, LUT {shapes[:1]}"


def placement(path: Path) -> tuple[str, float]:
    """(placement flags, compiled package MB) from the newest cache entry of this model."""
    digest = (path / "main.hash").read_bytes().hex()
    manifests = sorted(
        CACHE.glob(f"*/*/{digest}/*/model.aimodelx/**/manifest.plist"),
        key=lambda p: p.stat().st_mtime,
    )
    if not manifests:
        return "no cache entry", 0.0
    man = manifests[-1]
    text = man.read_bytes()
    flags = [f for f in ("mps.fullyPlacedOnANE", "mps.noGPUActivity") if f.encode() in text]
    if b"ANE_region" in text and b"mps.fullyPlacedOnANE" not in text:
        flags.append("partial ANE region")
    elif b"ANE_region" not in text:
        flags.append("NO ANE region")
    pkg = man.parent
    mb = sum(f.stat().st_size for f in pkg.rglob("*") if f.is_file()) / 1e6
    try:
        plistlib.loads(text)
    except Exception:
        pass
    return ", ".join(flags), mb


async def run(path: Path, x: np.ndarray, compute: str, iters: int = 50, warmup: int = 5):
    model = await AIModel.load(path, specialization_options=specialization_for(compute))
    fn = model.load_function("main")
    name = list(fn.desc.input_names)[0]
    nd = NDArray(x)
    for _ in range(warmup):
        out = await fn(inputs={name: nd})
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = await fn(inputs={name: nd})
        times.append(time.perf_counter() - t0)
    y = to_numpy(list(out.values())[0] if isinstance(out, dict) else out)
    return statistics.median(times), y


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("variants", nargs="*", default=list(VARIANTS), help=", ".join(VARIANTS))
    ap.add_argument("--channels", type=int, default=2048)
    ap.add_argument("--hw", type=int, default=4, help="spatial H=W (small = bandwidth bound)")
    ap.add_argument("--stack", type=int, default=32)
    ap.add_argument("--compute", choices=("ane", "gpu"), default="ane")
    ap.add_argument("--op", choices=("conv", "linear"), default="conv",
                    help="1x1 Conv2d chain, or nn.Linear chain on (H*W, C)")
    ap.add_argument("--group-size", type=int, default=0,
                    help="per-grouped-channel LUTs of this many output channels (0 = per-tensor)")
    ap.add_argument("--lut-dtype", choices=("fp16", "int8", "fp8"), default="fp16",
                    help="LUT value type (int8/fp8 add a per-tensor scale)")
    ap.add_argument("--force", action="store_true", help="re-export cached models")
    args = ap.parse_args()

    c, hw, s = args.channels, args.hw, args.stack
    n_weights = c * c * s
    flops = 2 * hw * hw * c * c * s
    torch.manual_seed(SEED)
    x = torch.randn(*input_shape(c, hw, args.op))
    rows = []
    for name in args.variants:
        if name not in VARIANTS:
            raise SystemExit(f"unknown variant {name!r}")
        cfg = VARIANTS[name]
        bits = 16.0 if cfg is None else cfg[1] / cfg[0]
        print(f"== {name}: {'dense FP16' if cfg is None else f'cluster_dim={cfg[0]} n_bits={cfg[1]}'}"
              f" ({bits:g} bits/weight) C={c} HW={hw}x{hw} S={s}", flush=True)
        t0 = time.perf_counter()
        path, model, ir = export(name, c, hw, s, args.force, args.op, args.group_size, args.lut_dtype)
        print(f"  export {time.perf_counter() - t0:.1f}s  IR: {ir_summary(ir)}", flush=True)
        med, y = asyncio.run(run(path, x.to(torch.float16).numpy(), args.compute))
        with torch.no_grad():
            ref = model(x).numpy().ravel()
        yf = y.astype(np.float32).ravel()
        cos = float(yf @ ref / (np.linalg.norm(yf) * np.linalg.norm(ref) + 1e-30))
        place, pkg_mb = placement(path)
        gbs = n_weights * bits / 8 / med / 1e9
        rows.append((name, bits, med, gbs, cos, place, pkg_mb))
        print(f"  {med * 1e3:.3f} ms  {flops / med / 1e12:.2f} TFLOPS  "
              f"{gbs:.1f} GB/s compressed weights  cos={cos:.4f}  [{place}]  "
              f"compiled {pkg_mb:.0f} MB", flush=True)

    dense = next((r for r in rows if r[0] == "dense"), None)
    print(f"\n{'variant':7s} {'bits/w':>6s} {'ms':>8s} {'x dense':>7s} {'GB/s':>6s} "
          f"{'cos':>7s} {'compiled MB':>11s}  placement")
    for name, bits, med, gbs, cos, place, pkg_mb in rows:
        speed = f"{dense[2] / med:7.2f}" if dense else f"{'-':>7s}"
        print(f"{name:7s} {bits:6g} {med * 1e3:8.3f} {speed} {gbs:6.1f} {cos:7.4f} "
              f"{pkg_mb:11.0f}  {place}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
