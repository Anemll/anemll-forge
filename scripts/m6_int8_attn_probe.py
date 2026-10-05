"""INT8 x INT8 at attention shapes on the M6 ANE: the history QK and PV matmuls alone, both operands runtime tensors.

Per KV head group (4 heads), one tile: QK = Q [4, M, 256] x K^T [4, 256, L] -> scores [4, M, L], PV = P [4, M, L] x
V [4, L, 256] -> [4, M, 256], with M = 6 query heads x rows (48 for the 8-row verify, 384 for the 64-row prefill) and
L the tile width. A package repeats the op over S tiles with distinct inputs, so overhead and merging do not dominate.
fp16: FP16 operands. a8a8: both operands INT8 (quantize -> dequantize for the FP16 side, INT8 runtime codes for the
cache side) and the output requantized, scales chosen so data is dense and does not clip.

    <coreai venv>/bin/python scripts/m6_int8_attn_probe.py --out DIR [--rows 48,384] [--tile 2048,4096] [--tiles 16]"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "scripts"))
import ane_compile_mode  # noqa: E402
ane_compile_mode.apply(log=lambda m: None)  # the SoC's bonded compile mode (M6: 2, M5: 1) unless set explicitly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai"))
sys.path.insert(0, str(ROOT / "coreai" / "swift_bridge"))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import coreai_bridge as B  # noqa: E402
import inspect_coreai_cache as IC  # noqa: E402
import m6_entry_sweep as S  # noqa: E402

f16, H, D = torch.float16, 4, 256


def qdq(x, unit, zero):
    q = torch.ops.coreai.quantize(x, unit, torch.int8, zero_point=zero)
    return torch.ops.coreai.dequantize(q, unit, zero_point=zero, output_dtype=f16)


class Tiles(nn.Module):
    def __init__(self, op: str, variant: str, m: int, tile: int, tiles: int):
        super().__init__()
        import coreai_torch._compression.custom_layers  # noqa: F401
        self.op, self.variant, self.m, self.tile, self.tiles = op, variant, m, tile, tiles
        # dense non-clipping scales: activations ~N(0,1) at 1/16; cache codes uniform in [-127, 127] scaled so a
        # product keeps unit variance; outputs ~N(0,1) at 1/16
        inner = D if op == "qk" else tile
        self.register_buffer("unit", torch.tensor(1 / 16, dtype=f16))
        self.register_buffer("cunit", torch.tensor(1 / (inner ** 0.5 * 73.6), dtype=f16))
        self.register_buffer("zero", torch.tensor(0, dtype=torch.int8))

    def input_names(self):
        return [f"a{i}" for i in range(self.tiles)] + [f"c{i}" for i in range(self.tiles)]

    def output_names(self):
        return [f"y{i}" for i in range(self.tiles)]

    def forward(self, *xs):
        a, c, outs = xs[:self.tiles], xs[self.tiles:], []
        for i in range(self.tiles):
            if self.variant == "fp16":
                x, w = a[i], c[i]
            else:
                x = qdq(a[i], self.unit, self.zero)
                w = torch.ops.coreai.dequantize(c[i], self.cunit, zero_point=self.zero, output_dtype=f16)
            y = x @ (w.transpose(1, 2) if self.op == "qk" else w)
            outs.append(qdq(y, self.unit, self.zero) if self.variant == "a8a8" else y)
        return tuple(outs)

    def shapes(self):
        if self.op == "qk":  # Q [H, M, D], K [H, L, D] (cache layout), scores [H, M, L]
            return (H, self.m, D), (H, self.tile, D)
        return (H, self.m, self.tile), (H, self.tile, D)  # P [H, M, L], V [H, L, D]

    def example(self):
        sa, sc = self.shapes()
        cd = f16 if self.variant == "fp16" else torch.int8
        a = [torch.randn(sa).to(f16) for _ in range(self.tiles)]
        c = [(torch.randn(sc).to(f16) if cd == f16 else torch.randint(-127, 128, sc, dtype=torch.int8))
             for _ in range(self.tiles)]
        return tuple(a + c)


def build(mod, dst: Path):
    import coreai_torch
    from coreai_opt.casting import cast_to_16_bit_precision
    ep = torch.export.export(mod, mod.example(), strict=False).run_decompositions(coreai_torch.get_decomp_table())
    cast_to_16_bit_precision(ep)
    conv = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    conv.add_exported_program(ep, input_names=mod.input_names(), output_names=mod.output_names(), entrypoint_name="main")
    prog = conv.to_coreai()
    prog.optimize()
    prog.save_asset(dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", default="48,384", help="query rows per KV head group: 48 = 8-row verify, 384 = 64-row prefill")
    ap.add_argument("--tile", default="2048,4096")
    ap.add_argument("--tiles", type=int, default=16)
    ap.add_argument("--iters", type=int, default=30)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    build_id = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()
    rows = []
    for op in ("qk", "pv"):
        for m in [int(x) for x in a.rows.split(",")]:
            for tile in [int(x) for x in a.tile.split(",")]:
                res = {}
                for variant in ("fp16", "a8a8"):
                    mod = Tiles(op, variant, m, tile, a.tiles).eval()
                    dst = a.out / f"{op}_{variant}_m{m}_l{tile}_t{a.tiles}.aimodel"
                    if not dst.exists():
                        build(mod, dst)
                    model = B.Model(dst, compute="ane")
                    place = IC.inspect_package(dst, model.function_names, Path.home() / "Library/Caches/coreai-cache",
                                               build_id, sys.executable)["status"]
                    fn = model.function("main")
                    rng = np.random.default_rng(0)
                    ins = {}
                    for n in fn.input_names:  # dense data, no zeros
                        b = fn.buffer("input", n)
                        if b.np.dtype == np.int8:
                            b.np[...] = rng.integers(1, 128, b.np.shape, dtype=np.int8) * rng.choice(np.array([-1, 1], np.int8), b.np.shape)
                        else:
                            b.np[...] = rng.standard_normal(b.np.shape).astype(np.float16)
                        ins[n] = b
                    outs = {o: fn.buffer("output", o) for o in fn.output_names}
                    plan = B.Plan([fn.bind(ins, outs)])
                    plan.run()
                    ms = S.timed(plan, a.iters, 3)["median_ms"]
                    tops = 2 * a.tiles * H * m * D * tile / (ms / 1e3) / 1e12
                    res[variant] = (ms, tops, place)
                    rows.append({"op": op, "rows": m, "tile": tile, "tiles": a.tiles, "variant": variant,
                                 "placement": place, "median_ms": ms, "tops": tops})
                f, q = res["fp16"], res["a8a8"]
                print(f"{op} rows {m:3} tile {tile}: fp16 {f[0]:6.3f} ms {f[1]:5.1f} TOPS [{f[2]}] | int8 {q[0]:6.3f} ms "
                      f"{q[1]:5.1f} TOPS [{q[2]}] | int8/fp16 time {q[0] / f[0]:.2f}", flush=True)
    (a.out / "results.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
