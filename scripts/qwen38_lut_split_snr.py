"""Weight SNR of LUT4 + per-channel scale for one per-tensor LUT vs separate LUTs per row block (DeltaNet in_proj_qkv
q / k / v) or per group of rows (grouped scalar LUTs, anemll per-group-8 at the extreme). bf16 checkpoint, no
calibration (RTN codebooks); tells whether splitting LUTs can pay off before touching the export / ANE builder.
    MODEL=~/Models/Qwen3.8-27B python qwen38_lut_split_snr.py"""
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from qwen3_lut_common import FORMATS, make_rounder

M = Path(os.path.expanduser(os.environ.get("MODEL", "~/Models/Qwen3.8-27B")))
WMAP = json.loads((M / "model.safetensors.index.json").read_text())["weight_map"]
SPEC = FORMATS["LUT4 per-tensor + pcs"][1]


def get(n):
    with safe_open(M / WMAP[n], framework="pt") as f:
        return f.get_tensor(n).float()


def snr(w, q):
    return 10 * np.log10(float((w * w).sum() / ((w - q) ** 2).sum()))


def main():
    for l in (0, 12, 30, 48, 62):  # q / k / v blocks of in_proj_qkv: 2048 / 2048 / 6144 rows
        w = get(f"model.language_model.layers.{l}.linear_attn.in_proj_qkv.weight")
        one = make_rounder(w, SPEC)(w)
        parts = [w[:2048], w[2048:4096], w[4096:]]
        three = torch.cat([make_rounder(p, SPEC)(p) for p in parts])
        print(f"L{l:02d} in_proj_qkv: one LUT {snr(w, one):.2f} dB | q/k/v LUTs {snr(w, three):.2f} dB", flush=True)
    for name in ("model.language_model.layers.30.linear_attn.in_proj_qkv.weight",
                 "model.language_model.layers.40.mlp.down_proj.weight"):
        w = get(name)
        out = []
        for g in (0, 2048, 512, 128, 8):  # rows per group (0 = per-tensor)
            grp = g or w.shape[0]
            if w.shape[0] % grp:
                continue
            q = make_rounder(w, ("group", grp, 4, "pcs"))(w)
            out.append(f"{'tensor' if not g else f'{w.shape[0] // grp} groups'} {snr(w, q):.2f}")
        print(name.split("layers.")[1], " | ".join(out), flush=True)


if __name__ == "__main__":
    main()
