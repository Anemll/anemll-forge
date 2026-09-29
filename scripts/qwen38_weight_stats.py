"""Weight-only analysis of Qwen3.8-27B linear layers for ANE LUT / VQ formats (no activations needed).

Per matrix: RMS, max|w|/RMS, excess kurtosis, spread (coefficient of variation) of per-output-channel
and per-input-channel RMS, and the weight SNR (dB) of each format with round-to-nearest, optionally after
a Bonsai-style block Hadamard on the input axis (block 1024, random signs; applied to activations at
runtime). Appends one JSON line per matrix to OUT and skips matrices already there.

    MODEL=/path/to/data/Qwen3.8-27B OUT=/path/to/data/vq27b/weight_stats.jsonl \
    LAYERS=0,1,2,31,62,63 python qwen38_weight_stats.py
"""
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from scipy.linalg import hadamard

from qwen3_lut_common import FORMATS, make_rounder, snr

MODEL = Path(os.environ.get("MODEL", "/path/to/data/Qwen3.8-27B"))
OUT = Path(os.environ.get("OUT", "/path/to/data/vq27b/weight_stats.jsonl"))
PATTERN = re.compile(os.environ.get(
    "PATTERN", r"^model\.language_model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight$"))
LAYERS = {int(x) for x in os.environ["LAYERS"].split(",")} if os.environ.get("LAYERS") else None
PLAIN = ["LUT4 per-group-8 (anemll)", "LUT4 per-tensor + pcs", "vector 2x16", "vector 2x16 + pcs",
         "vector 4x64 + pcs", "vector 4x16 + pcs", "ternary + pcs"]
ROTATED = ["LUT4 per-tensor + pcs", "vector 2x16 + pcs", "ternary + pcs"]


def block_hadamard(w, block=1024, seed=0):
    """W R^T with R = blockdiag(H diag(signs)) / sqrt(block) on the input axis (orthogonal)."""
    cout, cin = w.shape
    h = torch.tensor(hadamard(block) / np.sqrt(block), dtype=torch.float32)
    signs = torch.tensor(np.random.default_rng(seed).choice([-1.0, 1.0], cin), dtype=torch.float32)
    return ((w * signs).view(cout, cin // block, block) @ h).reshape(cout, cin)


def stats(w):
    rms = w.pow(2).mean().sqrt()
    row, col = w.pow(2).mean(1).sqrt(), w.pow(2).mean(0).sqrt()
    return {"rms": rms.item(), "max_over_rms": (w.abs().max() / rms).item(),
            "kurtosis": (w.pow(4).mean() / w.pow(2).mean().pow(2) - 3).item(),
            "row_cv": (row.std() / row.mean()).item(), "col_cv": (col.std() / col.mean()).item()}


def main():
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    names = []
    for name, shard in index.items():
        m = PATTERN.match(name)
        if m and (LAYERS is None or int(m.group(1)) in LAYERS):
            names.append((int(m.group(1)), name, shard))
    done = set()
    if OUT.exists():
        done = {json.loads(line)["name"] for line in OUT.read_text().splitlines() if line.strip()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for layer, name, shard in sorted(names):
        if name in done:
            continue
        t = time.time()
        with safe_open(MODEL / shard, framework="pt") as f:
            w = f.get_tensor(name).float()
        rec = {"name": name, "layer": layer, "kind": name.split(".")[-2], "shape": list(w.shape), **stats(w)}
        for fmt in PLAIN:
            rec[fmt] = snr(w, make_rounder(w, FORMATS[fmt][1])(w))
        wr = block_hadamard(w)
        rec["rotated"] = stats(wr)
        for fmt in ROTATED:
            rec[f"had1024 {fmt}"] = snr(wr, make_rounder(wr, FORMATS[fmt][1])(wr))
        with OUT.open("a") as fo:
            fo.write(json.dumps(rec) + "\n")
        print(f"L{layer:02d} {rec['kind']:9s} kurt {rec['kurtosis']:6.2f} max/rms {rec['max_over_rms']:6.1f} "
              f"rowCV {rec['row_cv']:.2f} | " + " ".join(f"{rec[k]:5.2f}" for k in PLAIN) + " | had " +
              " ".join(f"{rec['had1024 ' + k]:5.2f}" for k in ROTATED) + f"  ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
