"""Add closed-form low-rank error factors to an export: for every token-mixer matrix in PARTS (gdn = DeltaNet
in_proj_qkv / in_proj_z / out_proj, attn = q / k / v / o), the rank-LR_RANK SVD of W_bf16 - W_quant is stored as
`{key}.lr_a` (Cout, r) and `{key}.lr_b` (r, Cin), fp16, next to the LUT / INT8 tensors (qwen38_ane_model.as_quant
adds a @ (b @ x)). Same factors as qwen38_kl.py eval LR_RANK / LR_PARTS computes on the fly. Other files are symlinked.
    EXPORT_DIR=/path/to/data/vq27b/runs/export/mix25_aw_cal OUT_DIR=/path/to/data/vq27b/runs/export/mix25_aw_cal_lr64mix \
    MODEL=/path/to/data/Qwen3.8-27B LR_RANK=64 PARTS=gdn,attn python qwen38_lowrank_export.py"""
import json
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from qwen38_kl import dequant

SRC = Path(os.environ["EXPORT_DIR"])
DST = Path(os.environ["OUT_DIR"])
MODEL = Path(os.environ.get("MODEL", "/path/to/data/Qwen3.8-27B"))
RANK = int(os.environ.get("LR_RANK", "64"))
PARTS = set(os.environ.get("PARTS", "gdn,attn").split(","))
torch.set_grad_enabled(False)


def main():
    DST.mkdir(parents=True, exist_ok=True)
    wmap = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    t0, extra = time.time(), 0
    for f in sorted(SRC.iterdir()):
        out = DST / f.name
        if not f.name.endswith("_mixer.safetensors"):
            if not out.exists():
                out.symlink_to(f.resolve())
            continue
        i = int(f.name.split("_")[1])
        t = load_file(f)
        with safe_open(f, framework="pt") as fh:
            meta = fh.metadata()
        for key in sorted({k.rsplit(".", 1)[0] for k in t}):
            part = "attn" if key.startswith("self_attn") else "gdn"
            if part not in PARTS or f"{key}.lr_a" in t:
                continue
            name = f"model.language_model.layers.{i}.{key}.weight"
            with safe_open(MODEL / wmap[name], framework="pt") as fh:
                w = fh.get_tensor(name).float()
            e = w - dequant(t, key).float()
            u, s, v = torch.svd_lowrank(e, q=RANK + 16, niter=4)
            a, b = (u[:, :RANK] * s[:RANK]).half().contiguous(), v[:, :RANK].T.half().contiguous()
            rel = float((e - a.float() @ b.float()).norm() / e.norm())
            t[f"{key}.lr_a"], t[f"{key}.lr_b"] = a, b
            extra += (a.numel() + b.numel()) * 2
            print(f"layer {i:02d} {key}: {tuple(w.shape)} error left {rel:.3f}", flush=True)
        save_file(t, str(out), metadata=meta)
    print(f"done: {extra / 2**30:.2f} GiB of factors, rank {RANK}, parts {sorted(PARTS)} ({time.time() - t0:.0f}s) -> {DST}",
          flush=True)


if __name__ == "__main__":
    main()
