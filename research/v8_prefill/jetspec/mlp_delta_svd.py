"""C3 (20 Nov 2026): can the JetSpec++ adapter's MLPs be stored as the target's layer + a low-rank delta? For each of the
12 blocks, its gate / up / down weights minus the target layer its FFN was initialized from (ffn_source target_init,
resolve_ffn_layer_map: 5, 19, 33, 37, 40, 44, 47, 50, 53, 56, 59, 61; bf16 checkpoint), singular values of the delta
(CPU, float32). Reports |delta| / |W|, the rank for 50 / 90 / 99% of the delta's energy, and the error of
W ~ W_target + rank-r(delta) relative to |W| for r = 64 ... 2048.
    PY mlp_delta_svd.py OUT.json"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ADAPTER = "/Volumes/SSD4TB/anemll-forge-research/jetspec_pp/release/adapter/head.safetensors"
TARGET = Path("/Volumes/SSD4TB/Qwen3.8-27B")
BLOCKS = [("streams.0", 5), ("streams.1", 19), ("streams.2", 33), ("streams.3", 37), ("extra_stream_layers.3.0", 40),
          ("extra_stream_layers.3.1", 44), ("extra_stream_layers.3.2", 47), ("streams.4", 50), ("extra_stream_layers.4.0", 53),
          ("extra_stream_layers.4.1", 56), ("extra_stream_layers.4.2", 59), ("extra_stream_layers.4.3", 61)]
RANKS = [64, 128, 256, 512, 1024, 2048]
torch.set_num_threads(8)
idx = json.loads((TARGET / "model.safetensors.index.json").read_text())["weight_map"]
fa = safe_open(ADAPTER, "pt")
out, res = Path(sys.argv[1]), []
t0 = time.time()
for name, L in BLOCKS:
    for proj in ("gate_proj", "up_proj", "down_proj"):
        tk = f"model.language_model.layers.{L}.mlp.{proj}.weight"
        with safe_open(str(TARGET / idx[tk]), "pt") as ft:
            wt = ft.get_tensor(tk).float()
        w = fa.get_tensor(f"{name}.mlp.{proj}.weight").float()
        d = w - wt
        s = torch.linalg.svdvals(d)
        e = torch.cumsum(s ** 2, 0); tot = float(e[-1])
        rank_at = {p: int(torch.searchsorted(e, p * tot)) + 1 for p in (0.5, 0.9, 0.99)}
        nw = float(w.norm())
        err = {r: float(torch.sqrt(torch.clamp(e[-1] - e[r - 1], min=0))) / nw for r in RANKS}
        row = {"block": name, "target_layer": L, "proj": proj, "shape": list(w.shape), "delta_rel": float(d.norm()) / nw,
               "rank_50_90_99": [rank_at[0.5], rank_at[0.9], rank_at[0.99]], "err_rel_at_rank": err}
        res.append(row)
        print(json.dumps(row), flush=True)
        out.write_text(json.dumps({"seconds": round(time.time() - t0), "rows": res}, indent=1))
