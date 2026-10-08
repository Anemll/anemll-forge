#!/usr/bin/env python3
"""Export stock Qwen3.5-0.8B with the Jeff 6-chunk FP16 pipeline plus a tied LM head.

    COREAI_PYTHON scripts/qwen35_stock_convert.py \
        --model /Users/anemll/Models/qwen35-0.8b-stock \
        --output /Users/anemll/Models/qwen35-0.8b-stock-coreai

Backbone chunking matches jeff-coreai: 4 layers per chunk, fp16, prefill 256, context 2048.
The LM head is embed_tokens (tied), final RMSNorm included, vocab split into 16 convs.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai"))

from jeff_coreai_build import build_prefill_chunk, load_builder, save_dense_program
from qwen35_stock import CHUNK_LAYERS, CTX, LM_HEAD_PARTS, PREFILL_ROWS, StockCheckpoint, vocab_slices


def make_slice(builder, embed: np.ndarray, norm: np.ndarray, start: int, end: int):
    hid = embed.shape[1]
    rows = embed[start:end]

    class Slice(nn.Module):
        def __init__(self):
            super().__init__()
            self.T = 1
            weight = (1.0 + np.asarray(norm, np.float32)).astype(np.float16).reshape(1, -1, 1, 1)
            self.register_buffer("normw", torch.from_numpy(np.ascontiguousarray(weight)))
            self.proj = nn.Conv2d(hid, rows.shape[0], 1, bias=False)
            self.proj.weight = nn.Parameter(
                torch.from_numpy(np.ascontiguousarray(rows)).view(rows.shape[0], hid, 1, 1),
                requires_grad=False,
            )

        def forward(self, x):
            h = builder.rms_hidden(x, self.normw)
            return self.proj(h).reshape(rows.shape[0], self.T).transpose(0, 1)

        def example(self):
            return (torch.zeros(1, hid, 1, self.T, dtype=torch.float16),)

    return Slice().eval().to(torch.float16)


def export_stock(model: Path, out_dir: Path) -> dict:
    # These are read when the 27B builder module is imported. Match jeff-coreai (fp16, p256).
    os.environ["TPS"] = "256"
    os.environ["KV_CACHE_DTYPE"] = "fp16"
    os.environ["SILU"] = "tanh"
    os.environ["MLP_SILU"] = "tanh"
    os.environ["GDN_FAST"] = "1"
    os.environ["ATT_BLOCK"] = "2048"
    os.environ["ATT_BLOCK_PREFILL"] = "4096"
    os.environ["MODEL"] = str(model)
    ck = StockCheckpoint(model)
    cfg = ck.cfg
    vocab = int(cfg["vocab_size"])
    spans = vocab_slices(vocab, LM_HEAD_PARTS)
    builder = load_builder(cfg)
    builder.TPS = [PREFILL_ROWS]
    coreai = out_dir / "coreai"
    model_dir = out_dir / "model"
    if coreai.exists():
        shutil.rmtree(coreai)
    coreai.mkdir(parents=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    ck.write_embedding(model_dir / "embed_tokens_fp16.npy")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "vocab.json", "merges.txt"):
        src = ck.model / name
        if src.is_file():
            shutil.copy2(src, model_dir / name)
    n_layers = int(cfg["num_hidden_layers"])
    plan = [list(range(a, min(a + CHUNK_LAYERS, n_layers))) for a in range(0, n_layers, CHUNK_LAYERS)]
    man = {
        "version": "qwen35-stock1",
        "kind": "qwen35-stock",
        "base_model": "Qwen/Qwen3.5-0.8B",
        "base_revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "T": 8,
        "TP": PREFILL_ROWS,
        "pend": builder.P,
        "taps": [],
        "ctxs": [CTX],
        "pctxs": [CTX],
        "kv_len": {str(CTX): builder.kv_len(CTX, 8)},
        "pkv_len": {str(CTX): builder.kv_len(CTX, PREFILL_ROWS)},
        "kv_cache": {"format": "fp16", "keys": "float16", "values": "float16", "scales": None},
        "quant": "fp16",
        "dflash2": False,
        "tie_word_embeddings": True,
        "chunks": [],
        "head": {},
        "numerics": {"SILU": builder.SILU, "MLP_SILU": builder.MLP_SILU, "GDN_FAST": builder.GDN_FAST,
                     "ATT_BLOCK": builder.ATT_BLOCK, "ATT_BLOCK_PREFILL": builder.ATT_BLOCK_PREFILL},
    }
    man_path = coreai / "manifest.json"
    chunks = []
    for layers in plan:
        dest = coreai / f"chunk_L{layers[0]:02d}-{layers[-1]:02d}.aimodel"
        print(f"export {dest.name} layers {layers[0]}-{layers[-1]}", flush=True)
        info = build_prefill_chunk(builder, ck, layers, CTX, PREFILL_ROWS, "fp16", dest)
        chunks.append(info)
        man["chunks"] = chunks
        man_path.write_text(json.dumps(man, indent=1))
        gc.collect()
    embed = np.asarray(ck.embed_table(), np.float16)
    norm = np.asarray(ck.norm_weight(), np.float32)
    ck._weights.clear()
    gc.collect()
    slices = []
    for i, (start, end) in enumerate(spans):
        dest = coreai / f"head_lm_{i:02d}.aimodel"
        print(f"export {dest.name} rows {start}:{end}", flush=True)
        head = make_slice(builder, embed, norm, start, end)
        mb = save_dense_program(builder, [("h1", head, ["x"], ["logits"])], dest)
        del head
        gc.collect()
        slices.append({"file": dest.name, "entry": "h1", "start": start, "end": end,
                       "rows": end - start, "mb": round(mb, 1)})
    man["head"] = {
        "kind": "tied-lm-head",
        "tied_to": "model.language_model.embed_tokens.weight",
        "file": slices[0]["file"],
        "entry": "h1",
        "entries": ["h1"],
        "vocab": vocab,
        "parts": len(slices),
        "rows_per_slice": spans[0][1] - spans[0][0],
        "slices": slices,
        "mb": round(sum(s["mb"] for s in slices), 1),
    }
    man_path.write_text(json.dumps(man, indent=1))
    summary = {
        "model": str(model),
        "output": str(out_dir),
        "layers": n_layers,
        "chunks": [c["file"] for c in chunks],
        "head_slices": len(slices),
        "vocab": vocab,
        "prefill": PREFILL_ROWS,
        "ctx": CTX,
        "quant": "fp16",
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=Path("/Users/anemll/Models/qwen35-0.8b-stock"))
    p.add_argument("--output", type=Path, default=Path("/Users/anemll/Models/qwen35-0.8b-stock-coreai"))
    args = p.parse_args(argv)
    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    export_stock(args.model.expanduser().resolve(), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
