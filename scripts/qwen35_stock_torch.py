#!/usr/bin/env python3
"""HF torch fp32 reference for stock Qwen3.5-0.8B. CPU only.

Writes prompt ids, per-position argmax, final logits, last hidden states, and
zero-shot Snake scores. The Core AI script reads this file; it does not import transformers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai"))

from qwen35_stock import load_jsonl, parity_messages, snake_messages

OPTION_WORDS = ("up", "down", "left", "right")


def chat_ids(tokenizer, messages: list[dict]) -> list[int]:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def option_token_ids(tokenizer) -> dict[str, int]:
    ids = {}
    for word in OPTION_WORDS:
        for surface in (word, " " + word):
            enc = tokenizer.encode(surface, add_special_tokens=False)
            if len(enc) != 1:
                raise ValueError(f"{surface!r} is not one token: {enc}")
            ids[surface] = int(enc[0])
    return ids


def forward_row(model, input_ids: list[int], device: torch.device):
    """Return final logits, per-position argmax, pre-norm last row, post-norm last row."""
    captured: dict[str, torch.Tensor] = {}

    def pre_hook(_module, inputs):
        captured["pre"] = inputs[0].detach()

    def post_hook(_module, _inputs, output):
        captured["post"] = output.detach()

    norm = model.model.language_model.norm
    pre_handle = norm.register_forward_pre_hook(pre_hook)
    post_handle = norm.register_forward_hook(post_hook)
    try:
        tokens = torch.tensor([input_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            out = model(tokens, use_cache=False, logits_to_keep=len(input_ids))
    finally:
        pre_handle.remove()
        post_handle.remove()
    logits = out.logits[0].detach().float().cpu().numpy()
    pre = captured["pre"][0, -1].float().cpu().numpy()
    post = captured["post"][0, -1].float().cpu().numpy()
    return logits, logits.argmax(axis=-1).astype(np.int32), pre, post


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("/Users/anemll/Models/qwen35-0.8b-stock"))
    p.add_argument("--snake", type=Path, default=Path("/Users/anemll/Models/jeff-snake-data/heldout.jsonl"))
    p.add_argument("--out", type=Path, default=Path("/Users/anemll/Models/qwen35-0.8b-stock-coreai/eval"))
    p.add_argument("--threads", type=int, default=int(os.environ.get("TORCH_NUM_THREADS", "4")))
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device("cpu")
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    print("loading fp32 on CPU", flush=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(args.model), dtype=torch.float32, low_cpu_mem_usage=True)
    model.to(device)
    model.eval()
    ids_map = option_token_ids(tokenizer)
    prompts = []
    final_logits = []
    argmaxes = []
    pre_norm = []
    post_norm = []
    for spec in parity_messages():
        ids = chat_ids(tokenizer, spec["messages"])
        print(f"torch prompt {spec['name']} tokens={len(ids)}", flush=True)
        logits, argmax, pre, post = forward_row(model, ids, device)
        prompts.append({"name": spec["name"], "input_ids": ids, "n": len(ids)})
        final_logits.append(logits[-1])
        argmaxes.append(argmax)
        pre_norm.append(pre)
        post_norm.append(post)
    snake_rows = []
    correct = 0
    rows = load_jsonl(args.snake)
    bare_index = [ids_map[w] for w in OPTION_WORDS]
    for i, row in enumerate(rows):
        ids = chat_ids(tokenizer, snake_messages(row))
        if i == 0 or (i + 1) % 32 == 0:
            print(f"torch snake {i + 1}/{len(rows)} tokens={len(ids)}", flush=True)
        logits, _argmax, _pre, _post = forward_row(model, ids, device)
        last = logits[-1]
        bare = {w: float(last[ids_map[w]]) for w in OPTION_WORDS}
        spaced = {w: float(last[ids_map[" " + w]]) for w in OPTION_WORDS}
        pick = OPTION_WORDS[int(np.argmax([bare[w] for w in OPTION_WORDS]))]
        pick_spaced = OPTION_WORDS[int(np.argmax([spaced[w] for w in OPTION_WORDS]))]
        label = row["label"]
        correct += int(pick == label)
        snake_rows.append({
            "i": i,
            "label": label,
            "pred": pick,
            "pred_spaced": pick_spaced,
            "input_ids": ids,
            "bare_logits": bare,
            "spaced_logits": spaced,
        })
    meta = {
        "model": str(args.model),
        "device": "cpu",
        "dtype": "float32",
        "threads": args.threads,
        "option_ids": ids_map,
        "bare_index": bare_index,
        "prompts": prompts,
        "snake_accuracy": correct / len(rows),
        "snake_correct": correct,
        "snake_n": len(rows),
        "eos_token_id": int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None,
    }
    (out / "torch_meta.json").write_text(json.dumps({**meta, "snake": snake_rows}))
    # The meta file with every token id is large but still text. Arrays stay in npy.
    np.save(out / "torch_final_logits.npy", np.stack(final_logits).astype(np.float32))
    np.save(out / "torch_argmax.npy", np.array(argmaxes, dtype=object), allow_pickle=True)
    np.save(out / "torch_pre_norm.npy", np.stack(pre_norm).astype(np.float32))
    np.save(out / "torch_post_norm.npy", np.stack(post_norm).astype(np.float32))
    print(json.dumps({"snake_accuracy": meta["snake_accuracy"], "prompts": [p["name"] for p in prompts]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
