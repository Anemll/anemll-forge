#!/usr/bin/env python3
"""Write RESULTS.md from the torch and ANE eval files."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval", type=Path, default=Path("/Users/anemll/Models/qwen35-0.8b-stock-coreai/eval"))
    p.add_argument("--model", type=Path, default=Path("/Users/anemll/Models/qwen35-0.8b-stock"))
    p.add_argument("--out", type=Path, default=Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-qwen-stock/RESULTS.md"))
    args = p.parse_args(argv)
    ev = args.eval.expanduser().resolve()
    ane = json.loads((ev / "ane_report.json").read_text())
    meta = json.loads((ev / "torch_meta.json").read_text())
    place = json.loads((ev / "placement.json").read_text())
    tok = AutoTokenizer.from_pretrained(str(args.model))
    gen = tok.decode(ane["generation"]["new_token_ids"], skip_special_tokens=False)
    timing = ane["timing"]
    snake = ane["snake"]
    snake_rows = meta["snake"]
    torch_spaced = sum(r["pred_spaced"] == r["label"] for r in snake_rows) / len(snake_rows)
    bare_counts = Counter(r["pred"] for r in snake_rows)
    lines = [
        "# Stock Qwen3.5-0.8B on the Apple Neural Engine",
        "",
        "Baseline for the Colab Unsloth run: the stock decoder and tied LM head, not Jeff's 255-way readout.",
        "",
        "## Source",
        "",
        "- Weights: `Qwen/Qwen3.5-0.8B` revision `2fc06364715b967f1860aea9cf38778875588b17` (the revision in Jeff v1.3 `base_model`).",
        "- Unsloth `FastDecisionModel` example loads `unsloth/Qwen3.5-*`. For this size that repo is `unsloth/Qwen3.5-0.8B` revision `23c69c53358a07516b5827588b3fdb12ae78fd65`.",
        "- Safetensors SHA-256 of both repos: `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` (1,746,942,600 bytes). The weight files match. Chat-template text can still differ; this run uses the Qwen snapshot.",
        "- Local copy: `/Users/anemll/Models/qwen35-0.8b-stock` (hardlink of the existing snapshot). Build: `/Users/anemll/Models/qwen35-0.8b-stock-coreai`.",
        "",
        "## Build",
        "",
        "Same backbone as `jeff-coreai`: 6 chunks of 4 layers, FP16, prefill entry `p256_2k`, context 2048, `SILU=tanh`, `GDN_FAST=1`, `ATT_BLOCK=2048`, `ATT_BLOCK_PREFILL=4096`.",
        "Tied embeddings: no `lm_head` tensor. The head is final RMSNorm plus `embed_tokens`, split into 16 convs of 15,520 rows (248,320 / 16) so each output channel count stays under 16,384.",
        "Token embedding lookup stays on the host, as in the Jeff runtime. There is no separate verify/decode graph; a new token reuses `p256_2k`.",
        "Inference uses the Swift bridge. Each tensor is one IOSurface, bound once and reused. The Python Core AI runtime allocates a new IOSurface per call and runs out of surfaces on this eval.",
        "",
        "## Placement",
        "",
        "Every backbone chunk and every LM-head slice is `fully_ane`: one ANE region and zero GPU regions.",
        "",
        "| Package | Status | ANE regions | GPU regions |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in place:
        lines.append(f"| `{row['package']}` | {row['status']} | {row.get('ane_regions', 0)} | {row.get('gpu_regions', 0)} |")
    lines += [
        "",
        "## Parity vs Hugging Face torch fp32 (CPU)",
        "",
        "Per-position top-1 is argmax agreement of the full vocabulary at every prompt position. KL is KL(fp32 ‖ ANE) on the final position. Cosine is the last backbone row before the final RMSNorm.",
        "",
        "| Prompt | Tokens | Per-position top-1 | Final top-1 | KL(fp32 ‖ ANE) | Last-hidden cosine |",
        "| --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in ane["parity"]:
        top = "match" if row["final_top1_match"] else "differ"
        acc = row.get("per_position_top1")
        acc_s = f"{100.0 * acc:.1f}%" if isinstance(acc, float) else "n/a"
        lines.append(
            f"| {row['name']} | {row['tokens']} | {acc_s} | {top} | {row['kl_fp32_ane']:.3e} | {row['pre_norm_cosine']:.6f} |"
        )
    lines += [
        "",
        "## Latency",
        "",
        f"Load {ane['load_s']} s. Prompt `{timing['prompt']}` ({timing['tokens']} tokens).",
        "",
        "| Measurement | ms |",
        "| --- | ---: |",
        f"| Cold prefill (first call after load) | {timing['cold_prefill_ms']} |",
        f"| Cached prefill (median of 5) | {timing['cached_prefill_median_ms']} |",
        f"| Prefix hit (same prompt, no backbone calls) | {timing['prefix_hit_ms']} |",
        f"| Decode backbone, one new token on `p256_2k` (median of 8) | {timing['decode_backbone_median_ms']} |",
        f"| Decode LM head, 16 slices (median of 8) | {timing['decode_head_median_ms']} |",
        "",
        timing["decode_note"],
        "",
        "Cached prefill calls (ms): " + ", ".join(str(x) for x in timing["cached_prefill_ms"]) + ".",
        "",
        "## Greedy smoke",
        "",
        f"Prompt `{ane['generation']['prompt']}`, up to 24 new tokens, argmax, stop on eos.",
        "",
        "```",
        gen,
        "```",
        "",
        "## Zero-shot Snake",
        "",
        "256 rows from `/Users/anemll/Models/jeff-snake-data/heldout.jsonl`. Chat prompt contains the rules, the board, and the four options. The score is the LM-head logit of each option word.",
        "",
        "| Scorer | Accuracy |",
        "| --- | ---: |",
        f"| Torch fp32, bare tokens `up/down/left/right` | {fmt_pct(snake['torch_bare_accuracy'])} |",
        f"| Torch fp32, leading-space tokens | {fmt_pct(torch_spaced)} |",
        f"| ANE, bare tokens | {fmt_pct(snake['ane_bare_accuracy'])} |",
        f"| ANE, leading-space tokens | {fmt_pct(snake['ane_spaced_accuracy'])} |",
        f"| ANE bare vs torch bare (same prediction) | {fmt_pct(snake['ane_vs_torch_bare'])} |",
        "",
        f"Torch snake correct {meta['snake_correct']} / {meta['snake_n']}.",
        "Torch bare-token prediction counts: " + ", ".join(f"{k} {bare_counts[k]}" for k in ("up", "down", "left", "right")) + ".",
        "",
    ]
    args.out.write_text("\n".join(lines))
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
