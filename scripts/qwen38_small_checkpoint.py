"""Write the part of the original Qwen3.8-27B checkpoint that the Core AI builder reads besides the quantized export:
config / tokenizer files and the small BF16 tensors (per-layer norms, conv1d, A_log, dt_bias, in_proj_a / b, q / k
norms, and the final norm), unchanged, in one safetensors file with a model.safetensors.index.json. With it, MODEL can
point at this folder instead of the 52 GB checkpoint (coreai/qwen38_coreai_build.py with the quantized export).

    python scripts/qwen38_small_checkpoint.py --src ~/Models/Qwen3.8-27B --out <dir>"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

SMALL = ("input_layernorm.weight", "post_attention_layernorm.weight", "linear_attn.conv1d.weight",
         "linear_attn.A_log", "linear_attn.dt_bias", "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight",
         "linear_attn.norm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight")  # as qwen38_coreai_build
EXTRA = ("model.language_model.norm.weight",)
FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
         "merges.txt", "chat_template.jinja")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="original checkpoint (config, index, shards)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    wmap = json.loads((a.src / "model.safetensors.index.json").read_text())["weight_map"]
    keep = sorted(k for k in wmap if k in EXTRA or (k.startswith("model.language_model.layers.") and
                                                       k.split(".", 4)[4] in SMALL))
    tensors = {}
    for shard in sorted({wmap[k] for k in keep}):
        with safe_open(a.src / shard, framework="pt") as f:
            for k in keep:
                if wmap[k] == shard:
                    tensors[k] = f.get_tensor(k).contiguous()
    a.out.mkdir(parents=True, exist_ok=True)
    name = "small.safetensors"
    save_file(tensors, a.out / name, metadata={"source": "Qwen/Qwen3.8-27B", "content": "small tensors, unchanged BF16"})
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    (a.out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": {k: name for k in keep}}, indent=2))
    for f in FILES:
        if (a.src / f).exists():
            shutil.copy2(a.src / f, a.out / f)
    digest = hashlib.sha256((a.out / name).read_bytes()).hexdigest()
    print(f"{len(keep)} tensors, {total / 1e6:.1f} MB -> {a.out / name} (sha256 {digest[:16]})")


if __name__ == "__main__":
    main()
