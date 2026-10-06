---
base_model:
  - Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
pipeline_tag: text-generation
tags:
  - coreai
  - apple-neural-engine
  - ane
  - quantized
  - gptq
  - vector-quantization
  - lut
---

# ANEMLL · Qwen3.8-27B, quantized for Core AI export

**The quantized weights behind [ANEMLL Forge](https://github.com/Anemll/anemll-forge)'s Qwen3.8-27B for the Apple Neural Engine.** This repository holds the quantized export `mix25in_mixr_lr64mix` and the small part of the original checkpoint the converter needs, so the published Core AI packages ([anemll/anemll-forge-qwen3.8-27B](https://huggingface.co/anemll/anemll-forge-qwen3.8-27B)) can be rebuilt, checked or varied without the 52 GB original checkpoint. It is a research artifact for conversion: the files hold lookup-table indices, codebooks and scales in ANEMLL's own layout, not a checkpoint that transformers or other runtimes can load directly.

To run the model, use the prepared Core AI bundle: [anemll/anemll-forge-qwen3.8-27B](https://huggingface.co/anemll/anemll-forge-qwen3.8-27B) and the [ANEMLL Forge README](https://github.com/Anemll/anemll-forge). To rebuild those packages from this repository, follow [EXPORT.md](EXPORT.md).

## Contents

| Path | What it is |
| --- | --- |
| `export/mix25in_mixr_lr64mix/layer_NN.safetensors` (64) | each layer's MLP (gate, up, down): LUT indices, codebooks, per-channel scales, online-rotation seeds in the header |
| `export/mix25in_mixr_lr64mix/layer_NN_mixer.safetensors` (64) | each layer's token mixer: Gated DeltaNet projections (48 layers) or full-attention q / k / v / o (layers 3, 7, ..., 63), with rank-64 FP16 residual corrections |
| `export/mix25in_mixr_lr64mix/lm_head.safetensors` | the output head (LUT4, per-output-channel scales) |
| `model/` | the original `config.json`, tokenizer and chat template, and `small.safetensors`: the 449 small BF16 tensors the converter also reads (norms, DeltaNet conv1d, `A_log`, `dt_bias`, `in_proj_a` / `in_proj_b`, q / k norms, final norm), copied unchanged from the original checkpoint, with a `model.safetensors.index.json` |
| `weights_digest.json` | SHA-256 of all 2,097 weight arrays the converter builds from this repository, to verify a download ([EXPORT.md](EXPORT.md)) |
| `inventory.json` | size and SHA-256 of every file |
| `config.json` | a copy of `model/config.json` at the root, for Hub discovery and download counting |
| `LICENSE`, `NOTICE`, `MODIFICATIONS.md`, `QWEN_SOURCE.json` | the upstream license, attribution, modification notice and the pinned upstream source |

## Quantization recipe

Mixed precision with GPTQ, per weight matrix ([recipe and math](https://github.com/Anemll/anemll-forge/blob/main/docs/QUANTIZATION.md)):

- **MLP:** `vector 2x16 + pcs` (vector lookup table: pairs of weights, 16 codebook entries, about 2 bits per weight, per-channel scaling) in 38 layers and `LUT4 per-tensor + pcs` in 26 layers, with online Hadamard rotations of the MLP basis.
- **Token mixers:** vector LUTs in layers 0 to 23 and scalar LUT4 in layers 24 to 63; full-attention K / V projection weights use per-channel INT8. Rank-64 FP16 residual factors correct the DeltaNet and attention projections.
- **Output head:** scalar LUT4 with per-output-channel scaling. Embeddings stay FP16 and are looked up on the host (the inference bundle's `model/embed_tokens_fp16.npy`).

"2-bit model" is an incomplete description: activations, recurrent state and the KV cache stay in floating point (the Core AI graph adds an INT8 value cache and 8-bit attention arithmetic of its own).

## Quality

Teacher-forced KL divergence against the BF16 model, on a short trace (64 sequences, 40,023 positions), for the Core AI target built from this export: mean KL **0.184**, top-1 agreement **86.0%**, perplexity 2.414 against the teacher's 2.136. This is not a coding, reasoning, retrieval or long-context evaluation.

## Reproducibility

Converting this repository with the published settings gives the published packages' weights exactly: the converter consumes the same 2,097 arrays from this repository as from the original checkpoint (`weights_digest.json`). The package bytes still differ slightly from build to build, because the Core AI converter's serialization is not byte-deterministic. [EXPORT.md](EXPORT.md) lists the environment, the commands and the checks.

## Attribution and licenses

The original model is [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) at revision [`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0), developed by the **Qwen Team**; its weights carry **Copyright 2026 Alibaba Cloud** and the **Apache License 2.0**, preserved unchanged in [LICENSE](LICENSE). ANEMLL independently quantized these weights; see [MODIFICATIONS.md](MODIFICATIONS.md) and [NOTICE](NOTICE). Qwen, Alibaba Cloud and Apple are named for credit and context; no affiliation or endorsement is claimed.

The weights remain under Apache-2.0. The [ANEMLL Forge](https://github.com/Anemll/anemll-forge) source code that produces and converts them is MIT-licensed. Retain LICENSE, NOTICE and MODIFICATIONS.md when redistributing.
