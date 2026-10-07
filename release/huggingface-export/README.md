---
base_model: Qwen/Qwen3.8-27B
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
  - speculative-decoding
  - dflash2
---

# ANEMLL · Qwen3.8-27B, quantized for Core AI export

**The quantized weights behind [ANEMLL Forge](https://github.com/Anemll/anemll-forge)'s Qwen3.8-27B for the Apple Neural Engine.** This repository holds the quantized target export `mix25in_mixr_lr64mix`, the quantized export of its paired DFlash2 speculative drafter, and the small parts of both original checkpoints the converter needs, so the published Core AI packages ([anemll/anemll-forge-qwen3.8-27B](https://huggingface.co/anemll/anemll-forge-qwen3.8-27B)), target and drafter, can be rebuilt, checked or varied without the original checkpoints (52 GB and 3.6 GB). It is a research artifact for conversion: the files hold lookup-table indices, codebooks and scales in ANEMLL's own layout, not a checkpoint that transformers or other runtimes can load directly.

To run the model, use the prepared Core AI bundle: [anemll/anemll-forge-qwen3.8-27B](https://huggingface.co/anemll/anemll-forge-qwen3.8-27B) and the [ANEMLL Forge README](https://github.com/Anemll/anemll-forge). To rebuild those packages from this repository, follow [EXPORT.md](EXPORT.md).

## Contents

| Path | What it is |
| --- | --- |
| `export/mix25in_mixr_lr64mix/layer_NN.safetensors` (64) | each layer's MLP (gate, up, down): LUT indices, codebooks, per-channel scales, online-rotation seeds in the header |
| `export/mix25in_mixr_lr64mix/layer_NN_mixer.safetensors` (64) | each layer's token mixer: Gated DeltaNet projections (48 layers) or full-attention q / k / v / o (layers 3, 7, ..., 63), with rank-64 FP16 residual corrections |
| `export/mix25in_mixr_lr64mix/lm_head.safetensors` | the output head (LUT4, per-output-channel scales) |
| `model/` | the original `config.json`, tokenizer and chat template, and `small.safetensors`: the 449 small BF16 tensors the converter also reads (norms, DeltaNet conv1d, `A_log`, `dt_bias`, `in_proj_a` / `in_proj_b`, q / k norms, final norm) and the embedding row of the drafter's mask token, copied unchanged from the original checkpoint, with a `model.safetensors.index.json` |
| `drafter/drafter_quant.safetensors` | the DFlash2 drafter's 5 layers quantized by ANEMLL (LUT4 GPTQ, `q7_cal` calibration) |
| `drafter/small.safetensors`, `config.json`, `selector.safetensors` | the drafter checkpoint's small tensors (norms, convolution base kernels, selector hidden projection), its configuration and its two candidate-selector codebooks, copied unchanged from the upstream drafter |
| `drafter/LICENSE`, `drafter/NOTICE`, `drafter/DFLASH2_SOURCE.json` | the upstream drafter's license and notice, unchanged, and the pinned drafter provenance |
| `weights_digest.json`, `drafter/weights_digest.json` | SHA-256 of all 2,097 target and 165 drafter weight arrays the converter builds from this repository, to verify a download ([EXPORT.md](EXPORT.md)) |
| `inventory.json` | size and SHA-256 of every file |
| `config.json` | a copy of `model/config.json` at the root, for Hub discovery and download counting |
| `LICENSE`, `NOTICE`, `MODIFICATIONS.md`, `QWEN_SOURCE.json` | the upstream license, attribution, modification notice and the pinned upstream source |

## Quantization recipe

Mixed precision with GPTQ, per weight matrix ([recipe and math](https://github.com/Anemll/anemll-forge/blob/main/docs/QUANTIZATION.md)):

- **MLP:** `vector 2x16 + pcs` (vector lookup table: pairs of weights, 16 codebook entries, about 2 bits per weight, per-channel scaling) in 38 layers and `LUT4 per-tensor + pcs` in 26 layers, with online Hadamard rotations of the MLP basis.
- **Token mixers:** vector LUTs in layers 0 to 23 and scalar LUT4 in layers 24 to 63; full-attention K / V projection weights use per-channel INT8. Rank-64 FP16 residual factors correct the DeltaNet and attention projections.
- **Output head:** scalar LUT4 with per-output-channel scaling. Embeddings stay FP16 and are looked up on the host (the inference bundle's `model/embed_tokens_fp16.npy`).

"2-bit model" is an incomplete description: activations, recurrent state and the KV cache stay in floating point (the Core AI graph adds an INT8 value cache and 8-bit attention arithmetic of its own).

**Drafter.** The DFlash2 speculative drafter (5 layers, a 2,048-token sliding window) proposes 7 tokens per cycle for the target to verify. It reads the target's hidden features at layers 5, 19, 33, 47 and 61 and uses this export's LM head, so it pairs with this target only. Its linear weights are LUT4 GPTQ with `q7_cal` calibration; the published Core AI drafter scales the mask-token row by 0.7.

## Builds to try

The same weights give several Core AI builds; [EXPORT.md](EXPORT.md) has the commands. The 8-bit attention forms (INT8 scores, FP8 softmax and PV) and the KV-cache format apply to activations at run time, not to the weights: every build below uses these weights unchanged.

- **One package for M6 and M5** (the published build): M6 functions with INT8 scores and FP8 softmax and PV, M5 functions with FP16 softmax; an M5 derives its own build on first start.
- **Package without FP8, for M5 and M6:** INT8 scores with FP16 softmax and PV (`ATT_INT8MM=s8,s8b`, `KV_KEYS_T=1`). It compiles directly on M5 Macs (no first-start derivation, no second copy of the chunks; the M5 ANE compiler has no FP8) and runs on M6 too, where it trades the FP8 speedup for a base that combinations not yet validated with FP8 can start from (kv8, contexts above 64K). On an M5 Max, against the previous V8 packages: prefill up to +7.4% at 64K, decode +3.7 to +8.2%.
- **M6-only package with FP8:** the published M6 functions without the M5 set, so nothing extra to compile on an M6.
- **Longer context ladders on Macs with more memory:** `--ctx` / `--pctx` take any list of entries. Entries above 64K are research so far (an 80K-only package used 25.7 GB of wired memory on a 32 GB M6).
- **INT8 keys and values** (`--kv-cache-dtype kv8`): about half the cache of FP16. Not yet validated together with the 8-bit attention forms and the transposed key cache; run the long-context evals first.
- **The plain V8 graph** as a baseline for comparisons, and **the drafter** from its own export.

Results from other Macs (chip, memory, context, prefill and decode tok/s) are welcome in this repository's Community tab.

## Quality

Teacher-forced KL divergence against the BF16 model, on a short trace (64 sequences, 40,023 positions), for the Core AI target built from this export: mean KL **0.184**, top-1 agreement **86.0%**, perplexity 2.414 against the teacher's 2.136. This is not a coding, reasoning, retrieval or long-context evaluation.

## Reproducibility

Converting this repository with the published settings gives the published packages' weights exactly: the converter consumes the same 2,097 target arrays and 165 drafter arrays from this repository as from the original checkpoints (`weights_digest.json`, `drafter/weights_digest.json`). The package bytes still differ slightly from build to build, because the Core AI converter's serialization is not byte-deterministic. [EXPORT.md](EXPORT.md) lists the environment, the commands and the checks.

## Attribution and licenses

**Target.** The original model is [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) at revision [`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0), developed by the **Qwen Team**; its weights carry **Copyright 2026 Alibaba Cloud** and the **Apache License 2.0**, preserved unchanged in [LICENSE](LICENSE). ANEMLL independently quantized these weights; see [MODIFICATIONS.md](MODIFICATIONS.md) and [NOTICE](NOTICE).

**Drafter.** The drafter derives from [ProCreations/Ternary-Bonsai-2-27B-DFlash2](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2) at revision [`4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b`](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2/tree/4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b) (Apache-2.0; its [LICENSE](drafter/LICENSE) and [NOTICE](drafter/NOTICE) are preserved unchanged). Its notice credits the donor [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) (revision `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`) and the training target [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) (revision `6ed5e12bf84b7a63069882c91dd9e9218647d17b`); [drafter/DFLASH2_SOURCE.json](drafter/DFLASH2_SOURCE.json) records the pinned identities. The full BF16 drafter checkpoint is not redistributed here; download it from ProCreations if you need it.

Qwen, Alibaba Cloud, ProCreations, z-lab, Prism ML and Apple are named for credit and context; no affiliation or endorsement is claimed.

**Why Apache-2.0:** these weights are a quantized derivative of Apache-2.0 models (Qwen3.8-27B and the ProCreations drafter), so they keep that license. Apache-2.0 asks a redistributed derivative to include the license, keep the upstream notices and mark what was changed; LICENSE, NOTICE and MODIFICATIONS.md do that here. The MIT license of the [ANEMLL Forge](https://github.com/Anemll/anemll-forge) source code that produces and converts the weights covers that code, not the weights. Retain LICENSE, NOTICE, MODIFICATIONS.md and the drafter's LICENSE and NOTICE when redistributing.
