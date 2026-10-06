# Modification notice

**ANEMLL has modified the weights of Qwen3.8-27B by quantizing them.** The files in `export/` are derivatives of [Qwen/Qwen3.8-27B at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0), whose weights carry Copyright 2026 Alibaba Cloud and the Apache-2.0 license preserved in [LICENSE](LICENSE).

## Modified files

- **`export/mix25in_mixr_lr64mix/layer_NN.safetensors`, `layer_NN_mixer.safetensors`, `lm_head.safetensors`:** the text model's linear weights replaced by quantized representations: GPTQ lookup-table indices and codebooks (vector 2x16 and scalar LUT4), per-channel scales, per-channel INT8 for the full-attention K / V projections, online Hadamard rotation seeds for the MLP basis, and rank-64 FP16 residual factors for the token-mixer projections. Recipe: [docs/QUANTIZATION.md](https://github.com/Anemll/anemll-forge/blob/main/docs/QUANTIZATION.md).

## Unmodified files

- **`model/small.safetensors`:** 449 small tensors (layer norms, DeltaNet conv1d, `A_log`, `dt_bias`, `in_proj_a` / `in_proj_b`, attention q / k norms, final norm) copied byte for byte in BF16 from the upstream checkpoint, with a new `model.safetensors.index.json` listing only them.
- **`model/` configuration, tokenizer and chat template, root `config.json`, `LICENSE`:** copies of the upstream files.

Vision weights, multi-token prediction and the embedding table are not included. Retain this notice, [NOTICE](NOTICE) and [LICENSE](LICENSE) when redistributing.
