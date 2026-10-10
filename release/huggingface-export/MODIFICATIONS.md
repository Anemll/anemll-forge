# Modification notice

**ANEMLL has modified the weights of Qwen3.8-27B and of the paired ProCreations DFlash2 drafter by quantizing them.** The files in `export/` are derivatives of [Qwen/Qwen3.8-27B at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0), whose weights carry Copyright 2026 Alibaba Cloud and the Apache-2.0 license preserved in [LICENSE](LICENSE).

## Modified files

- **`export/release_vq3pA_mixh_s600_k1_mat/` (release 0.2) and `export/mix25in_mixr_lr64mix/` (first release), each with `layer_NN.safetensors`, `layer_NN_mixer.safetensors`, `lm_head.safetensors`:** the text model's linear weights replaced by quantized representations: GPTQ lookup-table indices and codebooks (vector 2x64, vector 2x16 and scalar LUT4), per-channel scales, per-channel INT8 for the full-attention K / V projections, online Hadamard rotation seeds for the MLP basis and, in release 0.2, for the token-mixer projections (weights stored in the rotated basis), and rank-64 FP16 residual factors for the token-mixer projections. Recipe: [docs/QUANTIZATION.md](https://github.com/Anemll/anemll-forge/blob/main/docs/QUANTIZATION.md). Release 0.2 stores every MLP matrix as `vector 2x64 + pcs` (three bits per weight) and is fitted by GPTQ with the residual factors inside the sequential pass, then by quantization-aware training (QAT) against the BF16 model that trains the lookup-table values, scales, factors and LM head with every index frozen.

- **`drafter/drafter_quant.safetensors`:** the linear weights of [ProCreations/Ternary-Bonsai-2-27B-DFlash2 at `4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b`](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2/tree/4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b) (Apache-2.0, [drafter/LICENSE](drafter/LICENSE)) replaced by LUT4 GPTQ indices, codebooks and scales, calibrated on the `q7_cal` traces.

## Unmodified files

- **`model/small.safetensors`:** 449 small tensors (layer norms, DeltaNet conv1d, `A_log`, `dt_bias`, `in_proj_a` / `in_proj_b`, attention q / k norms, final norm) and the embedding row of the drafter's mask token (`dflash.mask_token_embedding`), copied byte for byte in BF16 from the upstream checkpoint, with a new `model.safetensors.index.json` listing only them.
- **`drafter/small.safetensors`:** the drafter checkpoint's 33 small tensors (norms, convolution base kernels, candidate-selector hidden projection) copied byte for byte. **`drafter/config.json`, `drafter/LICENSE`, `drafter/NOTICE`:** unchanged upstream files. **`drafter/selector.safetensors`:** the two candidate-selector codebook tables extracted with their BF16 values unchanged. **`drafter/DFLASH2_SOURCE.json`:** an ANEMLL provenance record.
- **`model/` configuration, tokenizer and chat template, root `config.json`, `LICENSE`:** copies of the upstream files.

Vision weights, multi-token prediction and the embedding table are not included. Retain this notice, [NOTICE](NOTICE), [LICENSE](LICENSE) and the drafter's [LICENSE](drafter/LICENSE) and [NOTICE](drafter/NOTICE) when redistributing.
