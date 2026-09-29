# Modification notice

**ANEMLL has modified the original Qwen3.8-27B model weights for ANE inference.** The converted artifacts are derivatives of the Qwen Team's model, not original upstream distributions.

Source: [Qwen/Qwen3.8-27B at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0). The upstream weights carry Copyright 2026 Alibaba Cloud and the [upstream Apache-2.0 license](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/LICENSE).

## Changes represented by the planned bundle

- **Core AI `.aimodel` and optional `.aimodelc` packages:** model graph conversion, partitioning and runtime packaging for ANE inference, with quantized weights and the selected numerical implementation. Compiled packages additionally reflect a particular compiler/toolchain; record that provenance in the release metadata.
- **`coreai/manifest.json`:** copied from the deployed research build; the historical machine-local `export` path is replaced by the portable export identifier `mix25in_mixr_lr64mix`. The staged manifest adds an ANEMLL modification notice for this metadata edit. Runtime package names and context entries are preserved.
- **`embed_tokens_fp16.npy`:** the upstream embedding tensor converted to FP16 and packaged as a NumPy array for memory-mapped host lookup.
- **Optional quantized export:** transformed weight representations, codebooks, indices, scales, rotation metadata, and residual factors where present. The deployed research recipe uses mixed two-bit/four-bit GPTQ, per-channel scaling, online rotations, and rank-64 residual corrections. The inventory and actual export metadata must identify which transformations apply to each released artifact.
- **Configuration and tokenizer assets:** intended to remain verbatim copies from the pinned upstream revision. Verify their hashes before release. If a text asset is changed, explicitly identify that change rather than describing the file as an unchanged upstream copy.

The default target is text-only Core AI inference on M6 through the Swift bridge. Vision, MTP, and a speculative drafter are not included in the default bundle. Final runtime contexts, tensor inventory, export provenance, and full hardware validation remain pending.

## Recording changes in the release

Retain this prominent notice alongside the original upstream license and ANEMLL's added [NOTICE](NOTICE). The release inventory must include per-file `modification_notice` metadata tying modified binary families to this explanation and identifying verbatim upstream assets.

Where the model package format supports descriptive metadata, include the modification notice there before computing final hashes. Do not insert arbitrary text into compiled binaries or alter package contents in ways that invalidate their structure. For modified text files, include an appropriate prominent modification notice in the file where its format permits, with the corresponding inventory entry recording the change; preserve parseable configuration and tokenizer formats.

These records document provenance and release preparation. They are not a claim that all licensing, attribution, redistribution, or hardware-validation work has been completed. ANEMLL Forge code licensing remains separate from the Apache-2.0 model-weight license.
