# Modification notice

**ANEMLL has modified the Qwen3.8-27B target and the paired ProCreations DFlash2 drafter for Core AI inference on the Apple Neural Engine.** Converted artifacts are derivatives of their upstream models.

The target source is [Qwen/Qwen3.8-27B at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0). Its weights carry Copyright 2026 Alibaba Cloud and the original Apache-2.0 license, preserved in [LICENSE](LICENSE).

The drafter source is [ProCreations/Ternary-Bonsai-2-27B-DFlash2 at `4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b`](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2/tree/4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b), with its original Apache-2.0 [LICENSE](drafter/LICENSE) and [NOTICE](drafter/NOTICE) copied unchanged. Its source notice credits the z-lab DFlash2 donor and Prism Bonsai training target; pinned identities remain in [drafter/DFLASH2_SOURCE.json](drafter/DFLASH2_SOURCE.json).

## Target conversion and packaging

- **`coreai/*.aimodel`:** model graph conversion, partitioning and packaging with quantized weights and selected numerical implementations. The target recipe uses mixed two-bit/four-bit GPTQ, per-channel scaling, online rotations and rank-64 residual corrections. Per-artifact inventory and export metadata identify the applicable transforms.
- **`coreai/manifest.json`:** prepared from the selectable V8/FP16 build. The local export path was replaced by the portable target identifier `mix25in_mixr_lr64mix`, preserving the unchanged drafter/head pairing. V8 is the declared default. Physical function maps and 8K/16K/32K/48K/64K context entries were preserved; this update has no 24K entry.
- **`model/embed_tokens_fp16.npy`:** the original target embedding tensor converted to FP16 and packaged as a NumPy array for host lookup.
- **Target configuration/tokenizer assets:** copies of the pinned upstream files, retained without content changes.
- **Optional quantized exports:** transformed weight representations, codebooks, indices, scales, rotations and residual factors, where included. These are separate reproduction artifacts rather than required runtime assets.

## Paired drafter conversion and packaging

- **`drafter/dflash2_lut4_gptq.aimodel`:** the selected Core AI source package carries LUT4 GPTQ drafter weights, the paired mixr target head and mask-row scale 0.7. It uses the historical `q7_cal` calibration export, a five-layer drafter, a 2,048-row context ring and a block of one anchor plus seven proposals. The existing package was copied without changing its bytes; these notices identify the upstream weight transformations already represented in it.
- **`drafter/dflash2_lut4_gptq.json`:** source numerical/entry metadata was retained. Local `export`, `target_export` and `head_export` paths were replaced by portable identifiers, and an ANEMLL modification notice was added. Target/head association is `mix25in_mixr_lr64mix`.
- **`drafter/selector.safetensors`:** ANEMLL extracted only `candidate_selector.predecessor_codebook` and `candidate_selector.successor_codebook` from the pinned upstream BF16 checkpoint. Both tables retain their original BF16 values and shapes; raw tensor hashes were compared after serialization. This new container adds source and modification metadata and excludes all other checkpoint tensors.
- **`drafter/config.json`:** the pinned upstream configuration was copied without modification. It preserves target feature taps 5, 19, 33, 47 and 61, selector rank 256/top-k 16 and sliding window 2,048.
- **`drafter/LICENSE` and `drafter/NOTICE`:** unchanged upstream files. **`drafter/DFLASH2_SOURCE.json`** is an ANEMLL-added provenance record describing the source hashes, selector extraction and remaining reconstruction gaps.

The default release includes both Core AI target and speculative drafter. Vision and MTP remain outside this text-generation bundle. Original BF16 checkpoints are not required for prepared inference; they remain necessary for separate rebuilding workflows.

## Verification and remaining gaps

The complete local drafter BF16 checkpoint was rehashed and matched the pinned upstream LFS SHA256. Configuration and source license/notice bytes were checked against upstream Git blobs. Compact selector tensor values were checked for identical raw bytes after extraction.

The Core AI drafter body has not been independently reconstructed from the source checkpoint and GPTQ export. Head/target and calibration linkage are currently supported by the source sidecar/export metadata. The paired target/drafter has separate M6 prefill, decode and short KL experiments summarized with limitations in the model card. Packaging integrity checks do not run inference again or establish complete hardware or quality validation.

## Selectable historical value-cache update

The target packages now contain both FP16-cache and V8-cache entry families for 8K, 16K, 32K, 48K and 64K contexts, using the same quantized weights and unchanged output head. V8 preserves FP16 keys and stores historical values as INT8 with FP16 scales per token/head. The host quantizes accepted rows and ANE attention reconstructs historical values; newly returned activations remain FP16. V8 uses stable attention arithmetic, which differs from the stock short-context FP16 arithmetic. The manifest selects V8 by default; explicit FP16 fallback remains.

The 16 target source packages are new graph derivatives. Their binary program bytes were copied unchanged from the completed selectable export; only the external runtime manifest was sanitized. Tokenizer, embedding and tested Core AI DFlash2 assets remain unchanged from the previous paired release. Updated source code is required for physical entry selection, cache growth and speculative commit handling. A manifest flag alone does not retrofit older FP16-only packages.

## Retaining notices

Retain this prominent modification notice, root [NOTICE](NOTICE), original target [LICENSE](LICENSE), and the drafter's original [LICENSE](drafter/LICENSE)/[NOTICE](drafter/NOTICE). `release.json` records per-file hashes and modification notices, including changed binary families and unchanged upstream assets.

Keep model packages structurally valid: do not insert arbitrary text into binary contents. Mark modified JSON/container metadata where supported, and retain accompanying inventory notices. ANEMLL Forge source-code licensing remains separate from the Apache-2.0 model-weight license.
