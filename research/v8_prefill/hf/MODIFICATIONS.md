# Modification notice

**ANEMLL has modified the Qwen3.8-27B target and the paired ProCreations DFlash2 drafter for Core AI inference on the Apple Neural Engine.** Converted artifacts are derivatives of their upstream models.

The target source is [Qwen/Qwen3.8-27B at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0). Its weights carry Copyright 2026 Alibaba Cloud and the original Apache-2.0 license, preserved in [LICENSE](LICENSE).

The drafter source is [ProCreations/Ternary-Bonsai-2-27B-DFlash2 at `4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b`](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2/tree/4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b), with its original Apache-2.0 `LICENSE` and `NOTICE` copied unchanged into each drafter folder. Its source notice credits the z-lab DFlash2 donor and Prism Bonsai training target; pinned identities are recorded in each drafter folder's `DFLASH2_SOURCE.json`.

This repository (release 0.2.1) carries the release 0.2 target weights of [anemll/anemll-forge-qwen3.8-27B](https://huggingface.co/anemll/anemll-forge-qwen3.8-27B) (export `release_vq3pA_mixh_s600_k1_mat`) in new packages for contexts up to 192K, with two new drafter packages. The quantized target weights are unchanged; the graph, the cache format and the context entries are new.

## Target conversion and packaging

- **`coreai/m6-192k/chunk_*.aimodel`:** new graph derivatives of the release 0.2 target export, with the same quantized weights (three-bit vector lookup tables for every MLP matrix, two-bit/four-bit lookup tables with INT8 K/V projections and rank-64 factors for the token mixers, online Hadamard rotations, GPTQ followed by quantization-aware training; see the release 0.2 notice). Changes against the release 0.2 packages:
  - **Cache format `kv8`:** historical keys and values are both INT8 with FP16 scales per token and KV head (release 0.2: FP16 keys, INT8 values). Keys stay transposed (KV head, head dimension, token). The host quantizes accepted rows; ANE attention multiplies each history tile's scores by the key scales and reconstructs the values.
  - **Context entries** 8K, 16K, 32K, 64K, 96K, 128K and 192K (release 0.2: 8K, 16K, 32K, 48K and 64K). Entries above 64K evaluate history attention in tiles without forming the concatenated history and block tensor, so they hold their full length; the 64K entry holds 65,472 history rows.
  - **Prefill entries** take 64 rows per call for 8K to 96K with 4,096-wide history tiles, and 32 rows for 128K and 192K with 8,192-wide tiles. Verify entries (8 rows) use 2,048-wide tiles up to 128K and 4,096-wide tiles at 192K.
  - **M6 functions only:** INT8 attention scores and FP8 softmax and PV probabilities. No M5 function set is included (the M5 ANE compiler rejects FP8).
- **`coreai/m6-192k/head_T8.aimodel`:** the release 0.2 four-bit LUT head, re-exported with this build (same weights).
- **`coreai/m6-192k/manifest.json`:** prepared from the build. The local export path was replaced by the portable target identifier `release_vq3pA_mixh_s600_k1_mat`, and a modification notice was added. `ptp` records the prefill rows of each context entry and `kv_len` / `pkv_len` their history rows; each chunk's `numerics` record the graph options.
- **`model/`:** byte-identical to release 0.2: the original target embedding converted to FP16 as a NumPy array for host lookup, and unchanged copies of the pinned upstream configuration and tokenizer files. The root `config.json` is an exact copy of `model/config.json` for Hub discovery.

## Paired drafter conversion and packaging

- **`drafter/all3/dflash2_lut4_gptq_nohead.aimodel` and `drafter/gu2a3/dflash2_lut4_gptq_nohead.aimodel`:** new Core AI packages of the five-layer drafter body without an LM head, quantized by GPTQ from the pinned ProCreations BF16 checkpoint with the `qmixr_cal` calibration traces. `all3` stores every projection as three-bit vector lookup tables (64 two-component centroids per matrix, per-channel scales); `gu2a3` stores the gate and up projections as two-bit vector tables (16 centroids) and every other projection as three-bit. Both keep the kernel projection in INT8. Instead of carrying a copy of the target's LM head (the release 0.2 drafter carried a LUT4 GPTQ body with its own head copy), they feed their hidden state through the paired target's head. Mask-row scale 0.7, a 2,048-row context ring and a block of one anchor plus seven proposals, as before. The file name keeps the drafter family's `lut4` prefix; these weights are not LUT4.
- **`drafter/*/head_normw_fp16.npy`:** the final RMSNorm gain (1 + g) of the target checkpoint in FP16, with which the drafter scales its hidden state for the target head. Release 0.2's training did not change this gain.
- **`drafter/*/dflash2_lut4_gptq_nohead.json`:** source numerical and entry metadata retained. Local `export` and `target_export` paths were replaced by portable identifiers, and an ANEMLL modification notice was added. The packages were built beside the `release_u15_data600_mat` target export; a body-only package carries no target weights, and the runtime pairs it with this repository's target and head.
- **`drafter/*/selector.safetensors` and `drafter/*/config.json`:** byte-identical to release 0.2: two unmodified BF16 selector tables extracted from the pinned checkpoint, and the unmodified upstream configuration (target feature taps 5, 19, 33, 47 and 61, selector rank 256 and top-k 16, sliding window 2,048).
- **`drafter/*/LICENSE` and `drafter/*/NOTICE`:** unchanged upstream files. **`drafter/*/DFLASH2_SOURCE.json`:** an ANEMLL-added provenance record for each package (source hashes, selector extraction, package files, remaining gaps).

Vision and MTP remain outside this text-generation bundle. Original BF16 checkpoints are not required for prepared inference; they remain necessary for separate rebuilding workflows.

## Verification and remaining gaps

The target weights are those of release 0.2, whose KL and perplexity were measured on its 64K packages (V8 cache). For these packages: on the previous weights, `kv8` against V8 measured a direct KL of 6.6e-5 nats with the same top token at 99.75% of positions (KL-512, 64 sequences, 40,023 positions); this was not repeated with release 0.2 weights. At long context, recall of six planted codes was correct at every context size up to 195K tokens; no long-context KL or perplexity was measured above 64K. The drafters' acceptance and speed were measured on M6 with these target packages (within 1% of the release 0.2 drafter's speed). The drafter body tensors have not been independently reconstructed from the pinned checkpoint and GPTQ exports. Packaging integrity checks do not run inference again or establish complete hardware or quality validation.

## V8 research build (this repository, November 2026)

The packages in `coreai/m6-96k-v8/` keep the release 0.2 target weights and their quantization unchanged. Compared
with the release 0.2.1 packages, the graphs were rebuilt as follows:

- Context ladder 8K, 16K, 32K, 64K and 96K (98,304 tokens), INT8 keys and values (`kv8`, keys dim-major) with FP8
  attention probabilities (M6 only).
- Segmented KV history: each attention layer's cache is five buffers (8K, 8K, 16K, 32K and 32K rows) shared by every
  context entry, so growing the context binds more buffers and copies nothing.
- Additional entry points on the same weight constants for the V8 prefill schedule: each layer's FFN alone (with the
  FFN split into 8,192 + 9,216 intermediate channels), each DeltaNet mixer alone, and FFNs fused with the attention
  projections, in 64- and 128-row forms. The verify, prefill and head functions are those of the regular target.

The drafter and the host files in `model/` are byte-identical to release 0.2.1.

## SPLIT+GPU decode build (this repository, 17 November 2026)

`coreai/m6-96k-dec8/` replaces the 16 four-layer packages with 8 packages of 8 layers built from the same release 0.2
export (`v8/build_dec8_chunk.py` at `3d96cb9`). Each package carries, on one copy of its weights, the target's verify
and prefill entries and the V8 prefill entries of its two four-layer groups (`g0_` / `g1_`), plus three decode entries
that split each verification step around its attention layers: `dva` (the layers up to the attention layer's q / k / v
projections), `dvba` (the rest of that layer fused with the next group's part A) and `dvb` (the rest). They bind no KV
cache and no mask; the GPU computes the attention over the KV history between them. The weights, quantization, cache
format and numerics of every other entry are unchanged.

## Tree8 + sliced decode build (this repository, 22 November 2026)

`coreai/m6-96k-dec8ts/` is built from the same release 0.2 export as `coreai/m6-96k-dec8/` (`v8/build_dec8t_chunk.py`
with `TREE_FORM=corrx0 TREE_SLICED=24576,40960,49152,57344`). Every entry point of the 17 November build is kept with
the same weights, quantization, cache format and numerics. Added per 4-layer group, on the same weight constants:

- tree verify entries `t8_<ctx>k` (8K to 64K) and the decode-split tree entries `tdva` / `tdvba`: the 8-row verify
  with a per-row ancestor mask, so 8 rows check a tree of the drafter's continuations instead of one chain (row 0 =
  the anchor; each row attends to the KV history and to its ancestors only);
- sliced verify entries `v8s_<w>k` (chain) and `t8s_<w>k` (tree) for w = 24K, 40K, 48K, 56K: they bind the KV
  segments of the next context entry (32K for 24K, 64K for the others) and attend over the first w keys only
  (bit-exact against the larger entry with the keys past w masked).

`gpu/w_dec8ts/` holds the GPU engine's tables for this build: norms, rotations, low-rank adapters, DeltaNet
constants and INT8 k / v projections taken from the export, plus `.tab` offsets into the compiled Neural Engine
programs of the development M6 (macOS 26C5047e), with sha256 checks (`kit.json`).

## Retaining notices

Retain this prominent modification notice, the root [NOTICE](NOTICE), the original target [LICENSE](LICENSE), and each drafter folder's original `LICENSE` and `NOTICE`.

Keep model packages structurally valid: do not insert arbitrary text into binary contents. Mark modified JSON and container metadata where supported, and retain accompanying notices. ANEMLL Forge source-code licensing remains separate from the Apache-2.0 model-weight license.
