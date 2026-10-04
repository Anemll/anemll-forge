---
base_model:
  - Qwen/Qwen3.8-27B
  - ProCreations/Ternary-Bonsai-2-27B-DFlash2
base_model_relation: quantized
license: apache-2.0
pipeline_tag: text-generation
library_name: anemll-forge
tags:
  - coreai
  - ane
  - quantized
  - speculative-decoding
  - dflash2
  - kv-cache
---

# ANEMLL Forge · Qwen3.8-27B for ANE

**Research project for inference of large dense models on the M6 Apple Neural Engine.** ANEMLL Forge shares quantization, conversion, Core AI inference and measured limitations. This update replaces the 16 target chunks with a **faster exact ANE graph**: the same weights and model function, evaluated with less serial work. On M6 a full target verify is **19 to 31% faster** and a 64-row prefill call **29 to 35% faster** than the previous packages. The target now carries only the **V-only INT8 KV cache** (FP16 keys, INT8 values), which was already the default. Model-weight quantization is unchanged.

The normal inference path uses the included, tested **Core AI DFlash2 speculative drafter**. Each T=8 verifier cycle checks one anchor and seven draft proposals. T=8 target functions are verification functions; the separate `drafter/` package generates the proposals. Do not substitute a Core ML or unpaired drafter.

ANEMLL independently converts and quantizes [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Credit for the target belongs to the **Qwen Team and Alibaba Cloud**, with upstream **Copyright 2026 Alibaba Cloud** and Apache-2.0 licensing. The paired drafter derives from [ProCreations/Ternary-Bonsai-2-27B-DFlash2](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2), whose upstream notice preserves the z-lab DFlash2 and Prism Bonsai provenance. No endorsement by these projects or Apple is claimed.

## Files and runtime compatibility

- [coreai/](coreai): 16 target chunks (faster graph, V8 cache), the unchanged output head and the manifest. Target recipe: `mix25in_mixr_lr64mix`, with mixed two-bit/four-bit GPTQ, per-channel scaling, online rotations and rank-64 residual corrections.
- [model/](model): matching tokenizer/configuration files and FP16 host embedding table.
- [drafter/](drafter): `dflash2_lut4_gptq.aimodel`, numerical metadata, configuration, compact BF16 selector codebooks and source licenses/notices. These assets are unchanged from the previous paired release.
- [release.json](release.json): complete per-file SHA-256/size inventory and target/drafter pairing.
- [config.json](config.json): exact copy of `model/config.json` for Hub discovery and download counting, recorded separately as `hub_config` in the release inventory. The runtime continues to use the matching files under `model/`.

The target's supported entries are **8K, 16K, 32K, 48K and 64K**. There is no 24K entry in this update. The 64K entry holds 65,472 history rows. These compiled sizes do not establish long-context quality.

V8 caches store historical values as INT8 with FP16 scales per token and KV head. The host quantizes accepted rows; ANE attention reconstructs historical values. New graph outputs remain FP16. Keys and the recurrent GDN state retain their existing precision. Only the 16 full-attention layers grow this cache. These packages contain only V8 entries. The FP16-cache fallback remains in the previous revision [`cd7dfc6`](https://huggingface.co/anemll/anemll-forge-qwen3.8-27B/tree/cd7dfc605ccad091b961f7788939c30d01c3793e), which has selectable FP16/V8 packages with the earlier graph.

**Source:** use the current [ANEMLL Forge source](https://github.com/Anemll/anemll-forge/tree/main), which includes the V8 runtime; the faster graph needs no runtime change. The V8 runtime preserves V codes/scales through context growth and speculative commits and uses the Swift bridge. The source README provides setup and installation steps. The legacy Python binding is not the V8 runtime.

This update replaces the 16 target chunks and the manifest, and updates the inventory, [MODIFICATIONS.md](MODIFICATIONS.md) and this card. The output head, model assets and drafter are byte-identical to the previous revision. **A precision flag alone cannot add V8 support to the earlier FP16-only model files.** Original BF16 checkpoints are unnecessary for prepared inference; rebuilding requires the separate source weights and conversion environment. Vision and MTP are outside this text-generation bundle.

## Faster exact graph (this update)

Two rewrites change how the ANE evaluates the target, not its weights or math:

- **Gated DeltaNet (48 of 64 layers).** Verification and prefill process tokens in 8-row blocks, and each block needs a small triangular solve: every token's delta-rule update depends on the earlier tokens in the block. The previous graph solved it row by row (7 dependent steps). The new graph uses the exact closed form `(I + N)^-1 = (I - N)(I + N^2)(I + N^4)` for the strictly lower-triangular block (3 steps and a matrix product), a native depthwise convolution, and one fewer state product per prefill block. Background on the chunkwise delta rule: [DeltaNet Explained (Part II)](https://sustcsonglin.github.io/blog/2024/deltanet-2/).
- **Attention history tiles.** The 16 full-attention layers read the KV history in 2,048-wide tiles when verifying and 4,096-wide tiles when prefilling (previously 16,384). Smaller tiles cost the ANE less per call, most at long context.

Full 16-chunk target on M6, previous packages (V8 entries) to this update, idle machine, median of repeated calls:

| Context entry | Verify (8 rows) | Prefill call (64 rows) | Prefill rows/s |
| --- | --- | --- | --- |
| 8K | 113.1 → 91.5 ms (−19.1%) | 284.5 → 202.8 ms (−28.7%) | 225 → 316 |
| 16K | 123.5 → 97.4 ms (−21.1%) | 351.5 → 230.9 ms (−34.3%) | 182 → 277 |
| 32K | 144.8 → 108.1 ms (−25.4%) | 412.6 → 285.1 ms (−30.9%) | 155 → 225 |
| 48K | 168.2 → 119.5 ms (−28.9%) | 497.7 → 338.1 ms (−32.1%) | 129 → 189 |
| 64K | 190.3 → 131.5 ms (−30.9%) | 603.4 → 394.1 ms (−34.7%) | 106 → 162 |

These are target-only call times, not end-to-end generation rates. With the DFlash2 drafter, a full server measured cold prefill **26% (8K) to 41% (64K)** faster and decode **21 to 25%** faster at 32K to 64K, with identical replies at 48K and 64K, on one synthetic coding workload. That server run used an earlier two-format build of the same graph (2,048-wide tiles everywhere), not these exact packages.

Quality: after a 64,000-token prefill through the whole context ladder, perplexity on the next 1,024 WikiText-2 positions was **5.0673** (previous packages 5.0686); direct KL between the previous and new targets averaged **5.0e-5** nats, and the top token agreed at 99.7% of positions. On the short 64-sequence KL-512 trace, the two-format build of the same graph measured mean KL to BF16 **0.184168** (previous V8 0.184270). The differences are floating-point rounding order. This is not a coding, reasoning or retrieval evaluation.

**First start compiles the new packages.** The first `quick-test` or server start on a Mac compiles the target for the ANE once: **22 min 48 s** for the 16 chunks on M6 (measured, with the unchanged head and drafter already cached from the previous revision; on a fresh Mac they add an estimated 1 to 2 minutes). Later starts load from the compile cache in seconds. The cache is per macOS build, so a macOS update compiles again. Stopping is safe: each finished package stays cached and the next start resumes. Updated Forge source prints an `[ANE compile]` readout during this step: packages left, an estimate, a line per package with the time left, and options that compile faster. Earlier source prints each chunk as it finishes.

## V8 cache measurements (previous update)

The measurements in this section compare FP16 and V8 caches on the previous revision's packages (earlier graph). The V8 cache format is unchanged in this update. Measurements below are M6 experiments with the matching Core AI DFlash2 drafter and a 3 ms draft gap. They are research results, not a general capability score or a comparison against published Qwen leaderboard results.

Full-server prefill throughput, FP16 values → INT8 values:

- 8K: **181.8 → 181.0 tok/s** (−0.4%).
- 16K: **157.8 → 161.1 tok/s** (+2.1%).
- 32K: **126.5 → 144.5 tok/s** (+14.2%).
- 48K: **107.4 → 128.5 tok/s** (+19.7%).
- 64K: **99.8 → 111.5 tok/s** (+11.8%).

Prefill covers all 64 target layers, output head, host cache handling/context growth and drafter context ingestion; it excludes model loading, tokenization, HTTP and generated tokens. Each context has one paired trial. The prefill experiment used prototype V8 packages and stock FP16 packages, with different physical-function counts. Stable V8 attention arithmetic also differs from stock short-context FP16 attention.

Full-server decode median throughput on the final shared selectable packages, FP16 → V8:

- 8K: **57.17 → 55.37 tok/s** (−3.1%); speculative acceptance **88.49% → 88.49%**.
- 16K: **47.43 → 51.99 tok/s** (+9.6%); acceptance **85.71% → 91.43%**.
- 32K: **42.21 → 48.00 tok/s** (+13.7%); acceptance **88.49% → 88.49%**.
- 48K: **37.13 → 39.55 tok/s** (+6.5%); acceptance **88.49% → 84.56%**.
- 64K: **31.38 → 38.86 tok/s** (+23.9%); acceptance **91.43% → 91.43%**.

Decode uses three adjacent cached repeats per mode/context after a cold prompt, greedy decoding, thinking disabled and 256 generated tokens per request. It includes the whole model/drafter/verification/host-cache path and excludes prefill, loading and HTTP. One synthetic public coding workload was used at each context. Within-mode replies were identical across repeats. FP16/V8 replies matched at four contexts and differed at 48K. Acceptance and output differences affect throughput; this is not an isolated cache-bandwidth measurement. The 8K repeat ranges overlap. A token-limited synthetic response is not a scored coding benchmark. No equivalent full-server M5 result is included in this matrix.

Current quality evidence remains a **short teacher-forced KL-512 trace**, using the byte-verified original BF16 teacher, teacher top-512 plus an aggregate tail, 64 sequences and 40,023 next-token positions; the longest sequence is 857 tokens:

- Mean KL: **0.1846247 → 0.1842700** (FP16 → V8).
- Median / p99 KL: **0.030304 / 2.28602 → 0.030238 / 2.27443**.
- Top-1 agreement: **86.0455% → 86.0205%**.
- Trace perplexity: **2.41475 → 2.41210**; teacher trace perplexity **2.13554**.
- Direct target FP16-versus-V8 KL on the teacher partition: mean **0.000073707**.

The small mean-KL decrease does **not** establish better INT8 quality: top-1 agreement slightly decreases, attention arithmetic changes, and this is a short trace. This is not a MirAI-matched public protocol or a long-context, coding, reasoning or retrieval evaluation. The final shared packages reproduced the prototype's partition log-probability arrays and KL **bit-exactly for the first three sequences (2,299 positions) at 8K**, in both modes. The complete 64-sequence KL evaluation was not rerun on the final shared packages.

The logical persistent K/V/scale payload falls from **64 KiB to 48.125 KiB per history position**, a **24.8047%** saving. At 65,472 rows this is about **3.996 → 3.005 GiB**, saving **0.991 GiB**. This is logical cache-buffer size, **not** measured total resident/wired memory, compiler scratch size or leak mitigation.

Validation covered all five paired contexts, finite logits, selected-function cached placement audits with no GPU regions, compile mode 2 and clean exits of the benchmark-owned servers. Cached placement audits do not prove live physical bonded-cluster allocation, native integer-MAC precision or dequantization fusion. No power/thermal conclusion is included. Further benchmarks will be added with their measured methods and limitations.

The [dated KV-cache quantization research trace](https://github.com/Anemll/anemll-forge/blob/main/docs/research/KV_CACHE_QUANTIZATION_2026-10-02.md) preserves complete aggregate results, repeat ranges, methods and remaining work.

## Download and start

First install the current source and environment using the source README above. When replacing an older bundle, use a fresh output directory; the helper rejects a different release inventory in an existing destination. Pin `HF_COMMIT` to this updated bundle's actual commit, not the upstream Qwen revision or an older release:

```sh
python forge.py download \
  --repo anemll/anemll-forge-qwen3.8-27B \
  --revision HF_COMMIT \
  --runtime coreai \
  --output ./models/qwen3.8-27b-ane

python forge.py quick-test \
  --bundle ./models/qwen3.8-27b-ane \
  --runtime coreai --check-only

bash coreai/swift_bridge/build.sh

python forge.py quick-test \
  --bundle ./models/qwen3.8-27b-ane \
  --runtime coreai --tokens 16

FORGE_BUNDLE="$PWD/models/qwen3.8-27b-ane" \
PY="python" CTX=64K \
bash scripts/qwen38_server.sh start
```

The default Core AI download includes the paired drafter and selector assets. The runtime reads the V8 cache format from `coreai/manifest.json`. The full `quick-test` performs the one-time ANE compile (about 23 minutes on M6, above), so the server then starts in seconds. `CTX=64K` sets the growth cap; the runtime starts with a smaller entry and grows its KV state. For the FP16-cache fallback, download revision `cd7dfc605ccad091b961f7788939c30d01c3793e` instead and set `KV_CACHE_DTYPE=fp16` on the wrapper command, or use `--kv-cache-dtype fp16` with `forge.py serve`. `--plain` is a target-only diagnostic. The integrity-only check does not run inference; a short smoke generation does not establish sustained performance or broad quality.

## Attribution and licenses

- Target: [`Qwen/Qwen3.8-27B` at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0). Original Apache-2.0 [LICENSE](LICENSE) and ANEMLL-added [NOTICE](NOTICE) are included; [QWEN_SOURCE.json](QWEN_SOURCE.json) records source checks and remaining provenance limitations.
- Drafter: [`ProCreations/Ternary-Bonsai-2-27B-DFlash2` at `4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b`](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2/tree/4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b). Its original [LICENSE](drafter/LICENSE) and [NOTICE](drafter/NOTICE) remain unchanged.
- The drafter notice records donor [`z-lab/Qwen3.8-27B-DFlash2` at `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/tree/50307d4c4cde6860d4eee73e2547cd786fe8e8a4) and training target [`prism-ml/Ternary-Bonsai-2-27B-gguf` at `6ed5e12bf84b7a63069882c91dd9e9218647d17b`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/tree/6ed5e12bf84b7a63069882c91dd9e9218647d17b).

[MODIFICATIONS.md](MODIFICATIONS.md) describes the converted derivatives and V8 graph changes. The original BF16 drafter checkpoint was rehashed against the pinned upstream LFS digest, and compact selector tables retain identical tensor bytes. The Core AI drafter body has not been independently rebuilt from that checkpoint and GPTQ export; calibration and head linkage remain supported by its sidecar/export metadata. Reusing this tested pairing does not resolve that reconstruction gap.

**Model-derived assets follow their upstream Apache-2.0 licenses and notices. ANEMLL Forge source code and documentation are MIT-licensed.** Preserve the applicable model licenses, copyright and modification notices when redistributing derivatives.
