---
base_model:
  - Qwen/Qwen3.8-27B
  - ProCreations/Ternary-Bonsai-2-27B-DFlash2
base_model_relation: quantized
license: apache-2.0
pipeline_tag: text-generation
tags:
  - coreai
  - ane
  - quantized
  - speculative-decoding
  - dflash2
  - kv-cache
---

# ANEMLL Forge · Qwen3.8-27B for ANE

**Research project for inference of large dense models on the M6 Apple Neural Engine.** ANEMLL Forge shares quantization, conversion, Core AI inference and measured limitations. This update adds selectable **V-only INT8 KV caching, enabled by default**, with an explicit FP16-cache fallback in the same target packages. Keys remain FP16. Model-weight quantization is unchanged.

The normal inference path uses the included, tested **Core AI DFlash2 speculative drafter**. Each T=8 verifier cycle checks one anchor and seven draft proposals. T=8 target functions are verification functions; the separate `drafter/` package generates the proposals. Do not substitute a Core ML or unpaired drafter.

ANEMLL independently converts and quantizes [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Credit for the target belongs to the **Qwen Team and Alibaba Cloud**, with upstream **Copyright 2026 Alibaba Cloud** and Apache-2.0 licensing. The paired drafter derives from [ProCreations/Ternary-Bonsai-2-27B-DFlash2](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2), whose upstream notice preserves the z-lab DFlash2 and Prism Bonsai provenance. No endorsement by these projects or Apple is claimed.

## Files and runtime compatibility

- [coreai/](coreai): 16 target chunks, the unchanged output head and a selectable-cache manifest. Target recipe: `mix25in_mixr_lr64mix`, with mixed two-bit/four-bit GPTQ, per-channel scaling, online rotations and rank-64 residual corrections.
- [model/](model): matching tokenizer/configuration files and FP16 host embedding table.
- [drafter/](drafter): `dflash2_lut4_gptq.aimodel`, numerical metadata, configuration, compact BF16 selector codebooks and source licenses/notices. These assets are unchanged from the previous paired release.
- [release.json](release.json): complete per-file SHA-256/size inventory and target/drafter pairing.

The target's supported entries are **8K, 16K, 32K, 48K and 64K**. There is no 24K entry in this update. The 64K entry holds 65,472 history rows. These compiled sizes do not establish long-context quality.

V8 caches store historical values as INT8 with FP16 scales per token and KV head. The host quantizes accepted rows; ANE attention reconstructs historical values. New graph outputs remain FP16. Keys and the recurrent GDN state retain their existing precision. Only the 16 full-attention layers grow this cache. Selectable packages contain both physical cache-format entry families; the runtime binds only the selected family.

**Required source update:** use the V8-capable [KV-Cache-compression source branch](https://github.com/Anemll/anemll-forge/tree/KV-Cache-compression), or a newer merged revision with this support. Do not assume an older checkout or GitHub `main` supports these packages. A V8-capable runtime must select physical entries from `entries_by_kv`, preserve V codes/scales through context growth and speculative commits, and use the Swift bridge. The updated source's README provides setup and installation steps. The legacy Python binding is not the V8 runtime.

This release replaces target chunks and the cache manifest, and updates the inventory and this card. **A precision flag alone cannot add V8 support to the earlier FP16-only model files.** Original BF16 checkpoints are unnecessary for prepared inference; rebuilding requires the separate source weights and conversion environment. Vision and MTP are outside this text-generation bundle.

## Measured performance and evaluation

Measurements below are M6 experiments with the matching Core AI DFlash2 drafter and a 3 ms draft gap. They are research results, not a general capability score or a comparison against published Qwen leaderboard results.

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

The [dated KV-cache quantization research trace](https://github.com/Anemll/anemll-forge/blob/KV-Cache-compression/docs/research/KV_CACHE_QUANTIZATION_2026-10-02.md) preserves complete aggregate results, repeat ranges, methods and remaining work.

## Download and start

First install the required V8-capable source revision and environment using the source README above. When replacing an older bundle, use a fresh output directory; the helper rejects a different release inventory in an existing destination. Pin `HF_COMMIT` to this updated bundle's actual commit, not the upstream Qwen revision or the older FP16-only release:

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

The default Core AI download includes the paired drafter and selector assets. Automatic cache selection follows `coreai/manifest.json`: **V8 is the default** for this update. `CTX=64K` sets the growth cap; the runtime starts with a smaller entry and grows its KV state. Set `KV_CACHE_DTYPE=fp16` on the wrapper command for the explicit FP16-cache fallback, or use `--kv-cache-dtype fp16` with `forge.py serve`. `--plain` is a target-only diagnostic. The integrity-only check does not run inference; a short smoke generation does not establish sustained performance or broad quality.

## Attribution and licenses

- Target: [`Qwen/Qwen3.8-27B` at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0). Original Apache-2.0 [LICENSE](LICENSE) and ANEMLL-added [NOTICE](NOTICE) are included; [QWEN_SOURCE.json](QWEN_SOURCE.json) records source checks and remaining provenance limitations.
- Drafter: [`ProCreations/Ternary-Bonsai-2-27B-DFlash2` at `4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b`](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2/tree/4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b). Its original [LICENSE](drafter/LICENSE) and [NOTICE](drafter/NOTICE) remain unchanged.
- The drafter notice records donor [`z-lab/Qwen3.8-27B-DFlash2` at `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/tree/50307d4c4cde6860d4eee73e2547cd786fe8e8a4) and training target [`prism-ml/Ternary-Bonsai-2-27B-gguf` at `6ed5e12bf84b7a63069882c91dd9e9218647d17b`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/tree/6ed5e12bf84b7a63069882c91dd9e9218647d17b).

[MODIFICATIONS.md](MODIFICATIONS.md) describes the converted derivatives and V8 graph changes. The original BF16 drafter checkpoint was rehashed against the pinned upstream LFS digest, and compact selector tables retain identical tensor bytes. The Core AI drafter body has not been independently rebuilt from that checkpoint and GPTQ export; calibration and head linkage remain supported by its sidecar/export metadata. Reusing this tested pairing does not resolve that reconstruction gap.

**Model-derived assets follow their upstream Apache-2.0 licenses and notices. ANEMLL Forge source code and documentation are MIT-licensed.** Preserve the applicable model licenses, copyright and modification notices when redistributing derivatives.
