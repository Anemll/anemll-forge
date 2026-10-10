---
base_model: Qwen/Qwen3.8-27B
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

**Research project for inference of large dense models on the M6 Apple Neural Engine.** ANEMLL Forge shares quantization, conversion, Core AI inference and measured limitations. **This update (release 0.2) replaces the quantized weights.** Every MLP matrix is now a **three-bit vector lookup table** (previously two-bit in 38 layers and four-bit in 26), the token mixers get **online Hadamard rotations** like the MLP's, and the weights are fitted by GPTQ and **quantization-aware training (QAT)** against the BF16 model. On the M6 ANE, KL to BF16 falls from **0.184 to 0.052** (KL-512) and top-1 agreement rises from 86.0% to **91.9%**, at the previous packages' speed and with 0.35 GB less compiled ANE memory. The packages keep the previous update's 8-bit attention, transposed key cache and **M6 and M5 function sets in one download**: on M6 the attention scores are INT8 and the softmax and PV probabilities FP8; the M5 functions keep the INT8 scores and run the softmax in FP16, because the M5 ANE compiler does not support FP8.

The normal inference path uses the included, tested **Core AI DFlash2 speculative drafter**. Each T=8 verifier cycle checks one anchor and seven draft proposals. T=8 target functions are verification functions; the separate `drafter/` package generates the proposals. Do not substitute a Core ML or unpaired drafter.

ANEMLL independently converts and quantizes [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Credit for the target belongs to the **Qwen Team and Alibaba Cloud**, with upstream **Copyright 2026 Alibaba Cloud** and Apache-2.0 licensing. The paired drafter derives from [ProCreations/Ternary-Bonsai-2-27B-DFlash2](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2), whose upstream notice preserves the z-lab DFlash2 and Prism Bonsai provenance. No endorsement by these projects or Apple is claimed.

## Files and runtime compatibility

- [coreai/](coreai): 16 target chunks (8-bit attention with M6 and M5 function sets, transposed keys, V8 cache, faster graph), the output head and the manifest. Target weights: `release_vq3pA_mixh_s600_k1_mat` (release 0.2): three-bit vector LUTs for every MLP matrix, two-bit/four-bit LUTs for the token mixers with rank-64 low-rank factors, INT8 attention K/V projections, a four-bit LUT head, per-channel scaling and online Hadamard rotations on the MLP and token-mixer inputs, fitted by GPTQ and then quantization-aware training (QAT, below).
- [model/](model): matching tokenizer/configuration files and FP16 host embedding table.
- [drafter/](drafter): `dflash2_lut4_gptq.aimodel`, numerical metadata, configuration, compact BF16 selector codebooks and source licenses/notices. The drafter's own quantized layers are unchanged; the package was rebuilt because it drafts with the target's LM head, which release 0.2 retrained.
- [release.json](release.json): complete per-file SHA-256/size inventory and target/drafter pairing.
- [config.json](config.json): exact copy of `model/config.json` for Hub discovery and download counting, recorded separately as `hub_config` in the release inventory. The runtime continues to use the matching files under `model/`.

The target's supported entries are **8K, 16K, 32K, 48K and 64K**. There is no 24K entry in this update. The 64K entry holds 65,472 history rows. These compiled sizes do not establish long-context quality.

V8 caches store historical values as INT8 with FP16 scales per token and KV head. The host quantizes accepted rows; ANE attention reconstructs historical values. New graph outputs remain FP16. Keys and the recurrent GDN state retain their existing precision. Only the 16 full-attention layers grow this cache. These packages contain only V8 entries. The FP16-cache fallback remains in the previous revision [`cd7dfc6`](https://huggingface.co/anemll/anemll-forge-qwen3.8-27B/tree/cd7dfc605ccad091b961f7788939c30d01c3793e), which has selectable FP16/V8 packages with the earlier graph.

**Quantized weights:** the target and drafter exports these packages were converted from, with everything needed to rebuild them without the original checkpoints, are in [anemll/anemll-quantized-qwen3.8-27b-for-CoreAI](https://huggingface.co/anemll/anemll-quantized-qwen3.8-27b-for-CoreAI) (see its EXPORT.md).

**Source:** these packages need the ANEMLL Forge source with the transposed-key runtime and the M5 build derivation (October 5, 2026 or later); earlier source refuses them at load (key-layout check). On an M5, the first start also needs the Core AI authoring package: `python -m pip install coreai-core`. The V8 runtime preserves V codes/scales through context growth and speculative commits and uses the Swift bridge. The source README provides setup and installation steps. The legacy Python binding is not the V8 runtime.

This update replaces the 16 target chunks, the output head, the drafter package and the manifest, and updates the inventory, [MODIFICATIONS.md](MODIFICATIONS.md) and this card. The tokenizer, embedding table, drafter configuration and selector are byte-identical to the previous revision. **A precision flag alone cannot add V8 support to the earlier FP16-only model files.** Original BF16 checkpoints are unnecessary for prepared inference; rebuilding requires the separate source weights and conversion environment. Vision and MTP are outside this text-generation bundle.

## Release 0.2: three-bit MLP, token-mixer rotations, distilled weights (this update)

| Part | Previous packages | **Release 0.2** |
| --- | --- | --- |
| MLP gate / up / down | two-bit vector LUT (2x16) in 38 layers, four-bit LUT in 26 | **three-bit vector LUT (2x64) in all 64 layers** |
| Token mixers (DeltaNet and attention projections) | two-bit vector LUT in layers 0-23, four-bit LUT in 24-63, attention K/V INT8, rank-64 factors | same formats, **plus online Hadamard rotations** of the projections' inputs |
| LM head | four-bit LUT | four-bit LUT, retrained |
| Quantized weights, excluding the FP16 embedding | 9.07 GiB (9.36 with the FP16 factors) | 9.44 GiB (9.74), 3.17 bits per weight |
| Compiled ANE memory on M6 (chunks and head) | 13.32 GB | 12.97 GB |

- **Three bits per MLP weight:** 64 two-component centroids per matrix and one six-bit index per pair of output channels, with per-channel scales. In tests on eight layers it removes about 80% of the gap between the two-bit and four-bit formats, in every layer and matrix type; at the same size as a two-bit/four-bit split (`u48`) it lowered KL by 23%. On the ANE it decodes faster per byte than the four-bit tables, and it avoids a second compiled copy the four-bit `down_proj` needs, so the model uses less ANE memory than before.
- **Token-mixer rotations:** the inputs of DeltaNet's `in_proj_qkv`, `in_proj_z` and `out_proj` and of attention's `q_proj`, `k_proj`, `v_proj` and `o_proj` are multiplied online by 1,024-wide Hadamard blocks with seeded signs, the weights stored in the rotated basis, as the MLP already did. Against a matched control (two seeds) this lowered KL by 6 to 10%; it costs about 1% of decode time on M6 and 2 to 3% on the M5 Max.
- **Fitting:** GPTQ with the rank-64 factors inside the sequential pass and 128 calibration rows (WikiText, BF16 chat traces with thinking, rendered agentic coding sessions), then 600 steps of quantization-aware training (QAT, by distillation to BF16) that train the lookup-table values, per-channel scales, low-rank factors and LM head to match BF16's next-token distribution (top-256 plus tail), with every index frozen. Training data: the calibration rows plus about 640 rows of BF16 responses, thinking included, to new prompts across code, agentic tool use, multilingual text, math and rare tokens; the best step was chosen on 128 further held-out prompts.

Quality on the ANE (compiled packages, the same harness as the previous updates):

| | Previous packages | **Release 0.2** |
| --- | ---: | ---: |
| KL-512 to BF16, M6 (64 chats, 40,023 positions) | 0.1838 | **0.0518** |
| KL-512 median / p99 | 0.030 / 2.28 | 0.009 / 0.50 |
| Top-1 agreement with BF16 | 86.0% | 91.9% |
| KL-512 to BF16, M5 Max (M5 functions) | | 0.0517 |
| Perplexity, KL-512 chats (BF16 2.136) | 2.414 | 2.131 |
| Perplexity, verify path (64 + 4,096 tokens) | 6.645 | 6.052 |
| Perplexity, 8K (7,600 + 512) | 8.096 | 7.552 |
| Perplexity, 64K (64,000 + 1,024) | 5.068 | 4.537 |

The ANE results match the PyTorch evaluation of the same weights (KL 0.0515). In PyTorch against BF16, with the shared system prompt excluded, KL falls from 0.1512 to 0.0536 on the development chat trace and from 0.1617 to 0.0565 on the held-out trace; on WikiText KL is 0.0707, and on the last 2K tokens of 16K-token windows 0.762 (a larger experimental model with four-bit late MLP layers measured 0.0784 and 0.811). The training rows were checked against every evaluation set: their 13-gram overlap covers 0.04 to 0.12% of evaluation windows, all generic phrasing, and per-sequence gains do not track overlap.

Full server, the same synthetic coding workload as the previous updates, greedy, thinking off, 256 tokens, DFlash2 (previous packages in parentheses):

| Context | M6: prefill / decode tok/s | M5 Max: prefill / decode tok/s |
| --- | --- | --- |
| 8K | 299 / 59.5 (306 / 60.8) | 175 / 27.0 (172 / 28.8) |
| 16K | 290 / 53.3 (297 / 60.3) | 170 / 26.0 (171 / 27.8) |
| 32K | 269 / 49.7 (275 / 54.5) | 153 / 23.8 (156 / 24.5) |
| 48K | 250 / 45.5 (253 / 46.1) | 140 / 20.1 (143 / 24.1) |
| 64K | 235 / not measured (237 / 47.9) | 129 / 20.3 (132 / 22.6) |

Prefill is within 2 to 3% of the previous packages. Decode with a drafter depends on how many drafted tokens the target accepts, which depends on the text being generated: on this fixed prompt release 0.2 accepts fewer (79% at 8K against 87 to 89% for earlier weights), so its benchmark decode is lower. Over 24 varied chat and coding prompts on M6, its acceptance and decode match the previous weights' layout (37.5 against 37.6 tok/s). On the M5 Max, over the same 24 prompts, release 0.2 decodes at 16.6 tok/s against 17.2 for weights in the previous formats (-3.5%); there the token-mixer rotations cost 1.8 to 3.3% per chunk (1.2% on M6), and the bit width of the lookup tables does not change speed. The M5 Max's 64K evaluation gives perplexity 4.534 (M6 4.537) at 130 tok/s prefill.

64K on a 32 GB M6: the 64K evaluation runs without swapping; serving a 64K prompt with the drafter needs about 2 GB of swap headroom, as the previous packages do.

### Comparison with other quantizations of Qwen3.8-27B

Teacher-forced KL divergence against the BF16 model, the same evaluation for every model: 64 chats (40,023 positions; "dev", with the shared 45-token system prompt excluded unless marked full), held-out chats (37,076 positions), WikiText (16 x 1,024 tokens) and the last 2K tokens of 16 windows of 16,384 tokens from 12 long sessions. GGUF files were scored with llama.cpp on an M3 Ultra (the BF16 GGUF reproduces the PyTorch reference within 0.0004), Mirai S decoded to BF16 and scored in PyTorch, ANEMLL in PyTorch.

| Model | Size (GiB, bpw) | dev KL (excl. prompt) | held-out KL | dev KL, full | top-1 | p99 | WikiText KL | 16K windows, last 2K |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **ANEMLL release 0.2 (this)** | 9.44 (3.17)* | **0.0536** | **0.0565** | **0.0515** | **91.9%**† | **0.50**† | 0.0707 | **0.762** |
| [Unsloth UD-IQ3_XXS](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | 9.46 (3.17) | 0.0619 | 0.0651 | 0.1625 | 89.4% | 3.34 | **0.0506** | 1.108 |
| [DT-IQ3_XXS](https://huggingface.co/drawthingsai/Qwen3.8-27B-GGUF) | 8.94 (3.00) | 0.0789 | 0.0833 | 0.0763 | 90.4% | 0.75 | 0.1246 | 1.204 |
| Mirai S (2.4-bit trellis) | 7.56 (2.54) | 0.1226 | 0.1279 | 0.2716 | 85.1% | 5.73 | 0.1674 | not run |
| ANEMLL release 1 (previous packages) | 9.07 (3.04)* | 0.1512 | 0.1617 | 0.1852 | 86.1% | 2.25 | not run | 1.514 |

\* Size counts the quantized linear weights and the head, excluding the embedding (and MTP for the GGUF files); ANEMLL adds 0.30 GiB of rank-64 FP16 factors. † From the ANE run of the same trace.

At the same size as UD-IQ3_XXS, release 0.2 is 13% closer to BF16 on the chats, 31% on the 16K windows and has a far smaller worst-case tail; UD-IQ3_XXS is closer on WikiText. UD-IQ3_XXS and Mirai S carry most of their full-trace KL on the shared system prompt, which BF16 predicts almost deterministically. KL measures closeness to BF16 on these texts; it is not a coding, reasoning or retrieval benchmark, and the GGUF models run on the GPU, not the ANE.

## 8-bit attention for M6 and M5 (previous update)

The 16 full-attention layers now run most of their history attention in 8-bit, with every 8-bit operand as an explicit quantize / dequantize pair so the ANE compiler fuses it:

- **M6 functions:** scores in INT8 (step 1/4) and the softmax probabilities, softmax sum and PV probabilities in FP8 e4m3 (scale 1/64). These are attention activations computed at run time; the model weights are the same in both function sets.
- **M5 functions** (`<entry>_m5`, mapped in the manifest's `entries_by_soc`): the same INT8 scores, softmax and PV in FP16. The M5 ANE compiler rejects FP8: the M6 functions would fail to compile there and run on the GPU at about 2 s per call.
- **Transposed key cache:** keys are stored as (KV head, head dimension, token), the operand QK reads, so the ANE no longer transposes every key tile before QK.

Both function sets share the weights, so the packages are no larger than before. Core AI compiles a package as a whole, so on first start an M5 derives its own build once: each chunk with only its M5 functions, written to the Forge state folder (the download is not modified; about 20 seconds on an M5 Max and one more copy of the chunks, about 10 GB, on disk), then compiled as usual. An M6 uses the packages as they are.

Quality on M6 (M6 functions), against the previous V8 packages: KL-512 to BF16 **0.1838** (previous 0.1842), direct KL 0.0004 with the same top token at 99.3% of positions; 64K perplexity **5.0685** (5.0673), 8K perplexity 8.0957 (8.0952). The M5 functions compute what a build with those forms computes on the host; their device quality has not been measured separately.

Full server, the same synthetic coding workload, greedy, thinking off, 256 tokens, DFlash2:

| Context | M6: prefill / decode tok/s | M5 Max: prefill / decode tok/s (previous packages) |
| --- | --- | --- |
| 8K | 306 / 60.8 | 172 / 28.8 (179 / 28.8) |
| 16K | 297 / 60.3 | 171 / 27.8 (172 / 27.7) |
| 32K | 275 / 54.5 | 156 / 24.5 (151 / 24.9) |
| 48K | 253 / 46.1 | 143 / 24.1 (136 / 23.7) |
| 64K | 237 / 47.9 | 132 / 22.6 (124 / 22.1) |

The M6 numbers come from the same build with only the M6 functions (identical weights and M6 functions); the M5 Max numbers from these packages, derived on the M5 Max, which had other applications running. Against the previous packages this run measured prefill -4% at 8K to +6.5% at 64K and decode 0 to +2%; an earlier run of the same M5 functions (built as a separate package) measured prefill +1.2 to +7.4% and decode +3.7 to +8.2%. One workload with three decode repeats is not a general benchmark. On M6 the 8-bit attention used about 30% less whole-machine energy per prompt token than a GPU runtime on the same Mac (details and limits in the source repository's research notes).

**First start on M6 compiles both function sets** of each package: about 46 minutes for release 0.2 (37 for the previous packages; an M6-only build about 23). On an M5 Max the derived M5 build compiled in about 28 minutes (previous packages 26).

## Faster exact graph (previous update)

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

The default Core AI download includes the paired drafter and selector assets. The runtime reads the V8 cache format from `coreai/manifest.json`. The full `quick-test` performs the one-time ANE compile (about 46 minutes on M6, above), so the server then starts in seconds. `CTX=64K` sets the growth cap; the runtime starts with a smaller entry and grows its KV state. For the FP16-cache fallback, download revision `cd7dfc605ccad091b961f7788939c30d01c3793e` instead and set `KV_CACHE_DTYPE=fp16` on the wrapper command, or use `--kv-cache-dtype fp16` with `forge.py serve`. `--plain` is a target-only diagnostic. The integrity-only check does not run inference; a short smoke generation does not establish sustained performance or broad quality.

## Attribution and licenses

- Target: [`Qwen/Qwen3.8-27B` at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0). Original Apache-2.0 [LICENSE](LICENSE) and ANEMLL-added [NOTICE](NOTICE) are included; [QWEN_SOURCE.json](QWEN_SOURCE.json) records source checks and remaining provenance limitations.
- Drafter: [`ProCreations/Ternary-Bonsai-2-27B-DFlash2` at `4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b`](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2/tree/4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b). Its original [LICENSE](drafter/LICENSE) and [NOTICE](drafter/NOTICE) remain unchanged.
- The drafter notice records donor [`z-lab/Qwen3.8-27B-DFlash2` at `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/tree/50307d4c4cde6860d4eee73e2547cd786fe8e8a4) and training target [`prism-ml/Ternary-Bonsai-2-27B-gguf` at `6ed5e12bf84b7a63069882c91dd9e9218647d17b`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/tree/6ed5e12bf84b7a63069882c91dd9e9218647d17b).

[MODIFICATIONS.md](MODIFICATIONS.md) describes the converted derivatives and V8 graph changes. The original BF16 drafter checkpoint was rehashed against the pinned upstream LFS digest, and compact selector tables retain identical tensor bytes. The release 0.2 drafter package was rebuilt from the published drafter GPTQ export with the release 0.2 LM head; the export's build inputs match the original drafter and target checkpoints exactly (see the quantized-weights repository).

**Model-derived assets follow their upstream Apache-2.0 licenses and notices. ANEMLL Forge source code and documentation are MIT-licensed.** Why Apache-2.0: these packages are converted from Apache-2.0 models (Qwen3.8-27B and the ProCreations drafter), so they keep that license; LICENSE, NOTICE and MODIFICATIONS.md carry the license, the upstream notices and what ANEMLL changed. The MIT license covers the ANEMLL Forge code, not the weights. Preserve the applicable model licenses, copyright and modification notices when redistributing derivatives.
