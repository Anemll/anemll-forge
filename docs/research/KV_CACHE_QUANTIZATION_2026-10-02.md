# KV-cache quantization research trace: V8 on Apple M6

**Model:** Qwen3.8-27B, mixed-bit quantized text-inference target  
**Experiment dates:** 1–2 October 2026  
**Report date:** 2 October 2026  
**Status:** Experimental research results; full-server measurements and bounded numerical validation complete.

## Research record and implementation

This dated record preserves the aggregate results of the 1–2 October 2026 V-only INT8 investigation. It distinguishes the earlier prototype prefill/KL measurements from the final selectable-model decode measurements and records validation failures that were repaired. It contains no decoded prompts, token traces or machine-specific paths.

The implementation and usage guide are [V8 KV cache](../KV_CACHE_V8.md), [host cache quantization](../../scripts/qwen38_kv_cache.py), [Core AI runtime](../../scripts/qwen38_coreai_model.py), and [Core AI conversion](../../coreai/qwen38_coreai_build.py). The [README model update instructions](../../README.md#2-download-the-model-and-drafter) explain the required runtime and bundle update.

The selected V8 startup default was applied after evaluation. Measurements explicitly selected each mode; changing the manifest default did not change the tested model programs. The default reflects a release choice, not broader quality validation.

## Main findings

V-only INT8 history compression is implemented in the Core AI inference server with the tested Core AI DFlash2 speculative drafter. Keys remain FP16.

- At the **64K context entry**, full-server decode throughput increased from **31.38 to 38.86 tokens/s (+23.87%)**. Generated replies and draft acceptance matched between formats in that case.
- The earlier complete-model prefill sweep measured **99.78 to 111.52 tokens/s (+11.77%)** at 64K, with its largest observed gain at 48K: **+19.65%**.
- Logical persistent K/V/scale payload is **24.80% smaller**. Total resident-memory savings have not been measured.
- On a short 64-sequence quality trace, mean KL changed from **0.184625 to 0.184270 nats (−0.192%)**. This supports closely preserved distributional fidelity on that trace; it does not establish a quality improvement from INT8 or validate long-context capabilities.

These are measurements of the overall serving option. They include attention-arithmetic changes and, for decode, speculative acceptance and response differences.

## What “FP16” and “V8” mean here

Both target modes use the **same mixed-bit quantized model weights** and the same DFlash2 drafter. “FP16” in the tables refers to the **KV-cache baseline**, not an original full-precision target model. The original BF16 checkpoint supplies the separate reference distributions for KL evaluation.

| Component | FP16 cache baseline | V8 option |
| --- | --- | --- |
| Historical keys | FP16 | FP16 |
| Historical values | FP16 | Signed INT8 |
| Value scales | None | FP16, per token and KV head |
| Newly emitted K/V and local causal block | Existing FP16 path | Existing FP16 path |
| GDN recurrent state and projection weights | Unchanged | Unchanged |
| Drafter | Tested Core AI DFlash2 | Same drafter |

Qwen3.8-27B has 64 language-model layers, including 16 full-attention layers with four KV heads and head dimension 256. The other 48 layers use Gated DeltaNet. V8 affects the history buffers of those 16 full-attention layers. [Upstream Qwen model card](https://huggingface.co/Qwen/Qwen3.8-27B).

### Data path

The host compresses only valid prefill rows or accepted speculative V rows. It uses symmetric INT8 codes in the range −127 to 127 and an absolute-maximum scale per token/head; quantization uses the scale actually stored in FP16. Keys are copied without quantization. Rejected speculative rows do not enter committed history.

ANE attention consumes the INT8 history and FP16 scales. Native dequantization produces `codes / 128`; the dynamic `scale × 128` factor is applied to the unnormalized attention scores before the value projection, while the denominator remains unscaled. This avoids a host-side expansion of the full V history. Native scratch and fusion behavior still require measurement.

Context growth copies keys, V codes and scales together. Snapshot/restore retains position-mask and recurrent-state semantics. Compression is selected at process startup; switching formats requires a restart and an empty prompt/KV cache.

## Test conditions and measurement boundaries

| Setting | Recorded configuration |
| --- | --- |
| Hardware | Apple M6 |
| Operating system | macOS 27.0.1, build 26A434 |
| Target | Qwen3.8-27B, `mix25in_mixr_lr64mix` quantized export |
| Runtime | Core AI with Swift bridge |
| Serving scope | All 64 target layers, output head and Core AI DFlash2 |
| Context ladder | 8,192 / 16,384 / 32,768 / 49,152 / 65,536 |
| Decode settings | Greedy; thinking disabled; 256-token generation cap |
| Drafter pacing | 3 ms draft gap |
| ANE configuration | Bonded compile mode 2; strict cached-graph checks |
| Prefill trials | One measured cold prompt per mode/context after a distinct tiny warmup |
| Decode trials | One cold prompt request, then three cached requests per mode/context |

The prefill sweep used the original FP16-cache export and a prototype V8 export. The later decode matrix used **one final shared-weight export containing both formats**, selected at startup. These are separate experiments with different prompts; the tables should not be combined into a single end-to-end speedup.

V8 uses stable global exp/sum attention with 16K history tiles at every context. The normal FP16 export retains its shorter-context softmax calculation. A matched FP16 build with the same stable calculation is needed to isolate the effect of compression alone.

Throughput change is `100 × (V8 tokens/s ÷ FP16 tokens/s − 1)`. It is not the same percentage as latency reduction.

## Full-server prefill results

| Context entry | Prompt tokens | FP16 V (tok/s) | V8 (tok/s) | FP16 time (s) | V8 time (s) | Throughput change |
| --- | --- | --- | --- | --- | --- | --- |
| 8K | 8,048 | 181.79 | 181.00 | 44.27 | 44.46 | -0.43% |
| 16K | 16,240 | 157.78 | 161.09 | 102.93 | 100.81 | +2.10% |
| 32K | 32,624 | 126.53 | 144.52 | 257.83 | 225.74 | +14.22% |
| 48K | 49,008 | 107.39 | 128.50 | 456.34 | 381.39 | +19.65% |
| 64K | 65,328 | 99.78 | 111.52 | 654.72 | 585.79 | +11.77% |

**Timing scope:** The server prefill timer covers all target layers, projections, MLP, GDN, output head, host KV writes, context transitions, snapshots and DFlash2 context ingestion. Model loading, prompt rendering, HTTP transport and generated tokens are excluded. Each pair used identical prompt token IDs.

The V8 prototype packages contained five unused FP16 prefill-control entries: 15 physical entries versus 12 in the original FP16 source. Both servers bound ten functions on the same context ladder. Extra physical entries can affect program memory and scratch, so this sweep is not a clean measurement of compression alone or total-memory savings.

These are single-trial observations with no confidence intervals. The −0.43% result at 8K should not be treated as an established performance regression.

## Full-server decode results

| Context entry | Prompt tokens | FP16 V (tok/s) | V8 (tok/s) | Throughput change | Same generated reply |
| --- | --- | --- | --- | --- | --- |
| 8K | 7,673 | 57.17 | 55.37 | -3.14% | Yes |
| 16K | 15,865 | 47.43 | 51.99 | +9.62% | Yes |
| 32K | 32,249 | 42.21 | 48.00 | +13.72% | Yes |
| 48K | 48,633 | 37.13 | 39.55 | +6.51% | No |
| 64K | 64,953 | 31.38 | 38.86 | +23.87% | Yes |

**Timing scope:** Each number is the median decode throughput of three cached requests after one cold prefill. The server's monotonic decode timer includes the drafter, all target layers, verification, sampling, host cache work and detokenization during generation. It excludes prefill, loading and HTTP transport.

Both formats used the same final target packages, quantized weights, prompt token IDs, drafter and sampling settings. Every measured request generated 256 tokens. The 256-token cap may truncate the coding response; this experiment reports throughput, not a coding-task success score.

### Speculative acceptance and repeat ranges

| Context entry | FP16 draft acceptance | V8 draft acceptance | FP16 repeat range (tok/s) | V8 repeat range (tok/s) |
| --- | --- | --- | --- | --- |
| 8K | 88.49% | 88.49% | 52.09–58.83 | 55.24–55.51 |
| 16K | 85.71% | 91.43% | 45.80–48.58 | 50.80–52.41 |
| 32K | 88.49% | 88.49% | 40.59–42.28 | 46.86–48.05 |
| 48K | 88.49% | 84.56% | 33.93–38.02 | 38.91–41.03 |
| 64K | 91.43% | 91.43% | 30.87–32.77 | 38.82–38.89 |

Within each mode, the cold request and all three cached requests produced identical reply hashes. Between formats, replies matched at 8K, 16K, 32K and 64K; they differed at 48K.

The 16K gain includes increased draft acceptance. At 48K, both response and acceptance differ, so its percentage cannot isolate cache bandwidth. At 32K and 64K, reply equality and matching acceptance provide stronger controls for the overall serving comparison, but do not prove identical logits or establish the mechanism of acceleration.

At 8K, the repeat ranges overlap. Three adjacent repeats on one synthetic coding workload per context do not establish general performance or statistical significance.

## KL-512 quality results

The compiled target models were teacher-forced against a cached, byte-verified original **BF16 reference** generated on a separate machine. The reference and ANE target did not need to be resident simultaneously.

The trace covers **40,023 next-token positions in 64 sequences**; the longest sequence is **857 tokens**. Scoring uses the teacher's top-512 tokens plus one aggregate tail bucket, with a floor of `1e-12`. It includes prompt and generated positions and weights positions equally. Log-softmax and KL reductions use FP64; cached teacher log-probabilities are FP32.

| Metric | FP16 V | V8 | Change |
| --- | --- | --- | --- |
| Mean KL (nats) | 0.184625 | 0.184270 | -0.192% |
| Median KL (nats) | 0.030304 | 0.030238 | -0.000066 |
| p99 KL (nats) | 2.286022 | 2.274433 | -0.011589 |
| Top-1 agreement with BF16 | 86.0455% | 86.0205% | -0.0250 percentage points |
| Trace perplexity | 2.414751 | 2.412098 | -0.110% |

Reference trace perplexity: **2.135539**.

Direct **KL(FP16-cache target || V8 target)** on the same teacher top-512 partition plus tail was:

- Mean: **0.00007371 nats**
- Median: **0.00001023 nats**
- p99: **0.00094076 nats**

This direct comparison helps distinguish a small actual distribution change from coincidentally similar average distances to BF16. It still includes the attention-arithmetic change, and the grouped tail is not a full-vocabulary comparison.

The slightly lower mean KL does **not** establish that INT8 improves the model. Top-1 agreement decreased by **0.0250 percentage points**, and the arithmetic differs between settings. KL measures distributional fidelity, not coding, reasoning, factual correctness or retrieval quality. This trace does not implement the Mirai public scoring protocol and does not test long contexts.

The complete KL passes used the prototype V8 and original FP16 exports. Both formats of the final shared export were separately compared against those measured prototypes on **2,299 positions in the first three sequences**: teacher top-512 partition log-probability arrays and KL values were bit-exact. The complete 64-sequence KL pass was not repeated on the final export.

## Cache payload and model packaging

Across the 16 full-attention layers:

| Cache format | Persistent K/V/scale bytes per history position | Payload at 65,472 positions |
| --- | --- | --- |
| FP16 K + FP16 V | 64 KiB | 3.996 GiB |
| FP16 K + INT8 V + FP16 scales | 48.125 KiB | 3.005 GiB |
| Difference | 15.875 KiB, or 24.80% | Approximately 0.991 GiB |

These figures describe the logical cache payload. They exclude weights, recurrent state, output buffers, snapshots where applicable, native scratch, compiler allocations and system wired memory. They are **not** measured reductions in total memory pressure and do not establish leak freedom.

The final selectable target consists of **16 four-layer chunks plus an output head**. Each target chunk contains 20 physical functions: five context sizes × prefill/verification × two cache formats. At startup the server loads the ten functions for the selected format. Shared on-disk weights do not prove shared native resident allocations.

The runtime flag alone cannot enable V8 in a package exported only for FP16 KV. Matching V8 or selectable target packages are required.

## Runtime validation completed

- Both selectors passed native cache append checks, including accepted counts 0, 1, 7 and 8; rejected tails remained untouched.
- Native growth and restore checks passed 8K → 16K → 8K with exact K-prefix copies and exact replay in the bounded state test.
- Full-target HTTP generation, cache restore and streaming after shrink passed with the tested Core AI DFlash2.
- All ten decode cases passed finite-logit, cache-reuse and within-mode reply-equality checks.
- Strict selected-function cached-graph audits passed for all 18 target/head/drafter packages, including expected ANE regions, no GPU regions and compile mode 2.
- Every owned test server exited; the final 17 target/head packages have a per-file SHA-256 inventory.
- Both final formats passed the three-sequence numerical control described above.

Cached placement and compile mode are configuration/compiler evidence. Physical bonded-cluster allocation was not independently measured. This experiment also does not establish integer attention MAC execution.

An audit-manifest error and a final report-serialization error were corrected without altering inference settings or rerunning the completed decode measurements. Original failure records and measurements were preserved.

## Using the selectable export

With a matching build, tokenizer/model assets, the tested drafter and the Swift bridge installed, select V8 at startup:

```sh
python forge.py serve --runtime coreai \
  --model /path/to/model-assets \
  --build /path/to/shared-kvselect-build \
  --kv-cache-dtype v8 --ctx 65536 \
  --draft /path/to/tested-dflash2.aimodel
```

Use `--kv-cache-dtype fp16` for the baseline, or `auto` for the build's declared default. The wrapper equivalent is `KV_CACHE_DTYPE=v8` with `BUILD` pointing to the matching target. Restarting discards current prompt/KV state.

To build shared packages, the converter supports `--kv-cache-dtype both`, which now declares V8 as the startup default; `--kv-cache-default fp16` retains an FP16 default when wanted. The tested local export's default was switched to V8 after evaluation without changing its model packages. Benchmark cases explicitly selected each format, so their measurements are unchanged. A separate FP16 stable-attention control can be exported with `--kv-cache-dtype fp16 --stable-attention`.

## Remaining work

1. Compare V8 against FP16 with matching stable attention to separate arithmetic and compression effects.
2. Evaluate long-context retrieval and representative coding/reasoning tasks with matched BF16 and ANE settings.
3. Measure ANE/native resident allocations, scratch, memory pressure and repeated growth/restore behavior.
4. Repeat serving measurements with more prompts, randomized ordering and longer runs. Complete comparable full-server measurements on other supported hardware before extending these results.

## Attribution and evidence

**Original model:** Qwen3.8-27B, developed by the Qwen Team; original model copyright belongs to Alibaba Cloud. The upstream model is Apache 2.0 licensed. ANEMLL provides independent quantization, conversion and Apple Neural Engine inference research. [Qwen source model](https://huggingface.co/Qwen/Qwen3.8-27B).

**Drafter source:** The tested Core AI DFlash2 pairing is derived from the separately attributed ProCreations checkpoint. Its lineage is separate from the target model. [ProCreations/Ternary-Bonsai-2-27B-DFlash2](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2).

All performance and quality figures above come from this experiment's completed ledgers, not the upstream model cards or vendor benchmark claims. The report includes aggregate results only. Local paths, machine hostnames, credentials, decoded prompts and raw traces are excluded.

Underlying ledgers, model inventories and the quality trace are retained by the experiment owner. They are not attached here, so this document alone does not reproduce the exact quality trace or constitute a public benchmark harness.
