# Experimental V8 KV cache

This option keeps full-attention **keys in FP16** and stores historical **values in signed INT8**, with one FP16 absolute-maximum scale per token and KV head. It affects the 16 full-attention layers. GDN state, projection weights, the local causal block and newly emitted K/V rows retain their existing precision.

The [dated KV-cache quantization research trace](research/KV_CACHE_QUANTIZATION_2026-10-02.md) preserves the complete aggregate prefill, decode and KL results with their measurement boundaries and limitations. Raw prompts and traces are excluded.

A separate [M6 NIAH retrieval pilot](research/NIAH_PILOT_2026-10-03.md) reports four paired FP16/V8 placements through 64K context. All eight runs passed exact match. It is one trial per case, and the source artifacts remain on M6.

## Model update and compatibility

The selectable model update uses the same [Hugging Face repository](https://huggingface.co/anemll/anemll-forge-qwen3.8-27B/tree/main/coreai). Update both this runtime and the target bundle: the new chunks contain both cache formats, and `coreai/manifest.json` declares `format: selectable` with `default: v8`. It provides 8K/16K/32K/48K/64K entries and retains the matching head, tokenizer/embeddings and Core AI DFlash2 drafter. Earlier FP16-only revisions remain supported but cannot provide V8 through a flag. Download a different release into a fresh directory and keep its release inventory intact.

## Convert a matching target

Use the Core AI conversion environment described in [ENVIRONMENT.md](ENVIRONMENT.md), the original Qwen checkpoint and the same quantized export used for the target. A prepared inference bundle alone does not contain all conversion inputs.

```sh
MODEL="/path/to/Qwen3.8-27B" \
EXPORT_DIR="/path/to/mix25in_mixr_lr64mix" \
OUT="/path/to/new-coreai-build-parent" \
SILU=tanh MLP_SILU=tanh GDN_SQ=16 GDN_SV=64 MLP_DS=1 \
python coreai/qwen38_coreai_build.py all \
  --kv-cache-dtype v8 \
  --ctx 8192,16384,32768,49152,65536 \
  --pctx 8192,16384,32768,49152,65536
```

Since 3 October 2026 the builder also defaults to the faster exact graph (`GDN_FAST=1`, attention tiles `ATT_BLOCK=2048` / `ATT_BLOCK_PREFILL=4096`) and to `--kv-cache-dtype v8`; add `GDN_FAST=0 ATT_BLOCK=16384` to reproduce the graph measured on this page. See [M6 compute acceleration](research/M6_COMPUTE_ACCELERATION_2026-10-03.md). The destination is `<OUT>/<export-name>_kvv8`; use a new build parent. The builder writes standard `v8_<context>k` verification and `p64_<context>k` prefill entries and a manifest declaring their cache layout. Their new `vs<layer-index>` inputs carry the token/head scales. `v8_` in the entry name still means a block of eight verifier tokens; the manifest, not that name, identifies the cache precision. Prefer source `.aimodel` specialization when installed-OS and Xcode package versions differ.

To select either format from one export, replace `--kv-cache-dtype v8` with `--kv-cache-dtype both`. The destination then ends in `_kvselect`, and its startup default is V8. Each chunk contains FP16 and V8 functions sharing its weights; the manifest maps each layout onto its physical entry names. The server loads functions only for the selected format. Set `--kv-cache-dtype fp16` or `v8` at startup; `auto` uses the declared default. To retain an FP16 startup default in a new conversion, add `--kv-cache-default fp16`. Existing FP16-only bundles still select FP16 automatically. Changing formats requires restarting the process and rebuilding prompt/KV state. This is startup selection, not conversion of an existing cache during a request.

V8 uses the stable global exp/sum attention calculation at every context, with 16K history tiles. For a matched FP16 experiment, build separately with `--kv-cache-dtype fp16 --stable-attention`; its destination ends in `_stable`. A normal FP16 build retains the existing shorter-context softmax calculation. Compare that arithmetic change separately when measuring quantization quality.

## Start the server

```sh
python forge.py serve --runtime coreai \
  --model "$FORGE_BUNDLE/model" --build "$V8_BUILD" \
  --kv-cache-dtype v8 --ctx 65536 \
  --draft "$FORGE_BUNDLE/drafter/dflash2_lut4_gptq.aimodel"
```

The same DFlash2 drafter is used. The server requires the Swift bridge, verifies the requested format against the manifest before allocating models, and checks actual KV buffer dtypes and shapes while initializing each context. `--kv-cache-dtype auto` selects the declared layout; a manifest without `kv_cache` is treated as the existing FP16 format. There is no host expansion of the full INT8 V history to FP16. Native dequantization inside attention may still allocate temporary scratch; measure resident memory separately.

The wrapper accepts the same option through `KV_CACHE_DTYPE=v8` and validates it before `restart` stops a running instance. Select the matching target with `BUILD`. With a build outside the bundle, specify `DRAFT` explicitly as in the README. `/health` includes `kv_cache_dtype`, available `kv_cache_formats`, `active_context_entry` and the most recent prefill's new tokens, cached tokens, elapsed seconds and tokens/s. That prefill timer includes all target layers, the head, host cache writes, context transitions and drafter context ingestion; it excludes model loading, prompt rendering, HTTP transport and generated tokens. The separate `decode` record reports elapsed seconds, output tokens, tokens/s, speculative cycles, acceptance and phase times. `draft_accept_histogram[k]` counts cycles accepting exactly `k` draft tokens (0–7); `tokens_per_call` is null when no speculative cycle ran. Startup logs describe the selected K/V precision.

Across the 16 full-attention layers, the logical K/V/scale payload falls from 64 KiB to 48.125 KiB per history position. This saves about 0.99 GiB at 65,472 positions. These are cache payload figures; model weights, GDN state, output buffers and native scratch still contribute to resident memory.

## Commit, grow and restore

- After prefill, only its valid V rows are compressed. Keys are copied without quantization.
- After speculative verification, `accept(k)` compresses only the first `k` committed rows. Rejected or padded rows never enter the stored history. The zero-acceptance path writes nothing.
- Context growth copies the committed K prefix, V codes and FP16 scales into their corresponding new buffers and invalidates old binding plans. It preserves recurrent states and pending counts.
- Snapshot/restore retains the existing position-mask and GDN-state semantics. KV prefixes are not duplicated; a snapshot is valid only while its required prefix remains cached and has not been overwritten by a divergent continuation.

For history attention, `codes / 128` is native-dequantized and the dynamic `scale × 128` factor is applied to the unnormalized exp scores before PV. The softmax denominator uses the unscaled scores. This avoids multiplying every V channel by a dynamic scale. The identity is exact in real arithmetic; FP16 execution and quantization error require model evaluation.

## Validation scope

The isolated M6 attention-core sweep passed 8K/16K/32K/48K/64K with full ANE/noGPU cached placement and finite outputs. V8 reduced single-layer T64 latency 15–28% and K/V/scale input bytes 24.8%, with approximately 0.7% synthetic output relative RMSE. These measurements exclude projections, MLP, GDN, the head and drafter; do not report them as whole-model prefill or decode throughput.

Host tests cover accepted counts 0/1/7/8, partial prefill writes, exact key copies, untouched tails, zero-value scales, layout rejection, code/scale growth, binding invalidation and snapshot restore. An M6 smoke using the actual production Swift runtime and a real four-layer chunk also passed 64-row prefill, native T8 acceptance of 0/1/7/8 rows, 8K→16K→8K cache transitions and exact replay after restore. A separate matched-arithmetic four-layer pilot had identical first-block outputs and 0.0452% relative output RMSE after 512 tokens. These are narrow runtime and numerical checks, not full-model quality measurements.

The complete 64-layer target and the tested DFlash2 drafter also passed an M6 HTTP smoke: an 8,389-token public prompt grew from 8K to 16K, the identical request restored 8,382 cached tokens and returned the same greedy reply, and a subsequent short streaming request shrank back to 8K. End-of-prefill logits were finite. The cached V8 restart loaded in about one second in that bounded test; cold source specialization took about 20.6 minutes.

A short coding smoke also exercised multiple real speculative cycles with both formats. Each generated the same 71-token Python reply with 87% draft acceptance and valid syntax. Logged decode rates were 50.6 tokens/s for FP16 and 51.0 for V8. This used the 8K entry with a 16K cap; it is a serving smoke, not evidence of a decode speedup or general coding quality.

A separate V8 smoke with a 64,985-token public prompt reached the 64K entry and generated a valid 71-token Python reply. Its logged decode rate was 30.6 tokens/s, with 87% draft acceptance. The strict cached-graph audit passed after generation. This earlier smoke had no matched FP16 control; the paired decode measurements below use a different public prompt and generation budget.

## Whole-server prefill measurements

On M6, macOS 27.0.1 build 26A434, the complete target and DFlash2 passed ten paired HTTP cases on 1 October 2026. Both modes used identical quantized weights, drafter, prompt token IDs and the five-step 8K/16K/32K/48K/64K ladder. Each measured a cold public synthetic prompt after a distinct tiny warmup. The timer covers all 64 target layers, projections, MLP, GDN, head, host cache writes, context growth, snapshots and DFlash2 context ingestion. Loading, prompt rendering, HTTP transport and generated tokens are excluded.

Measured prefill throughput, **FP16 V → V8**:

- 8K entry, 8,048 prompt tokens: **181.8 → 181.0 tokens/s** (−0.4%).
- 16K entry, 16,240 tokens: **157.8 → 161.1 tokens/s** (+2.1%).
- 32K entry, 32,624 tokens: **126.5 → 144.5 tokens/s** (+14.2%).
- 48K entry, 49,008 tokens: **107.4 → 128.5 tokens/s** (+19.7%).
- 64K entry, 65,328 tokens: **99.8 → 111.5 tokens/s** (+11.8%).

All cases had finite end-of-prefill logits and passed strict cached-graph checks for all 16 chunks, the head and drafter: expected ANE regions, no GPU regions and compile mode 2. Physical cluster allocation was not independently measured.

These are single trials without confidence intervals, and they measure the overall option. At short contexts, the existing FP16 export uses softmax while V8 uses stable global exp/sum. Experimental V8 packages also contained five unused FP16 prefill controls (15 physical entries versus 12 in the FP16 source); both servers bound ten functions. Extra entries can affect program memory and scratch. Resident memory must be checked separately with a V8-only export; the logical cache saving is not a measured total-memory saving.

## Whole-server decode measurements

On 2 October 2026, the full 64-layer target and tested Core AI DFlash2 drafter completed all ten FP16/V8 cases on the same M6 and OS as the prefill sweep. Both formats selected functions from the same shared-weight `_kvselect` packages. Each context pair used identical public synthetic testing notes followed by a small Python coding task, prompt token IDs, quantized weights, drafter and inference settings. Decoding was greedy with thinking disabled, a 256-token generation cap and a 3 ms draft gap.

Each mode/context ran one cold prompt prefill followed by three identical cached requests. The figures below are medians of the three cached decode rates. All requests generated 256 tokens; the cold generation is excluded. The monotonic server timer includes the drafter, all target layers, verification, sampling, cache writes and other generation work. It excludes prefill, model loading and HTTP transport.

Measured decode throughput, **FP16 V → V8**:

- 8K entry, 7,673 prompt tokens: **57.17 → 55.37 tokens/s** (−3.14%).
- 16K entry, 15,865 tokens: **47.43 → 51.99 tokens/s** (+9.62%).
- 32K entry, 32,249 tokens: **42.21 → 48.00 tokens/s** (+13.72%).
- 48K entry, 48,633 tokens: **37.13 → 39.55 tokens/s** (+6.51%).
- 64K entry, 64,953 tokens: **31.38 → 38.86 tokens/s** (+23.87%).

The generated reply was identical between formats at 8K, 16K, 32K and 64K; replies differed at 48K. Draft acceptance, FP16/V8, was **88.49%/88.49%, 85.71%/91.43%, 88.49%/88.49%, 88.49%/84.56%, and 91.43%/91.43%**, respectively. This measures the overall serving option, including acceptance and output differences; it does not isolate cache bandwidth. Within each mode, all four requests had identical reply hashes. All cases passed finite-logit, cache-reuse and strict selected-function cached-placement checks for all 18 packages with compile mode 2; every owned test server exited.

These are three adjacent repeats on one synthetic workload per context, not a general performance estimate or confidence interval. At 8K the repeat ranges overlap: FP16 52.09–58.83 tokens/s and V8 55.24–55.51. The short-context attention-arithmetic difference remains, and identical generated text does not establish identical logits or broader quality. The 256-token cap can truncate the coding response; no capability score is reported. Physical cluster allocation, native scratch, long-run latency and total resident memory remain unmeasured. The initial audit-manifest error was corrected without rerunning its completed inference measurements; the original failure evidence is retained outside the repository.

## Compiled-model KL-512

Both complete target exports were evaluated on M6 using compiled Core AI T8 teacher forcing against a byte-verified cached BF16 reference from M3U. The trace contains 40,023 next-token positions across 64 sequences; its longest sequence is 857 tokens. Scoring uses the teacher's top-512 tokens plus one aggregate tail bucket, includes prompt and generated positions, and weights positions equally. Log-softmax and KL reductions use FP64; cached teacher log-probabilities are FP32.

Measured **FP16 V → V8**:

- Mean KL: **0.184625 → 0.184270 nats** (−0.192%).
- Median / p99 KL: **0.030304 / 2.286022 → 0.030238 / 2.274433**.
- Top-1 agreement with BF16: **86.0455% → 86.0205%** (−0.0250 percentage points).
- Perplexity: **2.414751 → 2.412098**; reference perplexity **2.135539**.

Direct **KL(FP16 target ∥ V8 target)** on the same teacher top-512 partition plus tail has mean **0.0000737**, median **0.0000102** and p99 **0.000941 nats**. This checks distribution changes directly rather than relying only on cancellation in the two averages against BF16. It still includes the attention-arithmetic change and groups all other vocabulary tokens into a single tail bucket.

Both passes completed all positions and passed strict target/head cached-placement checks. These results cover the prototype V8 export and stock FP16 export. The small difference does not establish an accuracy improvement from compression: the short-context attention arithmetic also changes. This trace does not test long-context quality or general capabilities and does not match the Mirai public scoring protocol.

Both formats of the final shared-weight selectable export were also checked against their measured prototype on the first three teacher-forced sequences, covering 2,299 positions. The teacher top-512 partition log-probability arrays and KL values were bit-exact in this check. The complete 64-sequence KL evaluation was not repeated on the final export, and this check does not validate long-context quality.

Long-context retrieval, representative speculative acceptance and resident memory must still be evaluated before calling this a validated release option. Benchmark artifacts remain outside the source repository.
