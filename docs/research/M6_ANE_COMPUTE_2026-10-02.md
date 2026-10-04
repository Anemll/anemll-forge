# M6 ANE compute: V8 attention dtypes and ~2x options

- **Scope:** Apple M6 Neural Engine / Core ML / Core AI compute in ANEMLL Forge, with a ~2x compute-throughput target versus the current baseline path.
- **Baseline:** the current Core AI serving path with mixed-bit LUT weights and selectable V8 KV (FP16 keys, INT8 values). Not an unquantized BF16 model.
- **Status:** research notes plus a host-side prototype. This checkout is a Linux VM without an M6, Core ML, Core AI, or model weights. No ANE timing was collected here.
- **Hypothesis (non-binding):** the first high-leverage target is V8 attention, because attention matmuls may still run in a wider dtype than the stored weights or V codes.

Evidence labels used below:

- **Source-verified:** confirmed by reading Forge code or checked-in docs in this checkout.
- **Measured (prior):** wall-clock or quality numbers already recorded in this repository from earlier M6/M5 work. Not rerun here.
- **Inferred:** architectural or arithmetic reasoning. Not a device measurement.
- **External doc:** Apple or coremltools public documentation. Not an M6 ANE result for this model.

Do not read any speedup rank as a measured 2x.

## What the current path actually executes

Qwen3.8-27B in Forge is 64 layers: 48 Gated DeltaNet (linear attention) and 16 gated full-attention layers (every fourth layer). Hidden 5120, 24 query / 4 KV heads, head dim 256. [QUANTIZATION_NOTES.md](../../QUANTIZATION_NOTES.md), [upstream Qwen card](https://huggingface.co/Qwen/Qwen3.8-27B).

Approximate weight traffic per generated token, from the imported notes: MLP 17.1B, DeltaNet projections 5.5B, attention projections 1.5B, head 1.27B. **Source-verified** as the documented breakdown. That mix is why decode is described as weight-bandwidth bound once weights are already at about 2 bits.

### Weights versus attention math

| Tensor | Storage on the deployed path | Compute seen in the graph | Evidence |
| --- | --- | --- | --- |
| MLP / most mixers | vector 2x16 or LUT4 + per-channel scale | 1x1 conv after LUT or dense reconstruct | Source-verified ([`QConv`](../../coreai/qwen38_coreai_build.py), [QUANTIZATION_NOTES.md](../../QUANTIZATION_NOTES.md)) |
| Attention Q/O projections | LUT4 + pcs | same conv path | Source-verified |
| Attention K/V projections | INT8 per-channel | dequantized to FP16, then conv | Source-verified (`QConv` INT8 branch multiplies codes by scale into FP16) |
| Full-attention Q and K | FP16 activations | FP16 matmul QK | Source-verified (`AttnW.forward`: `qg4 @ k_st.transpose`) |
| Full-attention V (FP16 cache) | FP16 | FP16 matmul PV | Source-verified |
| Full-attention V (V8 cache) | signed INT8 + FP16 token/head scale | `coreai.dequantize(..., 1/128)` to FP16, then FP16 matmul PV | Source-verified ([`AttnW.forward`](../../coreai/qwen38_coreai_build.py) lines 330-354) |
| GDN recurrent state | FP16 host I/O | FP16 matmuls / forward substitution | Source-verified |

The V8 graph does **not** keep INT8 codes in the PV matmul. It dequantizes historical V to FP16, then applies `scale * 128` to the unnormalized exp scores so the softmax denominator stays unscaled. [KV_CACHE_V8.md](../KV_CACHE_V8.md) states that this identity is exact in real arithmetic. The same note and the dated V8 trace say the experiment **does not establish integer attention MAC execution**. **Source-verified.**

So the hypothesis is right at the graph level: stored V codes are INT8, and stored weights are LUT/INT8, but the attention matmuls in the Core AI program are written as FP16. Whether the M6 ANE compiler later fuses that dequant into a native INT8 or FP8 MAC is **unmeasured** on this model.

### Bonded versus non-bonded, and FP8

Forge's bonded flag is `MPSGRAPH_ANE_BONDED_COMPILE_MODE`. Mode 2 keeps the bonded procedure variant and is the Core AI default. On one 4-layer chunk at 24K, mode 0 and mode 2 had essentially the same verify/prefill times (7.50 / 23.56 ms versus 7.46 / 23.52 ms); mode 2 used less wired memory. **Measured (prior)** in [COREAI_PORT_NOTES.md](../../COREAI_PORT_NOTES.md). Mode 2 versus 0 is not a 2x compute switch.

Apple's AFM package is described as shipping bonded and non-bonded variants of every procedure. That is a compile/cluster choice, not an FP8 dtype. This checkout has **no** public Apple document that names a distinct "bonded FP8" versus "non-bonded FP8" compute format. Treat "bonded FP8" as unconfirmed naming unless an M6 specialization dump shows it.

FP8 in Forge today is **E4M3 storage** for weight or LUT codes:

- ANE weight convention: codes clipped to 240 (IEEE-style E4M3 max finite). **Source-verified** ([`qwen3_lut_common.py`](../../scripts/qwen3_lut_common.py), [`lut_fp8_coreml.py`](../../scripts/lut_fp8_coreml.py)).
- Optional `fp8x448` probe uses the full E4M3FN max of 448.
- [VECTOR_LUT_README.md](../history/VECTOR_LUT_README.md): "The compiler turns the small FP8 table into fp16 at compile time." **Measured (prior)** as same LUT speed regardless of LUT value type (INT8 / FP16 / FP8) on M6.
- Core AI export notes: no FP8 / INT8 LUT values; INT8 / FP8 per-channel (non-LUT) weights do export. **Source-verified** ([COREAI_PORT_NOTES.md](../../COREAI_PORT_NOTES.md), [QUANTIZATION_NOTES.md](../../QUANTIZATION_NOTES.md)).
- Open item already recorded: vector-LUT weights combined with FP8 **activations**. **Source-verified** as still open.

INT8 dense 1x1 convs on M6 were about 2.13x faster per layer than FP16 dense in a weight-bandwidth probe (0.218 to 0.103 ms/layer at 4096). **Measured (prior)** in [RESULTS_M6_INT8.md](../../RESULTS_M6_INT8.md). That is storage/bandwidth, not a proof of INT8-INT8 MAC. The same file: LUT value dtype does not change M6 speed.

### Apple's stated INT8-INT8 path

coremltools documents W8A8 activation quantization (iOS 17 / macOS 14+) and says that on A17 Pro / M4-class Neural Engine, quantizing both weights and activations to INT8 can use optimized INT8-INT8 compute. Sources:

- [Quantization overview](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-overview.html)
- [Quantization performance](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html)
- [API overview](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-api.html) (`linear_quantize_activations` + `linear_quantize_weights`)

Those pages show ResNet50-class vision results, not Qwen attention, and they do not mention M6. Apple also recommends activation quantization only when the graph stays on the Neural Engine. **External doc.** Whether M6 ANE attention matmuls actually run INT8-INT8, and whether Core AI `dequantize` + FP16 `matmul` lowers to that path, must be measured on device.

M5/M6 GPU TensorOps INT8 matmul (Metal cooperative tensors) is a different unit from the ANE. Forge's release path is ANE / Core AI, compile mode 2, cached placement with no GPU regions. **Source-verified.** Do not treat GPU INT8 TensorOps as the baseline path.

## Does V8 attention look like the 2x lever?

Prior V8 serving measurements on M6 (same mixed-bit weights, Core AI + DFlash2):

- Decode at the 64K entry: 31.38 to 38.86 tokens/s (+23.87%).
- Prefill at 64K: 99.78 to 111.52 tokens/s (+11.77%); largest prefill gain +19.65% at 48K.
- At 8K decode the change was -3.14% with overlapping repeats.

**Measured (prior)** in [KV_CACHE_QUANTIZATION_2026-10-02.md](KV_CACHE_QUANTIZATION_2026-10-02.md). Those runs include attention-arithmetic changes and, for decode, speculative acceptance. They are not an isolated compute-dtype experiment. Isolated attention-core T64 latency was reported 15-28% lower with V8, still excluding MLP/GDN/head. **Measured (prior)** in [KV_CACHE_V8.md](../KV_CACHE_V8.md).

**Inferred** from that evidence:

- Long-context attention/KV cost is real and growing. V8 already bought bandwidth, not a new MAC dtype.
- A further 2x on **attention compute only** would not automatically 2x end-to-end decode, because most bytes per token are still LUT weights (MLP + GDN).
- Prefill is more compute-heavy than decode in the imported notes. Attention-dtype work is more plausible there, and at 32K-64K decode where V8 already moved the needle.
- The highest-leverage *next* experiment is still the V8 attention kernel: it is the only full-attention path, Forge already inserts `coreai.dequantize`, and Apple documents an INT8-INT8 NE path that this graph may not be using.

The hypothesis is kept as the first experiment, not as a promised 2x.

## Ranked options

Expected speedup is qualitative (high / medium / low / none) for **compute throughput of the targeted kernel**, then a separate note on end-to-end likelihood. Quality risk is for Qwen3.8-27B serving, not ImageNet.

| Rank | Option | Expected kernel speedup | End-to-end ~2x? | Quality risk | Forge hook | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | INT8-INT8 attention compute (Q and K, and PV, both operands + accum), not storage-only V | **High if** M6 ANE keeps an INT8 MAC; **none/low** if it dequants to FP16 first | Unlikely for decode at short context; possible large fraction of the long-context attention slope | Medium (scores and PV). May need calibration ranges | V8 dequant + `--kv-cache-dtype`; no Q/K compute-dtype flag yet. Prototype: `scripts/m6_attn_compute.py` | Graph: source-verified FP16 after dequant. INT8-INT8 on M6 attn: unmeasured. Apple W8A8: external doc (M4-class) |
| 2 | FP8-FP8 attention (E4M3 both operands; check 240 vs 448, accum dtype) | **High if** a native FP8 MAC exists; **none** if the compiler casts LUT/acts to FP16 as it does for FP8 LUT values | Same caveat as row 1 | Medium-high. Narrow range, softmax intermediates | Weight FP8 recipes and `lut_fp8_coreml.py`; no activation-FP8 attention hook. Open item in the vector-LUT notes | LUT FP8 -> fp16 at compile: measured (prior). FP8 activations: open / inferred |
| 3 | INT8 keys (K8) plus current V8, still FP16 QK unless Q is quantized too | Medium as **bandwidth** (another ~25% logical KV payload if scales stay token/head); low as compute unless QK also changes dtype | Complements V8; V8 itself was +24% decode at 64K, not 2x | Medium-high. Keys enter scores | Host `quantize_values` is V-only; keys stay FP16 by contract | V8 payload 24.80% smaller: measured (prior) arithmetic on logical bytes. K8 unmeasured |
| 4 | Fused / blocked attention (`sdpa`, flash-style tiles already used above 16K) | Low-medium (less traffic / better fusion), not a 2x MAC | Helps the attention slope only | Low if the identity stays global-softmax | [`qwen38_attn_chunk_probe.py`](../../scripts/qwen38_attn_chunk_probe.py); Core AI already tiles at `ATT_BLOCK=16384` | Probe exists; timings not in this checkout. ANEC failed 32K/64K single-softmax: source-verified |
| 5 | Zero-skip / structured sparsity on masked KV | Low-medium, only if the ANE skips zero or masked tiles | Helps small positions more than full-cache decode | Low if only masked tails are skipped | [`qwen38_zeroskip_test.py`](../../scripts/qwen38_zeroskip_test.py) | Hook exists. Result of the probe is not checked in. Inferred |
| 6 | Lower-precision GDN / linear-attn mixers (compute, not just LUT4 weights) | Medium on GDN time if state/projections become INT8/FP8 compute; weights are already LUT-bound | GDN is 48/64 layers and ~5.5B weights/token, so this is the other large slice. Still not an automatic 2x | **High.** Recurrent error accumulation is already the painful mixer story | LUT4 + rank-64 residual; `GDN_SQ` / `GDN_SV`; `qwen38_hybrid_eval.py` `MIXER_FP8` is a host dequant-to-bf16 experiment | KL damage of 4-bit DeltaNet and the rank-64 rescue: measured (prior). Native low-prec GDN MAC: unmeasured |
| 7 | Compile flags (`MPSGRAPH_ANE_BONDED_COMPILE_MODE`, placement, iOS 26 opset) | None for throughput versus mode 2 | No | Low (mode 2 outputs were bit-identical to default on the tested chunk) | Already default mode 2 in `forge.py serve --runtime coreai` | Measured (prior): same ms, less memory |
| 8 | Winograd | None for this model | No | n/a | None | Inferred. Forge mixers are 1x1 convs and matmuls. Winograd is a 3x3-class conv transform. No Forge hook and no M6 measurement |

Discarded as a decode 2x path, kept for accuracy or prefill only:

- Replacing LUT weights with dense INT8 or FP8: more bytes per token. The imported notes call this slower for decode. **Measured (prior)** bandwidth behavior; **source-verified** as a documented dead end for decode.
- Per-group vector LUTs or Cin grouping: fall off the ANE. **Measured (prior).**

## Prototype

Top candidate implemented as a host harness, not an M6 result:

- [`scripts/m6_attn_compute.py`](../../scripts/m6_attn_compute.py)
- Runbook: [M6_ATTN_COMPUTE_README.md](M6_ATTN_COMPUTE_README.md)

It encodes the current V8 score-scale identity, INT8-INT8 QK/PV with INT32 accumulation, and a software E4M3 path (240 vs 448). It can emit a Core ML MIL skeleton when `coremltools` is installed. On this VM it only runs the NumPy host path.

## What this VM verified

- Graph-level dtypes in `coreai/qwen38_coreai_build.py` `AttnW` and `QConv`.
- V8 host quantizer and tests already in `scripts/qwen38_kv_cache.py`.
- Existing probes for zeroskip, chunked attention, FP8 LUT bandwidth, bonded compile mode.
- New unit tests for the harness (`tests/test_m6_attn_compute.py`): V8 identity, INT8 matmul scaling, E4M3 encode/decode, recipe validation, logical flop/byte counts.
- `uname` is Linux x86_64. `coremltools` and `torch` are not installed. No `.aimodel` weights are in the checkout.

## What still needs an M6

1. Compile the MIL / Core AI attention-core variants and record `MLComputePlan` / cached-specialization placement (ANE versus GPU, compile mode).
2. Compare wall-clock T=8 and T=64 attention-core latency for `fp16_baseline`, `v8_current`, and `int8_int8_attn` at 8K/16K/32K/64K. Report whether INT8-INT8 is faster, equal, or slower than dequant-to-FP16.
3. Confirm the executed dtype: fused INT8 MAC, dequant-then-FP16, or CPU fallback. Placement text alone is not enough; pair it with timing and, if available, `aned` / compute-plan operation dtypes.
4. Repeat the same matrix for FP8-FP8 (E4M3 240 and 448) and record whether the compiler casts to FP16.
5. Measure bonded compile mode 2 versus 0 on those same attention graphs (not the full server) so "bonded FP8" is either observed or dropped.
6. Only after (2) shows a real kernel gain, repeat a paired full-server prefill/decode like the V8 study. Do not advertise 2x from the host prototype.

## Attribution

Qwen3.8-27B is developed by the Qwen Team; original model copyright belongs to Alibaba Cloud (Apache 2.0). ANEMLL provides independent quantization, conversion, and ANE research. Apple, coremltools, and third-party ANE writeups are cited as documents, not as this project's measurements.
