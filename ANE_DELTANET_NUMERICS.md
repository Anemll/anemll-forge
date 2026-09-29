> Historical research notes imported on 2026-09-29. Paths were generalized; results were not rerun. Later sections may supersede earlier findings. See the current [release workflow](docs/WORKFLOW.md).

# ANE DeltaNet numerics: two fp16 bugs that made every DeltaNet layer ~50% wrong

Found 2026-09-27 on the M6 (macOS 27), Qwen3.8-27B v4 build (`mix25in_aw_cal_lr64mix`, 16 x 4-layer chunks, T=8, 16K).
Every build before `ane7f` (ane4 ... ane7) has both bugs. Fixed in `scripts/qwen38_ane_chunk.py` (and the prototype
`scripts/dflash2_gdn_lazy.py`).

## Summary

| stage of a DeltaNet layer (layer 9) | ANE vs CPU (same compiled chunk) before | after the fix |
| --- | ---: | ---: |
| normed layer input h (RMSNorm) | 0.05% | 0.05% |
| depthwise conv sum (before silu) | 0.07% | 0.07% |
| silu(conv) = q / k / v source | **6.0%** | 0.14% |
| q, k after L2 norm | 7-9% | 0.16% |
| recurrent state S | **15%** | 0.35% |
| raw output q . S (before the gated norm) | **33%** | 0.25% |
| output into out_proj (after gated RMSNorm * silu(z)) | **51%** | 0.58% |

Against the HF-exact fp32 reference (same quantized weights), the tensor entering `out_proj` goes from **51% to 1.3%**
rel-L2 error, the level of the attention layers (1.4%). Attention, RMSNorm and the MLP were already fine.

## Root causes

1. **The ANE's native `silu` has ~1e-3 absolute error near 0.** The DeltaNet causal-conv outputs are tiny: more than
   99% of the (8, 10240) values of layer 9 lie in [-0.5, 0.5], where silu(x) is about +-0.01. A 1e-3 absolute error is
   a 12% relative error on q / k / v. The CPU silu is exact to 0.08%. The MLP silu is not affected (its inputs are
   large). Error by input range (layer 9, 1.3M values): |x| < 0.5: mean |error| 0.0011 against mean |silu| 0.008-0.010;
   |x| > 2: error 0.0002-0.0003.
   - `x * sigmoid(x)` does **not** help: `mil_backend::fuse_activation_silu` (coremltools backend pipeline) fuses it back
     into `silu`.
   - Fix: `silu(x) = 0.5 x (1 + tanh(x / 2))` (`qwen38_ane_chunk.silu`, `SILU=tanh`, default; `SILU=native` for A/B),
     used for the conv and for the gate silu(z) in every DeltaNet path.
2. **fp16 subnormals in q . S.** q is L2-normalized and scaled by dk^-0.5 (median |q| 0.002) and the recurrent state
   is small (median |S| 3e-4, 17% of it below fp16's smallest normal 6.1e-5), so the raw output q . S has median 4.4e-5
   and 61% of its values are subnormal. The ANE loses them (flush / reduced precision): 20% error, 46% after the gated
   RMSNorm rescales the output (its eps = 1e-6 is also larger than mean(o^2) ~1e-8, so the true model's gated norm is
   eps-dominated - the scale matters).
   - Fix: scale v by `GDN_SV = 64` (so u, the pending rows and the recurrent state scale by 64) and q by `GDN_SQ = 16`;
     the gated RMSNorm absorbs it exactly with eps * (GDN_SQ * GDN_SV)^2:
     rms(s o, eps s^2) = s o / sqrt(s^2 mean(o^2) + s^2 eps) = rms(o, eps). After scaling: |o| median 0.045, max ~21;
     state max ~50 (fp16 max 65504). The state is host-owned I/O (starts at 0, only read by the layer), so the scale is
     internal to the DeltaNet; the host code does not change. Env: `GDN_SQ`, `GDN_SV`.

Neither bug is visible in cosine-similarity checks of the whole model (the model still generates fluent text), in the
M3U KL runs (PyTorch), or in small test graphs (they run on the CPU). Look at per-stage tensors of the ANE program,
compare CPU_ONLY vs CPU_AND_NE of the SAME compiled model, and check magnitudes (subnormal fraction) and near-zero
activation accuracy.

## How it was found (reproducible)

All on the M6, `~/venvs/vq27b`, run from `~` (never from the coremltools repo root).

1. fp32 CPU reference of the same quantized weights (dflash2_target_ref, matches the HF modules: DeltaNet layer vs
   `Qwen3_5GatedDeltaNet` rel 3.5e-7; `delta_chunk` vs a naive recurrence 1e-15), hidden state after every layer:
   `TOKENS=~/Models/vq27b/tests/div_pi2k.npy NTOK=2048 EXPORT_DIR=~/Models/vq27b/export/mix25in_aw_cal_lr64mix python scripts/qwen38_divergence.py ref`
   (`bf16` instead of `ref`: the unquantized checkpoint). Token rows: `SEQ=2048 MAX_ROWS=4 OUT=... python scripts/qwen38_calib_pi.py`.
2. Per-chunk / per-layer ANE divergence, each chunk fed the reference input of its first layer (isolated):
   `ANE_OUT=~/Models/vq27b/<build> CTX=16384 (same env) python scripts/qwen38_divergence.py ane`.
   With a one-layer-per-chunk build (`CHUNK_PLAN=0-0,1-1,...,63-63`) this gives every layer's own error: attention
   layers 0.2-0.7% of the hidden state, **DeltaNet layers 1.5-8% (7-37% of the layer's update)**, flat over positions.
3. Mixer-input capture: a `DBG_MIXER_IN=1` build adds each mixer matrix's input as an output (`dbg<n>`, map in
   `<chunk>.dbg.json`): `ANE_OUT=~/Models/vq27b/ane7D ... LAYERS=1,9,11,33 python scripts/qwen38_ane_capture.py` ->
   normed input 0.05%, attention core 1.4%, **DeltaNet core 35-51%** vs the reference.
4. Same compiled chunk on the CPU: `UNITS=CPU_ONLY` (qwen38_ane_model `_load`) -> DeltaNet core 1.9%: the graph is right,
   the ANE executes it wrong.
5. Intermediate taps: `DBG_MIXER_IN=1 DBG_GDN=1` build of one layer (`CHUNK_PLAN=9-9`) exposes conv_pre, conv, q/k/v
   heads, beta, g, cum, pair, u, wk, the committed state and the raw output; CPU_ONLY vs CPU_AND_NE per tap (table
   above) located the first divergence at silu(conv), then the subnormal outputs.

## Side findings

- A 2K-token pi session row seemed to show the ANE model copying repeated context (a second system prompt + tool
  schemas at distance 512-1024) far better than the fp32 reference (NLL 0.9-1.1 vs 3.5-4.8), and even better than
  the unquantized bf16 model (2.95-3.80). Synthetic copies (300 / 1000 random tokens, repeated) are copied perfectly by
  all three (NLL 0.01-0.06, ANE = reference within 0.02 nats), so the reference has no long-range bug and the ANE no
  leak; the pi-row effect came with the broken DeltaNet (see the results below for the fixed build).
- The first divergence reference added the low-rank factors twice (`dflash2_target_ref.Weights.layer` already adds
  them); fixed in `qwen38_divergence.py`.
- 32K v4 chunks: every chunk's ANE compile fails after ~200 s and Core ML silently runs it off the ANE (wired memory
  flat, E5 cache dir empty). 16K / 64K compile fine. Unrelated to this bug; see the memory notes.

## Attention and MLP (same method, layers 9 and 11)

CPU_ONLY vs CPU_AND_NE of the same one-layer chunk (`DBG_MIXER_IN=1 DBG_TAPS=att,mlp`), then vs the fp32 reference:

| stage | ANE vs CPU |
| --- | ---: |
| attention q / k / v after norm + RoPE | 0.1% |
| softmax probabilities / output / sigmoid gate | 0.3% / 0.2% / 0.4% |
| MLP normed input, gate, up | 0.4-1.2% |
| MLP silu(gate) * up (native silu) | **5-6%** |
| MLP output (down projection) | **5.7-6.3%** |

Attention is clean. The MLP has the same silu problem (the gate values are small too: median |g| 0.1) plus
subnormal products in the down projection (input median |a| 1e-2):

| MLP output vs fp32 reference (128 tokens) | layer 9 (DeltaNet layer) | layer 11 (attention layer) |
| --- | ---: | ---: |
| CPU fp16 | 2.3% | 0.06% |
| ANE, native silu | 4.6% | 5.6% |
| ANE, tanh silu (`MLP_SILU=tanh`) | 0.7% | 2.9% |
| ANE, tanh silu + down input x64 (`MLP_DS=64`) | 0.7% | **0.8%** |

- `MLP_DS=1024` overflows (inf) on some tokens: the down conv's partial sums exceed fp16 at about 3.5x the output.
  Per-layer scales (`MLP_DS_TABLE=~/Models/vq27b/tests/mlp_ds_mix25in.json`): the largest power of 2 <=
  min(64, 2048 / max |layer update|) over the reference runs; 64 for most layers, 2 for layers 54, 58, 59, 63
  (updates up to 832), 8-32 for the other late layers.

## Results after the fix

`ane7f` = ane7 with the DeltaNet fix (SILU=tanh, GDN_SQ=16, GDN_SV=64); `ane7g` = ane7f + `MLP_SILU=tanh` +
`MLP_DS_TABLE`. Same export (`mix25in_aw_cal_lr64mix`), 16 x 4-layer chunks, 16K.

Per-chunk divergence on the 2K pi row (`qwen38_divergence.py ane`):

| | ane7 | ane7f |
| --- | ---: | ---: |
| isolated chunk error (share of the chunk's update) | 5-25% | 0.7-2.9% |
| end-to-end cos at the last layer | 0.822 | 0.986 |
| top-1 agreement with the fp32 reference | 0.742 | 0.922 |
| NLL vs the reference | -0.906 (the "copying" artifact) | -0.053 |

In-domain trace perplexity on the ANE (`qwen38_ane_trace_ppl.py`, 64 bf16 chat answers, 40K tokens; bf16 2.136):

| build | export | PyTorch ppl | ANE ppl |
| --- | --- | ---: | ---: |
| ane6 | mix25_aw_cal_lr64mix | 2.533 | 2.833 |
| ane7 | mix25in_aw_cal_lr64mix | 2.428 | 2.651 (+0.088 nats) |
| **ane7f** | mix25in_aw_cal_lr64mix | 2.428 | **2.420** (-0.003) |
| **ane7i** (+ MLP tanh silu) | mix25in_aw_cal_lr64mix | 2.428 | **2.426** (-0.001) |

Same weights, DeltaNet fix only: 2.651 -> 2.420 (-0.091 nats, -8.7% perplexity).

**Cost of the workarounds: none measurable.** Full 16-chunk verify-8 call on the M6 ANE (host included, 60 calls
after 5 warm-up, 16K, `verify_time.py`): ane7 109.8 / 109.4 ms median (two runs), ane7f 109.6 ms, ane7g 109.9 ms.
The extra elementwise ops (tanh form, scalings) disappear in the weight-bandwidth-bound chunk time.

**MLP down-projection input scaling is unsafe - dropped.** Layer 34 (one-layer chunk, MLP output vs fp32 reference):
native silu 1.58%, **tanh silu 0.23%**, tanh + static x64: inf on some tokens, tanh + dynamic per-token scale: garbage
on some tokens - although the scaled input max is only 115 and the true output max 21.6. The overflow is inside the
down conv. Most likely explanation (not verified): the ANE accumulates (unscaled LUT values x input) and applies the
per-channel scale afterwards, so the pre-scale partial sums are ~1/scale (~100x) larger than the output; that would
also explain the remaining ~3% down-projection error as fp16 cancellation in those large partial sums. The tanh silu alone gives most of the MLP gain. Details of the failed variants:

**Static MLP down scales overflow (ane7g):** layers 0-31 improved (isolated chunk error 0.15-0.9% of the hidden
state, 0.7-1.0% of the update, vs 0.6-1.3% / 0.7-2.9% for ane7f), but from layer 32 on some tokens overflow to inf and
chunks 8 and 10 show 9-11% median error: in those layers the down input itself has large channels on many tokens, so
the input x 64 overflows (the max-update table did not capture it). End to end +0.71 nats - unusable. Replaced by a
per-token dynamic scale (`MLP_DS_DYN=1`): f = min(MLP_DS, MLP_DS_C / max|a|), output divided by f (ane7h).

The ~0.1-nat "ANE gap" in QUANTIZATION_NOTES.md was this bug: with the fix the ANE matches the PyTorch simulation.
The dynamic variant (ane7h: f = min(64, 64 / max|a|) per token) broke the same layers further (end to end NLL 12,
trace ppl ~1e5); it also cost ~1.5 ms per verify call (111.2 ms).

### Final Core ML recipe: ane7i (DeltaNet fix + MLP tanh silu, no down scaling)

Build: `MLP_SILU=tanh CTX=16384 CHUNK_PLAN="0-3,...,60-63" EXPORT_DIR=~/Models/vq27b/export/mix25in_aw_cal_lr64mix
ANE_OUT=~/Models/vq27b/ane7i python scripts/qwen38_ane_model.py build_v3` (SILU=tanh, GDN_SQ=16, GDN_SV=64 are the
defaults; MLP_SILU still defaults to native in the builder).

| on the 2K pi row | ane7 | ane7f | **ane7i** |
| --- | ---: | ---: | ---: |
| isolated chunk error (share of the update) | 5-25% | 0.7-2.9% | **0.3-1.0%** |
| end-to-end cos at the last layer | 0.822 | 0.986 | **0.996** |
| top-1 agreement with the fp32 reference | 0.742 | 0.922 | **0.965** |

ane7i trace ppl 2.426 vs ane7f 2.420: both at the PyTorch simulation (2.428) within noise; ane7i tracks the
reference much more closely (top-1 / cos above), so it is the build to deploy. Verify-8 timing in the same
session: ane7i 109.8 ms, ane7f 107.7 ms (run-to-run noise ~2 ms): the MLP tanh costs at most ~1-2%.

## Core AI port (coreai/qwen38_coreai_build.py)

Same bugs, same fixes (checked by a one-layer Core AI harness, `~/Models/vq27b/coreai_fixtest/`; v8_16k and p64
entries, state / KV carried):

| Core AI one-layer chunk | layer 9 (DeltaNet) share of update | layer 9 core | layer 11 (attention) share of update |
| --- | ---: | ---: | ---: |
| before (native silu, no scaling) | 0.372 | 51% | 0.018 |
| SILU=tanh only | 0.343 | 48% | |
| GDN_SQ / GDN_SV only | 0.156 | 24% | |
| tanh + scaling (MLP silu native) | 0.030 | 1.3% | |
| **fixed (+ MLP_SILU=tanh)** | **0.011** | **1.3%** | **0.004** |

- Core AI keeps the tanh form (`xcrun coreai-build inspect --ops`: tanh 3, silu 0) and flushes subnormals like Core ML.
- MLP down scaling is not needed in Core AI: that port applies the per-channel scale as a mul after the conv (LUT
  values median 0.86), so the products stay normal. Idea for Core ML: the same structure (scale after the conv) might
  fix the remaining down-projection error without the overflow risk of input scaling.
- The Core AI CPU path is broken for this graph (L2-normalized k reaches 10.7, u / wk inf): use a torch fp32 mirror as
  the graph reference there.
