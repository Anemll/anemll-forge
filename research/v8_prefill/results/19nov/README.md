# 19 Nov 2026: 48K accuracy, FP8 K, DeltaNet profile, drafter on the GPU, tree drafting, Q.K epilogue e2e, GDN GEMMs

Forge stayed up on 8765 (FP8 x prefill + decode keep-alive) for steps 1-3: step 1 reused the 18 Nov captures, and
steps 2 and 3 are GPU-only runs on one-layer manifests (no model load) while forge was idle. Items 5-6 ran against the
served decode. Forge was restarted twice with timing-only logging flags and once by the swap watchdog (section 5). It
went back to the standard launcher at 05:52.

| dir | what |
|---|---|
| `kl48/` | step 1: teacher-forced KL at 48K, FP8 x vs plain (`power/kl48.py`) |
| `qk/` | step 2: Q.K variants alone (`power/qk_f8_bench.swift`), full attention call A/B (`power/qk_chain_a.sh`, `alone/`) |
| `gdn/` | step 3: DeltaNet call profile, LUT GEMM and recurrence forms, shape probe, DRAM-load runs, ablations |
| `drafter_gpu/` | queue 5: served decode baseline + draft breakdown (DRAFT_PROF), GPU drafter cost twin, overlap with the verify |
| `qk/e2e/` | queue 8: staged Q.K epilogue end to end, interleaved x / tg traces |
| `gdn/proto/` | queue 9: DeltaNet projection GEMM prototypes (fold, 256-token pass, hidden decode), alone and under DRAM load |
| `qkf8/deploy` (on the SSD) | QKE=tg deploy check: pre / post-deploy 32K prefill logits |
| `gdn/ws_e2e/` | queue 9a: hidden-decode LUT GEMM at 95K end to end |
| `gdn/g256/` | queue 9b: 256-row DeltaNet calls, the ANE same-function concurrency finding, debug logs |
| `tree/` | queue 6: tree drafting offline (DRAFT_TREE_LOG logs, `power/tree_sim.py`), per-length projection |

## 1. 48K teacher-forced KL (`kl48/kl48.json`)

A causal prefill's rows do not depend on later tokens. On 18 Nov, H_32768 was bit-identical to the first 32,768 rows
of H_95232, so positions [0, 49152) of the 95K teacher-forced run are the 48K run. Cross-check: KL at position 49,151 is
0.025975 here, and 0.02598 from the standalone 48K traces' last-token logits (`18nov/f8/pts`).

| 49,152 positions | value |
|---|---|
| KL mean / median | 1.08e-3 / 3.4e-4 |
| KL p99 / p99.9 / max | 7.0e-3 / 0.035 / 4.56 (position 14,192, the same as at 95K) |
| top-1 agreement | 98.67% |
| top-10 overlap / all 10 the same | 9.77 / 78.9% |
| NLL of the real next token, plain -> x | 1.8280 -> 1.8291 |
| KL mean per sixth | 6.7e-4, 1.7e-3, 1.5e-3, 9.7e-4, 8.5e-4, 8.4e-4 |
| positions above 0.026 / 0.1 / 1 | 74 / 18 / 3 |

The last token (49,151) is a near-tie. Plain gives 0.495 / 0.351 to its top two tokens and x gives 0.550 / 0.272, with
the same top token. Its KL of 0.026 ranks 75th of 49,152 positions. The two following positions have KL 4.6e-5 and
1.3e-6. At 4K and 16K the last token is a 99.7% top-1, and the last-token KL is 3e-6 and 2e-6.

## 2. FP8 K for Q.K (`qk/`)

The M6 matrix units run INT8 x INT8 at 39.5 TOPS and e4m3 x e4m3 at 36.9 (18 Nov `gpu_mm_peak`). Served Q.K is already
INT8 (`attn_qk8isk`: INT8 Q, INT8 K straight from the ANE cache layout [kv][dim][key], per-key scale epilogue, fp16
scores), so FP8 K cannot gain from a faster rate.

### Q.K alone (`qk_f8_bench.swift`, 3072 rows x 256 dims, 2048-key tiles as served, 2 s GPU warm-up, median of 12)

| variant | 32K ms | 95K ms | 95K TOPS |
|---|---|---|---|
| `i8s` served | 2.235 | 6.214 | 24.1 |
| `i8s_ne` served without the epilogue (lower bound) | 1.557 | 4.488 | 33.4 |
| `i8d` INT8, dense K copy | 2.244 | 6.137 | 24.4 |
| `f8d` FP8 Q x FP8 K copy (per-key scale), served epilogue | 2.446 | 7.106 | 21.1 |
| `f8t` FP8, K copy [key][256], transpose_right | 2.522 | 7.346 | 20.4 |
| `f8g` FP8, K copy with one global scale | 2.394 | 6.970 | 21.5 |
| `i8s_n128` / `i8s_m128` (64x128 / 128x64 per threadgroup) | 2.348 / 2.178 | 6.758 / 6.277 | |
| **`i8s_tg` served INT8, scores staged in threadgroup memory, coalesced half4 stores** | **1.586** | **4.655** | **32.2** |
| `i8d_tg` INT8 dense K copy, staged | 1.549 | 4.516 | 33.2 |
| `f8d_tg` FP8 K copy, staged | 1.545 | 4.509 | 33.2 |
| `f8g_tg` FP8 K global scale, staged | 1.527 | 4.441 | 33.7 |

- **The served Q.K loses its time in the epilogue.** It does one scattered 2-byte store per cooperative-tensor
  element. Staging the scores through threadgroup memory brings INT8 Q.K to 4% above the no-store bound, -25%.
- **FP8 K adds nothing.** With the same epilogue, FP8 K and a dense INT8 K copy are equal (4.509 / 4.516 ms). Their 3%
  over the strided ANE layout is the dense layout, not FP8. The global-scale FP8 form is 0.21 ms per call faster at 95K
  than staged INT8: 1% of the 20.8 ms attention call, below the 3% gate. It would also cost a second K copy (1.6 GB at
  96K, the memory the FP8 V copy saved) and an e4m3 rounding of K. FP8 K was therefore not integrated and gets no e2e.

### Full attention call (`qk_chain_a.sh`; f8_check.py, one-layer manifest, F8=x, `libgpu_v8_f8q.dylib`)

`gpu_v8_f8q.swift` is `gpu_v8_f8.swift` plus `kernels_qk.metal`; `QKE=tg` selects `attn_qk8isk_tg`. The banner reports
`qk=` from the loaded dylib, and `EXPECT_QK` aborts on a mismatch. Runs were interleaved x / tg / x / tg, 16 calls each,
median ms per 128-row pair call:

| keys | x a / b | x + tg a / b | change |
|---|---|---|---|
| 32K | 8.446 / 8.433 | 8.472 / 8.111 | -1.8% |
| 61K | 14.078 / 14.017 | 13.733 / 13.457 | -3.1% |
| 95K | 20.775 / 20.794 | 20.171 / 19.704 | -4.1% |

The output is bit-identical (relative RMS 0 at every length). Profile at 95K, per call: Q.K 6.74 -> 5.75 ms, softmax
4.17 -> 4.44, P.V 4.32 -> 4.49, total 18.75 -> 18.31 (-2.3%). In the full call, Q.K saves 1 ms rather than the 1.56 ms
it saves alone, and the next kernels give part of that back. Not deployed. It is a bit-exact candidate for the next
e2e window; in the 95K prefill (attention about 63% of the wall) it would be worth about 2-2.5%.

## 3. DeltaNet prefill profile (`gdn/`)

`gdn_prof.py`: one GDN layer (`manifest_L0.json`), the served environment (FUSE=all, I8, I8T `_k256`, I8A, LRM, CONC
on, NT=128), 128 tokens per call. The cost does not depend on the position (fixed-size state), so the alone numbers
hold at 32K and 95K alike.

| | 2K | 8K | 32K | 64K | 95K |
|---|---|---|---|---|---|
| e2e GPU ms per DeltaNet call (traces, deciles) | 2.54-2.58 | 2.52-2.60 | 2.47-2.56 | 2.44-2.59 | 2.44-2.56 |
| DeltaNet total s / wall s | 1.9 / 3.1 | 7.6 / 11.8 | 30.0 / 57.8 | 61.3 / 145 | 88.1 / 243 |
| GPU busy / ANE busy | 83% / 96% | 95% / 97% | 99% / 85% | 99% / 69% | 99% / 58% |

Alone, a call takes 1.96 ms; end to end it takes about 2.5 ms (1.3x), while attention calls run only 4% over their
alone time. A CPU memory load of about 140 GB/s (`dram_hog 2`, a stand-in for the ANE's weight stream) reproduces the
end-to-end call at 2.49 ms.

### Per kernel (ms per call)

Ablation numbers are in place: call time with minus without the group (`SKIP`), on the real command buffer. Profile
numbers put each dispatch in its own command buffer. Under the memory load, profile mode is not valid (the GPU clock
drops in the gaps, so every kernel reads about 3x).

| group | ablation, alone | ablation, 140 GB/s load | profile, alone | share |
|---|---|---|---|---|
| LUT GEMMs: qkv 5120->10240, z 5120->6144, out 6144->5120 (`lut_gemm32_i8t_k256`, 2-bit vector LUT x INT8) | 1.093 | 1.318 | 1.23-1.40 | 56% |
| `gdn_rec` (the recurrence) | 0.228 | 0.272 | 0.28-0.32 | 12% |
| glue: `gdn_norm_fwht4`, `rms_fwht4`, `quant8_blk` x2, `gdn_conv_qkv`, `rms_partial`, conversions, gates | 0.266 | 0.290 | 0.27 | 14% |
| low-rank correction (LRM): `dense_splitk`, `splitk_reduce`, `dense_gemm`, 3 each | 0.164 | 0.179 | 0.19 | 8% |
| whole call | 1.960 | 2.492 | 2.06-2.23 | |

### LUT GEMM forms (whole call, ms)

| served k256 | k256 double-buffered | k128 | k512 | k64 | m64k128 | k64 no decode* | k64 no matmul* |
|---|---|---|---|---|---|---|---|
| 1.964 | 1.988 | 2.032 | 1.999 | 2.215 | 2.123 | 2.002 | 1.042 |

\* timing diagnostics, wrong output.

Under the 114 / 140 GB/s loads: served 2.25 / 2.50, double-buffered 2.27 / 2.56, k128 2.34 / 2.58, m64k128 2.49 / 2.80.
The served form is the best of the existing forms under load too.

Recurrence forms (whole call): `gdn_rec` 1.964, `rec16` 1.998, `rec8` 2.027, `rec_pf` 1.973, `rec_px` 1.965.

### Occupancy and limits

- **No register limit.** Every DeltaNet pipeline allows 1024 threads per threadgroup (`gdn_pso_info.swift`); the
  dispatches use 128.
- **LUT GEMM.** 320 / 192 / 160 threadgroups x 4 simdgroups (13-27 per core on the 12-core GPU), 9.8 KB of threadgroup
  memory. The INT8 matmul plus activation load at the served tile shape (TM 32 x N 128 x KS 256, A from threadgroup
  memory) runs at 36.2 TOPS (`gpu_mm_shape.swift`): 0.82 ms for the three GEMMs (29.5 GFLOP per call). In place they
  take 1.09 ms (27 TOPS, 68% of the INT8 peak). The gap is the LUT decode (about 0.2 ms) and the epilogue. Weights are
  28.8 MB per layer call (2 bits per weight).
- **`gdn_rec`.** 192 threadgroups x 128 threads, 1 KB of threadgroup memory. Its time is linear in the head count from
  6 heads up (0.059 ms at 6, 0.152 at 24, 0.276 at 48, 0.539 at 96; `gdn_rec_scale.swift`), so the GPU is full: it is
  throughput-bound, not occupancy- or latency-bound. That is 0.8 GFLOP fp32 per call at about 2.9 TFLOPS, about 55% of
  the fp32 FMA peak (12 cores x 128 lanes x 2 x 1.73 GHz = 5.3 TFLOPS).

### Top 2 candidates

1. **The projection GEMMs (56% of the call, and most of its growth under memory load).**
   - Hide the LUT decode behind the matmul: a producer/consumer split inside the threadgroup, instead of decode ->
     barrier -> matmul (the existing double-buffered form did not help). Worth up to 0.27 ms of the 0.82 ms ceiling.
   - Fold the low-rank correction (0.16 ms in place) into the GEMM as one more 64-wide K chunk.
   - Under contention the GEMMs grow 0.23-0.32 ms per call. Their weight stream per token halves if one GDN call covers
     256 tokens (two pairs), which needs the pipeline to group two blocks per GDN layer.
   - Estimated -0.3 to -0.45 ms per e2e call: DeltaNet -12 to -18%, which is -11 to -16 s at 95K (wall about -5 to -6.5%,
     GPU-bound) and -4 to -5.5 s at 32K (wall -6 to -9%; the ANE at 85% busy limits it to about -15%).
2. **The recurrence on the matrix units (12-15%).** A chunkwise (WY) form of the gated delta rule: 64-token chunks,
   fp16 inputs and fp32 accumulate in matmul2d. About 1.2 GFLOP per call instead of 0.8 GFLOP of serial fp32 FMA. 0.23
   -> about 0.1 ms: -0.13 to -0.15 ms per call, -5 to -6 s at 95K, -2 s at 32K. Risk: the state's precision over 95K
   tokens, so it needs a teacher-forced KL check.
3. **Next after those: glue fusion** (quant8 into the rms/norm producers, conv into the GEMM epilogue). Bit-exact, about
   -0.1 ms per call.

## 5. DFlash2 drafter on the GPU (`drafter_gpu/`)

Served decode (V8 + dec8 split verify, keep-alive on). Each request is a long technical report on the docs text with 256
generated tokens; one cold turn plus one follow-up turn per length (`power/dec_base.py`). DRAM is IOReport over the
decode window. Draft timing comes from a restart with `DRAFT_PROF=1` (timing-only logging added to `pipe/v8_server.py`,
off by default; copy in `pipe/v8_server_dprof.py`).

| ctx | cycle ms | draft | verify | GPU active | DRAM ANE / GPU / total GB/s |
|---|---|---|---|---|---|
| 2K | 108-110 | 13.1-14.0 | 93.9-94.9 | 0.6-3.6% (502 MHz) | 98-99 / 5 / 107 (69% of peak) |
| 32K | 134-136 | 13.0-13.3 | 120.4-121.6 | 0.6-3.5% | 94 / 4-5 / 101-103 |
| 64K | 130-132 | 13.4-13.6 | 116.5-118.2 (split, GPU attention 18) | 100% (keep-alive) | 80 / 19 / 100 |

The draft phase, per propose (507 proposes, the same at every length): drafter body (ANE `draft` entry) 6.5 ms, the
target's head on the 7 draft rows (ANE) 4.3 ms, host work (top-k, selector, rope / mask fill, ctx64 calls) 1.8-2.0 ms,
and the 1 ms draft gap.

**GPU cost twin** (`power/drafter_gpu_twin.swift`). It runs every drafter matrix at its real shape and 16 rows: fc
25600->5120, 5 x (q, k, v, o, gate, up, down, 2 INT8 conv projections). The indices are synthetic (no weights loaded),
6-bit packed pairs as in the package (674 MB). It uses the engine's LUT pattern: 8 indices per 64-bit read, an INT8
pair palette and an INT8 x INT8 matmul.
- **Alone:** 8.9 ms per draft call (76 GB/s). Gate and up stream at 98 GB/s, down at 67, and the narrow k / v / conv
  matrices (32-40 threadgroups) at 36-41.
- **Ceiling:** with split-K about 5.5-7 ms, against the ANE body's 6.5 ms. The first naive kernel took 16 ms.
- **Gapped:** one call every 135 ms (one per cycle) takes 17.6-18.2 ms, because the GPU drops to its low clock between
  calls. It needs a keep-alive below 64K (+2 W), where the served keep-alive is not active.

**Overlap with the verify** (`power/dec_overlap.py`: the twin in its own process during the served decode).

| ctx | arm | verify ms | twin ms per call |
|---|---|---|---|
| 2K | none | 93.6-94.4 | |
| 2K | twin back to back | 119.5 (+27%) | 14.0 |
| 2K | twin once per 135 ms / + keep-alive | 95.8 / 95.9 (+1.6%) | 18.2 / 14.5 |
| 32K | none (cold turn) | 121.3 | |
| 32K | twin back to back | 151.3 (+25%) | 13.3 |
| 32K | twin once per 135 ms / + keep-alive | 123.2 / 123.2 (+1.6%) | 17.6 / 13.8 |
| 64K | none | 116.2-118.0 | |
| 64K | twin once per 135 ms | 123.1 (+5%; GPU attention 19.6) | 13.0 |
| 64K | twin back to back / once per 135 ms + external keep-alive | 203.1 / 196.1 (+68-73%) | 12.6 / 16.3 |

**Verdict: keep the drafter on the ANE.**
- **No overlap is possible.** The draft needs the verify's accepted rows (their tap features) and the bonus token.
- **Serial GPU drafting gains almost nothing.** At best it saves about 1 ms per 108-134 ms cycle (5.5 vs 6.5 ms body);
  today's kernel would lose 2.4 ms. The head stays the same either way (a GPU head would stream the same 636 MB).
- **Forced overlap costs more than it saves.** At 2K / 32K the decode is DRAM-bound: the ANE verify uses 94-99 GB/s
  and the total is 100-107 GB/s. Against it, a GPU drafter takes 13.3-14.5 ms (as long as the whole ANE draft phase)
  and slows the verify 1.6% per call. At 64K, GPU work from another process also queues against the split verify's
  attention dispatches.
- **Where the headroom is.** The GPU is idle in 99% of the decode at 2K / 32K (attention 18 ms of the 117 ms verify at
  64K), but the dependency chain leaves no place to use it.

Note: during the first overlap run, the swap watchdog stopped forge at 05:37:28, at the end of a cold 64K prefill
(swap 672 -> 2276 MB) after 14 turns at 2K / 32K. It was restarted at 05:38:30. The 64K arms were rerun on a fresh
server.

## 6. Tree drafting on DFlash2, offline (`tree/`)

`DRAFT_TREE_LOG` (same restart) logged, per propose, the top-16 candidates per draft position, their unary scores and
the selector's hp, plus the committed stream: 12 requests, 964 cycles, temperature 0, so the stream is the greedy
ground truth. `power/tree_sim.py` builds fixed trees with the served selector score s_i(c | parent) (top-w children per
depth, given each branch's own parent). Acceptance = the longest root-to-leaf prefix equal to the ground truth (exact
for greedy). Check: the flat chain reproduces the server's accepted count in 964 / 964 cycles.

| tree (children per depth) | nodes | tok/verify | vs flat |
|---|---|---|---|
| flat (served) | 7 | 3.064 | |
| [2,1,1,1,1,1,1] | 14 | 3.279 | +7.0% |
| [1,2,1,1,1,1,1] | 13 | 3.276 | +6.9% |
| [3,1,1,1,1] | 15 | 3.241 | +5.8% |
| [2,2,1,1] | 14 | 3.176 | +3.7% |
| [4,1,1] | 12 | 2.893 | -5.6% |
| beam 2 / beam 4 x 3 (selector score) | 9 / 7.7 | 3.058 / 2.884 | -0.2% / -5.9% |

Per request the [2,1,...] gain is +4% to +18%.

**Verify cost on this target.** DeltaNet layers are recurrent, so an exact tree needs one state per branch. The
served dec8 entries take 8 rows of one sequence, so a 2-branch tree is two verify calls. A batched 16-row entry (two
8-row sequences with their own states) would stream the weights once. It needs a new ANE build (parked under the
17 Nov package rule); its cost is estimated at +0% (compute fully hidden under the weight stream) to +12% (linear
between the 8-row verify and a 64-row call).

| ctx | flat: tok/verify, tok/s | [2,1,...]: tok/verify | two served calls: cycle, tok/s | batched 16-row: tok/s |
|---|---|---|---|---|
| 2K | 3.347, 31.0 | 3.553 | 202 ms, 17.6 | 32.9 to 29.8 |
| 32K | 2.947, 21.9 | 3.136 | 255 ms, 12.3 | 23.3 to 21.1 |
| 64K | 3.029, 23.2 | 3.253 | 247 ms, 13.1 | 24.9 to 22.5 |

A tree only pays with a batched 16-row verify entry costing under about 7% more than the 8-row verify. That is a
build (parked); first measure a 16-row ANE verify on one chunk if the package rule allows it. These requests are low
acceptance (24-37%); with high-acceptance text (81%, most cycles accepting all 7), a tree adds less.

## 8. Staged Q.K epilogue end to end (`qk/e2e/`, `power/qk_chain_e2e.sh`)

The same binary (`libgpu_v8_f8q.dylib`, F8=x) in interleaved fresh processes, x a / tg a / x b / tg b (tg = `QKE=tg`,
`attn_qk8isk_tg`). Each process ran 95K, then 64K, then 32K, with v8_trace's 45 s idle before each run. The banner
checks F8=x and the Q.K kernel (`EXPECT_QK`). Forge was down 05:54-06:35, then restarted on the standard launcher.

| prompt | x a / b tok/s | x + tg a / b | change | attention s, x -> tg | DeltaNet s, x -> tg | ANE busy | DRAM total / GPU GB/s |
|---|---|---|---|---|---|---|---|
| 95K | 390.0 / 393.6 | 401.1 / 401.2 | +2.4% | 154.1 / 152.9 -> 148.6 / 148.7 (-3.2%) | 88.7 / 88.0 -> 87.5 / 87.5 | 58.2 / 57.6 -> 58.5 / 58.4% | 116.4 / 73.3 -> 118.7 / 74.8 |
| 64K | 456.8 / 463.1 | 467.0 / 467.2 | +1.6% | 81.5 / 80.3 -> 78.8 / 78.6 (-2.7%) | 60.9 / 60.4 -> 60.1 / 60.3 | 66.6 / 67.1 -> 68.4 / 68.3% | 118.1 / 68.3 -> 120.1 / 69.5 |
| 32K | 570.7 / 572.5 | 579.8 / 580.1 | +1.5% | 26.3 / 26.2 -> 25.7 / 25.7 (-2.1%) | 30.5 / 30.3 -> 30.2 / 30.2 | 81.4 / 82.0 -> 82.7 / 82.7% | 121.9 / 60.7 -> 123.4 / 61.4 |

**Accuracy:** the last-token logits are bit-identical at every length, across x a, x b, tg a and tg b (KL 0, the same
top-10), as the kernel-level check predicted (relative RMS 0). The repeats are tight: tg a and b are within 0.1%, x a
and b within 1.4%. Not deployed; the switch is `QKE=tg` with `V8_SRV_DYLIB=libgpu_v8_f8q.dylib`.

## 9. DeltaNet projection GEMM prototypes (`gdn/proto/`, `power/gdn_gemm_proto.swift`, `engine/kernels_gdnp.metal`)

Kernel level only. The setup is layer 0's real projections (qkv / z / out `lut_gemm32_i8t_k256` on the build's HWX
weights, mapped zero-copy as the engine does, plus the LRM low-rank correction), random INT8 / fp16 activations, in
one serial command buffer per 128-token call. "Load" is the 2-thread CPU memcpy (about 140 GB/s), which reproduces
the end-to-end DeltaNet call time (section 3). All rows are ms per 128 tokens, median of 150, two rounds.

| variant | alone | under load | output |
|---|---|---|---|
| served (3 LUT GEMMs + B x split-K + 3 dense_gemm) | 1.417-1.427 | 1.869-1.878 | |
| low-rank folded into the GEMM epilogue (`k256lr`) | 1.532-1.547 (+0.12) | 1.894 (+0.02) | fp32 rounding (1e-7 rel) |
| 256 tokens per weight pass (served kernels at T=256) | 1.393-1.430 (-0.01) | 1.866-1.886 (0) | bit-exact |
| hidden decode (`k256ws`: simdgroups 4-7 decode chunk c+1 while 0-3 run the matmul on chunk c) | 1.448-1.451 (+0.03) | 1.740-1.744 (-0.13) | bit-exact |
| hidden decode + 256 tokens per pass | 1.391-1.393 (-0.03) | 1.664-1.665 (-0.21) | bit-exact |
| bound: no dense_gemm (a perfect fold) | 1.322 (-0.10) | 1.71 (-0.17) | |
| bound: decode removed (k128 vs k128 no-decode) | -0.25 | -0.35 | |
| bound: hidden decode + 256 tokens + no dense_gemm | 1.270 (-0.15) | 1.448 (-0.43) | |

**Against the 0.2 ms gate:**
- **Hidden decode + 256 tokens per weight pass** saves 0.21 ms per call under load, only 0.03 alone. It needs a
  pipeline change before any e2e A/B: the GDN calls would have to take two pairs (256 rows), with a T=256 library for
  the DeltaNet kernels next to the T=128 attention path. Estimated 95K gain: 0.21 ms x 35,712 calls = 7.5 s of DeltaNet
  GPU time, about -3% of the wall. Not run end to end; it needs a go-ahead for the pipeline work.
- **Hidden decode alone** is a drop-in kernel: same tables, same outputs (bit-exact), 256 threads per threadgroup.
  It saves 0.13 ms under load, below the gate. Estimated 95K: about 4.6 s of DeltaNet time, about -2% of the wall.
- **The low-rank fold** as built loses: its 20 KB of extra threadgroup memory (the A rows and a 32 x T fp32 tile) cuts
  occupancy. Its ceiling is 0.10-0.17 ms. A version that aliases the tile onto W in four 32-token passes would keep
  served occupancy, but cannot reach 0.2 ms on its own.

## 17. Pi coding-agent session on V8.2 (`pi_session/summary.json`, logged 16:30)

**Session:** Anemll's interactive `pi` session against the served forge (8765, the V8.2 configuration), 12:46-12:58.
- **Setup:** provider `ane-qwen38`, thinking medium, working directory `ML_playground/11/11.a`.
- **Task:** "make a game of tetris in HTML save in ./tetris-v8-2/, test and fix".
- **Sources:** the server log `coreai_seg_logs/v8_server.20261119-125807.log` and the session file in `~/.pi` (read
  only; only counts, times and the outcome are kept here).

| | |
|---|---|
| model requests | 21 (21 tool calls: 13 bash, 5 write, 3 edit) |
| generated | 19,174 tokens in 696 s of decode, 27.5 tok/s overall |
| per request | 19.0-51.9 tok/s. The 4,613-token `index.html` write ran at 51.9 tok/s (5.84 tok / verify, 69% accepted); thinking-heavy turns ran at 19-25 tok/s (18-27% accepted). |
| context | 6.1K-27.6K tokens: no split verify and no keep-alive (no dec8 lines in the log) |
| cycle | 112-113 ms in the 16K entry, 120-124 ms in the 32K entry (from about 16.4K tokens) |
| prefill | 2,375 new tokens in 10.8 s in all. Each follow-up re-prefilled 19-553 new tokens in 0.2-1.4 s from the prompt cache. |
| outcome | a single-file Tetris. The model tested it in Node against a stub DOM (17 assertions, all passing) and fixed a bug it found. |

- **Speed tracked acceptance, not the hardware:** the cycle is flat within each entry.
- **No Splash run:** this session was forge only; the last head-to-head Pi runs are 17 Nov, against Splash 1.3.2.
- **Before it:** a 5,933-token "hi" (prefilled cold in 8.7 s) and a 3 s aborted request, at 12:45-12:46, in an
  earlier session.

## 7. JetSpec++ Qwen3.8-27B drafter (download verified 15:40; not run)

**Download check (15:40, read-only):** `hf download` to `jetspec_pp/release` finished at 12:33 (11 files, 12.66 GB, no
partial files, no process left). Every file is at revision 3147a3513f7cc7d81bc82e40e4ce3f28ad7d8047.
- **sha256 against the repo's own `manifest.json`:** 7 of 8 match: model.safetensors 20361600..., adapter/head
  0384faef..., adapter/converter 89e58efc..., config.json, dflash.py, adapter/config.json, and
  calibration/decoder/manifest.json.
- **The exception, `calibration/thresholds.json`:** 110,004 bytes, sha256 d08f1182..., while the manifest lists 110,027
  bytes and c746c327.... The file is HF's own blob at this revision (git blob beffda70 = its etag), so the repo's manifest
  entry is stale; it is not a download error. The thresholds serve the reference engine's anchor predictor. Use them
  as is, but note the mismatch.
- **Tensors (headers only):**
  - Drafter: 58 BF16 tensors, 3.46 GB: fc [5120, 25600] over 5 taps, hidden_norm, 5 Qwen3 layers (32 q / 8 kv heads x
    128, MLP 17408), norm. No embedding or lm_head: the engine takes those from the target.
  - Converter: fc plus hidden_norm, 262 MB, metadata FROZEN.
  - Head: 144 tensors, 8.93 GB: 5 streams plus 7 extra stream layers (depths 1, 1, 1, 4, 5 = 12 target-shaped blocks:
    24 q heads x 256 with the output gate, 4 kv heads, MLP 17408). Their attention is initialised from target layers 3 /
    19 / 31 / 35-47 / 47-59, their FFNs from layers 5 / 19 / 33 / 37-47 / 50-61.
- **The target slice:** `calibration/decoder/manifest.json` pins the target's config.json, the index, and shards 16-18
  by sha256. All 5 match `/Volumes/SSD4TB/Qwen3.8-27B` (revision 1d4bf0f2, the manifest's target) exactly, rehashed.
- **Memory for a replay:** drafter 3.46 GB, head 8.93 GB, converter 0.26 GB, plus the target's embedding, lm_head
  (2.54 GB each in BF16) and layers 62-63: about 18-19 GB in torch. It does not fit beside the served forge, so it
  runs with forge down.

`JetSpec/jetspec-plus-plus-qwen3p8-27b` (public, revision 3147a351; configs in
`/Volumes/SSD4TB/anemll-forge-research/jetspec_pp/meta`):
- **Files:** 12.7 GB in all: the drafter (3.46 GB, 5 Qwen3 layers, block 16, the same taps 5 / 19 / 33 / 47 / 61 as
  our DFlash2), an 8.9 GB "ctxkv" adapter head built from copies of target layers (stream depths 1, 1, 1, 4, 5), and a
  converter. A decoder is also cut from the BF16 target's shards 16-18 (present locally).
- **Engine:** its foresight drafting and self-refinement live in the GitHub engine, which needs CUDA (two GPUs with
  peer access, custom kernels).
- **Scope:** an offline acceptance check needs the download, a port of the reference decode to MPS / CPU (about 20 GB
  of bf16 weights in torch, so forge has to be down), and fresh tap captures from our target (V8 taps). Estimated 3-5
  hours; parked for a go / no-go.

### Port, forge-up part done (16:15; `jetspec/` and `pipe/v8_server_tapdump.py`, sources in `v8_work/jetspec`)

Forge stayed up on 8765 throughout; no real JetSpec++ weights were loaded.

**Approach: replay.** The public torch reference (JetSpec 41a30357, `draft/reference`) runs unchanged through its own
acceptance reader (`bench/jetspec_plus_plus/reader.py` read_passes -> `core/llm.py` generate_tree, width-one B16,
`paper_eval` J3-R3, 16 keyed rows). Only the target is replaced, by a replay of a captured trace of our served model:
prompt + greedy stream token ids, and the 5 tap features of every position.
- **Exact at temperature 0:** the verify reads only the target's argmax at each node and the taps of accepted nodes.
  In a chain, an accepted node is always on the stream, so both are in the trace. Off-stream nodes get argmax V-1 and
  zero taps, and neither is ever read.
- **The target object** holds only what the drafter side reads: embed (on the CPU), layers 62 (GDN) and 63 (full),
  norm and lm_head, from `/Volumes/SSD4TB/Qwen3.8-27B`. `target._cache` stays None, so the reference anchor decoder
  builds its layer 62/63 prefix state from the tap-61 features, as the reference does without a live cache. The
  reference's own `build_drafter` checks the 5 sha256 pins.

**MPS fixes:** `anchor_offers`, `blend` and the tree top-k helper compute in float64, which MPS lacks; the harness runs
those three on the CPU. Everything else runs on MPS.

**Two cost and memory changes (both checked to change nothing):**
- **`--rows outcome`:** the paper protocol drafts 16 keyed rows per round, each a full J3-R3 schedule (4 drafter
  passes, 4 adapter passes, 3 decoder refreshes over the whole context), and keeps one. Each row's random block comes
  from a fresh generator (seed ^ prompt_index), so rows are independent. The replay knows the stream, so it drafts
  only the rows that can be the true outcome (drafts 1..k on the stream, anchor = the next stream token) and gives the
  others a placeholder that raises if read.
  - Tiny test: the same accept lengths, hits / misses (6 / 57 and 2 / 61) and passes as `--rows all`, with 252 instead
    of about 4,270 drafter passes.
  - A unit test with k > 0 keeps the true key.
- **Low-memory adapter loader:** the reference `load_adapter` builds the 4.47 B-parameter adapter head in fp32 on the
  CPU (17.9 GB) beside the 8.9 GB checkpoint, about 27 GB on a 32 GB Mac. The replacement builds on the meta device and
  streams each tensor to the device. The only buffers (12 rotary `inv_freq`) come from a narrow copy: they are the same
  at widths 64 and 160 and equal the closed form. It matches the reference loader bit for bit (156 tensors, CPU and
  MPS).

**Checks done with forge up:**
- **Tiny release-shaped artifacts** (`make_tiny.py`: hidden 64, vocab 512, 64 target layers, the same taps / tail /
  B16 / loaders / pins / thresholds): the whole J3-R3 loop runs on CPU and MPS. The replay is lossless (output = trace),
  and MPS gives the same hits / misses as CPU. tau = 1.0 is expected with random weights.
- **Dry check of the real artifacts** (modules on the meta device against the safetensors headers): drafter 58 / 58
  tensors, adapter head 144 / 144, replay target 28 tensors (shards 3, 16, 17, 18). No missing, extra or shape
  mismatch.
- **Capture wrapper** (`pipe/v8_server_tapdump.py`, imports `v8_server.py`, which is not edited), unit-tested against
  a fake engine: the prompt is swapped to the pinned ids, snapshots are cleared (cold), every position's taps land at
  its row, and duplicates are counted.
- **Capture inputs:** the served build v8_dec8 has no residual rotation (`rot1_runtime`: basis None), so the taps that
  `add_context` sees are in the basis JetSpec++ was trained on. The tap order is 5 / 19 / 33 / 47 / 61 in the DFlash2
  config, the V8 manifest and JetSpec++.
- **Pitfall:** without thinking, the server's default presence penalty is 1.5 (`--presence-nothink`). The capture is
  still greedy, because at temperature 0 `dist()` returns argmax before any penalty; the client sends temperature 0,
  presence 0, DRY 0, min_p 0 and thinking off anyway.

**Next (needs forge paused; about 1-1.5 h estimated, the per-round time is measured in step 2 first):**
1. **Capture** (about 15-20 min, port 8771, `jsp_capture_server.sh`, which refuses to start while 8765 listens):
   gsm8k prompts 0-31 of `27b-prompts_gsm8k128.json` (the paper's first group), max 2,112 tokens.
   - The same server gives our DFlash2 tokens per verify on the same prompts (meta decode_stats), the baseline to
     compare against.
2. **Replay** (memory about 19 GB, so forge must stay down; swap watchdog):
   - Check `--rows all` against `--rows outcome` on 1 trace x 64 tokens (equality plus s / round).
   - Then J3-R3 over the 32 traces at the paper cap of 2,048.
   - Estimate about 1-1.6 s per round (about 71 GB of weight reads per driver run, 1-2 runs per round with
     `--rows outcome`), so about 1-1.5 min per gsm8k answer.
3. **Restart the served forge** (`PORT=8765 v8_server_dec8_qk.sh`). math500 0-31 is optional after that.

## Deploy: QKE=tg serving (19 Nov 06:52)

`pipe/v8_server_dec8_qk.sh` is `v8_server_dec8.sh` with `V8_SRV_DYLIB=libgpu_v8_f8q.dylib` and `QKE=tg`. Rollback is
the plain launcher. The banner reads `qk=attn_qk8isk_tg ... dylib .../libgpu_v8_f8q.dylib sha256 ea591258bce...`, on
top of FP8 x and the keep-alive (`libdec_gpu_clk.dylib` 2b64e979a91a414f).

Check: the same cold 32K request (32,751 tokens, fixed salt, temperature 0) before and after the deploy, with the
prompt's last-token logits dumped by the server (`PREFILL_LOGITS_DIR`, a new env-gated hook in `v8_server.py`). The
logits are bit-identical (KL 0) and the 64-token text is identical (`qkf8/deploy/`). Prefill was 577 / 573 tok/s,
single cold runs.

## 9a. Hidden-decode LUT GEMM at 95K, end to end (`gdn/ws_e2e/`)

`libgpu_v8_gdnw.dylib` (`gpu_v8_gdnw.swift` = f8q + `kernels_gdnp.metal`; `I8TV=_k256ws` runs every INT8 LUT GEMM as
`lut_gemm32_i8t_k256ws`). Both arms use the same binary, `QKE=tg`, interleaved k / ws / k / ws.

| arm | tok/s a / b | DeltaNet s | attention s | DRAM total / GPU GB/s |
|---|---|---|---|---|
| served k256 | 397.2 / 398.2 | 88.3 / 87.5 | 150.3 / 149.8 | 118.5 / 74.5 |
| k256ws | 401.8 / 401.0 (+0.9%) | 86.6 / 87.0 | 149.0 / 149.2 | 119.5 / 75.5 |

The last-token logits are bit-identical. In the engine, a DeltaNet call takes 2.029 vs 1.942 ms alone and 2.490 vs
2.489 ms under the 140 GB/s load; the harness's -0.13 ms does not survive the real command buffer, which already
overlaps the GEMMs with the other kernels.

## 9b. 256-row DeltaNet calls (`gdn/g256/`)

**Implementation:** a second engine instance for the DeltaNet layers only (`gpu_v8_g256.swift`, module `gpuv8g`, T=256,
GDN-only manifest, `v8_submit_quad` over four 64-row ANE blocks), plus the scheduler copy
`unit/power/v8_engine_trace_g256.py` (generated by `make_g256_engine.py`; units (0,1) and (2,3) form quads for DeltaNet
layers, attention stays as 128-row pairs on the served engine).

**Exactness:** `rms_partial`'s summation order depended on T (`NG = 256 / T`), so `rms_partial_g256` restores the
T=128 order. After that, one quad matches two pair calls bit-exactly, as do six chained DeltaNet layers over two spans
and a mixed 8-layer chain (DeltaNet quads + attention pairs).

**The pipeline was not bit-exact, and the outputs varied run to run.** Dumps of every job's inputs and outputs
(`V8_DUMP`) showed the input of layer L+1 differing while the GPU output of layer L matched. The cause: both units of
a quad are released by the same GPU completion, so two ANE calls of the same `ffn2_L` function start at the same instant
from two threads, and they corrupt each other's outputs. A per-layer lock around the ANE call (`V8_ANE_LOCK=1`)
makes the pipeline deterministic and bit-exact against served.

The served schedule overlaps the same function only staggered and was exact in every test, but the hazard is latent
there too.

**Speed:**
- **Kernel level** (`gdn_quad_check.py`), ms per 256 tokens, quad vs two pairs: alone 3.98 vs 3.91 (+1.8%); with
  hidden decode, 4.02 vs 3.90 alone and 4.92 vs 5.00 under load (-1.6%, 0.04 ms per 128-token call).
- **End to end, 4K:** 651 vs 705 tok/s (-8%), bit-exact with the lock.
- **95K:** the 256-row arm stopped with a GPU command-buffer error at layer 12 (`v8_wait 12 failed`). Not debugged
  further.

**Verdict:** the 256-row pass does not pay. Weight traffic is not what limits the DeltaNet GEMMs, and the coupled
schedule costs ANE / GPU overlap. Not served.

## 12. FP8 in the DeltaNet prefill (analysis only)

Each estimate uses section 3's profile, scaled to e2e (2.45 ms per call, 35,712 calls at 95K, 12,288 at 32K, walls
237 / 56.5 s, GPU-bound).

| candidate | max saving, 95K / 32K wall |
|---|---|
| FP8 activations and FP8-decoded weights in the LUT GEMMs | 0: INT8 MMA 39.5 >= FP8 36.9 TOPS; activations already 1 B |
| FP8 in the recurrence as it is | 0: fp32 ALU FMA; no FP8 ALU path |
| FP8 operands in a chunked recurrence (the extra over fp16 chunked) | 0.6% / 0.8% |
| FP8 conv / gates (0.04 ms in total) | 0.4% / 0.5% |
| FP8 low-rank correction (overhead-bound, 26 us of math) | about 0 |
| for reference, not FP8: the chunked recurrence in fp16 on the matrix units | 2.4% / 3.5% |
| for reference, not FP8: fp16 glue intermediates | 1.5% / 2.2% |

No FP8 candidate reaches the 2% gate, so nothing was prototyped. The chunked recurrence is a separate (non-FP8)
candidate for a later item.

## 11. Short-context attention placement (`place/`, `power/place_chain.sh`, `power/ane_place.py`)

In the served stack every new span of >= 256 tokens already runs V8, with attention on the GPU (the same FP8 x +
QKE=tg causal kernels as at 32K-95K; e.g. a 2047-token cold prefill: `v8: 2047 tok ... 542 tok/s`). The placements that
put attention on the ANE are ANE-only (the bridge's p64 prefill entries) and V8 with the attention projections on the
ANE (attproj from 0, the core on the GPU). Fresh processes, two rounds, v8_trace's 45 s idle; forge was down.

| prompt | served V8, GPU attention (a / b) | V8 + projections on the ANE | ANE-only |
|---|---|---|---|
| 2K | 688 / 688 | 616 / 619 (-10%), KL 1.3e-4 | 326 / 325 (-53%), KL 0.014 |
| 4K | 710 / 713 | 618 / 637 (-12%), KL 5.6e-6 | 331 / 331 (-54%), KL 4.3e-4 |
| 8K | 696 / 698 | 637 / 654 (-7.5%), KL 3.1e-4 | 302 / 302 (-57%), KL 2.3e-3 |
| 16K | 652 / 654 | 636 / 633 (-3%), KL 4.5e-6 | 256 / 255 (-61%), KL 2.3e-4 |

**No crossover in 2K-16K: GPU attention (served) wins at every length.** The projections-on-ANE gap narrows with
length but does not cross. Served a vs b is bit-identical; every arm keeps the same top-1 token.

## 13. Single-call tree verify on the ANE (analysis, `tree/`)

**Math.** The ANE verify entries (`v8_<ctx>k`, 8 rows) already compute DeltaNet in the chunkwise WY / UT form over the
block. The causal structure is all `tri(T)` constants: the inclusive / strict masks, the `pair` decay matrix, the
Neumann-product triangular solve, and o = (q e^cum) S1 + ((q k^T) * pair) vn. The entries also already keep a lazy-commit
tape (pending k / u / wk / cum rows, `commit` / `commit_last` folded into the state by the next call).

Replacing `tri(T)` with an ancestor-mask input gives each row its own path's output from the committed state. In
topological order the solve stays strictly lower-triangular, so it is exact. This is Splash 1.7.0's
`gdn_decode_transform`: no per-branch state, and the accepted path (an ancestor mask) folded in by the next call. Two
more changes are needed:
- **Conv:** each row needs its 3 nearest ancestors, not the 3 previous rows: a per-tap selection-matmul input.
- **Attention:** the intra-block causal mask becomes the ancestor mask. RoPE positions (start + depth) are already
  per-row inputs, and the accepted path's K / V rows are appended host-side in path order.

So a single-call tree verify needs new verify entries (16 rows, mask inputs) on the same weight constants and the same
format: a rebuild, no second weight copy. Not built (no new ANE build without asking).

**Cost** (measured in one process, `ane_place.py`; the 16 / 32-row costs are linear in rows between the two measured
calls):

| ctx | 8-row verify | 64-row call | per extra row | 16 rows | 32 rows |
|---|---|---|---|---|---|
| 8K | 93.3 ms | 191 ms | 1.75 ms | +15% | +45% |
| 16K | 97.5 | 213 | 2.07 | +17% | |
| 32K | 104.9 | 259 | 2.75 | +21% | +63% |

**Trees** (`tree/tree_sim_report2.json`): dynamic top-N trees (EAGLE-2 / qwen-spec style: the N nodes of highest path
probability, p(child | parent) = softmax of the served selector score), 964 cycles of low-acceptance text.

| tree | nodes | rows | tok/verify |
|---|---|---|---|
| flat (served) | 7 | 8 | 3.064 |
| dynamic top-7 | 7 | 8 | 3.348 (+9.3%) |
| dynamic top-15 | 15 | 16 | 3.761 (+22.7%) |
| dynamic top-24 | 24 | 25 | 3.965 (+29.4%) |
| dynamic top-31 | 31 | 32 | 4.064 (+32.6%) |

Fixed shapes are weaker: [2,1,...] 3.279; [2,2,1,...] (26 nodes) 3.517.

**Projection** (the measured flat cycle; the verify grows by the per-row slope):

| ctx | flat | tree in 8 rows | tree in 16 rows | tree in 32 rows |
|---|---|---|---|---|
| 2K | 31.0 tok/s | 33.9 (+9.3%) | 33.7 (+8.7%) | 29.6 (-4.4%) |
| 32K | 21.9 | 23.9 (+9.3%) | 23.1 (+5.5%) | 19.5 (-11%) |
| 64K | 23.2 | 25.4 (+9.3%) | 24.4 (+5.0%) | 20.4 (-12%) |

16 rows pay while the 16-row verify costs under about +25% over 8 rows. With these linear estimates, a tree in the
current 8 rows (ancestor-mask inputs in place of the `tri` constants, the same row count) is as good as 16 rows.
qwen-spec reports trees at +40% per pass on open chat (about +10% on code / math), so higher-acceptance text should
favour bigger trees.

**First step if approved:** one chunk's 8-row and 16-row tree entries (a small test build) to measure the real cost
instead of the linear estimate.

## 14. Splash 1.7.0 vs forge (`splash170/`)

**Splash:** 1.7.0 in its own git worktree (`ML_playground/incoai_splash_170`, detached at 61f6d516, `make -j4` in 33 s,
its own `.venv` from the 1.3.2 bundle's Python; the 1.6.1 checkout and brew untouched). Model `unsloth/Qwen3.8-27B-GGUF:
UD-IQ3_XXS` at revision 4ca72078 and the cached DFlash2 draft, as on 18 Nov. Defaults: `--verify-rows auto`, i.e. tree
verify of up to 16 rows chosen by measured step costs, plus prompt lookup; `--offline`. It reports an ANE FFN split at
share 0.76. Its start-up measured decode step costs at one lane on this M6: 8 rows 111.8 ms, 15 rows 112.5, 16 rows
112.8, 2 rows 95.1.

**Forge:** `v8_server_dec8_qk.sh` (FP8 x + QKE=tg + keep-alive).

**Harness:** `scripts/m6_compare_bench.py` as on 18 Nov, each server alone; whole-machine W from mactop; idle 4.9 /
4.2 W.

### Clean, matched, cool run: the reference numbers (13:10-13:55, `splash170_cool/`)

**Setup:**
- **Machine:** external fan on the M6; the JetSpec++ download (network only) kept running.
- **Grid:** one continuous run, Splash 1.7.0 first, then forge, each server alone.
- **Bench:** `scripts/m6_compare_bench.py` once per context, so every context has its own 30 s idle baseline (3.7-4.3
  W). Cold prompt = entry minus 512 tokens, 256 greedy tokens, 3 identical repeats (decode = median repeat).
- **Cooldowns:** at least 180 s before each server, until the SoC was at most 45 C. Before each context, until the SoC
  was at most 48 C.
- **Temperatures:** recorded before each server and context (`temps.jsonl`, mactop). The script is
  `splash170/compare_cool.sh`, now in `splash170_cool/`.

**Banners** (`banner_*.txt`, printed before any timed request):
- **Splash 1.7.0:** git 61f6d51643a2cf8b, binary sha256 4c20b1d043a474cb..., UD-IQ3_XXS GGUF (rev 4ca72078) sha256
  c0b7c3038681ed2e..., DFlash2 draft 015e7956, defaults (`--verify-rows auto`) plus `--offline`.
- **Forge (served config):** `v8_server_dec8_qk.sh` sha256 a2d7ffc067e4f852..., build v8_dec8 manifest sha256
  b59b6319c78ed245.... Prefill libgpu_v8_f8q.dylib sha256 ea591258bceafe6d... (F8=x, `attn_qk8isk_tg`, e4m3 V copy).
  Decode `dec_attn_r4` in libdec_gpu_clk.dylib sha256 2b64e979a91a414f..., `dec_kernels.metal` sha256
  00f351b5edc893b7.... V8_DEC8=auto; keep-alive hold 60 ms; `ctl/ka` = on. Only the 96K requests ran split steps.

**Starting temperatures** (SoC / GPU C, fan 0 RPM; thermal state Nominal throughout):

| ctx | Splash | forge |
|---|---|---|
| before load | 36.8 / 36.5, 999 | 40.5 / 40.1, 1000 |
| 2K | 45.2 / 44.9, 1000 | 41.3 / 41.0, 1000 |
| 8K | 47.3 / 47.2, 1000 | 44.6 / 44.3, 1001 |
| 16K | 48.5 / 48.0, 1001 | 45.6 / 45.3, 999 |
| 32K | 47.1 / 46.9, 999 | 47.9 / 47.8, 999 |
| 64K | 48.1 / 48.1, 1000 | 47.5 / 47.4, 1001 |
| 96K | 47.3 / 47.2, 1397 | 47.1 / 46.8, 1000 |
| after the last request | 64.9 / 64.9, 2617 | 58.0 / 57.9, 2165 |

**Prefill** (cold request, from request to first token; W = mactop `total_power`, whole machine; net = minus that
context's own idle baseline):

| ctx | prompt | prefill tok/s, Splash / forge | W gross, S / F | W net, S / F | J per prompt token gross, S / F | J net, S / F | mactop samples, S / F |
|---|---|---|---|---|---|---|---|
| 2K | 1,529 | 481 / 599 (+24%) | 19.0 / 15.4 | 14.8 / 11.6 | 0.040 / 0.026 | 0.031 / 0.019 | 2 / 2 |
| 8K | 7,673 | 489 / 685 (+40%) | 43.1 / 48.2 | 38.8 / 44.1 | 0.088 / 0.070 | 0.079 / 0.064 | 13 / 9 |
| 16K | 15,865 | 464 / 650 (+40%) | 46.6 / 48.7 | 42.3 / 44.6 | 0.100 / 0.075 | 0.091 / 0.069 | 29 / 20 |
| 32K | 32,249 | 425 / 576 (+36%) | 45.7 / 48.0 | 41.5 / 43.8 | 0.108 / 0.083 | 0.098 / 0.076 | 64 / 47 |
| 64K | 64,953 | 360 / 465 (+29%) | 44.2 / 46.9 | 39.9 / 42.7 | 0.123 / 0.101 | 0.111 / 0.092 | 153 / 117 |
| 96K | 97,785 | 314 / 389 (+24%) | 44.0 / 45.4 | 39.6 / 41.5 | 0.140 / 0.116 | 0.126 / 0.107 | 264 / 211 |

**Decode** (median of 3 repeats, first to last token; cycle ms: forge from its server log, Splash = tok per cycle / tok/s):

| ctx | decode tok/s, Splash / forge | W gross, S / F | W net, S / F | J per generated token gross, S / F | J net, S / F | Splash tok / cycle, forge tok / verify | cycle ms, S / F |
|---|---|---|---|---|---|---|---|
| 2K | 59.8 / 62.2 (+4%) | 42.1 / 23.9 | 37.9 / 20.2 | 0.702 / 0.384 | 0.631 / 0.324 | 6.9 / 6.56 | 115 / 108 |
| 8K | 61.2 / 59.0 (-4%) | 43.6 / 24.6 | 39.3 / 20.6 | 0.710 / 0.416 | 0.640 / 0.347 | 7.3 / 6.24 | 119 / 108 |
| 16K | 57.8 / 58.3 (+1%) | 44.4 / 24.6 | 40.1 / 20.5 | 0.766 / 0.421 | 0.692 / 0.351 | 7.1 / 6.40 | 123 / 112 |
| 32K | 53.2 / 53.1 (0%) | 41.8 / 24.7 | 37.5 / 20.4 | 0.782 / 0.463 | 0.703 / 0.383 | 7.1 / 6.24 | 133 / 120 |
| 64K | 46.9 / 49.9 (+6%) | 42.7 / 24.6 | 38.5 / 20.4 | 0.909 / 0.492 | 0.818 / 0.407 | 6.7 / 6.56 | 143 / 134-135 |
| 96K | 44.4 / 45.3 (+2%) | 43.2 / 27.4 | 38.9 / 23.5 | 0.970 / 0.602 | 0.873 / 0.518 | 6.9 / 6.40 | 155 / 144-146 |

**Net savings, forge vs Splash 1.7.0** (1 - forge J / Splash J, net J per token):

| | 2K | 8K | 16K | 32K | 64K | 96K |
|---|---|---|---|---|---|---|
| prefill | 39% | 19% | 24% | 22% | 17% | 15% |
| decode | 49% | 46% | 49% | 46% | 50% | 41% |

**What the cool run changes against the earlier runs** (below):
- **Prefill:** forge is +24% to +40% at every length (earlier +20% to +36%). The 16K dip earlier (579) was thermal:
  650 now. Forge's prefill is 599 / 685 / 650 / 576 / 465 / 389 tok/s against v1's (16 Nov) 586 / 658 / 620 / 541 /
  420 / 347: +2%, +4%, +5%, +6%, +11%, +12%. FP8 x and QKE=tg grow with context.
- **Decode:** within -4% to +6% of Splash at every length. Splash 1.7.0's trees accept 0.1-1.1 more tokens per cycle;
  forge's cycle is 7-13 ms shorter at every length.
- **Decode energy:** forge uses 41-50% less net energy per generated token: 20-24 W net against 38-40.
- **The 96K forge cycle** (144-146 ms) is 2-4 ms slower than 16b's arm A1 (142 ms, a different nonce, so different
  text). The keep-alive was confirmed on (60 ms hold, switch on, 4 split requests).
- **Charts:** these two tables should replace the earlier ones; the energy-efficiency savings chart moves slightly
  (table above). Forge was left up on 8765 in the served config after the run.

### Earlier runs (08:39-08:56 at up to about 90 C after hours of benches; 96K: 12:04 Splash, 16b arm A1 forge)

**Prefill** (the cold request, from request to first token; W = mactop `total_power`, whole machine; net = minus that
run's own 30 s idle baseline):

| ctx | prompt | prefill tok/s, Splash / forge | W gross, S / F | W net, S / F | J per prompt token gross, S / F | J net, S / F | mactop samples, S / F |
|---|---|---|---|---|---|---|---|
| 2K | 1,529 | 484 / 598 (+24%) | 27.6 / 15.4 | 22.7 / 11.2 | 0.057 / 0.026 | 0.047 / 0.019 | 3 / 2 |
| 8K | 7,673 | 483 / 659 (+36%) | 47.2 / 45.1 | 42.3 / 40.9 | 0.098 / 0.068 | 0.088 / 0.062 | 13 / 10 |
| 16K | 15,865 | 448 / 579 (+29%) | 43.2 / 43.5 | 38.3 / 39.3 | 0.096 / 0.075 | 0.086 / 0.068 | 30 / 23 |
| 32K | 32,249 | 415 / 545 (+31%) | 43.9 / 45.4 | 39.0 / 41.2 | 0.106 / 0.083 | 0.094 / 0.076 | 66 / 49 |
| 64K | 64,953 | 358 / 459 (+28%) | 44.1 / 45.5 | 39.2 / 41.3 | 0.123 / 0.099 | 0.110 / 0.090 | 155 / 119 |
| 96K | 97,785 | 311 / 374 (+20%) | 43.1 / 44.5 | 38.9 / 39.5 | 0.139 / 0.119 | 0.125 / 0.106 | 268 / 220 |

**Decode** (the median of 3 repeats, first to last token; J per generated token = that repeat's energy / its tokens):

| ctx | decode tok/s, Splash / forge | W gross, S / F | W net, S / F | J per generated token gross, S / F | J net, S / F | Splash tok / cycle, forge tok / verify | mactop samples, S / F |
|---|---|---|---|---|---|---|---|
| 2K | 61.0 / 60.6 (-1%) | 43.1 / 24.4 | 38.2 / 20.1 | 0.705 / 0.400 | 0.625 / 0.331 | 7.11 / 6.40 | 3 / 4 |
| 8K | 60.6 / 59.1 (-2%) | 44.0 / 24.7 | 39.1 / 20.5 | 0.724 / 0.416 | 0.643 / 0.345 | 7.31 / 6.24 | 3 / 4 |
| 16K | 54.5 / 50.6 (-7%) | 41.1 / 24.6 | 36.2 / 20.4 | 0.751 / 0.484 | 0.661 / 0.401 | 6.92 / 5.57 | 4 / 4 |
| 32K | 54.1 / 57.5 (+6%) | 41.9 / 24.9 | 37.0 / 20.6 | 0.771 / 0.431 | 0.681 / 0.358 | 7.11 / 6.74 | 4 / 4 |
| 64K | 47.9 / 52.7 (+10%) | 42.6 / 24.5 | 37.7 / 20.3 | 0.886 / 0.464 | 0.784 / 0.384 | 6.92 / 6.92 | 5 / 4 |
| 96K | 43.3 / 46.4 (+7%) | 41.5 / 28.4 | 37.3 / 23.4 | 0.954 / 0.609 | 0.858 / 0.503 | 6.7 / 6.40 | 5 / 5 |

**Notes on these tables:**
- **Idle baselines** (each run's own 30 s, `total_power`; net = gross minus the baseline of the run the row comes from):
  - Splash: 4.906 W for 2K-64K (`splash170_ud_iq3xxs.json`); 4.213 W for 96K (`splash170_ud_iq3xxs_96k.json`).
  - Forge: 4.213 W for 2K-64K (`forge_qk.json`); 4.932 W for 96K (`dec_f8e2e/bench_A1.json`).
  - At 96K, forge's gross minus net reads 5.0 W only through rounding: prefill 44.454 - 4.932 = 39.522 W, decode
    28.354 - 4.932 = 23.422 W.
- **J values:** The earlier version of this table's J values used each context's mean W
  over its repeats; this one uses the median repeat's own energy, a difference of 0.5% at most.
- **96K row** (16d, 12:04-12:11, `splash170/splash170_ud_iq3xxs_96k.json`):
  - **Banner:** Splash git 61f6d516, binary sha256 4c20b1d043a474cb..., `Qwen3.8-27B-UD-IQ3_XXS.gguf` sha256
    c0b7c3038681ed2e..., draft DFlash2 015e7956, defaults (`--verify-rows auto`) plus `--offline`, context limit 256K.
  - **Splash decode repeats:** 44.2 / 43.2 / 43.3 tok/s, 6.9 / 6.7 / 6.7 tok/cycle.
  - **Forge side:** the served config at 96K from 16b's arm A1 (`dec_f8e2e/bench_A1.json`: v8_server_dec8_qk.sh, the
    96K entry's split verify, served `dec_attn_r4`), same bench and power method.
  - **Arm A2 at 96K:** prefill 373 tok/s at 44.3 W (net 40.5), 0.119 / 0.109 J per prompt token; decode 49.0 tok/s at
    29.1 W (net 25.2), 0.591 / 0.513 J per generated token, 6.74 tok/verify.
  - **Forge's first repeat** at 96K (43.5 / 43.1 tok/s) was slowed by 16b's logits dump. The medians are clean repeats.
- **Forge decode at 96K draws 28-29 W** against 24.5 at 64K and below: the GPU runs all 16 attention layers there,
  with the keep-alive.

- **Decode:** Splash 1.7.0's trees accept 0.4-1.4 more tokens per cycle than forge's chain. Splash is +1-8% faster at
  2K-16K; forge is +6% at 32K and +10% at 64K, at 41-48% less energy per generated token.
- **Prefill:** forge is +24-36% faster.
- **Splash 1.7.0 vs 1.6.1 (18 Nov):** decode 8K 56.3 -> 60.6, 16K 55.1 -> 54.5, 32K 51.1 -> 54.1, 64K 49.1 -> 47.9;
  prefill unchanged within 3%.
- **Forge vs 18 Nov:** server-side prefill is 4-10% below at 8K / 16K (659 vs 688, 579 vs 643) after hours of
  benches at about 90 C. The same-day V8 traces match 18 Nov's, so I read this as thermal, not an engine change.
- **Determinism:** Splash's three replies differed at 2K and 32K; its `--verify-rows auto` notes say greedy outputs can
  differ run to run. Forge's were identical.

## 13b. Single-call tree verify on the ANE, measured (`tree_probe/`)

`v8/build_tree_probe.py` built a SCRATCH package of one target group (layers 0-3: three DeltaNet layers + the attention
layer, the chunked FFN) on the served weight constants, with these entries at 8K / 32K / 64K of KV history:
- **chain8:** the served verify entry form (`Entry`, 8 rows).
- **tree8 / tree16:** the same with `anc` (T x T ancestor mask) and `ctap` (4 x T x (T + 3) conv-tap selection) inputs.
  While tracing, `tri()` returns the ancestor mask (DeltaNet WY / UT block and the attention's intra-block mask) and
  the causal conv reads its taps through `ctap`. tree16 is built with P = 16 pending rows.

The packages were compiled with forge.py compile (1 min) and timed with `power/tree_probe_time.py` (B.Plan.run, median
of 30). Banner: `tree_p8_L00-03.aimodel` sha 7bbf92ee86ac274f and `tree_p16_L00-03.aimodel`.

| KV ctx | chain, 8 rows | tree, 8 rows | tree, 16 rows |
|---|---|---|---|
| 8K | 5.06 ms | 5.34 (+5.6%) | 6.44 (+27%) |
| 32K | 5.79 | 6.08 (+4.9%) | 7.74 (+34%) |
| 64K | 6.75 | 7.04 (+4.3%) | 9.60 (+42%) |

- **Scale:** 16 groups plus the head is about the served 8-row verify (about 93 ms at 8K).
- **Masks:** a real tree mask times the same as a chain mask.
- **Correctness:** tree8 under a chain mask matches chain8 to 1.6-1.8e-3 relative in y (fp16 differences from the conv
  taps done as matmuls).
- **Cost:** the real 16-row cost (+27-42%) is above my linear estimate (+15-21%) and above the +25% break-even. On the
  ANE, 16 rows cost far more than on Splash's GPU step (+0.9% at one lane).

Projection with the measured costs and the tree_sim acceptance (964 cycles, low-acceptance text):

| ctx | flat chain | dynamic tree in 8 rows | dynamic tree in 16 rows |
|---|---|---|---|
| 2K | 31.0 tok/s | 32.3 (+4.2%) | 30.8 (-0.6%) |
| 32K | 21.9 | 22.9 (+4.6%) | 20.6 (-6%) |
| 64K | 23.2 | 24.4 (+5.2%) | about 21.2 (-9%) |

**Verdict:** a 16-row tree verify on the ANE does not pay. A dynamic tree inside the current 8 rows (top-7 nodes by path
probability, ancestor-mask inputs) is worth about +4-5% decode at these acceptance rates, and more on
higher-acceptance text. It would need a full build of tree8 verify entries (16 chunks, same weights).

The scratch build was deleted afterwards (its Core AI entries via `scripts/coreai_cache_delete.py --delete`, 1.2 GB
plus aned's programs, then the files). Free disk after: SSD 1.1 TB, internal 110 GB.

## 4. WL3A8 FFN on the ANE: prefill A/B and KL (`wl3a8/`; TEMPORARY build, deleted)

Setup:
- **WL3A8 build:** `v8/build_v8_ffn8x.py` built 16 FFN-only packages of `v8_ffn2c`'s layout (`ffn_L` / `ffn2_L`
  entries) with every FFN as w8a8/lut3 ServedL3 "i8gud": the release's 3-bit index constants, the palette as INT8
  codes, per-token INT8 activations into gate / up and down, INT8 x INT8. 6.3 GB, scratch, sha 357ad82cf45c37f3.
- **Baseline:** the FP16 FFN-only build `v8_ffn2c` (sha 97afdc3287778d07).
- **Harness:** `power/ffn_only.py` (FFN_ONLY_DIR in v8_trace.py / tf_kl.py) loads only the FFN-only packages plus the
  head, so one copy of the FFN weights is ever in memory and the served build was never loaded beside it. The KV is
  host memory (98,304 rows per attention layer).
- **GPU engine:** as served (libgpu_v8_f8q.dylib ea591258, F8=x, QKE=tg). Forge stayed down.

**Check:** the FP16 FFN-only harness is bit-identical to the served path at 4K, and base_a vs base_b are bit-identical
at 32K / 64K / 95K.

Prefill, interleaved fresh processes (base / wl / base / wl), tok/s:

| ctx | FP16 FFN (a, b) | WL3A8 (a, b) | WL3A8 vs FP16 |
|---|---|---|---|
| 2K (warm) | 693, 686 | 665, 678 | -2.6% |
| 4K | 713, 670 | 687, 692 | -0.3% |
| 8K | 694, 694 | 671, 674 | -3.1% |
| 32K | 571, 555 | 555, 549 | -2.0% |
| 64K | 461, 452 | 446, 444 | -2.5% |
| 95K | 396, 390 | 384, 388 | -1.8% |

**Why it loses (trace_summary):**
- **32K and longer** (the GPU is busy 99% of the run): WL3A8 frees the ANE (mid-run calls 3.6-3.8 ms vs 4.0-4.6 ms),
  but the GPU slows by DRAM contention (DeltaNet pair call 2.62-2.65 ms vs 2.50-2.57; attention +1-3%). The GPU is
  the bottleneck, so the run loses.
- **2K-8K** (the ANE is busy 97-99%): the INT8 FFN calls are no faster under contention (2K: 8.6-9.1 ms vs 8.1-8.5 in
  the first half). The FFN is bound by the weight stream (the same 3-bit index constants in both builds), so the
  INT8 x INT8 rate does not show up, and the activation quantization adds work.

**Accuracy:**
- **Last-token KL vs the FP16 FFN:** 6e-6 (4K), 2.4e-4 (8K), 2.5e-3 (32K), 1.3e-5 (64K), 6.8e-3 (95K); argmax the same
  everywhere.
- **Teacher-forced, all positions, vs the served F8=x capture:** KL mean 0.00125 / 0.00119 at 32K / 95K (p99 0.009,
  max 2.4); top-1 agreement 98.4%; NLL +0.0005 / +0.0001 nats. That is as much KL again as the served F8=x has
  against FP16 x (0.0012 / 0.0010).

**Verdict:** WL3A8 is 2-3% slower at every context and adds KL comparable to the whole FP8 x change. Not worth a
second package set.

**Cleanup:**
- **Build:** deleted after the test (its Core AI entries via `coreai_cache_delete.py --delete`, 6.8 GB plus aned's
  programs, then the 6.3 GB of files).
- **Captures:** the teacher-forced hidden-state capture (1.2 GB) was deleted too.
- **Free disk after:** SSD 1.1 TB, internal 98 GB. `v8_ffn2c`'s compiled cache (about 13 GB on the internal disk,
  recompiled for this baseline) is kept.

## 16. Real FP8 x FP8 P.V in decode GPU attention (`dec_f8pv/`)

**Kernel:** `dec_attn_r4f8` in a research copy of the decode kernels (`engine/dec_kernels_f8d.metal`; the served
`dec_kernels.metal` is untouched). It is the served `dec_attn_r4` with two changes:
- P = e4m3(256 exp(s - m)) in threadgroup memory.
- The V cache is read as an e4m3 copy: the same [kv][rows][256] layout, one byte, value = e4m3(code x vs), the prefill
  F8=x form.

The product goes through matmul2d FP8 x FP8 into FP32; the row sums are fp32 exp and LP carries the 256 factor.

**Bench:** `unit/dec_gpu_steady.py` timed one synthetic layer in steady state: 50 copies per command buffer, two
repeats, the same random INT8 cache, and the e4m3 copy made from it. Banner: libdec_gpu_clk.dylib sha 2b64e979a91a414f;
served metal sha 00f351b5edc893b7; research metal sha a5fcc944f4682a7f. DEC_SM=4 drops the P.V matmul and the V reads,
leaving the K side (INT8 Q.K, online softmax, the P store).

Exact 32K / 64K / 96K keys (rerun 11:07, `dec_f8pv/chain2.log`; ms per layer, two repeats each):

| keys | served r4 (FP16 P x INT8 V) | r4f8 (e4m3 x e4m3) | r4f8 vs r4 | r4 without P.V | r4f8 without P.V | V side, r4 / r4f8 | K side alone (r4 without P.V) |
|---|---|---|---|---|---|---|---|
| 32,768 | 0.548 / 0.562 ms | 0.552 / 0.553 | -0.5% | 0.320 / 0.321 | 0.318 / 0.320 | 0.235 ms = 143 GB/s / 0.234 = 144 GB/s | 105 GB/s |
| 65,536 | 1.069 / 1.069 | 1.070 / 1.068 | 0.0% | 0.626 / 0.628 | 0.625 / 0.624 | 0.442 = 152 / 0.445 = 151 | 107 |
| 98,304 | 1.606 / 1.612 | 1.589 / 1.592 | -1.1% | 0.921 / 0.914 | 0.919 / 0.919 | 0.692 = 146 / 0.672 = 150 | 110 |

- **Output checksums** (r4 / r4f8): 496.4 / 496.2 (32K), 369.8 / 369.6 (64K), 302.1 / 302.0 (96K).
- **r4f8 against r4:** rel L2 2.9e-2, cos 0.99958 at all three.
- **DEC_SM=2** (P rounded to FP8, FP16 x INT8): 1.1e-2, and 3-6% slower (0.581 / 1.106 / 1.658 ms).

The first run, at 64K / 95K / 120K (`dec_f8pv/chain.log`), gave the same picture:

| P (keys) | served r4 (FP16 P x INT8 V) | r4f8 (e4m3 x e4m3) | r4 without P.V | r4f8 without P.V | V side (full - no P.V) |
|---|---|---|---|---|---|
| 64K | 1.068 / 1.075 ms | 1.082 / 1.075 | 0.632 / 0.622 | 0.625 / 0.622 | r4 0.445 ms = 151 GB/s; r4f8 0.455 = 147 |
| 95K | 1.542 / 1.542 | 1.544 / 1.540 | 0.886 / 0.892 | 0.887 / 0.887 | r4 0.653 = 149 GB/s; r4f8 0.655 = 149 |
| 120K | 1.992 / 1.990 | 1.979 / 1.982 | 1.139 / 1.143 | 1.134 / 1.144 | r4 0.850 = 148 GB/s; r4f8 0.841 = 150 |

- **Speed:** r4f8 equals the served kernel within noise at every length (-1.1% to +0.7%). The whole layer streams at 126 GB/s of
  unique K / V.
- **Where the time goes:** the V side's marginal cost already runs at 147-151 GB/s, against the 156 GB/s plain read.
  The P.V matmul is not what limits the kernel, so a faster matmul form has nothing to win. The slow part is the K
  side: with the V reads removed, the kernel reads only K, at 107 GB/s (67 MB in 0.63 ms at 64K). The 17 Nov "103
  GB/s V side" figure was this split mis-attributed.
- **Error** against r4's output on the same cache (synthetic random codes):
  - r4f8: rel L2 2.9e-2, cos 0.99959. The V copy's 3 mantissa bits dominate; F8VS 16 gives the same cos.
  - DEC_SM=2 (P rounded to FP8, FP16 x INT8, the older dec_attn kernel): 1.0e-2, and 4% slower.

**Verdict:** the single layer does not win, so no e2e and no KL run (per the item's rule). FP8 x FP8 P.V has nothing to
give decode. The remaining decode-attention lever is the K side: Q.K plus the online softmax at 107 GB/s, where each
of the 3 query groups reads the same K tile.

## 13b with 32 rows: tree verify cost on the ANE, rebuilt (`tree_probe32/`)

**Build:** the scratch probe was rebuilt (`v8/build_tree_probe.py`, TREE_P = 32 / 8 / 16: one target group, layers
0-3, served weight constants) and compiled with forge.py compile (1m37s). Timing: `power/tree_probe_time.py`, median
of 30. Banner: `tree_p32_L00-03.aimodel` sha 284d7b23733b2842, plus the p8 / p16 packages (`tree_probe32/banner.txt`).

Per group:

| entry KV rows | chain8 | tree8 | tree16 | tree32 |
|---|---|---|---|---|
| 8K | 5.08 ms | 5.34 (+5.3%) | 6.38 (+25.8%) | 9.49 (+87%) |
| 32K | 5.81 | 6.09 (+5.0%) | 7.71 (+32.8%) | 11.68 (+101%) |
| 64K | 6.74 | 7.05 (+4.6%) | 9.39 (+39.3%) | 14.53 (+116%) |

- **Repeatability:** chain8 / tree8 repeat the first run within 1%, tree16 within 1-2%.
- **Masks:** a real tree mask times the same as a chain mask at 32 rows too.
- **Correctness:** tree8 under a chain mask matches chain8 to 1.6-1.8e-3.
- **Scaling:** the cost grows faster than the rows, and more so with context: the attention part grows as rows x KV.
  Splash's GPU step pays +0.9% for 16 rows.

**tok/s with tree_sim dynamic trees** (cost model as in section 15: the served cycle, with the verify scaled by the
measured ratio at the entry size):

| ctx | flat chain | top-7 (8 rows) | top-15 (16 rows) | top-31 (32 rows) |
|---|---|---|---|---|
| 2K, all 964 cycles | 28.1 | 29.4 (+4.5%) | 28.1 (0%) | 21.2 (-25%) |
| 2K, its own 150 cycles | 30.7 | 31.3 | 30.3 | 22.7 |
| 32K, all cycles | 22.7 | 23.8 (+4.9%) | 20.6 (-9.5%) | 14.8 (-35%) |
| 32K, its own 169 cycles | 21.8 | 23.2 | 19.7 | 14.1 |
| 64K, all cycles | 23.4 | 24.5 (+4.9%) | 21.2 (-9.4%) | 15.2 (-35%) |
| 64K, its own 645 cycles | 23.1 | 24.4 | 21.0 | 15.1 |

**Verdict:** on the ANE only a tree inside the current 8 rows pays (+4.5-5%). 16 rows break even at 2K and lose 9-10%
at 32K and longer; 32 rows lose 25-35%. The scratch build was deleted (Core AI entries 1.8 GB plus aned's programs,
then the 1.7 GB of files). Free disk after: SSD 1.1 TB, internal 98 GB.

## 15. Dynamic per-cycle verify size (`tree_dyn/`, analysis only)

**Simulator:** `power/tree_dyn.py` runs over the 964 logged cycles (temperature 0). Per cycle it builds tree_sim's
dynamic tree (best-first by path probability under the served selector score, top-4 children per node); its first 7 /
15 / 31 nodes are the 8 / 16 / 32-row trees (nested). Check: it reproduces tree_sim's 3.348 / 3.761 / 4.064 tok/verify.

**Expected acceptance:** E_N = the sum of the chosen nodes' path probabilities, with softmax(s / tau).
- **Uncalibrated (tau = 1):** overconfident, predicting 3.07 / 3.85 / 4.43 accepted against 2.35 / 2.76 / 3.06
  realized.
- **Calibrated:** tau fit by the log-likelihood of "node on the true path" on half of the requests and scored on the
  other half (2-fold). tau = 1.5 in both folds; it predicts 2.25 / 2.76 / 3.18.

**Cost model:** cost(rows, ctx) = served cycle(ctx) + verify8(ctx) x (ratio_rows(e) - 1).
- **Served cycle and verify8 (queue 5):** 2K 109 / 94.4 ms, 32K 135 / 121.0, 64K 131 / 117.3.
- **Ratios:** the linear fits over the entry's KV rows e of section 13b's measured ratios. Per 1K KV rows: 8 rows
  1.053 - 0.00012, 16 rows 1.243 + 0.0024, 32 rows 1.836 + 0.0051.
- **Entry sizes:** e = 8K at 2K, and 64K at both 32K and 64K: the served 32K decode runs the 64K-row entry, and the 64K
  split verify is assumed to scale like the ANE part.
- **Cycle costs (8 / 16 / 32 rows):** 2K 114.0 / 133.8 / 191.8 ms, 32K 140.6 / 183.0 / 275.5, 64K 136.4 / 177.6 / 267.2.

**Policies, tok/s over all 964 cycles** (count of cycles at 8 / 16 / 32 rows in brackets):

| ctx | fixed top-7 / 15 / 31 | per-cycle max (1 + E) / cost, tau 1 | the same, calibrated | throughput-optimal (Dinkelbach), calibrated | oracle (realized acceptance) |
|---|---|---|---|---|---|
| 2K | 29.38 / 28.12 / 21.19 | 28.88 [385 / 579 / 0] | 29.35 [596 / 368 / 0] | 29.85 (+1.6%) [694 / 270 / 0] | 31.96 (+8.8%) [701 / 233 / 30] |
| 32K | 23.82 / 20.55 / 14.75 | 23.56 [852 / 112 / 0] | 23.73 [912 / 52 / 0] | 23.88 (+0.3%) [917 / 47 / 0] | 25.08 (+5.3%) [859 / 89 / 16] |
| 64K | 24.54 / 21.18 / 15.21 | 24.28 [852 / 112 / 0] | 24.46 [912 / 52 / 0] | 24.61 (+0.3%) [916 / 48 / 0] | 25.85 (+5.3%) [859 / 89 / 16] |

On the context's own cycles, the calibrated per-cycle rule gives 31.14 (2K), 23.11 (32K) and 24.33 (64K), against
fixed top-7's 31.3 / 23.15 / 24.35.

**Verdict:** with the ANE's measured verify costs, choosing the size per cycle adds +0.3-1.6% over a fixed 8-row
dynamic tree. Even perfect foresight adds only +5-9%. 32 rows are never worth it.
- **The rule as specified** (per-cycle max expected tokens per ms) is not better than fixed top-7 (-0.1% to -1.7%).
  Taken alone, a per-cycle ratio over-picks big trees.
- **The throughput-optimal rule** (max (1 + E) - lam cost, with lam = the long-run tok/ms) is the right form if this
  is built. It needs calibrated probabilities (tau = 1.5 here).
- **What decides it:** the verify cost curve (16 rows +26-39%, 32 rows +87-116%). Splash's GPU step pays +0.9% for
  16 rows, which is why its auto verify-rows pays there and not on the ANE.
- **Recommendation:** a fixed dynamic top-7 tree in 8 rows (+4.5-5%, needs a full build of tree8 verify entries, 16
  chunks on the same weights), with the size rule left out.

## 16b. Why FP8 x FP8 P.V decode (`dec_attn_r4f8`) is slower than baseline `dec_attn_r4` (A1/B1 complete; A2/B2 & KL pending)

> **Note (12:13):** interim write-up from while A2 / B2 were running. The final results (all four arms, KL and output
> agreement aligned by generated token) are in the next section, "16b. Real FP8 x FP8 P.V in the end-to-end hybrid decode".
> The cold-reply comparison there uses the bench's stored text: A1 / B2 identical at all three contexts; B1 differs
> at 32K and 96K, and so does B2 against B1.

End-to-end hybrid decode A/B (`dec_f8e2e_chain.sh`, `m6_compare_bench.py`, arms A1 / B1 / A2 / B2 at 32K / 64K / 96K;
logs under `/Volumes/SSD4TB/anemll-forge-research/contention/dec_f8e2e/`). Arms A1 (served `dec_attn_r4`,
`libdec_gpu_clk.dylib` sha `2b64e979a91a414f`, `dec_kernels.metal` sha `00f351b5edc893b7`) and B1 (`dec_attn_r4f8`,
`libdec_gpu_f8d.dylib` sha `ad3783b24795d7c0`, `dec_kernels_f8d.metal` sha `e97f2c93f9cb1a18`, `DEC_F8V=1`, verified
live via `f8_rows_converted` and `f8_calls`) have completed all three context lengths; arms A2, B2, and step-logit KL
are still running and this section will be updated when they land.

### End-to-end hybrid decode results (A1 vs B1)

| ctx | A1 `dec_attn_r4` median tok/s (3 reps) | B1 `dec_attn_r4f8` median tok/s (3 reps) | A1 vs B1 verify ms/step | A1 vs B1 GPU attn ms/step | A1 vs B1 accept (tok/verify) | Output agreement (256 tok) |
|---|---|---|---|---|---|---|
| 32K (32,249) | **55.60** (51.5, 55.6, 56.7) | **54.78** (48.1, 55.0, 54.8; -1.5%) | 104.7-105.5 vs 105.4-106.0 | **9.5 vs 10.1** (+6.3%) | 81% (6.56) vs 77% (6.40) | Diverged at char 236: `min(high, value)` vs `min(value, high)` |
| 64K (64,953/54) | **49.97** (46.1, 50.0, 50.1) | **49.44** (19.9*, 49.4, 49.8; -1.1%) | 116.7-117.5 vs 118.1-119.6 | **18.0 vs 18.9-20.3** (+5.0-12.8%) | 78% (6.40) vs 78% (6.40) | Exact match |
| 96K (97,785/86) | **46.39** (43.5, 46.4, 46.7) | **47.60** (42.8, 47.7, 47.6; +2.6%**) | 126.2-127.2 vs 127.8-128.5 (cold 144.2) | **26.7 vs 27.9-28.1** (+4.5-5.2%) | 78% (6.40) vs 80% (6.56) | Exact match |

\* B1 64K rep 1 hit a 178-189 ms `ctx` stall (19.9 tok/s); steady cached reps 2-3 were 49.4 and 49.8 tok/s.
\** B1 96K median tok/s was higher only because acceptance moved from 78% (6.40 tok/verify) to 80% (6.56 tok/verify) on
that prompt; per-step verify (+1.2 ms) and GPU attention (+1.2-1.4 ms, +4.5-5.2%) were both slower than A1.

### Why `dec_attn_r4f8` loses in decode

1. **Decode attention is DRAM-bandwidth-bound, not compute-bound, and V is already 1 byte/element (INT8) near the DRAM ceiling.**
   In the single-layer steady-state microbench (`dec_f8pv/chain2.log`, 32K / 64K / 96K), the served `dec_attn_r4`
   (FP16 P x INT8 V) already reads V at **143-152 GB/s**—within a few percent of M6's **156 GB/s** plain-read
   ceiling—and runs at virtually identical latency to `dec_attn_r4f8` (0.555 vs 0.552 ms at 32K, 1.069 vs 1.069 ms at
   64K, 1.609 vs 1.591 ms at 96K). Meanwhile, the K side (`Q.K` + online softmax, `DEC_SM=4` no-P.V) is the slower
   half at **105-110 GB/s** (0.625 ms at 64K, 0.919 ms at 96K). Because both INT8 V and e4m3 V move the exact same
   1 byte per element, switching P.V compute to FP8 x FP8 saves zero V DRAM traffic and leaves the K-side bottleneck
   untouched.
2. **Decode's softmax P matrix is only `rows x seq_len` (8 verify rows, effectively 1-by-seq_len per query row) rather than prefill's `128 x seq_len` tile.**
   In prefill (`results/18nov/f8/README.md`), FP8 P.V is compute-bound over 128 query rows and doubles P.V throughput
   (47.2 -> 23.0 ms at 60K, 33 TFLOPS) while halving the 3.2 GB FP16 V copy to 1.6 GB e4m3. In decode, each query row
   attends to `seq_len` keys once with low arithmetic intensity, and V is already 1 byte/element in INT8, so FP8 P
   saves neither DRAM bytes nor compute cycles.
3. **Hardware matmul constraint forces e4m3 V conversion overhead with no mixed FP8 x INT8 path.**
   M6 GPU `matmul2d` has a fast FP8 x FP8 path (~33 TFLOPS) and no fast mixed FP8 x INT8 or FP16 x FP8 form. Using
   FP8 P therefore forces an e4m3 V copy (`f8_rows_converted` reached 10.97M rows across B1), which adds conversion
   work and slows end-to-end GPU decode attention by **+0.6 ms (+6.3%) at 32K**, **+0.9 to +2.3 ms (+5.0-12.8%) at 64K**,
   and **+1.2 to +1.4 ms (+4.5-5.2%) at 96K** compared to `dec_attn_r4`.
4. **Accuracy cost with no speed upside.**
   In the single-layer microbench (`dec_f8pv/chain2.log`), `dec_attn_r4f8` shows **rel L2 error 2.9e-2** (`cos 0.99958-0.99959`),
   dominated by e4m3 V's 3 mantissa bits (versus `DEC_SM=2` FP8 P x INT8 V at `rel L2 1.06e-2`, `cos 0.999944`, and
   prefill FP8 teacher-forced KL `4.3e-3` at 95K). End-to-end at 32K, that perturbation flipped a greedy token at
   character 236 (`min(high, value)` -> `min(value, high)`) and coincided with acceptance dropping from 81% (6.56
   tok/verify) to 77% (6.40 tok/verify). Step-logit KL from `lg_A1` vs `lg_B1` is pending completion of A2/B2.
5. **Audit notes on the prototype microbench code (`dec_kernels_f8d.metal`, `dec_gpu_clk.swift`, `dec_gpu_steady.py`).**
   Code audit of the initial item-16 microbench flagged three gaps: (a) `dec_prep` in `dec_kernels_f8d.metal` wrote
   INT8 V rather than e4m3 V for the 8 newly decoded tokens, (b) `dec_gpu_clk.swift` did not propagate `DEC_F8V` into
   `F8VS`, and (c) `dec_gpu_steady.py` printed its banner at the end without aborting on variant mismatch. For the e2e
   A/B (item 16b), Claude rebuilt a separate `libdec_gpu_f8d.dylib` (`ad3783b2`) + `dec_kernels_f8d.metal` (`e97f2c93`)
   with explicit startup banner checks (`EXPECT_DEC_ATTN`, `EXPECT_DEC_METAL`, `EXPECT_DEC_DYLIB`) and live `f8v=1` /
   `f8_rows_converted` counters.

## 16b. Real FP8 x FP8 P.V in the end-to-end hybrid decode (`dec_f8e2e/`)

**Setup:** the served hybrid decode: the ANE runs the 8-row verify FFN and DeltaNet plus the DFlash2 drafter, and the GPU
runs all 16 attention layers. The split verify is forced on at every context (V8_DEC8=on) in both arms. At 32K this is
not the served placement: served `auto` keeps the 64K entry's attention on the ANE. At 96K it is the served placement.

| arm | decode attention | dylib (sha256) | kernels (sha256) |
|---|---|---|---|
| A (served) | `dec_attn_r4`, FP16 P x INT8 V | libdec_gpu_clk.dylib 2b64e979a91a414fba14261a02db9db3a9e595cd008b3bece70db474ef74fdb2 | dec_kernels.metal 00f351b5edc893b73a8ef21433fe8fca425c51a14eac795e151c40e4f5767a1a |
| B (research) | `dec_attn_r4f8`, e4m3 P x e4m3 V copy | libdec_gpu_f8d.dylib ad3783b24795d7c07ded96464100579574a9a2de12409b85740eaf770c3f7ae1 | dec_kernels_f8d.metal e97f2c93f9cb1a184e38b784b2152f05369509018525ac04b8e86c5d16589d9a |

**The V copy:** B keeps an e4m3 copy of every INT8 V segment in GPU-private memory.
- **Upkeep:** `dec_v_to_f8` runs after `dec_prep` and converts the step's 8 new rows. After a position jump (new
  request) it converts the whole cache.
- **Check:** in the microbench, the in-GPU conversion reproduces the numpy one exactly (rel L2 2.870e-2, checksum 369.6
  at 64K). With F8V off, the research dylib is bit-exact to the served one.

**Guards:**
- **Banner:** `dec_split.py`'s banner prints at the start of every run, with dylib and kernels path, full sha256, and
  the variant reported by the dylib. EXPECT_DEC_* make the server refuse to start on a mismatch.
- **Counters:** per-request counters confirm the copy is maintained (f8_rows_converted, f8_calls).
- **Prefill:** both arms prefill as served: F8=x, `attn_qk8isk_tg`, libgpu_v8_f8q.

**Harness:**
- **Bench:** `scripts/m6_compare_bench.py` at 32,768 / 65,536 / 98,304: a cold prompt of the entry minus 512 tokens,
  256 greedy tokens, then 3 identical repeats (median).
- **Arms:** A1 B1 A2 B2, each a fresh server on port 8771 (`power/dec_f8e2e_chain.sh`). Forge on 8765 stayed down.
- **Timings** below are the server's per-request numbers, the median of the 3 repeats.

| ctx | arm | client decode tok/s (repeats) | server tok/s | tok/verify | accept | cycle ms | draft ms | verify ms | GPU attention ms / step |
|---|---|---|---|---|---|---|---|---|---|
| 32K | A1 | 55.6 (51.5 / 55.6 / 56.7) | 54.4 | 6.56 | 81% | 121 | 13.1 | 105.3 | 9.5 |
| 32K | A2 | 58.1 | 56.7 | 6.74 | 82% | 119 | 13.0 | 105.0 | 9.5 |
| 32K | B1 | 54.8 | 53.4 | 6.40 | 77% | 120 | 13.2 | 105.8 | 10.1 |
| 32K | B2 | 54.3 | 53.0 | 6.40 | 77% | 121 | 13.3 | 106.5 | 10.0 |
| 64K | A1 | 50.0 | 48.8 | 6.40 | 78% | 131 | 13.0 | 117.3 | 18.0 |
| 64K | A2 | 52.7 | 51.5 | 6.74 | 82% | 131 | 13.0 | 117.0 | 18.0 |
| 64K | B1 | 49.4 | 48.2 | 6.40 | 78% | 133 | 13.2 | 118.7 | 19.0 |
| 64K | B2 | 49.6 | 48.3 | 6.40 | 78% | 132 | 13.2 | 118.3 | 18.9 |
| 96K | A1 | 46.4 | 45.2 | 6.40 | 78% | 142 | 13.1 | 126.6 | 26.7 |
| 96K | A2 | 49.0 | 47.7 | 6.74 | 82% | 141 | 13.1 | 126.3 | 26.6 |
| 96K | B1 | 47.6 | 46.1 | 6.56 | 80% | 142 | 13.1 | 128.2 | 27.9 |
| 96K | B2 | 45.2 | 44.0 | 6.24 | 76% | 142 | 13.4 | 127.7 | 27.8 |

The prompt lengths were 32,249, 64,953 / 64,954 and 97,785 / 97,786 (the bench's per-run nonce).

**Like for like (same work per step), B against A:**

| ctx | verify ms | GPU attention ms / step | cycle ms |
|---|---|---|---|
| 32K | +1.0 ms (+0.9%) | +0.55 ms (+5.8%) | equal within 1 ms |
| 64K | +1.35 ms (+1.2%) | +0.95 ms (+5.3%) | +1.5 ms (+1.1%) |
| 96K | +1.5 ms (+1.2%) | +1.2 ms (+4.5%) | +0.5 ms |

- **Why B is slower:** the attention kernel is no faster end to end (as in the microbench), and the copy upkeep adds
  time. The whole-cache conversion at each request's first step works out to about 0.2 / 0.4 / 0.6 ms per step over a
  256-token reply, and the 16 per-step 8-row conversions add about 0.3 ms.
- **The tok/s spread** (A 56.9 / 51.4 / 47.7 vs B 54.6 / 49.5 / 46.4, client, mean of the two runs) follows
  tok/verify. Tok/verify varies with each run's continuation: A1 vs A2 alone spans 6.40-6.74 at 64K and 96K.

**Output agreement and KL:**
- **Method:** the cold requests' verify logits (DEC_LOGITS_DIR), followed along the committed path and aligned by
  generated-token index. KL(A1 || x) is over the positions up to and including the first divergence.
- **Prompts differ slightly:** the arms' prompts differ by the bench's nonce, so these KL values include that
  difference too.

| pair | identical tokens 32K / 64K / 96K | first divergence | KL mean (median, max) 32K | 64K | 96K |
|---|---|---|---|---|---|
| A1 vs A2 (noise floor) | 257/259 / 258/258 / 258/258 | 32K at token 74 | 2.2e-4 (1.0e-5, 2.8e-3) | 1.4e-4 (9.7e-6, 4.0e-3) | 8.8e-5 (2.8e-6, 5.1e-3) |
| A1 vs B2 | 259/259 / 259/259 / 258/258 | none | 3.2e-4 (3.2e-6, 1.7e-2) | 9.6e-5 (7.1e-6, 1.8e-3) | 1.1e-4 (3.9e-6, 3.2e-3) |
| A1 vs B1 | 256/259 / 259/259 / 257/258 | 32K and 96K at token 12 | 1.9e-3 (13 tokens) | 4.0e-4 | 2.8e-3 (13 tokens) |
| B1 vs B2 (same FP8 config) | 256/259 / 259/259 / 257/258 | 32K and 96K at token 12 | 1.4e-3 | 6.1e-4 | 2.7e-3 |

B1 diverges from B2 (the same FP8 config) exactly where it diverges from A1, so B1's run is the outlier (its prompt
differs by the nonce), not the FP8 path. A1 vs B2 is inside the A1 vs A2 noise. The cold replies' text is identical
for A1 / B2 at all three contexts.

**Verdict:** real FP8 x FP8 P.V gives no end-to-end gain. The verify step is +1.0-1.2% slower, the GPU attention +0.6-1.2
ms per step (copy upkeep, no kernel gain), and accuracy is within run-to-run noise. Not worth deploying.

**Files:**
- **Kept:** the logs, bench JSONs and `analysis.json` are in `dec_f8e2e/`.
- **Deleted:** the logit dumps (3 GB). The served `dec_kernels.metal` and libdec_gpu_clk.dylib are untouched.
- **dec_split.py:** gained only env-gated hooks (DEC_METAL, EXPECT_DEC_*, DEC_LOGITS_DIR, the variant line), all
  off by default. A backup is in `pipe/dec_split.pre_f8d.py`.

## 16c. What caps the decode attention's K side (`dec_kside/`)

**Bench:** one synthetic layer, `unit/dec_gpu_steady.py`, steady state (50 copies per command buffer), two repeats,
the served libdec_gpu_clk.dylib (sha 2b64e979a91a414f). The variants are in a research copy of the kernel,
`dec_attn_r4ks` in `engine/dec_kernels_ks.metal` (sha be0b561ab2618e37); the served `dec_kernels.metal` is untouched.
- **Correctness:** its full variant is bit-exact to the served kernel at 64K and 96K.
- **Key-major K:** the bench registers a transposed copy of the same random K. Its full variant is bit-exact at 64K, but
  not at 96K (rel L2 2.9e-2, checksum 301.4 vs 302.1), with the largest (65,536-row) segment in use. Unexplained, and
  moot because the layout is slower anyway.

ms per layer (mean of the two repeats; K bytes only for the K-side rows):

| variant | 64K | 96K | K read rate (64K / 96K) |
|---|---|---|---|
| served `dec_attn_r4` (full) | 1.068 | 1.600 | |
| research copy, full | 1.066 | 1.594 | |
| without P.V (the "K side") | 0.622 | 0.923 | 108 / 109 GB/s |
| **(a) Q.K only** (K stream + INT8 matmul2d, no softmax) | **0.563** | **0.833** | **119 / 121 GB/s** |
| **(d) softmax only** (no Q.K, no P.V) | **0.134** | **0.192** | |
| (b) key-major K, Q.K only | 0.664 (+18%) | 0.978 (+17%) | 101 / 103 GB/s |
| (b) key-major K, without P.V | 0.688 (+11%) | 1.019 (+10%) | |
| (b) key-major K, full | 1.113 (+4%) | 1.645 (+3%) | |
| (c') served, DEC_CK 1024 (2x the threadgroups) | 1.145 (+7%) | 1.700 (+6%) | |
| (c') served, DEC_CK 512 (4x) | 1.291 (+21%) | 1.936 (+21%) | |
| without P.V, DEC_CK 1024 | 0.689 (+11%) | 1.031 (+12%) | |
| Q.K only, DEC_CK 1024 | 0.603 (+7%) | 0.913 (+10%) | |

(c) DEC_M48 (one 48-row query group, K read once) was not rerun: on 17 Nov it measured 5.57 ms against 2.00 ms at 120K,
and the r4 kernel's lane mapping is fixed at 16 rows.

**Answer:**
- **The K side is the INT8 Q.K matmul2d pass itself.** Q.K alone takes 90% of the K side (0.563 of 0.622 ms at 64K) and
  streams K at about 120 GB/s, below the 156 GB/s plain read. The softmax epilogue costs 0.13 ms alone but only about
  0.06 ms net, since it overlaps the matmul.
- **Not the K layout.** The ANE's dims-major K, read as 256-byte strided rows, is the faster form for matmul2d. A
  contiguous key-major tile with a transposed operand is 17-18% slower.
- **Not too few threadgroups.** Smaller chunks (more threadgroups) are slower: +7% at CK 1024, +21% at CK 512.
- **Not the precision** (16 Nov): K is already INT8 and Q.K already INT8 x INT8.

**Ceiling of the lever:** at 156 GB/s the Q.K pass would take about 0.43 ms at 64K. That saves about 0.13 ms per layer,
2.1 ms per verify step (about 1.6% of a 131 ms cycle), and about 3.2 ms (about 2.3%) at 96K. Only where attention runs
on the GPU (the served 96K entry).

**Remaining candidates** (not tried):
- Overlap each K tile's load with the previous tile's compute: software pipelining, double-buffered threadgroup K.
- One threadgroup per KV head and chunk covering all 48 query rows with 12 simdgroups, so the K tile is read once
  instead of 3 times. This keeps today's thread count; 17 Nov's M48 used 6 simdgroups.

## QUEUE (live, updated 19 Nov 15:41: cool run and article v2 done; JetSpec++ next; forge on 8765 is UP, served config, ka on)

Rules for every item: banner with the effective flags and dylib hash; report only; no deploy and no commit without a
report; forge stays down unless asked.

1. ~~Queue 4, WL3A8~~ DONE 10:50 (section 4 above; build deleted, disk confirmed).
2. ~~Queue 16~~ DONE 10:54 (section 16 above: no gain, no e2e). Was: **Queue 16 (moved FIRST: the FP8 rollout / release comes before speculative-decoding work):** real FP8 x FP8 P.V
   in decode GPU attention (dec_gpu_steady.py, one layer at 64K / 95K / 120K, checksummed, banner): e4m3 P x e4m3 V
   through matmul2d FP8 x FP8, against the served FP16 x INT8 P.V (about 103 GB/s on the V side, against 156 GB/s
   plain read; 17 Nov DEC_SM=2 only rounded P to FP8). If the single layer wins: e2e decode + KL against the served
   FP16 x INT8. Prefill F8=x already keeps an e4m3 V copy.
3. ~~Queue 13b with 32 rows~~ DONE 11:03 (section above; scratch deleted). Was: **Queue 13b with 32 rows:** rebuild the scratch tree-verify probe (`v8/build_tree_probe.py`, TREE_P = 8 / 16 / 32,
   one group, layers 0-3); time chain8 / tree8 / tree16 / tree32 at 8K / 32K / 64K (`power/tree_probe_time.py`);
   project tok/s with tree_sim dynamic top-7 / 15 / 31; delete the scratch build afterwards.
4. ~~Queue 15~~ DONE 11:04 (section 15). Was: **Queue 15, dynamic verify size:** from 13b's measured costs, pick each cycle's rows (8 / 16 / 32) by maximum
   expected accepted tokens (tree_sim path probabilities) per ms of measured verify + draft cost, with a cost fit over
   context length. Simulate over the 964 logged cycles; report tok/s against fixed top-7 / 15 / 31 at 2K / 32K / 64K.
   Analysis only.
RUN ORDER from here (Anemll, 19 Nov ~12:05, supersedes ~11:50): 16b write-up -> 16d -> 16c -> JetSpec++.

5. ~~(1) Queue 16b~~ DONE (runs 11:58, write-up 12:04: section 16b, no gain): REAL end-to-end hybrid decode A/B.
   Arms: served dec_attn_r4 vs dec_attn_r4f8, the latter through the research dylib libdec_gpu_f8d.dylib and
   dec_kernels_f8d.metal with an e4m3 V copy. Both arms with V8_DEC8=on (all 16 attention layers on the GPU), at
   32,768 / 65,536 / 98,304, A1 B1 A2 B2, fresh servers on port 8771 (`power/dec_f8e2e_chain.sh`, analysis
   `power/dec_f8e2e_analyze.py`). Then finalize section 16b: A1 / B1 / A2 / B2 tok/s, cycle ms, acceptance, and KL /
   output agreement against A.
6. ~~(2) Queue 16d~~ DONE 12:11 (section 14 now has the 96K row and net columns). Was: **(2) Queue 16d: Splash 1.7.0 at 96K**, to fill the Splash side of a 96K row in section 14.
   - **Method:** exactly as `splash170/`: `scripts/m6_compare_bench.py` with mactop whole-machine power, the same
     UD-IQ3_XXS model and settings as `splash170_ud_iq3xxs.json` (worktree incoai_splash_170, 61f6d516, defaults
     plus --offline), `--ctx 98304`.
   - **Record:** prefill tok/s, prefill W, J per prompt token, decode tok/s, decode W, J per generated token, idle W,
     and the per-repeat details.
   - **Banner:** resolved flags, binary / model identity and sha256, printed outside the timed region.
   - **Output:** `splash170/splash170_ud_iq3xxs_96k.json`. Add the 96K row (Splash 1.7.0 vs forge), with prefill W and
     J/tok columns, to section 14.
   - **Forge side of the row:** 16b's A arms at 98,304 are the served config at 96K: the 96K entry splits by
     default, and they ran the same bench and power method.
7. ~~(3) Queue 16c~~ DONE 12:13 (section 16c: the INT8 Q.K matmul pass is the cap). Was: **(3) Queue 16c: K-side decode microbench**, one layer, `unit/dec_gpu_steady.py` at 64K, research kernels copy only:
   - (a) Q.K only, without the softmax.
   - (b) K in a contiguous key-major layout ([kv][rows][256], like V) instead of the ANE's dims-major layout.
   - (c) DEC_M48 (one 48-row query group, so K is read once).
   - (d) softmax only (no Q.K, no P.V), to price the epilogue.
   Question: what caps the K side at 105-110 GB/s? K is already INT8 and Q.K already INT8 x INT8; 17 Nov: removing
   exp -0.3%, removing all K work = removing all V work.
8. ~~Clean, matched, cool Splash 1.7.0 vs forge grid~~ DONE 13:55 (section 14, `splash170_cool/`; forge left up on
   8765).
9. ~~Article v2~~ DONE 15:40 (approved by Anemll): `ARTICLE_ANE_GPU_M6.md` is now Version 2 (19 Nov) with the cool-run
   tables and the 4 charts in `img/v2/`. The HTML is rebuilt; 15 images and 28 links checked, none broken. Not
   committed or pushed; the edits are local, for review.
   - **Restructured ~16:40** (Anemll's feedback): the article is now one V2 story against Splash 1.7.0.
     - A V8 -> V8.1 -> V8.2 table near the top, and every optimization step and decision tagged with its version.
     - V8 against Splash 1.3.2 is a short baseline subsection.
     - The coding-agent section is the 19 Nov Pi session on V8.2 (section 17).
     - The busy table carries the V8.2 shares.
     - HTML rebuilt: 15 images, 29 links, 0 broken; the tables match `sec14_tables.md`.
   - **Evening edits (~19:00):**
     - Step 9 is described as an on-chip scratch buffer for the Q·K write-back (Anemll's wording), in 4 places.
     - Anemll's diagram `img/v2/attn-fp8-island.png` ("Why FP8 only lives inside P·V") is wired into step 8 with a
       caption, and listed in "About the figures".
     - HTML rebuilt: 52 KB; a headless render shows 16 images and 10 tables; 0 broken links.
     - Anemll's diagram `img/v2/attn-qke-tg-scratchpad.png` ("Coalescing Q·K write-back with a 16 KB on-chip
       scratchpad") is wired into step 9 with a caption, and listed in "About the figures". HTML 53 KB; 17 images,
       10 tables; 0 broken links.
     - Anemll's "V8 to V8.2 TLDR" line added verbatim under "## From V8 to V8.2". Its numbers check out: prefill +12%
       at 95K; decode 150 / 133 = +12.8% tokens per second at 66K (cycle -11.3%). HTML 54 KB; 0 broken links.
10. **Queue 7, JetSpec++ port: the forge-up part is DONE 16:15** (section 7: replay harness, capture wrapper, tiny MPS
   test, dry check of the real artifacts). **WAITING for Anemll's OK on a forge-down window** (about 1-1.5 h): capture
   gsm8k 0-31 on port 8771, the replay, then restart forge on 8765.
   Was: keep forge serving while porting (it is down now by request); take the tap capture in one short window. Repo cloned at `/Volumes/SSD4TB/anemll-forge-research/jetspec_pp/src/JetSpec` (41a30357), HF
   meta in `jetspec_pp/meta`; the drafter is stock DFlash plus a ctxkv adapter and a target-layer decoder; the eager
   reference route needs the target in torch, so replace it with our tap captures.

Done today (see the sections above): 1-3, 5, 6, 8, 9a / 9b, 11, 12, 13 / 13b (8 / 16 rows), 14; QKE=tg deployed.
Parked: item 10 (hybrid drafter split: estimate 1-2% of the cycle, bandwidth-bound; a prototype only if asked).
