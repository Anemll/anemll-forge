# FP8 P.V on the served prefill attention (18 Nov 2026)

Prototype: `pipe/engine/kernels_f8.metal` + `pipe/engine/gpu_v8_f8.swift` -> `libgpu_v8_f8.dylib`. It is the served
srv3 engine (plus the fold switch) with an FP8 path on the KV8 + K8V16 attention: the softmax writes e4m3
probabilities, the GPU's V copy holds e4m3 values (one byte per value, half the fp16 copy's memory: 1.6 instead of 3.2
GB at 96K), and P.V runs FP8 x FP8 (matmul2d, fp32 accumulate). The M6 has FP8 x FP8 at about twice the FP16 rate and no
fast FP8 x INT8 or FP16 x FP8 form, so both operands are 8-bit. Q.K stays INT8 x INT8 and the scores stay fp16 (8-bit
scores before the max is subtracted are too coarse).

Variants (`F8=` at load; `F8=""` is the plain fp16 path in the same binary):

| `F8` | V copy | probabilities | softmax |
|---|---|---|---|
| `c` | e4m3 of the INT8 codes | e4m3(e x vs_j x c_tile), c_tile = 256 / the tile's largest value scale; P.V divides by c_tile | scalar, software e4m3 |
| `v` | e4m3 of the fp16 values x `F8VS` | e4m3(256 e), sum = 256 x F8VS x sum(e) | scalar, software e4m3 |
| `w` | as `v` | as `v` | half4 loads, uchar4 stores, e4m3 through half |
| `x` | as `v` | as `v` | as `w`, exp in half, mask only on the boundary chunk |

## Alone (`alone/`, `power/f8_check.py`, one-layer manifest `w_dec8/manifest_L3.json`)

GPU ms per 128-row attention pair call (layer 3, median of 12, seeded random INT8 KV), and the layer's mixer output
against the plain path (relative RMS, random data: near-uniform attention, the hard case for relative error):

| keys | plain | c | v | w | x |
|---|---|---|---|---|---|
| 2K | 2.405 | 2.314 | 2.281 | 2.286 | 2.293 (-4.7%) |
| 8K | 3.851 | 3.682 | 3.585 | 3.577 | 3.577 (-7.1%) |
| 16K | 5.792 | 5.467 | 5.283 | 5.308 | 5.234 (-9.6%) |
| 32K | 9.653 | 9.067 | 8.641 | 8.577 | 8.505 (-11.9%) |
| 60K | 16.446 | 15.256 | 14.589 | 14.188 | 14.198 (-13.7%) |
| 93K | 24.443 | 22.689 | 21.253 | 21.037 | 20.779 (-15.0%) |
| error vs plain | | 3.4% | 2.1% | 2.1% | 2.1% |

`F8VS=16` gives the same as `F8VS=1` on this data. The profile at 60K (8 calls, 248 tiles; `alone/p_*.log`):

| kernel | plain | v | x |
|---|---|---|---|
| P.V | 47.2 ms | 23.3 | 23.0 |
| softmax | 20.9 | 28.8 | 22.3 |
| Q.K | 36.5 | 37.2 | 37.3 |
| combine | 8.8 | 8.9 | 8.9 |
| total | 128.5 | 113.6 | 105.0 |

- **P.V is exactly twice as fast:** 33 TFLOPS, about 92% of the FP8 peak.
- **The first FP8 softmax lost part of that** to its software e4m3 conversion and byte stores. The vectorized
  half-precision form `x` is 1.4 ms above the plain softmax.

Where the error comes from, simulated in numpy (`ml_dtypes` e4m3; random scores and values): e4m3 values alone give
2.6% attention-output error, e4m3 probabilities alone 0.9-2.6% (less for peaked attention), both 2.8-3.8%. The values'
3 mantissa bits are the larger part. October's ANE `C2` form (FP8 probabilities only, INT8 values) had 2.8% attention
error and KL 0.0002-0.0005 to the previous build.

## Next to the ANE FFN (`f8_<K>k_x/`, `att8_win.py` + `coupling_run.py`)

| keys | attention alone: plain / x | next to the FFN: plain / x | FFN slowdown: next to plain / next to x |
|---|---|---|---|
| 32K | 9.52 / 8.15 ms (-14.4%) | 13.69 / 12.20 (-10.9%) | +13.0% / +9.7% |
| 60K | 16.22 / 14.09 (-13.1%) | 23.52 / 21.06 (-10.5%) | +5.4% / +3.1% |
| 88K | 22.76 / 19.25 (-15.4%) | 33.17 / 28.95 (-12.7%) | +2.4% / +3.2% |

FP8 keeps 11-13% of its gain under contention. It is not slowed more in absolute terms, but more in relative terms
(+50% against +45%), because it runs faster and is more bandwidth-bound.

## In the V8 prefill (`tr_plain_a`, `tr_f8x_a`, `tr_plain_b`, `tr_f8x_b`)

`power/v8_trace.py`, interleaved plain / FP8 / plain / FP8 processes, the same binary (`V8_DYLIB=libgpu_v8_f8.dylib`;
plain = `F8` unset), the same build, environment and prompt text:

| prompt | plain (a / b) | FP8 x (a / b) | change | GPU s, attention: plain / x | ANE busy: plain / x |
|---|---|---|---|---|---|
| 2K | 670.6 / 675.5 | 657.3 / 673.3 | -1.2% | 0.7 / 0.7 | 98.8 / 97.5% |
| 8K | 680.8 / 683.7 | 692.5 / 689.2 | +1.3% | 3.8 / 3.7 | 95.5 / 96.5% |
| 32K | 541.0 / 543.4 | 567.0 / 566.3 | **+4.5%** | 29.5 / 27.0 (-8.3%) | 80.6 / 85.0% |
| 64K | 416.2 / 419.1 | 451.6 / 452.4 | **+8.2%** | 93.8 / 82.6 (-11.9%) | 63.9 / 69.3% |

Attention per call, FP8 against plain, by decile of the prompt (the gain grows with the keys behind each call):

| prompt | 1st | 2nd | 3rd | 4th | 5th | 6th | 7th | 8th | 9th | 10th |
|---|---|---|---|---|---|---|---|---|---|---|
| 32K | -2.9% | -3.8% | -5.4% | -6.2% | -6.3% | -8.5% | -9.3% | -10.7% | -11.4% | -12.8% |
| 64K | -3.2% | -6.7% | -6.8% | -9.6% | -10.5% | -12.9% | -13.5% | -14.0% | -14.4% | -14.5% |

DeltaNet is unchanged (-0.1% to -0.9%).

- **Repeats agree:** plain runs within 0.5% of each other, FP8 runs within 0.2%, except 2K (657 / 673).
- **At 2K the change is noise:** the Neural Engine sets the pace there.
- **95K, traced after** (`tr_plain95_a`, `tr_f8x95_a`, `tr_plain95_b`, `tr_f8x95_b`, `chain_d.log`; one fresh
  process per run, swap steady at 735 MB): plain 353.8 / 351.7 tok/s, FP8 x 388.4 / 383.2, so **+9.4%**. The time model
  had estimated about 380.
  - GPU attention: 179.0 -> 156.4 s (-12.6%), the last tenth of the prompt -16% (25.9 -> 21.8 ms per call).
  - DeltaNet: unchanged (89.4 -> 88.7 s).
  - ANE busy: 52.9 -> 59.1%.

**Accuracy on real text, a first check** (`logits_kl.jsonl`, `power/logits_cmp.py`): the last-token logits of each run.
The runs are deterministic: plain a = plain b and x a = x b exactly.

| prompt | KL(plain \|\| x) | cosine | argmax | top-10 overlap |
|---|---|---|---|---|
| 2K | 4.9e-5 | 0.99991 | same | 10 / 10 |
| 8K | 1.6e-4 | 0.99985 | same | 10 / 10 |
| 32K | 4.6e-3 | 0.99971 | same | 10 / 10 |
| 64K | 5.2e-6 | 0.99993 | same | 10 / 10 |
| 95K | 4.3e-3 | 0.99932 | same | 10 / 10 |

One token per length is a weak measure; the teacher-forced check below covers every position. For scale,
KL(ANE verify || GPU split verify) on real text was 0.025 at 66K and 0.054 at 95K (17 Nov).

## Teacher-forced, every position (`tf_kl/`, `power/tf_kl.py`, `tf_chain.sh`)

The same prompt text, the served build, one fresh process per variant (plain, then `F8=x`), the same binary
(`libgpu_v8_f8.dylib` rebuilt with `v8_variant`, sha256 `50198bdf4f8b6b8b`; the timed binary is kept as
`libgpu_v8_f8.pre_info.dylib`, `e8fa74befb8ec380`, the only change an added C function). The engine copy
`v8_engine_tf.py` hands every block's final residual rows to a capture (`H_<N>.npy`, [N][5120] fp16, kept on the SSD:
`contention/fold/f8/tf_kl/<variant>/`); then the build's own head (`head_T8`, 8 rows per call) runs over both captures,
and each position's logits are compared. KL(plain || x), over all positions:

| prompt | positions | KL mean | median | p99 | p99.9 | max | top-1 same | top-10 overlap | top-10 all same | NLL plain / x |
|---|---|---|---|---|---|---|---|---|---|---|
| 32K | 32,768 | 1.20e-3 | 3.5e-4 | 6.8e-3 | 0.040 | 4.56 | 98.65% | 9.78 | 79.7% | 1.9478 / 1.9492 |
| 95K | 95,232 | 9.9e-4 | 3.2e-4 | 7.7e-3 | 0.033 | 4.56 | 98.68% | 9.76 | 78.1% | 1.6483 / 1.6494 |

- **Banner check:** each process printed the loaded variant from the dylib's own globals before any timed work
  (`banner.txt`): plain `softmax=attn_softmax8n pv=attn_pv_w`, fp16 V copy; x `softmax=attn_softmax8n_f8x
  pv=attn_pv_f8v`, e4m3 V copy, mask on the boundary chunk only, half exp; keep-alive not loaded; both MATCH.
- **Deterministic and causal:** in each process the 32K run's rows equal the 95K run's first 32,768 rows bit for bit,
  so the 32K numbers are also the 95K run's first third.
- **NLL of the real next token:** +0.0014 (32K) and +0.0012 (95K) nats, +0.07%; next-token top-1 hit rate 61.80% ->
  61.74% at 95K.
- **No growth with depth:** KL mean per tenth of the 95K prompt 6.8e-4 to 1.6e-3, highest in the 2nd and 3rd tenths,
  7.7e-4 in the last; top-1 agreement 98.4-99.0% in every tenth.
- **The tail is near-ties:** 24 of 95,232 positions above KL 0.1, 3 above 1. At the largest (position 14192) plain's
  top-1 was wrong and FP8's right (NLL 1.41 -> 0.06); at the next two plain was right and FP8 wrong. No region of
  the prompt is affected.
- **Throughput in these runs** (not a timing A/B: the capture copy runs in the ANE thread and the FP8 process started
  20 s after the plain one): 95K plain 360, x 352 tok/s; 32K plain 547, x 561. The interleaved traces above had x at
  +9.4% (95K); this run did not reproduce that, cause not checked.

## 95K repeat, 4K / 16K / 48K, fill path, and serving (18 Nov 2026, evening)

Both A/Bs use the same harness as the first table: interleaved plain / x / plain / x V8 traces, one fresh process per
run, binary `libgpu_v8_f8.dylib` sha256 50198bdf4f8b6b8b. Each run printed the loaded variant from the dylib
(`v8_banner.py`, `banner.txt`) before its timed work; all eight matched.

**95K** (`rep95/`, `fold_chain_e.sh`):

| run | wall s | tok/s | GPU attention s | DeltaNet s | ANE busy | DRAM GB/s total / GPU / ANE |
|---|---|---|---|---|---|---|
| plain a | 268.7 | 354.4 | 179.4 | 88.3 | 51.3% | 112.6 / 73.4 / 37.8 |
| x a | 243.2 | 391.6 | 153.7 | 88.1 | 58.1% | 116.4 / 73.4 / 41.6 |
| plain b | 264.7 | 359.8 | 175.5 | 87.5 | 53.4% | 113.2 / 73.4 / 38.3 |
| x b | 242.0 | 393.6 | 153.1 | 87.6 | 58.0% | 116.6 / 73.5 / 41.7 |

The mean is 357.1 -> 392.6 tok/s, **+9.9%**, the same as the first pair (+9.4%). The slower x in the tf_kl capture
runs came from that setup.

**4K / 16K / 48K** (`pts/`, `fold_chain_f.sh`; the three lengths in one process, 45 s idle before each):

| prompt | plain (a / b) | x (a / b) | change | GPU attention s, plain / x | ANE busy, plain / x | DRAM GB/s, plain / x |
|---|---|---|---|---|---|---|
| 4K | 705.7 / 701.6 | 705.5 / 705.6 | +0.3% | 1.58 / 1.52 | 97.7 / 98.4% | 125.0 / 124.8 |
| 16K | 635.8 / 636.7 | 652.9 / 650.3 | **+2.4%** | 9.93 / 9.29 | 88.8 / 91.1% | 123.7 / 124.4 |
| 48K | 479.8 / 480.8 | 510.6 / 509.9 | **+6.2%** | 55.9 / 50.0 | 69.8 / 74.1% | 118.1 / 119.9 |

Together with the first table, the gain by prompt length is:

| prompt | 2K | 4K | 8K | 16K | 32K | 48K | 64K | 95K |
|---|---|---|---|---|---|---|---|---|
| change | -1.2% (noise) | +0.3% | +1.3% | +2.4% | +4.5% | +6.2% | +8.2% | +9.9% |

Last-token KL plain vs x: 3.0e-6 at 4K, 2.0e-6 at 16K, 0.026 at 48K, the same argmax each time. 0.026 is one token at
the teacher-forced p99.9 level (0.033).

**Fill path** (`fill/`, `power/f8_fill_check.py`). A follow-up V8 prefill from a cached prefix refills the GPU's V copy
from the ANE's INT8 cache (`kv8_v_to_f8v` under FP8); no earlier test ran it. Each run prefilled [0, 32K), then ran
[32K, 40K) as a second span, once as is and once with all 32K rows refilled. Each was compared with the single-span
95K run, through the head over 8,192 positions:

| | plain: continued | plain: refilled | x: continued | x: refilled |
|---|---|---|---|---|
| H vs single span | bit-exact | 3.7% RMS | bit-exact | 4.4% RMS |
| KL mean / p99 / max | 0 | 4.3e-4 / 3.9e-3 / 0.042 | 0 | 6.3e-4 / 5.1e-3 / 0.31 |
| top-1 agreement | 100% | 98.9% | 100% | 98.8% |
| refill time for 32K rows | | 12 ms | | 47 ms |

The FP8 refill deviates by the same order as the served plain one (INT8 rounding of V plus the e4m3 rounding), so its
layout and scale are right.

**Served from 20:27:**
- **Launcher:** `v8_server_dec8.sh` defaults to `--v8-dylib libgpu_v8_f8.dylib` and `F8=x`. To revert, set `V8_F8=`
  (plain P.V, same binary) or `V8_SRV_DYLIB=libgpu_v8_srv3.dylib`. The previous launcher is `v8_server_dec8.pre_f8.sh`.
- **Variant in the log:** `v8_engine.py` now logs the loaded variant and the dylib's hash
  (`v8: variant F8=x ... softmax=attn_softmax8n_f8x pv=attn_pv_f8v ... sha256 50198bdf4f8b6b8b`).
- **Smoke test on 8765:**
  - A cold 16,050-token prompt ran the V8 prefill at 653 tok/s.
  - A follow-up of 4,027 new tokens continued from the cache (V fill 9 ms, 128 rows).
  - Both answers were coherent.
- FP8 applies to every V8 prefill. Below 8K the ANE sets the pace, so it changes nothing there.

## F8=x with the softmax folded into P.V (`f8fold/`, evening)

The kernels are in `pipe/engine/kernels_f8fold.metal`, built into `libgpu_v8_f8f.dylib` (`gpu_v8_f8f.swift`, `F8F=`;
the served FP8 binary is untouched):
- **Q.K:** `attn_qk8iskm`, the fold's Q.K, which also writes each 64-key tile's row maximum.
- **P.V:** `attn_pv_f8m_*`. It computes exp(s - m) in half on load, scales by 256, writes e4m3 into threadgroup
  memory and multiplies FP8 x FP8 against the e4m3 V copy.
- **No P pass:** there is no separate softmax pass and no P8 in device memory.
- **Wide:** 64 rows x 256 dims per threadgroup (8 simdgroups), so each score is exponentiated once.
- **Forms:** KC = 64 or 128 keys per chunk, single- or double-buffered (`w64`, `w64d`, `w128`, `w128d`).

**Alone** (`f8_check.py`, one-layer manifest, GPU ms per 128-row attention call, banner from the dylib):

| keys | x | w64 | w64d | w128 | w128d |
|---|---|---|---|---|---|
| 2K | 2.297 | 2.344 | 2.334 | 2.328 | 2.326 |
| 8K | 3.586 | 3.710 | 3.707 | 3.687 | 3.697 |
| 16K | 5.249 | 5.531 | 5.531 | 5.500 | 5.514 |
| 32K | 8.568 | 9.183 | 9.168 | 9.165 | 9.156 |
| 60K | 14.29 | 15.56 | 15.54 | 15.48 | 15.49 |
| 93K | 21.01 | 23.17 | 23.02 | 22.87 | 22.99 |

- **Output:** within 4-8e-5 relative RMS of x (the same maths; only the accumulation order differs).
- **Speed:** slower at every length, +7% at 32K and +9% at 93K.
- **Double buffering does not help:** the exponentials in the P.V kernel set its pace. Each tile runs 48
  threadgroups, against the 3072 rows x 256 threads of x's softmax.

**End to end at 64K** (`e2e64/`, V8 traces x / x+fold / plain / x / x+fold, fresh processes, `F8F=w128` on every call):

| run | tok/s | GPU attention s | DeltaNet s | ANE busy | DRAM GB/s total / GPU / ANE |
|---|---|---|---|---|---|
| plain | 428.4 | 91.3 | 60.4 | 63.4% | 115.7 / 68.9 / 45.2 |
| x (a / b) | 457.2 / 461.8 | 81.6 / 80.9 | 60.6 / 60.0 | 67.6 / 68.0% | 118.7 / 68.1 / 48.1, 118.6 / 68.5 / 48.7 |
| x + fold (a / b) | 451.6 / 453.2 | 83.9 / 83.6 | 60.4 / 60.3 | 64.3 / 64.1% | 108.4 / 59.3 / 47.6, 108.2 / 59.1 / 47.8 |

Attention ms per call by tenth of the prompt (mean of the two runs):

| | 1st | 2nd | 3rd | 4th | 5th | 6th | 7th | 8th | 9th | 10th |
|---|---|---|---|---|---|---|---|---|---|---|
| plain | 3.50 | 5.37 | 7.04 | 8.81 | 10.39 | 12.12 | 13.61 | 15.20 | 16.78 | 18.31 |
| x | 3.37 | 5.10 | 6.51 | 7.82 | 9.31 | 10.65 | 12.00 | 13.35 | 14.64 | 15.97 |
| x + fold | 3.34 | 4.96 | 6.51 | 8.01 | 9.53 | 11.06 | 12.53 | 14.07 | 15.44 | 16.86 |

- **Throughput:** x + fold is 1.5% slower than x end to end (452.4 against 459.5 tok/s), still +5.6% over plain; x is
  +7.3% over plain.
- **Where it pays:** under contention the fold is 1-3% faster than x in the first two tenths, ties at about 13K keys
  and is 5.5% slower in the last tenth.
- **DRAM:** the fold cuts GPU traffic by 9 GB/s (68.3 -> 59.2), but the GPU is compute-bound at these lengths, so
  that does not pay. DeltaNet is unchanged.
- **Accuracy:** last-token KL to plain 8.1e-6 (x 5.2e-6), the same argmax.
- **Decode:** not affected by any of these variants. Decode attention is `dec_attn_r4` over the ANE's INT8 cache and
  does not read the GPU's V copy.
- **Verdict:** not adopted. The window where it pays (under ~13K keys, at most 3% of attention there) is worth under
  0.5% end to end. A split-K form with more threadgroups per tile would be the next thing to try.
