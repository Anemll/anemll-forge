# Softmax folded into Q.K / P.V on the served prefill attention (18 Nov 2026)

Prototype: `pipe/engine/kernels_fold.metal` + `pipe/engine/gpu_v8_fold.swift` -> `libgpu_v8_fold.dylib`. This is the
served srv3 engine plus a fold on its attention path, KV8 + K8V16 (INT8 keys from the ANE's cache, the GPU's fp16 V
copy).

- **Plain path:** `attn_qk8isk` writes the scores, `attn_softmax8n` reads them and writes P, `attn_pv_w` reads P.
- **Folded:** `attn_qk8iskm` also writes each 64-key tile's masked row maximum. `attn_pv_wm8_{64,64s,128}` reads the
  scores and computes exp(s - m) while staging each key chunk in threadgroup memory. The softmax pass and its two
  score passes are gone.
- **Switches:** `FOLD=64|64s|128` picks the P.V chunk form, and `FOLD_MIN=<keys>` folds only calls that attend to at
  least that many keys. `v8_fold(mode, min)` sets both at run time.
- **The served files are unchanged.**

## Correctness (`alone/sweep.log`, `power/fold_check.py`)

The plain and folded forms run on the same random INT8 KV and the same inputs, in one process. The layer's mixer output
against plain: relative RMS 0.5-2.8e-4 (fp16 output rounding), largest difference 0.001-0.008, no NaN or Inf, at 1K to
95K keys.

## Alone (`alone/`)

GPU ms per 128-row attention pair call (layer 3), median of 12. Plain is the mean of the runs before and after.

| keys | plain | fold64 | fold64s | fold128 |
|---|---|---|---|---|
| 2K | 2.35 | 2.34 | 2.34 | 2.34 |
| 4K | 2.81 | 2.88 | 2.86 | 2.86 |
| 8K | 3.78 | 3.91 | 3.89 | 3.92 |
| 16K | 5.73 | 5.98 | 5.95 | 5.91 |
| 32K | 9.61 | 10.14 | 10.09 | 10.12 |
| 48K | 13.49 | 14.25 | 14.17 | 14.14 |
| 60K | 16.43 | 17.35 | 17.30 | 17.14 |
| 80K | 21.36 | 22.55 | 22.43 | 22.19 |
| 93K | 24.44 | 25.84 | 25.86 | 25.48 |

The profile at 60K (`alone/prof60k.log`, `PROFILE=1`, 8 calls, 248 tiles) shows where the time goes.

- **Plain:** P.V 45.0 ms, Q.K 34.6, softmax 20.0, combine 8.7.
- **Folded (128):** P.V 69.3 ms, Q.K with row maxima 41.8, combine 8.8.

So the folded P.V costs more than the plain P.V and softmax together (69.3 against 65.0 ms), and the row maxima add
21% to Q.K. Alone the fold is 3-5% slower from 8K keys up.

## Next to the ANE FFN (`fold_<K>k_128/`, `power/att8_win.py` + `coupling_run.py`)

Attention runs continuously at a fixed position, plain or folded (fold128), alone and next to the FFN (`ffn2_L0`,
v8_dec8). Slowdowns are against the adjacent alone phases.

| keys | attention alone: plain / fold | attention next to the FFN: plain / fold | FFN slowdown: next to plain / next to fold | GPU DRAM next to the FFN: plain / fold |
|---|---|---|---|---|
| 8K | 3.74 / 3.81 ms | 5.07 / 4.97 ms (-1.8%) | +13.6% / +4.2% | 79 / 67 GB/s |
| 32K | 9.44 / 9.90 | 13.60 / 13.18 (-3.1%) | +13.3% / +3.5% | 88 / 69 |
| 60K | 16.21 / 16.95 | 23.63 / 22.61 (-4.3%) | +7.4% / +2.8% | 90 / 70 |
| 88K | 22.84 / 24.30 | 33.68 / 32.72 (-2.8%) | +8.3% / +5.9% | 91 / 69 |

Under contention the fold wins on both units: attention is 2-4% faster, the FFN is slowed about half as much, and the
GPU moves 12-22 GB/s less. Plain always ran before the fold in each set, and the FFN warms over the set. The adjacent
baselines correct for most of that drift.

## In the V8 prefill (`trace_fold128/` against `../coupling/trace1/` and `trace_plain95/`)

`power/v8_trace.py` traced the served pipeline with `V8_DYLIB=libgpu_v8_fold.dylib FOLD=128 FOLD_MIN=0`, so every
attention call folds. The same build, environment and prompt text as the plain traces.

| prompt | plain tok/s | fold tok/s | GPU s: DeltaNet plain / fold | attention plain / fold | DRAM total plain / fold |
|---|---|---|---|---|---|
| 2K | 687 / 696 | 691 (-0.2%) | 1.9 / 1.9 | 0.7 / 0.7 | 122 / 121 GB/s |
| 8K | 684 | 691 (+1.0%) | 7.7 / 7.7 | 3.8 / 3.8 | 125 / 124 |
| 32K | 543 | 546 (+0.6%) | 30.8 / 30.5 | 29.0 / 28.9 | 120 / 113 |
| 64K | 420 | 416 (-1.0%) | 61.7 / 60.1 | 93.4 / 94.8 | 114 / 103 |
| 95K | 349 | 346 (-0.8%) | 87.2 / 88.1 | 180.9 / 182.7 | 112 / 97 |

Per position (decile of the prompt), fold against plain:

| prompt | measure | 1st | 2nd | 3rd | 4th | 5th | 6th | 7th | 8th | 9th | 10th |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 32K | attention per call | -1.1% | -3.7% | -2.7% | -1.8% | -0.6% | +0.1% | +0.9% | +1.6% | +1.6% | +1.9% |
| 32K | DeltaNet per call | 0 | +0.4% | -0.7% | -0.8% | -0.2% | -1.7% | -3.9% | -3.8% | -4.4% | -5.0% |
| 64K | attention per call | +1.5% | -1.4% | -0.4% | +0.3% | +1.4% | +2.4% | +2.1% | +2.3% | +3.2% | +2.4% |
| 64K | DeltaNet per call | -0.2% | -1.2% | -1.3% | -2.3% | -4.2% | -4.8% | -4.7% | -5.3% | -5.8% | -5.4% |

- **End to end it is a wash:** -1% to +1% at every length, inside the run-to-run spread (the two plain 2K runs differ by
  1.3%). Single runs per engine.
- **Per segment, the effects pull in opposite directions:**
  - The folded attention call is 1-4% faster up to about 13K keys and 1.5-3% slower past about 25K keys. Its extra
    compute outweighs the bytes saved there.
  - DeltaNet gets 4-6% faster in the second half of 32K and 64K prompts. Attention's lower DRAM traffic eases
    DeltaNet's contention.
  - At 32K and 64K the two cancel.
- **95K is not conclusive:** the swap watchdog killed the fold trace's process right after its 95K run (swap 930 ->
  2700 MB during it), so that run may have been slowed by memory pressure.
- **Small prompts (2K-8K):** neutral to +1%. Below 8K the ANE sets the pace, and attention's score traffic is a few
  percent of the ANE's bytes.

## What it would take to gain

The fold pays only if the folded P.V runs as fast as the plain P.V alone. Then the contention gains above would be
net: DeltaNet -4 to -6% late in long prompts, and the FFN less slowed. Two forms are drafted and compile-checked, not
yet run (`pipe/engine/kernels_fold_v2.draft.metal`):

- **Wide:** 64 rows x all 256 dims per threadgroup, so each score is read and exponentiated once instead of twice.
- **Online:** a running row max in P.V, with Q.K unchanged, which drops Q.K's +21%.

Repeated, interleaved A/B traces (plain, fold, plain, fold) are needed to resolve 1% effects.

## Decode (`dec_clock_partial/`)

Decode attention is already a fused kernel (`dec_attn_r4`, scores in registers and threadgroup memory), so the fold
does not apply there. The partial run of `power/dec_clock.py` (served kernel, one layer at 64K keys) gives GPU ms per
call:

| regime | ms per call |
|---|---|
| 50 copies per command buffer | 1.053 |
| back to back | 1.049 |
| after a 1 ms idle gap | 3.131 |
| after a 5 ms idle gap | 3.192 |
| 1 ms gap with a 10% duty ALU keep-alive beside it | 1.749 |

The run stopped at the next phase (the keep-alive process did not exit on SIGINT). The GPU clock in decode is being
investigated separately: `../gpu_clock/`.
