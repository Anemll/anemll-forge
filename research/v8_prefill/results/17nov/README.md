# Results of 17 Nov 2026 (and the 16 Nov night decode work)

Every measurement behind the 17 Nov notes, the article and the model card. Summaries and conclusions live in
[../../PREFILL_RESEARCH.md](../../PREFILL_RESEARCH.md), [../../pipe/unit/DECODE_GPU_ATTENTION.md](../../pipe/unit/DECODE_GPU_ATTENTION.md)
and [../../ARTICLE_ANE_GPU_M6.md](../../ARTICLE_ANE_GPU_M6.md). All on the M6 (32 GB, macOS 27.3, build 26C5035r).

## Server benchmarks (`pipe/unit/cmp_bench.py`, 96K context, greedy, thinking off)

| file | what |
|---|---|
| `../cmp_v8_96k.jsonl`, `../cmp_splash_96k.jsonl` | 16 Nov: V8 server and splash 1.3.2 (with its Neural Engine split), the same requests |
| `../cmp_v8dec8_96k.jsonl` | 17 Nov: the served SPLIT+GPU build (v8_dec8), same requests |
| `cmp_v8dec8r4_64k.jsonl` | 17 Nov: SPLIT+GPU with the `dec_attn_r4` kernel at 64K |
| `../cmp_splash_gpu_96k.jsonl` | 17 Nov: splash 1.3.2 GPU only (`--disable-ane`), same requests (`splash132/cmp_96k_gpu.sh`) |
| `v8_server_8765_dec8_bench.log` | the server log of the SPLIT+GPU benchmark (per request: `dec8: N split verify steps, X ms each (GPU attention Y ms)`) |

## SPLIT+GPU build and correctness checks

| file | what |
|---|---|
| `v8_dec8_all.log`, `v8_prep_dec8.log`, `v8_prep_kv8_dec8.log` | the 8-package build, its GPU tables and INT8 K/V tables |
| `dec8_check_L00.{json,log}` | split entries vs the plain verify on one package |
| `dec8_srv_check.{json,log}` | the server's split verify vs the ANE verify |
| `dec8_real_check.{json,log}` | real-text KL and NLL of the split verify (KL 0.0005 to 0.054) |
| `dec8_long_time.{json,log}` | verify step time by context, up to 131K keys (139 ms, no recompile) |
| `v8pf_dec8.log`, `v8pf_served.log` | V8 prefill on the dec8 build identical to the served build |
| `dec_att_probe.log`, `build_dec_att.log` | ANE + GPU attention by key range (rejected: about 1 ms per step) |

## Decode-attention kernels and bandwidth

| file | what |
|---|---|
| `decode_kernel_and_bandwidth_runs.txt` | 78 runs recovered from the session transcript, each with its command and output: kernel variants with output checksums (FP8 softmax, register softmax `DEC_SM`, shapes, prefetch, double buffering), `dec_bench` steady timings, `bw_gpu` / `bw_ane` bandwidth |
| `bw/gpu_conc.txt`, `bw/ane_conc.txt` | GPU streaming and the split's ANE calls, alone and at the same time (156, 121, 87 + 77 GB/s) |
| `decode_attn_flash/*.txt` | 16 Nov night: the standalone flash-decoding kernels (`decode_attn_flash*.swift`, `decode_attn_seg.swift`) |

## Pi coding sessions (same prompt per task, one server at a time)

| file | what |
|---|---|
| `pi_coding_sessions.json` | per run: wall time, turns, output and reasoning tokens, server-side prefill time (`../../bench/pi_session_summary.py` over the Pi session file and the server log) |
| `pi_runs/<run>/` | what each model wrote, as left on disk: `tetris_forge`, `tetris_splash_nocap`, `cron_forge`, `cron_splash_nocap` (rebuilt from its session: the capped rerun overwrote it), `cron_splash_cap6144` |
| `pi_runs/cron_*/grade.txt` | the hidden grader (`../../bench/cron/grade.mjs`): 1373 / 1373 for all three |
| `pi_runs/tetris_*/probe.json`, `keyseq.txt` | `../../bench/tetris/probe.mjs` and `keyseq.mjs`: piece sequence, O rotation, split clears, input while paused, a 20,000-action fuzz |
| `server_logs/` | the V8 and splash server logs of those runs (per request: prefill, generation, thinking) |
| `verbosity_{splash,forge}.jsonl`, `verbosity_result.json` | single-turn verbosity probe (`../../bench/verbosity/`), 14 paired samples |

## Palettized INT8 FFN on the ANE (`../../w8a8/lut8_a8_probe.py`, layer 36, dense synthetic input)

One real FFN with compile-time weights only: 3-bit indices with an INT8 palette and per-token INT8 activations.

| file | what |
|---|---|
| `lut8_a8/lut8_a8_build.log`, `lut8_a8_time.log`, `lut8_a8_L36.json` | first run: `lut16` (FP16, one conv), `lut16s` (FP16, 17 x 1024 blocks, K split), `lut8w` (INT8 palette, FP16 activations), `lut8a8` (full form W8A8), `lut8a8s` (split W8A8) at 128 and 1024 rows, with the HWX summaries |
| `lut8_a8/lut8_a8_build.try1.log`, `lut8_a8_time.try1.log` | first attempts: the build failed (the sub-byte pass needs operands named `indices` / `lut`), the timing ran before the `lut16s` baseline existed |
| `lut8_a8/hwx_cost_breakdown.txt` | `../../w8a8/hwx_cost_breakdown.py`: the compiled programs of the bisection, per function (two streams each), the compiler's ExeCycles by task class (INT8 / FP16 convs, the quantize's reduction and INT8 write, PE rescale and sums) |
| `lut8_a8/hwx_ne_shapes.txt` | `../../w8a8/hwx_ne_shapes.py`: NE cycles by operand format and channel shape at 128 and 1024 rows, and the count of tasks that read INT8 outside the NE (a separate dequantize) |
| `lut8_a8/bisect_build.log`, `bisect_time.log`, `bisect_L36.json`, `bisect_watchdog.log` | the bisection of the split W8A8 failure: `lut16s`, `lut8a8s`, `s8w` (INT8 weights only), `s8a8_noks` (no K split), `s8a8_gu` (INT8 activations on gate / up only), `s8a8_dn` (on down only) at 128, 512 and 1024 rows; each output against FP32 and against a host simulation of its own INT8 recipe |

## INT8 gate / up on the served FFN (`../../w8a8/srv_ffn_a8_probe.py`, `srv_ffn_decode_probe.py`, layer 36)

| file | what |
|---|---|
| `srv_ffn_a8/build.log`, `time.log`, `srv_ffn_a8_L36.json`, `watchdog.log` | the served chunked FFN (`srv16`, bit-identical to the served `ffn2_L36`, which is timed too) against INT8 gate / up in the served chunks (`srv8gu`) and in 17 x 1024 tiles (`srv8gu17`), at 64 rows and the 2 x 64 pair: pair 2.353 ms served, 2.005 ms `srv8gu` (-15%), FFN error 1.4% |
| `srv_ffn_a8/decode_build.log`, `decode_time.log`, `dec_ffn_a8_L36_x4.json`, `decode_watchdog.log` | synthetic decode: the same FFN applied 4 times per call at 8 and 16 rows: INT8 gate / up is 3-5% slower (0.772 vs 0.748 ms per FFN at 8 rows) |
| `ffn8_prefill/build.log`, `build_watchdog.log` | an INT8 FFN-only package set for a prefill A/B (`v8/build_v8_ffn8.py`), stopped after 3 of 16 packages: it duplicated every FFN weight (see the note in `../../PREFILL_RESEARCH.md`) |

## Redo with nothing INT8 stored: q/dq pairs on the builder's own constants (`../../w8a8/qdq.py`, `qdq_probe.py`, `qdq_run.sh`)

Every INT8 test above rebuilt with the weights as the builder's FP16 LUT constants (same 3-bit indices and FP16
palette, package sizes equal to FP16) and quantize -> dequantize pairs on the expanded weights and the activations.

| file | what |
|---|---|
| `qdq/build_probe.try1_dense.log` | first build: the palettizer skipped weights feeding `coreai.quantize` and stored them dense FP16 (539 MB); fixed by adding `coreai.quantize` to its consumer set (`qdq.allow_qdq_palettize`) |
| `qdq/build_probe.log`, `build_served_decode.log`, `build_report_*.json` | the three suites (probe shape, served FFN, synthetic decode); static activation scales |
| `qdq/time.log`, `time_*.json` | timings and errors: every program with q/dq on a weight fails the Neural Engine compile (`ANECCompile() FAILED`, "MLIR MPS to ANEC conversion failed"; 20 of 20) and runs off the ANE, 10-40x slower (probe 128 rows: 75-127 ms against 2.9 / 12.0 ms FP16; served pair 98-110 ms against 2.87) |
| `qdq/hwx.log`, `hwx_*.json` | no ANE program exists for any q/dq-weight package; the FP16 baselines' HWX for comparison |

## LUT3 dequantized for INT8 x INT8 on one index tensor (`../../w8a8/lut3.py`, `lut3_probe.py`, layer 36 served FFN)

The export's LUT3 matrix (6-bit pair indices, 64 x 2 FP16 palette, FP16 per-row scale) with ONE index buffer per matrix
chunk and two palettes: FP16 (the served path) and INT8 codes round(lut / s), s = max|lut| / 127, dequantized by
s x row scale. Packages `fp16`, `n` (FP16 + INT8 entries on the served chunk constants), `k` (both paths on K-split
index slices), `i8n` / `i8k` (INT8 only).

| file | what |
|---|---|
| `lut3/build.try1_graphslice.log`, `time.try1_graphslice.log`, `hwx.try1_graphslice.log` (+ `.json`) | first attempt: the K split sliced the one expanded weight inside the graph; `ANECCompile() FAILED` for every package holding such an entry (the FP16 entries in it fell off the ANE too); the rejected programs left 3.0 / 0.7 GB of Core AI cache (deleted) |
| `lut3/build.log`, `build_report.json` | storage per package: the same 133,693,440 6-bit indices in every package (stored once); the INT8 entries add 385 INT8 values (three 128-value palettes + a zero point) |
| `lut3/time.log`, `time.json` | same session: served `ffn2_L36` 2.676 ms; INT8 gate / up 2.43-2.59 ms (-3 to -9%); INT8 gate / up + down 2.23-2.25 ms (-16 to -17%); FFN error against the served FP16 FFN 1.4% / 1.8% |
| `lut3/hwx.log`, `hwx.json` | HWX: INT8-only 101.8 MB = FP16-only 101.5 MB (INT8 x INT8 tasks on palettized kernels, nothing expanded); FP16 + INT8 in one package 198.4-198.8 MB: the compiled program keeps each kernel format separately, so the indices are stored twice there |

## LUT3 INT8 x INT8 at decode sizes (`../../w8a8/lut3_decode_probe.py`, layer 36 served FFN, 4 FFN passes per call)

| file | what |
|---|---|
| `lut3_decode/build.log`, `build_report.json` | one format per package: `dec_fp16`, `dec_i8gu`, `dec_i8gud`, `dec_i8gudk`; the same 133,693,440 indices in each |
| `lut3_decode/time.log`, `time.json` | 8 rows: FP16 0.738 ms per FFN; INT8 gate / up 0.827 (+12%), + down 0.833 (+13%), + K split 0.845 (+15%); 16 rows the same; FFN error against the served FP16 0.9-1.2% |
| `lut3_decode/hwx.log`, `hwx.json` | HWX 101.6 MB (FP16) against 102.6-103.0 MB (INT8): nothing expanded |

## LUT3 INT8 against FP16 from 8 to 512 rows (`../../w8a8/lut3_decode_probe.py --rows 8,...,512`, 4 FFN passes per call)

| file | what |
|---|---|
| `lut3_rows/build.log`, `build_report.json` | one format per package: `fp16`, `i8w` (INT8-palette kernels, FP16 activations), `i8gu`, `i8gud`; seven functions r8 ... r512 |
| `lut3_rows/time.log`, `time.json` | ms per FFN, FP16 / i8w / i8gu / i8gud: 8 rows 0.736 / 0.738 / 0.830 / 0.849; 32 rows 0.765 / 0.780 / 0.884 / 0.921; 64 rows 1.263 / 1.209 / 1.179 / 1.059; 128 (one call) 3.79 / 3.67 / 4.09 / 4.10; 256 rows 5.83 / 5.67 / 7.79 / 5.01; 512 rows 11.62 / 11.30 / 9.99 / 7.55 |
| `lut3_rows/hwx.log`, `hwx.json` | HWX 306.5-311.6 MB for every format: equal, nothing expanded; seven row shapes compile to about three kernel copies (two-function packages: 101.6 MB) |

Pi session files (not copied; the user's Pi data): `~/.pi/agent/sessions/--Users-anemll-SourceRelease-GITHUB-ML_playground-11-11.a--/2026-11-17T18-12-53*` (Tetris, forge), `...T18-23-59*` (Tetris, splash),
`...-compare-forge--/2026-11-17T19-57-15*` (cron, forge), `...-compare-splash--/2026-11-17T20-09-10*` (cron, splash, no cap) and `...T20-40-39*` (cron, splash, cap 6144).
