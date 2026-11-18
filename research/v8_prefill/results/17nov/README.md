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

Pi session files (not copied; the user's Pi data): `~/.pi/agent/sessions/--Users-anemll-SourceRelease-GITHUB-ML_playground-11-11.a--/2026-11-17T18-12-53*` (Tetris, forge), `...T18-23-59*` (Tetris, splash),
`...-compare-forge--/2026-11-17T19-57-15*` (cron, forge), `...-compare-splash--/2026-11-17T20-09-10*` (cron, splash, no cap) and `...T20-40-39*` (cron, splash, cap 6144).
