# V8 prefill server (16 to 17 Nov 2026)

OpenAI-compatible Qwen3.8-27B server on the M6 with the V8 GPU + ANE prefill. It is the anemll-forge-v02 release
server (`scripts/qwen38_server.py`: Core AI target, DFlash2 all3 drafter, prompt-cache snapshots) with only
`Engine.prefill` replaced. Decoding is the release path; since 17 Nov the served script (`v8_server_dec8.sh`) adds
SPLIT+GPU: past 64K, the verify's attention over the KV history runs on the GPU between Neural Engine calls
(`dec_split.py`, write-up in `unit/DECODE_GPU_ATTENTION.md`).

## Run

```
PORT=8765 v8_work/pipe/v8_server_dec8.sh [options]   # served since 17 Nov: SPLIT+GPU, build v8_dec8, tables w_dec8
v8_work/pipe/v8_server.sh [v8_server.py options]      # port 8767 (PORT=...), ctx 98304 (CTX=...), v8_shared2, no split
v8_work/pipe/srv_stop.sh                              # stops whatever listens on $PORT (default 8767)
```

- Build: `/Volumes/SSD4TB/anemll-forge-research/v8_shared2` (shared-weight build: verify + prefill + ffn / ffn2 /
  ffnq2 / offn2 entries), 96K context ladder, KV8 (INT8 K and V, dim_token keys).
- GPU engine: `engine/libgpu_v8_srv3.dylib` (srv2 + `patch_fillfrom.py`), kernels compiled at load from `engine/kernels.metal`, GPU weights
  `w_shared2r/manifest_best.json`. GPU switches in the script env: FUSE=all I8=1 I8T=1 I8TV=_k256 LUTC=1 I8A=1
  I8KV=1 LRM=1 QK8=1 TKV=2048.
- Log `/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log`, swap watchdog +1536 MB, the load is
  journaled (deploy_journal).
- `v8_server_plain.sh`: the same with `--plain` (no drafter; deterministic greedy, used for equality checks).
- `v8_server_dec8.sh`: as `v8_server.sh` plus `--v8-dec8 1`, `--model-dir SSD/v8_dec8` (8 packages of 8 layers: the
  target and V8 entries plus the split entries `dva` / `dvba` / `dvb`) and `--v8-gpu-manifest pipe/w_dec8/manifest_best.json`;
  decode attention from `engine/libdec_gpu.dylib`. Both scripts keep every previous log as `v8_server.<time>.log`.

Options (defaults are the best measured):

| option | default | meaning |
|---|---|---|
| `--v8-min` | 256 | V8 for new spans of at least this many tokens (the bridge alone does ~145 tok/s at 64K+) |
| `--v8-kv8` | 1 | GPU attention reads and writes the ANE's INT8 KV cache (no KV copy, warm continuation) |
| `--v8-k8v16` | 1 | with KV8: the GPU also keeps an fp16 copy of V for its PV (+32 KB/token, 3.2 GB at 96K) |
| `--v8-attproj-from` | off | attention projections on the ANE from this position (no clear gain, see below) |
| `--v8-dec8` | 0 | 1: SPLIT+GPU decode (needs the v8_dec8 build; `v8_server_dec8.sh` sets it) |
| `V8_DEC8` env | auto | auto: split in the 96K entry only (past 64K tokens); on: always; off |
| `V8_DEC8_MID` env | 0 | with auto, also split in the 64K entry up to this many keys (measured about even there) |
| `DEC_ATTN` env | dec_attn_r4 | decode attention kernel (`dec_attn`: the previous one, 3.3% slower) |
| `V8_ALIGN` env | 64 | continuation start alignment (64 or 128) |
| `V8_REST_MIN` env | 128 | V8 after an alignment only when at least this many tokens are left (else all on the bridge) |
| `V8_TAIL_T` env | 8 | a last pair with this many real rows or fewer: one T-row bridge call instead of a padded pair |
| `V8_DEC_SNAP` env | 256 | a snapshot every this many decoded tokens (0: off) |
| `V8_DEC_KEEP` env | 3 | decode snapshots kept (the latest; about 97 MB each) |
| `V8_TEXT_MATCH` env | 1 | keep the state's own tokens where the prompt's text equals the state's |
| `V8_FILL_CHECK` env | 0 | 1: before every warm run, check the GPU's fp16 V copy against the ANE cache (validation only) |
| `V8_V02` env | anemll-forge-v02 | the release tree imported (`scripts/qwen38_server.py`, the bridge, the drafter) |
| `NGRAM_DRAFT` env | 0 | prompt-lookup drafts (release server `--ngram-draft`), off until the greedy divergence is explained |
| `DRAFT_GAP_MS` env | 1 on M6 | minimum gap between a target verify and the next drafter call |
| `V8_AB` env | off | decode A/B over requests, e.g. `g1l,g3l,g3e` (draft gap ms, drafter context lazy / eager); `unit/srv_ab_summ.py` |

## How a request runs

1. The prompt is rendered as in the release server. When the model state is not a token prefix of it, but the
   prompt's text starts with the state's text for more tokens (the client sent the model's own output back and it
   re-tokenizes differently), the state's tokens are kept up to there and only the rest of the text is tokenized.
2. Cold prompt: V8 from 0 to the end in one run, the last 128-row pair padded (`v8_submit_pair_nv`: DeltaNet state
   and conv history stop at the last real token), the logits from V8's own head. A last pair with 8 real rows or
   fewer is left to one T-row call of the bridge instead.
3. Snapshots:
   - 'system' and 'turn' (the last `<|im_start|>`) are taken during a V8 run at the pair end at or below each cut:
     each DeltaNet layer's state is copied when its pair ending there completes, the next pair of that layer is
     gated until then. The bridge path takes 'system' exactly and no longer splits the prompt at the last header.
   - 'gen' at the prompt's end, exact, on every path (with `preserve_thinking` the next request renders the
     assistant message after exactly this header).
   - 'dec0..2' every 256 decoded tokens (the latest 3), so a client edit inside the answer restores close to it.
   - After every prefill and every decode, snapshots that are not a prefix of the tokens now in the KV cache are
     dropped (their KV rows were rewritten by another conversation, a summary request or another continuation).
   - 'sysk': before a prompt that does not start with the 'system' snapshot's tokens rewrites the rows under it
     (Pi's summary request), that snapshot is kept with a copy of its KV rows (about 33 KB per token: 266 MB for
     8.4K tokens, 13 to 28 ms). The conversation's next prompt restores it: the context entry grows to hold the
     rows, they are written back (5 to 17 ms), the GPU's V copy is refilled from row 0, the copy is freed.
4. The drafter's context gets the tap features V8 captures (layers 5, 19, 33, 47, 61), windowed.
5. Continuation (a restored snapshot or the state after decoding): the bridge feeds prefill blocks of any 1 to 64
   rows up to one of the next 64- or 128-aligned points (fewest blocks + pairs; a 64 start only when the span crosses
   no KV segment boundary); V8 runs only if at least 128 tokens are left after that. The DeltaNet states go to the
   GPU (`v8_state_in`, on the GPU); with K8V16 the GPU's V copy gets the rows written since it was last valid
   (`v8_kv8_v16_fill_from`; the rows of earlier V8 runs keep the GPU's own fp16 V), then V8 runs to the end.
6. Decode streams text only up to the last complete UTF-8 character (a character split across tokens was streamed
   as U+FFFD before, and the client sent that back).
7. Any V8 failure falls back to the bridge from the last consistent position.

## Results (M6, macOS 27.3, speculative decode on)

Current (16 Nov evening, with every change below: 96K context, greedy, thinking off, the GPU otherwise idle;
`unit/cmp_bench.py`: a unique salt per cold prompt, then one follow-up turn of the answer + 300 new tokens; client
time to the first streamed token):

| prompt | cold prefill | follow-up | decode cycle | tokens per verify (cold / follow-up) | decode tok/s (cold / follow-up) |
|---|---|---|---|---|---|
| 2K | 586 tok/s, 3.6 s | 1.04 s | 108 ms | 3.66 / 6.40 | 34.4 / 61.0 |
| 8K | 658 tok/s, 12.5 s | 1.15 s | 112 ms | 2.72 / 7.31 | 24.6 / 67.2 |
| 16K | 620 tok/s, 26.5 s | 1.90 s | 120 ms | 2.42 / 4.20 | 20.4 / 35.5 |
| 32K | 541 tok/s, 61 s | 1.75 s | 134 ms | 2.56 / 3.77 | 19.3 / 28.7 |
| 64K | 420 tok/s, 156 s | 2.36 s | 153 ms | 2.78 / 3.10 | 18.5 / 20.7 |
| 95K | 347 tok/s, 274 s | 2.92 s | 155 ms | 2.81 / 7.76 | 18.9 / 53.7 |

At 2K the run itself does 655 tok/s: about 0.4 s per request is fixed (setup, the drafter's context after the run,
HTTP and template), and 17 pairs leave a pipeline fill / drain of about one pair. Results:
`coreai_seg_logs/cmp_v8_96k.jsonl`.

Earlier (16 Nov morning, the configuration matrix):

Cold prefill, server-side tok/s (whole request, includes handover and drafter context):

| prompt | K8V16 (default) | KV8 | fp16 KV (ctx 32K) | bridge only |
|---|---|---|---|---|
| 2K | 634 | | 576 | |
| 6K | 672 (663 to 700 across runs) | 625 | 590 | 314 |
| 16K | 601 | 591 | 564 | |
| 32K | 516 to 537 | 527 | 490 | |
| 64K | 405 to 415 | 399 | | |
| 95K | 348 to 352 | 327 | | |

- K8V16 + attproj from 32K: 64K 426, 95K 346: no clear gain, left off.
- fp16 KV in the server is the slowest at every length: the run itself is slower and the KV must be copied into the
  ANE cache afterwards (0.5 s at 6K, 1.7 s at 32K). It also cannot continue warm.
- Run-to-run spread is about +-5% at 6K.

Follow-up turns (new tokens = the previous answer + the new user text), prefill seconds:

| state + new | bridge only (old --v8-min 1024) | now |
|---|---|---|
| 6K + 573 | 2.03 | 1.27 to 1.36 |
| 6K + 723 | 2.64 | 1.84 to 1.88 |
| 64K + 477 | 2.93 | 1.92 to 2.03 |
| 64K + 723 | 4.93 | 3.54 to 3.74 |
| 64K + 1522 | 10.39 | 6.94 to 7.17 |
| 95K + 423 | | 2.46 (2.91 with 128-aligned starts) |
| 95K + 1223 | | 6.97 (bridge 12 rows 0.42 s, sync 6 ms, V fill 57 ms, V8 6.4 s) |

Decode: unchanged by the V8 prefill (ms/cycle about 110 at 6K, 128 at 32K, 152 at 96K; 19 to 35 tok/s depending on
draft acceptance). SPLIT+GPU (17 Nov, the same `cmp_bench.py` requests, `cmp_v8dec8_96k.jsonl`): decode cycle 107 /
111 / 119 / 133 / 141 / 149 ms at 2K / 8K / 16K / 32K / 64K / 95K (splash 1.3.2: 116 / 119 / 122 / 131 / 141 / 148), cold-turn
decode 20.6 tok/s at 64K and 20.4 at 95K; prefill unchanged within run noise; the 64K follow-up took 3.31 s against
2.36 s (one run, not investigated). A request past the context returns HTTP 400 and the server keeps running.

## KV8 findings

- KV8 is lossless for quality: per-block KL at 64K 0.0592 (fp16 KV 0.0593), 97K 0.2267 vs 0.2274.
- The KV8 attention core was 12% slower per call at 64K (PV half x INT8 +40%, softmax with per-key scales +22%);
  that is the 4 to 6% end-to-end loss at 64K / 96K. On-chip V dequant (KV8D) and a 256-wide PV (PVWW) did not help.
- K8V16 (keys INT8 from the ANE cache with the key scales folded into the INT8 score, values fp16 on the GPU):
  64K attention call 17.65 ms vs KV8 19.83 vs fp16 17.71; real end to end 64K 419.9 tok/s vs KV8 401.7, KL mean
  0.0598 (KV8 0.0592). The ANE cache still gets INT8 K and V for decode.

## Short prefills and the prompt cache (16 Nov, a Pi session)

A Pi coding session (12:23 to 13:55, 53 requests; `unit/srv_log_prefill.py` over the server log): prefill 497 s,
decode 4231 s.

| requests | prefill s | share | what |
|---|---|---|---|
| 5 compaction-related (summary prompts cold, compacted context again) | 245 | 49% | inherent in Pi's compaction |
| 6 restores after a prompt mismatch, 47.6K tokens | 129 | 26% | see below |
| 11 follow-ups through V8 (1.4K to 4.5K new) | 96 | 19% | run at 81 to 93% of the cold rate; 0.9 to 1.3 s around it |
| 29 short follow-ups (13 to 329 new) | 18 | 3.5% | 0.3 to 0.6 s for up to 66 tokens |

Losses on the short and follow-up prefills, and the fixes:

- Every bridge-only prefill was split before the last `<|im_start|>` for the 'turn' snapshot, so the 5 header tokens
  took their own T-row call (+0.10 to 0.13 s on 0.3 to 0.6 s). Now 'gen' at the prompt's end.
- The alignment target was at least 9 rows away (`i + 9`), so a state 1 to 8 rows below a 64 boundary needed 65 to
  72 rows in 2 blocks (4 cases). Now blocks of any 1 to 64 rows.
- V8 was re-checked against `--v8-min` after the alignment: 279, 294 and 329 new tokens at 12K to 17K paid 2
  alignment blocks and then went all to the bridge (1.5 to 1.7 s, about 1.2 s possible). Now decided before, on the
  rest (>= 128).
- Last pairs with 2 to 8 real rows (121 to 126 pad) cost a pair (0.27 to 0.47 s) where one T-row call costs 0.1 to
  0.12 s. Now on the bridge.
- The K8V16 fill rebuilt rows 0 to p0 on every warm run (60 to 400 ms at 22K to 61K; 3.2 s in the session). Now only
  the rows written since.

The mismatches (6 restores to the last header, 47.6K tokens re-prefilled):

- 3 with the same text on both sides at the first difference (positions 15304, 26659, 52037, two of them 16K-token
  answers cut at max_tokens): the client's re-rendered text tokenizes differently from the model's own tokens.
  37.8 + 75.4 + 6.0 s. The text match keeps the model's tokens.
- 1 streaming bug: `U+21D4` split across tokens was streamed as `U+FFFD`, the client stored that (1.5 s).
- 2 client edits of the answer (a trailing newline dropped from a tool argument, a malformed argument removed):
  5.2 + 2.7 s. Decode snapshots make these a few hundred tokens.

A correctness bug found on the way: a summary request (another system prompt, cold) rewrites KV rows 0 to 34K, but
the main conversation's 'system' snapshot stayed in the cache (a V8 cold run does not take an in-run snapshot for a
cut below 128, so the summary's own 'system' did not replace it). The next main request restored it: the DeltaNet
state of the main system prompt over the summary prompt's KV rows (13:14 and 13:36 in that session). Snapshots
whose tokens are not a prefix of the KV cache's are now dropped, so that request runs cold.

Validation (fill check on, `unit/srv_bench.py --tokens 20000 --follow 50,300,2600,40`): cold 20K 574 tok/s;
+72 / +63 tokens 0.4 / 0.3 s; +322 at 20.5K 1.4 s; +2622 at 21K 6.5 s (V fill 320 rows, 37 ms); fill check error
0.500 x scale on every warm run (the GPU's own fp16 rows against INT8; converted rows are exact). An answer edited
near its end restored 'dec2' 117 tokens back (0.4 s; before: about 630 tokens from the answer's start).
`unit/shortfix_check.py` (fake bridge and engine, the real tokenizer) covers the logged cases, the summary request,
the text match and the UTF-8 hold-back. Logs: `coreai_seg_logs/v8_server_8765_pre_shortfix.log` (the session),
`v8_server_shortfix_check.log`.

## Host speedups and the summary-proof system snapshot (16 Nov, evening)

The release tree the server imports (`anemll-forge-v02`, = `scripts/` of the `v8-gpu-ane-prefill` branch) now has
the host-side speedups from Greg Martin's fork (gregcmartin/anemll-forge-speedup, 8410a33 and ada23b3): head-free
prefill blocks, FP16 top-k / FP32-widened argmax in sampling, an incremental detokenizer (also holds a split UTF-8
character back; the V8 wrapper's own hold-back is used only without it), the drafter's FP16 top-k and lazily
written context, a 1 ms draft gap on M6; prompt-lookup drafts are ported but off. The V8 wrapper skips the head on
its alignment blocks too.

Measured against the same server before (20K cold + follow-ups of 50 / 300 / 2600 / 40, greedy, thinking off): decode
cycle 122 -> 120.5 ms (draft 14.5-15.1 -> 13.0-13.4 ms, sampling 0.9-1.3 -> 0.2-0.3 ms, verify unchanged at 106 ms:
no stalls from the 1 ms gap), the same tokens per verify in all 5 turns, bridge blocks about 5 ms faster. With
Pi's sampling (temperature 1, top_k 20) the sampling phase was 3 to 6.6 ms per cycle before.

'sysk', live (an 8.4K-token system prompt, then a 2.7K summary prompt with another system prompt, then the main
system prompt again): restored at 8433 with 26 new tokens in 0.2 s (cold: about 13 s), the same 24 greedy tokens as
the in-memory restore; a 628-token follow-up after it ran on V8 with the V copy refilled from row 0 (fill check
0.061 x scale: fp16 rounding of converted rows). The first live test found that a short summary shrinks the cache
to the 8K entry, below an 8.4K system prompt: the restore now grows the context first, and falls back to a cold
start if it cannot restore. Logs: `coreai_seg_logs/v8_server_speedup_check1.log`, `v8_server_speedup_check2.log`.

In a Pi session (17:14 to 18:03): Pi compacted at 65K; the summary request kept 'sysk' (5888 tokens, 185 MB, 33 ms),
the compacted context restored it (rows written back in 37 ms) and prefilled 14,262 tokens instead of 20,150 (10 to
12 s saved). The text match (one re-tokenized spot in the first answer, every follow-up) now uses a memo of the
stretch where the release tokens and the state's cover the same text: 0.3 ms per request (the general path: about
25 ms at 65K, on top of the release tokenizer's 28 ms).

Decode A/B (`V8_AB=g1l,g3l,g3e`, interleaved over the session's requests, 64K entry, by verify cycles):

| arm | requests | ms/cycle | draft | verify | sample |
|---|---|---|---|---|---|
| g1l (1 ms gap, lazy context: the default) | 5 | 142.8 | 13.7 | 127.4 | 0.87 |
| g3l (3 ms gap, lazy) | 6 | 144.3 | 15.9 | 127.1 | 0.71 |
| g3e (3 ms gap, eager: before the port) | 6 | 144.1 | 15.4 | 126.9 | 0.86 |

The 1 ms gap saves about 2 ms per cycle at 32K and 64K with the verify unchanged; lazy and eager context are the same.
The verify rose from about 121 ms (earlier sessions) to 125 to 127 ms in every arm: not the port but the machine. A
visible Chrome tab (the WebGL page under test) and Screen Sharing kept the GPU about 66% busy with the server idle; on
the M6 the GPU's load takes ANE speed (the shared power budget), and the V8 prefill's GPU work and K8V16 fills waited
too (fills 65 to 368 ms instead of 31 to 58 ms, a cold 45K summary at 387 tok/s). Benchmarks need that GPU idle.

## Files

- `v8_server.py`, `v8_server.sh`, `v8_server_plain.sh`, `v8_server_dec8.sh`, `srv_stop.sh`
- `dec_split.py` (SPLIT+GPU: runtime patch to load each 8-layer package once, the split verify, KV sync to the GPU),
  `engine/dec_gpu.swift`, `engine/dec_kernels.metal`; checks `unit/dec8_srv_check.py`, `unit/dec8_real_check.py`
- `v8_engine.py` (V8Prefill: run, snapshots, handover, sync, K8V16 fill, can_start)
- `engine/gpu_v8.swift`, `engine/kernels.metal`, patches `patch_nv.py`, `patch_srv.py`, `patch_state_gpu.py`,
  `patch_k8v16.py`, `patch_fillfrom.py` (each keeps `*.pre_<name>.*` backups)
- `unit/srv_bench.py` (cold + follow-up turns), `unit/srv_matrix.sh` (config x length matrix), `unit/srv_summ.py`,
  `unit/attn_kv8_prof.py` (attention core per call), `unit/state_check.py`, `unit/shortfix_check.py` (prefill /
  snapshot / sysk / text-match logic over a fake bridge, offline; `V8_V02=<tree>` checks another release tree), `unit/srv_log_prefill.py` (per-request prefill paths from a server log)
- results: `coreai_seg_logs/srv_matrix.jsonl`, `coreai_seg_logs/k8v16_65536.{log,json}`

## Open items

- A8 (runtime INT8 FFN on the ANE) is not in the server: +6% at 16K in the driver, slower at 64K+.
- Decode past 64K: done with SPLIT+GPU (141 / 149 ms per cycle at 64K / 95K, the same as splash 1.3.2). Left: the GPU's cold
  start after each ANE call (about 10 ms per step at 95K), splash 1.3.2's higher acceptance per cycle, and serving 128K
  (the split needs no ANE recompile there; the KV cache, the V8 prefill and the server limit stop at 96K).
- A bridge prefill block costs 0.2 to 0.45 s whatever its rows (one is needed after decoding: pending rows);
  V8 at any 64-unaligned position would save one block on some follow-ups (tile n rounded up, masked).
- Short follow-ups are now one bridge block (0.24 s at 13K, 0.37 s at 50K) plus a T-row call only when the rest
  does not fit the block; follow-ups of 1.4K+ still pay 1 to 2 alignment blocks and about 0.2 s of handover.
- Short V8 spans are latency-bound (a pair through 64 layers): about 0.25 s per pair at 6K, 0.36 s at 32K, 0.55 to
  0.64 s at 96K.
