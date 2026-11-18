# GPU + ANE prefill on the M6: what we tried (13 to 17 Nov 2026)

Goal: faster prefill for Qwen3.8-27B (release 0.2, vq3pA: 3-bit VQ MLP, LUT mixers) on the M6 32 GB by using the GPU
and the Neural Engine together, ideally 700 to 1000 tok/s. The model has 64 layers: 48 DeltaNet (Gated DeltaNet
mixers) and 16 full attention, each followed by an FFN (5120 -> 17408 -> 5120). The ANE-only target build (Core AI,
INT8 KV) prefills about 300 tok/s at 2K to 8K, 259 at 32K, 216 at 64K and 180 at 96K.

Final state: the V8 schedule (all FFN on the ANE, all mixers on the GPU) in an OpenAI-compatible server:

| prompt | V8 server, K8V16 | ANE-only | splash 1.3.2 (same M6) |
|---|---|---|---|
| 2K | 634 | about 300 to 320 | 496 to 507 |
| 6K | 663 to 700 | about 300 to 320 | 486 (8K) |
| 16K | 601 | 290 | 460 |
| 32K | 516 to 537 | 259 | 411 |
| 64K | 405 to 415 | 216 | 336 |
| 95K | 348 to 352 | 180 | 288 |

Server-side tok/s for a whole cold request (handover and drafter context included). Client time-to-first-token figures
are 3 to 4% lower; on that basis V8 leads splash 1.3.2 by 21% at 2K, about 35% at 6K to 8K, 30% at 16K, 25% at 32K and
20 to 23% at 64K to 95K. Splash 1.3.2 runs a different 3-bit model (unsloth UD-IQ3_XXS GGUF), so quality was not compared.

### The same requests at the same 96K context (16 Nov evening)

`splash132/cmp_96k.sh`: `pipe/unit/cmp_bench.py` on the V8 server, then on splash 1.3.2 served with
`--max-context 96K` (its earlier runs: 256K), one server at a time, the GPU otherwise idle. The same documents and
instructions (a random 12-character salt per cold prompt, 1 token of difference), 2 min idle before each cold prompt,
256 tokens greedy with thinking off, one follow-up turn (each model's own answer + 300 new tokens), both with a DFlash2
drafter. Client time to the first streamed token.

| prompt | cold prefill, tok/s: V8 | splash 1.3.2 | V8 lead | follow-up, time to first token: V8 | splash 1.3.2 | decode cycle: V8 | V8 + SPLIT+GPU | splash 1.3.2 |
|---|---|---|---|---|---|---|---|---|
| 2K | 586 tok/s | 476 | +23% | 1.04 s | 1.48 s | 108 ms | 107 ms | 116 ms |
| 8K | 658 | 480 | +37% | 1.15 s | 1.58 s | 112 ms | 111 ms | 119 ms |
| 16K | 620 | 454 | +37% | 1.90 s | 1.73 s | 120 ms | 119 ms | 122 ms |
| 32K | 541 | 404 | +34% | 1.75 s | 2.09 s | 134 ms | 133 ms | 131 ms |
| 64K | 420 | 329 | +28% | 2.36 s | 2.71 s | 153 ms | **141 ms** | 141 ms |
| 95K | 347 | 283 | +23% | 2.92 s | 3.34 s | 155 ms | **149 ms** | 148 ms |

- **Cold prefill**: prompt tokens divided by the client's time to the first streamed token of a new conversation.
- **Follow-up**: the second turn of the same conversation (the cold prompt, the model's own 256-token answer, then 300
  new tokens). Time from sending the request to its first streamed token at the client, which is mostly the
  prefill of the text not yet in the server's cache. V8 continues from its saved state and prefills only the new text
  (about 320 tokens with the chat template); splash 1.3.2 caches the prompt it was sent, not what it generated, so it
  also prefills its own previous answer (95,040 of 95,630 tokens cached at 95K: 590 tokens prefilled).
- **Decode cycle**: milliseconds per speculative cycle (the DFlash2 drafter proposes tokens, the target verifies them
  in one call); decode tok/s = accepted tokens per cycle / cycle time.
- **SPLIT+GPU** (17 Nov, `pipe/v8_server_dec8.sh`, `pipe/dec_split.py`): a decode-only change. In each verification
  step only the attention over the KV history of the 16 full-attention layers (QK, softmax, PV over the INT8 K/V
  cache) runs on the GPU; everything else stays on the Neural Engine (the 48 DeltaNet layers, every FFN and every
  projection). Per 8-layer package the Neural Engine runs part A (up to the attention layer's q / k / v projections,
  `dva`), the GPU the attention, the Neural Engine part B of that layer fused with part A of the next (`dvba`), the
  GPU the next attention, the Neural Engine the rest (`dvb`). The Neural Engine parts bind no KV cache and no mask, so
  one program serves every context length. It is on only in the 96K context entry, that is past 64K tokens
  (`V8_DEC8=auto`; the 64K row has 65.6K tokens, so it already runs there): in that entry the Neural Engine's own
  attention is the most expensive (122 ms per verification step against 85 ms in the 16K entry) and the split is 5 to
  8% faster. In the 16K and 32K entries it is 8 to 17% slower and in the 64K entry about even (-1.3% to +2.5%),
  because each Neural Engine / GPU hand-over costs about 0.65 ms, four per package. Running the attention on both
  units, split by key range, gave about 1 ms per step and was rejected (`pipe/unit/DECODE_GPU_ATTENTION.md`).

- Prefill: V8 leads at every length. Splash 1.3.2 at 96K context ran within 1 to 5% of its 256K-context runs.
- Follow-ups: both servers keep the prompt cache (splash reported 95,040 of 95,630 tokens cached); V8 is faster at 5
  of 6 lengths.
- Decode: with the Neural Engine verify alone (16 Nov), V8's cycle is cheaper to 16K, the same at 32K and 5 to 8%
  dearer from 64K (the ANE verify carries the long attention). SPLIT+GPU (above) closes that gap:
  141 ms at 64K and 149 ms at 95K against splash 1.3.2's 141 and 148. Below the 96K entry it is off and the cycle is unchanged
  (the 1 ms differences are run noise). Acceptance is close on cold turns (V8 2.4 to 3.7 tokens per cycle, splash 2.6
  to 4.4) and follows the text on follow-ups. Cold-turn decode is within 10% to 32K; at 64K / 95K splash decodes
  22.6 / 20.4 tok/s against 18.5 / 18.9 before SPLIT+GPU and 20.6 / 20.4 with it.
- End to end, a 95K cold prompt + 256 tokens: V8 287 s (278 s with SPLIT+GPU), splash 1.3.2 348 s.
- The 17 Nov rerun with SPLIT+GPU gave the same prefill within run noise (576 to 668 tok/s at 2K to 16K, 548 at 32K,
  429 at 64K, 358 at 95K) and follow-ups of 1.1 to 1.7 s to 32K and 2.55 s at 95K; the 64K follow-up took 3.31 s
  against 2.36 s (one run, not investigated).
- One run per point; different weights (the answers and their acceptance differ). Results:
  `results/cmp_v8_96k.jsonl`, `results/cmp_v8dec8_96k.jsonl` (SPLIT+GPU), `results/cmp_splash_96k.jsonl`.
- **Splash 1.3.2 on its GPU alone** (`--disable-ane`, 17 Nov, the same requests, `splash132/cmp_96k_gpu.sh`,
  `results/cmp_splash_gpu_96k.jsonl`): cold prefill 319 / 317 / 306 / 282 / 244 / 217 tok/s at 2K / 8K / 16K / 32K /
  64K / 95K. Its Neural Engine split adds 1.49 / 1.51 / 1.48 / 1.43 / 1.35 / 1.31 times; V8 adds 1.83 / 2.08 / 2.02 /
  1.92 / 1.72 / 1.60 times over the same baseline. Splash's GPU alone matches our ANE-only build up to 16K and leads it
  at long context (244 against 216 at 64K, 217 against 180 at 95K). Its decode barely changes without the split
  (within 10%): the split only takes prefill chunks of 512 rows or more. Why splash gains less: its Neural Engine takes
  76% of each FFN's channels inside each layer and waits while the GPU runs the mixers (busy about 42% of each
  2,048-row chunk, 13 Nov), while V8 pipelines prompt blocks so both units stay busy; its GPU also computes the other
  24% of each FFN and stages the Neural Engine's INT8 weights. With kernel tuning these explain the gap, but their
  shares are not isolated (no matched phase profiles or scheduling ablations yet, and the quantizations differ).

### Pi coding sessions against splash 1.3.2 (17 Nov)

The same Pi coding-agent prompt on each server, one server at a time: V8 with SPLIT+GPU (`pipe/v8_server_dec8.sh`,
port 8765) and splash 1.3.2 (`splash132/serve_pi.sh`, port 8000, DFlash2 drafter). Both at 96K context, thinking on
with effort medium, Pi compaction reserve 32,768 / keep 8,192, and the same sampling (neither request sets it, and both
servers then use temperature 1.0, top_p 0.95, top_k 20, no penalties). Both servers render the same 6,761-token first
prompt. Our server closes thinking at 6,144 tokens for effort medium with Qwen's budget phrase; splash has no cap
unless `SPLASH_THINKING_TOKEN_BUDGET` is set, so the second splash cron run used 6,144 and the same phrase
(`SPLASH_THINKING_END_TEXT`). The cron library is graded by a hidden test suite (`bench/cron/grade.mjs`, 1,373 checks:
`nextRuns` against a day-by-day scan of the whole 4,000-day window over 600 random expressions, fixed edge cases, 47
invalid expressions, argument errors and a 100 ms limit; prompt in `bench/cron/PROMPT.md`).

| task | server | wall | turns | output tokens (thinking) | prefill total | decode | result |
|---|---|---|---|---|---|---|---|
| Tetris in HTML, "test and fix" | V8 | 4.0 min | 10 | 8,259 (1,585) | 3 s | 34.8 tok/s | playable; the O piece shifts when rotated, Left/Right work while paused |
| | splash 1.3.2, no cap | 52.9 min | 66 | 69,655 (35,472) | 450 s | 29.9 tok/s | every piece after the second is the same type (`spawn()` never advances `next`) |
| Cron library, hidden grader | V8 | 10:17 | 12 | 18,877 (8,991) | 22 s | 31.8 tok/s | 1373 / 1373 |
| | splash 1.3.2, no cap | 22:25 | 12 | 38,389 (24,771) | 137 s | 31.8 tok/s | 1373 / 1373 |
| | splash 1.3.2, cap 6,144 | 16:56 | 27 | 29,746 (15,151) | 133 s | 33.7 tok/s | 1373 / 1373 |

- **Decode speed is a tie.** On the same first prompt (the capped cron pair) V8 wrote 8,265 tokens at 29.7 tok/s
  (3.3 tokens per 112 ms cycle) and splash 8,271 at 31.5 tok/s (3.8 per about 121 ms): V8's cycle is shorter, splash
  accepts more per cycle. Over the cron runs the two stay within 6% of each other; in Tetris V8 averaged 34.8 against
  29.9 tok/s, but splash's context grew to 64K against V8's 15K.
- **Prefill in agent loops favors V8.** V8 keeps the generated tokens in its state and prefills only each new tool
  result (0.2 to 1.6 s for most turns, 5.5 s for a 2.3K-token test log); splash caches the prompt it was sent and prefills its own previous answer again on
  every turn (20 s after an 8K-token answer, 65 s after a 24K-token one): 133 to 137 s per cron run against 22 s.
- **The time gaps come from the amount of work.** In the capped cron pair the first turns were the same size; splash's
  model then wrote twice the tests (76 against 34), needed 16 debugging turns against 7 and kept checking after its
  tests passed. Without the cap its first turn thought 21,780 tokens (V8: capped at 6,144) for the same 1373 / 1373.
  Splash's Tetris run also lost 4.6 min to a Pi compaction at 64K (a cold 48K-token prefill plus the summary) and
  4 min to two Chrome AppleScript timeouts. Tetris never reached the thinking cap on either server.
- **No per-response verbosity difference.** Single-turn probe (`bench/verbosity/`: 20 short coding prompts sent to the
  API directly, 14 paired samples completed in the time box): thinking median 904 (splash) against 900 (V8) tokens,
  answers 706 against 678; splash was longer on 7 of 14 and 6 of 14 (sign test p = 1.0 and 0.79).
- One run per cell at temperature 1.0 on two different 3-bit quantizations; only the cron task has a grader.

## How we measured

- **Prefill only**: driver timing from the first block to the last logits (`prototypes/v8_prefill.py --mode v8`).
- **Real end to end**: `--handover --ho-overlap`: the process also holds the whole target build. KV rows and DeltaNet
  states go to the ANE during the run, then 32 decode steps run on the ANE, teacher-forced against the ANE-only
  reference: decode KL and argmax agreement.
- **Server**: HTTP requests (`pipe/unit/srv_bench.py`, `srv_matrix.sh`), cold prompts plus follow-up turns.
- **Quality**: per-block last-token KL against the ANE-only reference (`--mode ref`) and the decode check. KL against
  the ANE-only path grows with length for every V8 variant (2K 0.010, 8K 0.020, 32K 0.030, 64K 0.058).
- **Thermal**: both units slow under sustained load with no pmset warning: the ANE FFN pair call goes from 2.6 to
  3.0-3.7 ms and the GPU mixer from 3.19 to 3.38 ms. A 5 min idle restores it. The first rep after a cool-down is
  about 5% fastest. Configs are compared only interleaved (A B A B) in one session, or rotated within one process.
- **Memory**: every long job runs under `tools/swap_watchdog.sh`. Weights must exist once (the GPU reads the ANE's
  compiled HWX zero-copy).

## Variants, in order

### V1: layer pipeline (hybrid), kept as the first baseline

The ANE runs target chunks 0..c-1 of block b+1 while the GPU runs the remaining layers of block b, reading the weights
zero-copy from the ANE's compiled programs (`pipe/engine/gpu_layers.swift`, `prep_engine.py`). Best split 9 chunks on
the ANE, 28 layers on the GPU ("9/28"). Driver `prototypes/hybrid_prefill.py`, long context and decode handover
`prototypes/hybrid_long.py`.

- 8K: 486 tok/s (ANE-only 303), 32K 412 (264), 64K 326 (217). KL vs ANE-only 0.0033 (2048 tokens). Decode after
  handover at 8K 32/32, KL 0.004.
- The GPU engine matches a float32 reference of the release math to 4e-4 (`pipe/ref_layers.py`).
- Splash 1.3.2 measured about the same: 2K 499, 8K 486, 32K 396-411, 64K 323-336.

### V2 to V7: analysis and probes

- **V2, attention offload to the GPU.** At 64K the ANE spends about 8.7 ms more per attention layer than at 8K. A GPU
  INT8 QK + PV would cost at least 3.1 ms at 64K. This needs ANE chunks split at the attention layers. Folded into V8.
- **V3 (mixers on the ANE, FFN on the GPU) and V4 (tensor parallel).** Rejected on arithmetic: GPU-bound, or a sync
  per matmul.
- **V6, GPU matmul units.** NAX (MPP matmul2d) runs at fp16 18, int8 x int8 30-33.6, uint8 32.8, fp8 e4m3 30-32, e5m2
  21, half x int8 19.4 and fp4 16.5 TOPS. simdgroup_matrix gives 4.8. FP8 needs Metal 4.1.
- **V7, splash-style W8A8 FFN** (runtime INT8 weights) in Core AI: about 10 TOPS, 2.4x slower than our compile-time
  LUT FFN. Rejected then. The A8 probe later showed the cause was the matmul shape, not the INT8 operands (see A8).

### V8: every FFN on the ANE, every mixer on the GPU

A per-layer comparative-advantage model gave the split. Per token per layer, the ANE LUT FFN costs 23.2 us against
36.3 on the GPU (1.56x better on the ANE), while the mixers cost about the same on both. The GPU mixes layer L of
block k once the GPU has mixed layer L for every earlier block and block k has finished the FFN of layer L-1 (an
earlier block's FFN of layer L is not needed: it feeds only that block's next layer). Buffers are the ANE's
IOSurfaces, used zero-copy.

- **First prototype was slower than V1** (2K 470, 8K 445, 32K 363). Fixes:
  - a separate GPU waiter thread, so the submit thread never blocks;
  - more streams;
  - **pair mode**: one 128-row GPU call per two 64-row ANE blocks;
  - **one ANE thread per pair**: two threads calling the same Core AI function concurrently (different bindings)
    corrupt outputs at random (KL spikes 0.6 to 4).
  - Result: 2K 574, 8K 494, 32K 402.
- **ANE per-prediction gap.** Instruments shows a 0.23 ms idle gap before every ANE evaluation. It comes from the
  driver or firmware: private _ANEClient calls and Metal shared-event chaining (splash's method) show it too. Only
  fewer, bigger predictions help: `ffn2` entries run two 64-row blocks in one function, saving one gap per pair.
  Result: 2K 566-582, 8K 515.
- **Chunked FFN** (`v8/chunkmlp.py`, builder `V8_SPLIT=8192,9216`): the intermediate is split into two chunks of the
  same LUT palette. The ANE FFN takes 1.31-1.35 ms instead of 1.61-1.66 (-18%). Result: 2K 626-633 (+17%), 8K
  544-560, 32K 439.5. Other splits gained less ([9216, 8192] -10%, 4 to 17 chunks -5 to -14%) or lost
  ([16384, 1024] +24%).
  128-row ANE calls stay 19% worse per token than 64-row ones.
- **Shared-weight build** (`v8/build_shared_chunk.py`). Separate FFN-only and mixer builds duplicated every FFN weight
  and did not fit in 32 GB above 8K with the decode target loaded. One package per target chunk now carries the target
  entries (with the chunked FFN) and the V8 `ffn` / `ffn2` / `mix` / `mix2` entries on the same modules. Core AI
  stores each constant once, so a package is the size of the plain target chunk (593 to 703 MB).
- **Attention projections on the ANE** (`V8_ATTPROJ=1`, build v8_shared2):
  - `ffnq2_L` = FFN + the next attention layer's norm, rotation and q / k / v projections; `offn2_L` = o projection +
    residual + FFN. The GPU runs only the attention core (`v8_submit_attcore`).
  - Per attention layer per pair: ANE +1.28 ms, GPU -2.0 ms.
  - Helps where the GPU is the bound: 32K real e2e 490 vs 474.
  - Hurts below about 8K, where the ANE is the bound: 2K 594 vs 653, 6K 539 vs 592.
- **ANE mixers** (`mix2` entries for 16 DeltaNet layers, "M16"): +2.8% at 64K, +1.9% at 96K. Neutral to -20% at
  8K to 32K. Not used in the server.
- **Position-dependent offload** (attproj only past a position, `--attproj-from`): about 0.5% over "always".
- **GPU engine work** (`pipe/engine/gpu_v8.swift`, `kernels.metal`, env switches). Kept:
  - `CONC`: concurrent encoder, the recurrence overlaps z (-2% GPU time).
  - `FUSE=all`: fused norm / rotation / conv / gates / conversions.
  - `I8=1`: W8A8 DeltaNet LUT GEMMs, int8 palettes, activations int8 per 1024-ch block; KL unchanged.
  - `I8T=1` and `I8TV=_k256`: channel-major int8 tiles, K step 256.
  - `LUTC=1`: fp16 decode, K-column-fastest.
  - `I8A=1`: attention q / o in int8. `I8KV=1`: the export's int8 k / v weights. `LRM=1`: one low-rank B for qkv + z.
  - `QK8=1 TKV=2048`: int8 Q x K scores, 2048-key tiles.
  - Result: the DeltaNet layer per 128 rows went from 2.59 to 1.97 ms. 2K prefill-only reached 727 mean / 745 best;
    real e2e 2K 694 / 747, 6K 681 / 712.
- **Rejected engine variants** (still opt-in switches in the sources, outputs bit-same or correct):
  - Flash attention `FA2` (S in MPP cooperative tensors): correct but 11.2 ms vs 6.8 at 16K. Single-simdgroup
    matmul2d runs about 2x slower than the 4-simdgroup tiles.
  - Two-pass QK `QX`: exact, 7.52 vs 6.81 ms. Tile shapes (QK 64x128 / 128x64 / 128x128, PV 64x256 / 128x128): none
    beats 64x64 QK / 64x128 PV.
  - `SMF` softmax forms and `LRF` (low-rank folded into the GEMM): no gain.
  - `EPI` (residual in the epilogue): no gain under contention, decode KL worse.
  - `gdn_rec8` / `gdn_rec16`: slower; the recurrence is bound by its serial chain.
  - Double-buffered LUT tiles (`LUTV`), 256-row calls (`NTV=256`), int8 decode double buffering, `REC` / `LUTP`: no gain.
  - The GPU FFN for 2 to 6 layers (`I8F`, `--gpu-ffn`): the GPU becomes the bound, no gain.
- **KV8: the GPU attention reads and writes the ANE's INT8 KV segments.** No fp16 GPU KV, no KV copy, warm
  continuation possible.
  - The first version lost quality (decode KL 0.10 vs 0.02 at 16K). Cause: `attn_softmax8` put p x value-scale into
    fp16, which goes subnormal (about 1% error). Fixed by scaling P' and the sum by 4096.
  - After the fix KV8 is lossless: per-block KL 0.0592 vs 0.0593 fp16 at 64K.
  - It is still 4 to 6% slower at 64K+: the PV is half x INT8, +40% over fp16 PV. On-chip V dequant (`KV8D`) and a
    256-wide PV (`PVWW`) did not help.
- **K8V16** (`K8V16=1`, the server default): keys INT8 from the ANE cache with the key scales folded into the INT8
  score; values also kept fp16 on the GPU (+32 KB/token, 3.2 GB at 96K) for the fp16 PV. A warm start rebuilds that
  copy from the ANE's INT8 V (`v8_kv8_v16_fill`). The 64K attention call takes 17.65 ms (KV8 19.83, fp16 17.71);
  64K e2e 419.9 vs 401.7, KL 0.0598.

### The coupling: why balanced estimates are not reached

- **The cost.** The GPU and ANE slow each other down. The GPU attention call at 60K takes 18.7 ms alone and 26.8-27.1
  ms (+43-45%) under an ANE load. The ANE FFN slows 3 to 20% under GPU load.
- **Not bandwidth.** A weight-light, compute-heavy ANE program and the weight-heavy FFN loop cause the same slowdown.
  A 153 GB/s GPU memory stream costs the ANE only 7% and ALU FMA 3%, but the int8 DeltaNet loop costs 20%.
  In V8 the GPU slows only once the ANE is busier than about 50%.
- **The likely cause:** a shared power or clock limit on the matrix units (not measured: no power or clock samples yet;
  shared system-cache contention for matrix work would fit the same tests). Moving work to an idle unit buys less than
  the arithmetic says. ANE prefix chunks at long context, ANE mixers and the A8 wave size all hit this.
- **Core AI's own GPU path** cannot be the V8 GPU side: `ffn2_L0` took 89.5 ms there (our Metal 4.6), the DeltaNet mix
  took 438 ms, and the 593 MB chunk took 118 s to load and expanded its weights into memory.

### A8: runtime-INT8 FFN on the ANE

The ANE's INT8 rate (about 53 TOPS on a 4096^2 matmul) is twice what our FP16-activation LUT FFN programs reach
(26 to 28). Probes in `w8a8/`:

- **Shape probe** (`a8_ffn_shape_probe.py`):
  - The full 5120 x 17408 matmul runs at 13 TOPS (V7's 10).
  - 17 blocks of 1024 outputs with the K dimension split in two 2560 parts runs at 48.5 TOPS.
  - Gate and up as separate K=5120 matmuls compute **wrong results** (rel 0.98) at any width, and so does 17x1024
    without the K split at 1024 rows (rel 1.24), with nothing to signal it. Always check ANE INT8 output against a
    host reference.
- **Real FFN** (`a8_ffn_real_probe.py`: SiLU, the H1024 rotation, per-row weight scales, dynamic activation scales):
  - Speed: 1024 rows take 13.95 ms = 39 TOPS, 13.6-13.9 us/token, against about 20 for the production ffn2 (1.5x).
    One package serves every layer.
  - Accuracy: FFN error 1.3-1.7% vs fp32 (fp16 LUT 0.21%), 0.11-0.31% after the residual.
- **Pipeline M1** (`pipe/v8_a8.py`, layer-major over 1024-row waves). The GPU expands each layer's LUT FFN into INT8
  weights on a second queue (`v8_a8_expand`; CPU NEON version in `engine/a8_cpu.c`).
  - KL unchanged, but slower: 2K 630 vs 742. It is GPU-bound once the ANE speeds up.
- **A8 + mixer projections on the ANE** (`w8a8/a8_att_build.py`: AQKV, AO programs; `--att all --out-layers gdn`):
  - Same-session A/B: 6K 691 vs 625 (+10.6%), 16K 641 vs 596 (+7.6%), KL equal. Real e2e 16K 641.7 vs 605.4.
  - 2K: 611-633, slower (only 2 waves). 32K: no gain (514).
  - 64K+: slower with `--kv8` (391.6 vs 426.3 at 64K). Without it, 64K did not fit in memory.
  - Not in the server.
- **The split pattern on V8's fp16 LUT FFN** (`w8a8/lut_split_probe.py`): no gain at V8's 128-row calls. At 1024 rows
  it is -8%, against -33% for the A8 int8 form.

### The server (`pipe/v8_server.py`, `pipe/V8_SERVER.md`)

It is the release server with `Engine.prefill` replaced. Decode, the DFlash2 drafter and prompt-cache snapshots are
unchanged.

- **Cold prompt:** V8 runs from 0 to the end; the last 128-row pair is padded (state and conv stop at the last real
  token), and the logits come from V8's own head.
- **Snapshots** are taken during the run at the aligned pair end below each cut. Each DeltaNet layer's state is copied
  as its pair completes, and that layer's next pair is gated until then.
- **DeltaNet state handover** runs on the GPU (`gdn_state_out` / `gdn_state_in`, about 10 ms). The remaining ~170 ms
  after a run is the drafter taking in its context.
- **Follow-up turns:**
  - The bridge feeds prefill blocks up to the next 64- or 128-aligned point. The start is chosen by fewest bridge
    blocks + V8 pairs. A 64 start is allowed only if the span crosses no KV segment boundary.
  - Then the states are synced to the GPU and V8 runs. V8 takes spans of 256 tokens or more (the bridge alone does
    about 145 tok/s at 64K+).
  - 64K + 723 new tokens: 4.93 -> 3.54 s; 64K + 1522: 10.39 -> 6.94 s.
- **Bugs found:**
  - A bridge feed that ended with a short call left pending rows, so V8 silently never ran on some follow-ups.
  - Speculative greedy outputs are not an equality test, because drafter state changes the verify grouping. Use
    `--plain`.
- **Decode** is unchanged by the V8 prefill: 110 ms per cycle at 6K, 152 at 96K (verify 135 ms), 19 to 35 tok/s
  depending on draft acceptance. SPLIT+GPU (17 Nov) brings 64K and 95K to 141 and 149 ms.
- **A real Pi session** (53 requests, `pipe/unit/srv_log_prefill.py`): prefill 497 s against decode 4231 s.
  - Compaction took 49% of the prefill: summary prompts are cold, and the compacted context is prefilled again.
  - Prompt mismatches took 26%: 6 restores to the last header re-prefilled 47.6K tokens. Three were the same text
    tokenized differently, one was a character the server had streamed as U+FFFD, and two were client edits.
  - Short follow-ups (29 requests, 13 to 329 tokens) were only 3.5% (0.3 to 0.6 s each). They still lost 20 to 35%,
    on an extra T-row call for the header snapshot, two alignment blocks where one does, V8 dropped after the
    alignment, and nearly empty last pairs.
- **Fixes (16 Nov, afternoon):**
  - an exact 'gen' snapshot at the prompt's end, plus decode snapshots every 256 tokens;
  - alignment blocks of any 1 to 64 rows;
  - V8 decided on the rest after the alignment;
  - one T-row call for a last pair with 8 real rows or fewer;
  - an incremental K8V16 fill (`v8_kv8_v16_fill_from`);
  - keeping the model's own tokens where the prompt's text matches them;
  - UTF-8 hold-back in streaming.
  - Measured: +63 tokens 0.3 s in one block; an edited answer restored 117 tokens back instead of about 630. Details
    are in `pipe/V8_SERVER.md`.
- **Evening:** the host-side speedups from Greg Martin's fork (head-free prefill blocks, faster sampling and
  detokenizing, the drafter's FP16 top-k, a 1 ms draft gap) took the decode cycle from 122 to 120.5 ms with the same
  tokens per verify. A 'sysk' snapshot keeps the system prompt's KV rows across Pi's summary request: the next prompt
  restores 8.4K tokens in 17 ms instead of prefilling them cold (about 13 s).
- **Bug found then:** a summary request (another system prompt) rewrites the KV rows under the main conversation's
  'system' snapshot. The V8 cold run did not replace that snapshot, so the next main request paired the main system
  prompt's DeltaNet state with the summary's KV rows. Snapshots that are not a prefix of the KV cache's tokens are now
  dropped.

## Lessons that carry over

1. **Fewer, bigger ANE predictions.** Every evaluation has a fixed 0.11-0.23 ms gap that no submission trick hides.
2. **Never call the same Core AI function from two threads at once**, even with different bindings.
3. **Fast ANE INT8 forms can be silently wrong.** Check every new form against a host reference.
4. **The GPU and ANE share a matrix-unit power budget on the M6.** Per-unit ceilings do not add up; measure the pair.
5. **Weights once.** The GPU reads the ANE's compiled HWX zero-copy, and shared-weight packages store constants once.
   Duplicated weights do not fit in 32 GB at long context.
6. **Compare only back-to-back or interleaved runs.** Thermal drift is 5 to 10% within minutes.
7. **Core AI limits.** A function fails at run time past about 51 bound tensors. A view that does not cover its
   whole IOSurface aborts.
   Segmented KV (commit 1) is the way to grow context without copies.
8. **Compile cache purges.** After a reboot aned can purge every compiled entry while `forge.py` still reports
   "compiled": use `forge.py compile --force`.

## Where the time goes now

- **2K to 8K: ANE-bound.** The ANE runs 98-100% busy. The FFN programs run FP16 activations x LUT at 26-28 TOPS;
  the ceiling for these programs is about 735-750 tok/s at 2K.
- **32K and up: GPU-bound on attention.** At 30K a call is pv 3.09 ms, qk 2.27 ms and softmax 1.96 ms. The two-pass
  score matrix (about 190 MB per call) is most of it.
- **Decode past 64K:** before SPLIT+GPU, bound by the ANE verify step (135 of 152 ms at 96K). With it, 141 ms per
  cycle at 64K and 149 at 95K; the GPU attention runs at about 90 to 100 GB/s in the server (127 back to back), held
  back by its cold start after each ANE call.

## Decode: where the verify time goes, and GPU attention for it (16 Nov, night)

**Superseded on 17 Nov:** the tuned kernel and the split build are served as SPLIT+GPU (see the 96K table above and
[pipe/unit/DECODE_GPU_ATTENTION.md](pipe/unit/DECODE_GPU_ATTENTION.md)). This section is the 16 Nov analysis.

The decode cycle is the drafter call (13 to 16 ms) + the target's 8-row verify on the ANE (85 to 90%) + about 1 ms of
host work. The verify costs about 90 ms without attention + about 0.5 ms per 1K rows of its context entry, whatever
the position (static shapes: the attention runs over the whole entry's KV, 32 KB per position for 16 layers): 94 ms
in the 8K entry, 106 in 32K, 120 in 64K, 138 in 96K. A 65.6K prompt is past the 64K entry's 65,472 rows, so it decodes
in the 96K entry: 153 ms per cycle at 65.6K, 155 at 95K (splash 1.3.2's GPU attention follows the actual length: 141 / 148).

Two levers:
- **Intermediate entries** (48K, 80K): save 0.5 ms per 1K of overshoot, at most 8 ms per cycle (5%) just past a
  boundary, nothing at the top of an entry. An entry has to end on a KV segment boundary, and the 96K entry already
  binds 5 segments per attention layer (the binding count limit: about 51 per function).
- **Attention on the GPU over the actual length** (the ANE verify split around the 16 attention layers): with a kernel
  at the read floor (the GPU streams 148 GB/s here; 32 KB per position) plus about 6 ms of handoffs per verify
  (16 GPU round trips at 0.14 ms + the extra ANE calls), the projection was 5 to 23 ms per cycle saved from 16K.

The standalone test (`pipe/unit/decode_attn_bench.swift`, `decode_attn_flash.swift`, `decode_attn_flash2.swift`:
synthetic KV in the ANE cache layout, 16 layers, 48 query vectors per KV head padded to 64, checked against a CPU
reference to 0.02%) did not get there:

| kernel | 96K, 16 layers | per 1K keys |
|---|---|---|
| the engine's prefill attention core, one 64-row tile per head | 82.7 ms | 0.86 ms |
| flash decoding, online softmax per 64-key sub-tile | 100 ms | 1.04 ms |
| flash decoding, two passes per chunk (scores in a cache scratch) | 92.7 ms | 0.97 ms |
| flash decoding, scores and P in threadgroup memory | 89.4 ms | 0.93 ms |
| the ANE today (96K entry) | about 48 ms | 0.5 ms per 1K of entry |
| the read floor (3.2 GB at 148 GB/s) | about 21 ms | 0.22 ms |

All of them stay near 35 GB/s: latency, not bandwidth or compute. A threadgroup walks its keys strictly in sequence
(a small int8 QK matmul, softmax, rescale, a small PV matmul, barriers), the prefill kernels get their parallelism from
row tiles (decode has one: the value product ran on 8 threadgroups), and the 64 x 256 fp32 accumulator per threadgroup
probably limits occupancy (not measured). Getting to about 0.3 ms per 1K needs a tuned kernel (the head dimension
split over threadgroups, double-buffered K / V loads, 128 to 256 keys per matmul), then the split build and the engine
work; parked. Larger key tiles made the prefill kernels slower (more score traffic), and INT8 V was not faster than
the fp16 copy in any of them.

## Open levers

- **A8 in the server for 6K to 16K** (+6 to 10%). It needs the memory work to fit at 64K+.
- **Long context:**
  - DeltaNet q / k / v / z projections on the ANE (estimated about 520 tok/s at 64K);
  - attention with the scores kept on chip, or fp8 P (small probabilities may underflow);
  - finer ANE context rungs.
- **Decode at long context:** GPU attention for the verify is done (SPLIT+GPU, 17 Nov: 64K and 95K now tie splash 1.3.2).
  Left: the GPU's cold start after each ANE call (the kernel reaches 127 GB/s back to back, about 90 to 100 in the
  server), splash 1.3.2's higher acceptance per cycle, and serving 128K (no ANE recompile needed for the split).
- **V8 continuation at any position**, which saves one bridge block (0.2-0.45 s) on some follow-ups.
- **1000 tok/s** needs about 53 TFLOPS effective (53 GFLOP per token). That means the ANE's INT8 rate for the FFN
  and more of the GPU's work moved to the ANE, under the shared power budget.

## Files behind each step

| step | code | main results |
|---|---|---|
| V1 hybrid | `prototypes/hybrid_prefill.py`, `hybrid_long.py`, `pipe/engine/gpu_layers.swift` | 8K 486, 64K 326 |
| V7 W8A8 probe | `w8a8/build_probe.py`, `prototypes/w8a8_mlp_probe.py`, `capture_hidden.py` | 10 TOPS, rejected |
| V8 driver | `prototypes/v8_prefill.py`, `pipe/unit/run_wd.sh`, `e2e_ab.sh`, `pos_sweep.sh`, `short_sweep.sh` | see V8 |
| builds | `v8/build_seg2_96.sh`, `build_shared2_all.sh`, `build_shared_chunk.py`, `chunkmlp.py` | v8_shared2 |
| GPU tables | `pipe/prep_engine.py`, `prep_kv8.py`, `prep_ffn.py` | w_shared2r |
| coupling | `pipe/unit/contention.sh`, `gpu_steady.py`, `gpu_stress.swift`, `v8/share/ane_load.py` | power budget |
| A8 | `pipe/v8_a8.py`, `prep_a8*.py`, `w8a8/a8_*.py`, `a8probe/*.sh`, `a8probe/results*.json` | 6K +10.6%, 16K +7.6% |
| KV8 / K8V16 | `pipe/engine/patch_kv8b.py`, `patch_k8v16.py`, `pipe/unit/attn_kv8_prof.py`, `results/k8v16_65536.json` | lossless, K8V16 = fp16 speed |
| server | `pipe/v8_server.py`, `v8_engine.py`, `pipe/unit/srv_bench.py`, `srv_matrix.sh`, `results/srv_matrix.jsonl` | table above |
| short prefills, prompt cache | `pipe/engine/patch_fillfrom.py`, `pipe/unit/shortfix_check.py`, `srv_log_prefill.py` (log: `SSD/coreai_seg_logs/v8_server_8765_pre_shortfix.log`) | `pipe/V8_SERVER.md` |
| splash 1.3.2 | `splash132/serve.sh`, `bench.py`, `long_bench.sh`, `results/splash132_*.jsonl`; GPU only: `splash132/cmp_96k_gpu.sh`, `results/cmp_splash_gpu_96k.jsonl` | 2K 500, 96K 288; GPU only 319 / 217 |
| V8 vs splash 1.3.2, 96K context | `splash132/cmp_96k.sh`, `pipe/unit/cmp_bench.py`, `results/cmp_{v8,splash}_96k.jsonl` | table above |
| SPLIT+GPU decode | `v8/build_dec8_*`, `dec8_srv_manifest.py`, `pipe/dec_split.py`, `pipe/engine/dec_gpu.swift`, `dec_kernels.metal`, `pipe/unit/DECODE_GPU_ATTENTION.md`, `results/cmp_v8dec8_96k.jsonl` | 64K 141 ms, 95K 149 ms per cycle |
| Pi coding sessions | `splash132/serve_pi.sh`, `bench/cron/` (prompt, hidden grader), `bench/verbosity/` | both 1373 / 1373; prefill 22 s vs 133 s |
