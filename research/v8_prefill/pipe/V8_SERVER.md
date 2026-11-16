# V8 prefill server (16 Nov 2026)

OpenAI-compatible Qwen3.8-27B server on the M6 with the V8 GPU + ANE prefill. It is the anemll-forge-v02 release
server (`scripts/qwen38_server.py`: Core AI target, DFlash2 all3 drafter, prompt-cache snapshots) with only
`Engine.prefill` replaced. Decoding is the release path, unchanged.

## Run

```
v8_work/pipe/v8_server.sh [v8_server.py options]      # port 8767 (PORT=...), ctx 98304 (CTX=...)
v8_work/pipe/srv_stop.sh                              # stops whatever listens on $PORT (default 8767)
```

- Build: `/Volumes/SSD4TB/anemll-forge-research/v8_shared2` (shared-weight build: verify + prefill + ffn / ffn2 /
  ffnq2 / offn2 entries), 96K context ladder, KV8 (INT8 K and V, dim_token keys).
- GPU engine: `engine/libgpu_v8_srv2.dylib`, kernels compiled at load from `engine/kernels.metal`, GPU weights
  `w_shared2r/manifest_best.json`. GPU switches in the script env: FUSE=all I8=1 I8T=1 I8TV=_k256 LUTC=1 I8A=1
  I8KV=1 LRM=1 QK8=1 TKV=2048.
- Log `/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log`, swap watchdog +1536 MB, the load is
  journaled (deploy_journal).
- `v8_server_plain.sh`: the same with `--plain` (no drafter; deterministic greedy, used for equality checks).

Options (defaults are the best measured):

| option | default | meaning |
|---|---|---|
| `--v8-min` | 256 | V8 for new spans of at least this many tokens (the bridge alone does ~145 tok/s at 64K+) |
| `--v8-kv8` | 1 | GPU attention reads and writes the ANE's INT8 KV cache (no KV copy, warm continuation) |
| `--v8-k8v16` | 1 | with KV8: the GPU also keeps an fp16 copy of V for its PV (+32 KB/token, 3.2 GB at 96K) |
| `--v8-attproj-from` | off | attention projections on the ANE from this position (no clear gain, see below) |
| `V8_ALIGN` env | 64 | continuation start alignment (64 or 128) |
| `V8_TAIL_MIN` env | 192 | a shorter rest after a snapshot goes to the bridge |

## How a request runs

1. Cold prompt: V8 from 0 to the end in one run, the last 128-row pair padded (`v8_submit_pair_nv`: DeltaNet state
   and conv history stop at the last real token), the logits from V8's own head.
2. Snapshots (system, turn) are taken during the run at the pair end at or below each cut: each DeltaNet layer's
   state is copied when its pair ending there completes, the next pair of that layer is gated until then.
3. The drafter's context gets the tap features V8 captures (layers 5, 19, 33, 47, 61), windowed.
4. Continuation (a restored snapshot or the state after decoding): the bridge feeds prefill blocks up to the next
   64- or 128-aligned point (fewest blocks + pairs; a 64 start only when the span crosses no KV segment boundary),
   the DeltaNet states go to the GPU (`v8_state_in`, on the GPU), with K8V16 the GPU's V copy is rebuilt from the
   ANE's INT8 V (`v8_kv8_v16_fill`), then V8 runs to the end.
5. Any V8 failure falls back to the bridge from the last consistent position.

## Results (M6, macOS 27.3, speculative decode on)

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

Decode: unchanged by V8 (ms/cycle about 110 at 6K, 128 at 32K, 152 at 96K; 19 to 35 tok/s depending on draft
acceptance). A request past the context returns HTTP 400 and the server keeps running.

## KV8 findings

- KV8 is lossless for quality: per-block KL at 64K 0.0592 (fp16 KV 0.0593), 97K 0.2267 vs 0.2274.
- The KV8 attention core was 12% slower per call at 64K (PV half x INT8 +40%, softmax with per-key scales +22%);
  that is the 4 to 6% end-to-end loss at 64K / 96K. On-chip V dequant (KV8D) and a 256-wide PV (PVWW) did not help.
- K8V16 (keys INT8 from the ANE cache with the key scales folded into the INT8 score, values fp16 on the GPU):
  64K attention call 17.65 ms vs KV8 19.83 vs fp16 17.71; real end to end 64K 419.9 tok/s vs KV8 401.7, KL mean
  0.0598 (KV8 0.0592). The ANE cache still gets INT8 K and V for decode.

## Files

- `v8_server.py`, `v8_server.sh`, `v8_server_plain.sh`, `srv_stop.sh`
- `v8_engine.py` (V8Prefill: run, snapshots, handover, sync, K8V16 fill, can_start)
- `engine/gpu_v8.swift`, `engine/kernels.metal`, patches `patch_nv.py`, `patch_srv.py`, `patch_state_gpu.py`,
  `patch_k8v16.py` (each keeps `*.pre_<name>.*` backups)
- `unit/srv_bench.py` (cold + follow-up turns), `unit/srv_matrix.sh` (config x length matrix), `unit/srv_summ.py`,
  `unit/attn_kv8_prof.py` (attention core per call), `unit/state_check.py`
- results: `coreai_seg_logs/srv_matrix.jsonl`, `coreai_seg_logs/k8v16_65536.{log,json}`

## Open items

- A8 (runtime INT8 FFN on the ANE) is not in the server: +6% at 16K in the driver, slower at 64K+.
- Decode at 96K is verify-bound on the ANE (135 ms of 152 ms per cycle); GPU attention for the verify would need a
  new ANE build split around the attention layers.
- A bridge prefill block costs 0.2 to 0.45 s whatever its rows (one is needed after decoding: pending rows);
  V8 at any 64-unaligned position would save one block on some follow-ups (tile n rounded up, masked).
- Short V8 spans are latency-bound (a pair through 64 layers): about 0.25 s per pair at 6K, 0.36 s at 32K, 0.55 to
  0.64 s at 96K.
