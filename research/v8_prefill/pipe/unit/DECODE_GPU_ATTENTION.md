# GPU attention for the decode verify (16 Nov 2026)

Question: can the 8-row decode verify run its 16 attention layers on the GPU (INT8 KV, flash decoding) while the ANE
keeps everything else, and is that faster than the ANE verify with its own attention?

Short answer: the kernel is fast and accurate. With the CPU between every ANE and GPU call, the split only wins at the
96K entry (-2% to -8% per verify step) and loses below it. If ANE and GPU work were ordered without the CPU, it would
win -10% to -20% from 49K keys up. On macOS 27.3 the Core AI ComputeStream does not order mixed ANE / GPU work, so
that path is not available. Parked.

## Kernel (tests 1-2)

* `engine/dec_kernels.metal`: `dec_prep` (INT8 Q with a per-row scale; the 8 new K / V rows quantized the host's way
  and written into the ANE's own KV segment buffers; a window may cross a segment), `dec_attn` (Metal 4 tensor ops:
  int8 QK, half P x int8 V, 48 query rows per KV head split in 3 threadgroups of 16, 4 simdgroups, 256-key sub-tiles,
  2048-key chunks), `dec_combine_gate` (chunks merged, x sigmoid(gate), written channel-major into part B's input).
* Back to back: 0.0167 ms per 1K keys per layer (0.27 ms per 1K keys for 16 layers), the standalone rate.
* Error vs FP32: the kernel's own 0.09-0.2% (rel max). The per-row INT8 Q quantization adds up to ~1.8% on o with
  real q (outlier channels). Through the layer it is still 2-5x closer to exact attention than the ANE verify's own
  attention: y vs exact 0.06-0.14% (GPU path) vs 0.20-0.38% (ANE verify).

## Split entries (tests 3-4, probe `v8/build_dec_split_probe.py`, chunk L00-03)

* `dva` = part A: the chunk's DeltaNet layers + FFNs, then the attention layer's ln1, rmix, q / k / v projections,
  q / k norm and RoPE. Outputs xr, q, g [24][8][256], kt, vt [4][8][256], the states. No KV inputs.
* `dvb` = part B: rout, o projection, + xr, the FFN.
* dva -> exact attention -> dvb vs v8_16k: cosine 0.99998-0.99999. The new K / V rows match the verify's bit for bit.
* With the real GPU kernels on zero-copy ANE buffers (`unit/dec_gpu_check.py`), the GPU-written KV rows (codes and
  scales) are bit-exact vs `quantize_values`, including a window across the 8K segment boundary.

## Step timing (probe 2 `v8/build_dec_split_probe2.py`, `unit/dec_chain_time.py`, Swift bridge)

16-chunk verify step; ref = 16 x the chunk's real `v8_<entry>` in one bridge run (how the server runs it). Fused =
`dvba` (part B of chunk k + part A of chunk k+1 in one call: saves ~0.25 ms per chunk vs separate calls).

| keys | entry | ref ms | split unfused | split fused |
|---|---|---|---|---|
| 8K | 16K | 85.3 | +12.5% | +8.1% |
| 16K | 16K | 83.9 | +22.0% | +16.9% |
| 32K | 32K | 92.5 | +16.4% | +11.4% |
| 49K | 64K | 107.7 | +2.6% | -1.3% |
| 65K | 64K | 107.4 | +6.2% | +2.5% |
| 82K | 96K | 122.3 | -4.9% | -7.8% |
| 98K | 96K | 122.5 | -2.0% | -4.8% |

The ANE's own attention is cheap at short entries (~5 ms per step at the 16K entry). Each GPU round trip costs
~0.65 ms on top of the kernel:

* commit -> GPU start 0.28-0.34 ms (the GPU idles during each ~4 ms ANE call), GPU end -> CPU wake ~0.1 ms;
  encoding 6 us (`unit/dec_gpu_lat2.py`, `unit/gpu_wake_lat.swift`).
* Tried and rejected:
  * Spinning on `cb.status` / `MTLSharedEvent.signaledValue`: the chain at 96K goes from 7.5 to 11.6-13.6 ms per chunk.
  * GPU-side spin on a CPU-written flag: the GPU never sees the store (bounded loop times out).
  * A warm-up kick (empty command buffer ~0.3 ms ahead): helps GPU-only (0.25 -> 0.09 ms), but with the ANE running,
    the GPU waking mid-call makes the ANE call ~1.3 ms longer. A spinning CPU core does the same (+1.5-2 ms).
  * A residency set for the wrapped buffers: no change.

## Core AI ComputeStream (`unit/dec_stream/`, Swift, `unit/dec_stream_run.py`)

`InferenceFunction.encode(inputs:outputViews:to: ComputeStream(commandQueue:))` encodes ANE calls asynchronously, the
CPU waits once per step. Measured with the stream reference (16 x v8_<entry> encoded, 95 / 117 / 131 ms at the
16K / 64K / 96K entries; ~0.6 ms per call slower than the bridge):

* Unordered upper bound (raw GPU command buffers on the stream's queue): 92-103 ms, i.e. -10% to -21% from 49K up.
  The results are wrong: the GPU runs concurrently with the ANE calls encoded before it.
* With Core AI GPU copy functions as bridges (`v8/build_sync_bridge.py`, cpq / cpo) and the cross-device outputs
  unbound (returned futures passed on as inputs): still unordered (y differs run to run), 103-109 ms.
* Even an all-Core-AI chain (ANE dva0 -> GPU stub `att` -> ANE dvba ..., only futures) is unordered: stream runs differ
  from each other and from the sync run, while ANE-only chains on the same stream (futures or bound buffers) are
  correct and deterministic. So on macOS 27.3 a ComputeStream orders ANE work but not ANE <-> GPU work (ANE and GPU
  functions from two AIModel loads). `encode` returns an empty dictionary when every output is bound via outputViews.

## What would change the answer

* A Core AI runtime that orders ANE <-> GPU work on a ComputeStream (or exposes an event for an encode's completion):
  the split would then be worth ~-10% to -20% per verify step from ~49K keys, with the GPU path more accurate than
  the ANE's attention. Re-test with `unit/dec_stream_run.py` after OS updates (the ordering check is built in).
* Without that, only a long-context mode (96K entry, 65K-96K tokens) is positive, ~-5% to -8% of the verify.

## Where the CPU-synced split is worth it (17 Nov 2026)

The ANE verify costs a step function of its entry (16K 84.6, 32K 92.5, 64K 107.6, 96K 122.4 ms per 16 chunks); the
split costs ~92 ms + ~0.24 ms per 1K keys. So the split wins just past an entry boundary: 33K-~50K keys (64K entry,
up to -4%) and 65K-98K keys (96K entry, -10% at 66K to -5% at 96K). A hybrid keeps the ANE verify elsewhere; the
entries share the package weights (served chunk_L00-03 567 MB, the one-chunk probe with v8_16k + dva + dvb 565 MB).

The bench prompts (`unit/cmp_bench.py`) land just past the boundaries: 16,435 > 16K, 32,822 > 32K, 65,588 > 64K, so
"64K" runs on the 96K entry. Decode cycle at those points (V8 from `cmp_v8_96k.jsonl` / V8_SERVER.md, splash 1.3.2
from its log as output tokens per cycle / tok/s; V8 + split projected from the probe's step savings):

| prompt | V8 now ms | V8 + fused split ms | splash 1.3.2 ms |
|---|---|---|---|
| 2K | 108 | 108 (ANE verify) | 116 |
| 8K | 112 | 112 | 119 |
| 16K | 120 | 120 | 122 |
| 32K | 134 | ~130 | 131 |
| 64K | 153 | ~141 | 141 |
| 95K | 155 | ~149 | 148 |

The fused call (part B of chunk k + part A of chunk k+1) crosses today's 4-layer packages. Without a re-cut
(unfused split): ~134 / 145 / 152 ms at 32K / 64K / 95K; 8-layer packages (7 of 15 boundaries unfused): ~131 / 142.5 /
150; a cut at the attention layer keeps all of it but needs the ANE verify entries shifted too. Cold-prompt draft
acceptance is a separate gap: splash 1.3.2 4.4 / 2.8 / 2.8 / 2.6 / 3.2 / 3.0 output tokens per cycle vs V8 3.66 / 2.72 / 2.42
/ 2.56 / 2.78 / 2.81.

## Files

* Kernels / dylib: `engine/dec_kernels.metal`, `engine/dec_gpu.swift` (`libdec_gpu.dylib`).
* Probes: `v8/build_dec_split_probe.py` / `.sh` (one chunk), `v8/build_dec_split_probe2.py` / `.sh` (two chunks, fused
  entry), `v8/build_sync_bridge.py` / `.sh`; outputs in `/Volumes/SSD4TB/anemll-forge-research/v8_decsplit_probe{,2}`,
  `v8_sync_bridge{,2}`.
* Tests: `decode_attn_seg.swift`, `decode_step_gpu.swift` (GPU side), `dec_split_check.py`, `dec_gpu_check.py`,
  `dec_gpu_long.py`, `dec_gpu_lat2.py`, `gpu_wake_lat.swift`, `dec_chain_time.py`, `dec_stream/` + `dec_stream_run.py`.
* Logs: `/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/{dec_*,build_dec_*,build_sync_bridge*}.log`.

## The 8-layer build, served (17 Nov 2026)

The split ANE entries bind no KV and no mask, so one set serves every context. Built once: `v8/build_dec8_all.sh`
(8 packages of 8 layers, `v8/build_dec8_chunk.py`: each group's existing entries as `g0_` / `g1_`, the V8 FFN / mixer
entries, and `dva`, `dvba` (part B of group 0 + part A of group 1, one ANE call), `dvb`; 55 entries, 1186 / 1406 MB
per package = two served chunks, so no weight duplication; export ~5-6 min per package, compile ~2 min each, run
alongside). Then `v8/dec8_srv_manifest.py` (16 logical chunks over the 8 packages) and `pipe/prep_kv8.py` for the
GPU tables (`w_dec8/manifest_best.json`). Build: `/Volumes/SSD4TB/anemll-forge-research/v8_dec8`, 9.9 GB.

Server: `pipe/v8_server_dec8.sh` (`--v8-dec8 1`, `pipe/dec_split.py`: loads each package once, the split where
`V8_DEC8=auto` says, the 96K entry only by default; `V8_DEC8=off` turns it off).

Checks:
* `unit/dec8_check.py` (package L00-07): the new g0 / g1 verify entries give output identical to the served v8_shared2
  packages; split chain vs reference cosine 0.99997.
* `unit/v8pf_check.py`: V8 prefill and ANE prefill logits on the new build identical to the served build's (2K, 6K).
* `unit/dec8_real_check.py` (real docs text, V8 prefill to each P, teacher-forced blocks): KL(ANE verify || split) 0.025
  at 66K, 0.0005 at 82K, 0.054 at 95K; next-token NLL ANE / split 2.630 / 2.593, 0.0019 / 0.0020, 0.049 / 0.025. The
  ANE reading the GPU-written KV rows agrees as well. (On random INT8 history `unit/dec8_srv_check.py` gives logits
  cosine 0.97-0.997 with top-1 8/8: near-uniform attention, the worst case for comparing two approximate attentions.)
* `unit/dec_gpu_128k.py`: the kernel at 100K-131K keys (5 segments, the last 64K): new K / V rows bit-exact with the
  host quantize (after `precise::divide` in dec_prep: fast-math division put 1-3 of 16,384 codes one off at rounding
  ties), o 0.28-0.56% of max vs exact FP32, 0.0170 ms per 1K keys per layer.

Verify step in the runtime (16 chunks + head, `unit/dec8_srv_check.py`, `unit/dec8_long_time.py`):

| keys | ANE verify | split |
|---|---|---|
| 33K (64K entry) | 119.0 | 118.1 |
| 49K (64K entry) | 119.5 | 121.6 |
| 66K (96K entry) | 134.4 | 124.5 |
| 82K | 134.0 | 127.9 |
| 95K | 133.8 | 130.2 |
| 110K | (no entry) | 133.9 |
| 131K | (no entry; ~149 extrapolated) | 139.0 |

Server benchmark (`unit/cmp_bench.py`, same as the V8 / splash runs; `coreai_seg_logs/cmp_v8dec8_96k.jsonl`,
`v8_server_8765_dec8_bench.log`), decode cycle ms (V8 before / dec8 / splash 1.3.2): 2K 108 / 107 / 116, 8K 112 / 111 / 119,
16K 120 / 119 / 122, 32K 134 / 133 / 131, 64K (65.6K tokens, 96K entry) 153 / 141 / 141, 95K 155 / 149 / 148. Prefill
576 / 668 / 631 / 548 / 429 / 358 tok/s. Cold decode at 95K 20.4 tok/s = splash 1.3.2.

Bandwidth (`unit/bw/`): GPU streaming read alone 156 GB/s; the split's ANE calls on one package alone 121 GB/s; both at
once 87 + 77 = 164 GB/s (of 170). The attention kernel reaches 126 GB/s back to back and ~90-100 GB/s in the server
(the GPU starts cold after each ~5 ms ANE call). Next levers: the kernel (each K / V tile read once for all 48 query
rows, chunk size) and the cold start, ~10 ms per step at 95K; ANE + GPU attention by key range at most ~5-8 ms and
needs per-segment ANE attention programs. 128K: no ANE recompile; needs KV allocation past 96K, the V8 prefill past
96K, and the server's context limit.

## ANE + GPU attention at the same time, by key range (17 Nov 2026): rejected

`v8/build_dec_att.py` (weight-free package `v8_decatt/dec_att.aimodel`, 0.44 MB: `att8k` / `att16k` / `att32k` /
`att64k` = the verify's own 8-bit history attention over the first 1-4 KV segments; output o, row max m, sum den), the
GPU the rest (`dec_submit_part`, `dec_combine_part`: the kernel now starts at the first registered segment), a CPU
merge by log-sum-exp. Probe `unit/dec_att_probe.py` (a real dva call before each, so the GPU starts cold as in the
server), per layer:

| keys | GPU full | best split (att32k) | per step |
|---|---|---|---|
| 66K | 1.63 ms | 1.56 | -1.0 ms |
| 82K | 1.86 | 1.78 | -1.3 ms |
| 95K | 2.09 | 2.01 | -1.3 ms |
| 131K | 2.65 | 2.57 | -1.2 ms |

The ANE reads 32K keys in 1.5 ms (~45 GB/s next to the GPU; 8K still 0.5 ms); the GPU's part takes ~1.94 ms from submit
whether it reads 63K or 87K keys (cold start + shared bandwidth). Accuracy: the GPU partial matches exact (lse within
0.005); the ANE partial carries the verify's 8-bit limits: true scores clip at 31.75 (s8b 0.25 x 127) and its o is
13-20% off on random data (64-88% with 4x larger queries). Not worth it; the GPU kernel is the lever.

## FP8 softmax on the GPU (17 Nov 2026): no gain

The GPU kernel computes scores INT8 x INT8 (INT32 accumulate), the softmax in FP32, PV with FP16 probabilities x INT8
V (FP32 accumulate): no FP8. The M6 matmul2d has no FP8 x INT8 form (only FP8 x FP8, FP16 x FP8). `DEC_SM` compile
switch in dec_kernels.metal (env DEC_SM of libdec_gpu; `unit/dec_gpu_128k.py`, DEC_DYLIB=libdec_gpu_t.dylib), one layer:
FP32 softmax 1.682 / 1.991 / 2.173 ms at 100K / 120K / 131K keys (error 0.28-0.56%); no exp at all (timing bound)
1.685 / 1.985 / 2.168; FP8 e4m3 probabilities 1.695 / 2.006 / 2.188 (error 0.52-1.07%). The softmax costs nothing
measurable as math (no exp = same time).

Corrected (17 Nov, `unit/dec_gpu_steady.py`: one command buffer of 50 copies, output checksum per variant; an earlier
batch had silently run the unmodified kernel because zsh does not word-split `env $cfg`), one layer at 120K keys:
full 2.00 ms (246 MB unique K / V, 123 GB/s); both matmuls without the softmax data path (no score store, no softmax,
no P write-back, fixed P) 1.84 (-8%: the S / P round trips through threadgroup memory); no exp 2.00; FP8-rounded P
2.00; K side only 1.18 (~104 GB/s), V side only 1.19 (~103 GB/s); one query group of three 1.88 (-6%). So ~8% is the
materialized softmax (P kept in registers can feed the PV matmul: cooperative f16 x device int8 is supported) and
~92% is the matmul ops' own K / V loads at ~104-123 GB/s vs 156 GB/s for a plain read (memory-level parallelism per
threadgroup, not DRAM bandwidth): next, P in registers and prefetching the next K / V tile.

## Softmax in registers, more loads in flight (17 Nov 2026)

Research kernels in `engine/dec_kernels_r.metal` (the served file unchanged), checksummed, steady clock, one layer:
* MPP limits on this SDK: `reduce_rows` and cooperative (register) INPUT tensors need a single-simdgroup matmul. A
  per-simdgroup flash-attention (`dec_attn_sg`, softmax fully in registers, P as the PV's register input, 4
  independent K / V streams per threadgroup) is correct but 2.7-4.4x slower (5.3 ms at SUBR 64 vs 2.0 at 120K keys):
  single-simdgroup matmuls are slow here.
* Threadgroup shapes (`DEC_MCFG`): 16 rows x 4 simdgroups (served) 2.00 ms; 16 x 8 2.11; 16 x 2 3.25; 8 rows x 6
  threadgroups 3.53; 48 rows in one threadgroup 5.57. The served shape is the best.
* `dec_attn_r4`: the served shape with the scores kept in the QK register tensor (probed layout: a lane holds rows r
  and r + 8, r = ((lane >> 1) & 3) + 4 (lane >> 4), shared by lanes ^ 1, ^ 8, ^ 9), row max / sum by simd shuffles
  plus a 4 x 16 float exchange, P written once to threadgroup memory (the 4-simdgroup PV needs a tensor operand):
  identical output, 1.107 -> 1.071 ms at 66K, 1.597 -> 1.543 at 95K, 1.994 -> 1.927 at 120K, 2.175 -> 2.106 at 131K
  (-3.2% to -3.4%; the bound for removing the whole softmax data path was -8%).

## Prefetch (17 Nov 2026): slower in every form

`engine/dec_kernels_r.metal`, checksummed, steady clock, one layer at 120K keys (95K the same pattern):
* Cache-warming touch of the next tile's lines (`dec_attn_pf`, plain loads consumed one tile later): K and V +25%
  (2.42 vs 1.93 ms), 2 tiles ahead +27%, K only +11%, V only +11%, compiled in but off 0%. Hundreds of threadgroups
  warming 128 KB each: the lines are evicted before the matmuls read them.
* Double-buffered K / V through threadgroup memory (`dec_attn_db`: next 32-key tile in 8 x uint4 registers per
  thread, matmuls on threadgroup operands; 32 keys is the most that fits the M6's 32 KB): 3.59 ms (+86%). The same
  kernel with direct device loads at 32-key tiles: 2.61, so the small tile costs +30% and the staging +37%.
* Tile size, direct loads: 32 keys 2.61, 64 2.41, 128 2.10, 256 2.00 (served), 512 2.36. The served shape and the MPP
  matmuls' own device loads are the best found; the kernel stays at ~127 GB/s (r4) vs 156 for a plain read.
* Prefetch placement (`dec_attn_pf` PF_AT / PF_CUR), 120K keys, r4 1.93 ms: next tile at the tile start 2.38 (+24%),
  after QK 2.25 (+17%), K after QK and V after PV 2.27 (+18%); control touching the CURRENT tile's lines (no extra
  DRAM traffic) at the tile start 2.34 (+22%), staggered 2.08 (+8%). Later placement helps, but the extra plain loads
  cost even without extra traffic: they slow the matmuls' own loads (and hold registers). No placement wins.
