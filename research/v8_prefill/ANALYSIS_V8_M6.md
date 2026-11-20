# Where V8 loses time on the M6: prefill and decode, measured with the new counters (18 Nov 2026)

*ANEMLL Forge research notes. The served V8 build (dec8: every FFN on the Neural Engine, every mixer on the GPU in
prefill; speculative decode with the DFlash2 drafter and the split verify) measured with IOReport and `powermetrics`
telemetry ([pipe/unit/power/IOREPORT_COUNTERS.md](pipe/unit/power/IOREPORT_COUNTERS.md)), a per-call trace of the
prefill pipeline, and isolated tests of each candidate lever. Raw logs: [results/18nov/](results/18nov/README.md).
Background: [ARTICLE_CONTENTION_M6.md](ARTICLE_CONTENTION_M6.md) (why the units slow each other: DRAM).*

## Summary

- **Prefill at 2K is set by the Neural Engine, from 32K by the GPU's attention.** The traced pipeline: the Neural
  Engine is 99% busy at 2K, both units 95% at 8K, the GPU 99-100% from 32K, where attention takes 49-60% of its time.
- **Everything runs contended.** From 8K up DRAM carries 77-80% of its peak and the GPU's memory latency is 2-3 times
  its alone value; the Neural Engine's FFN pair costs 19% more and DeltaNet 20% more than alone.
- **Attention's DRAM traffic is its own score matrix,** about 90% of 1.4 GB per call at 60K; a kernel that keeps the
  scores on chip and reads the keys and values once is the largest lever, and the larger the longer the prompt (a
  time model fitted to the trace: 64K about 420 to 515-585 tokens per second, 95K about 350 to 445-525). The engine's
  existing flash forms do not do that and lose.
- **Decode is bound by the Neural Engine's weight stream (more than 128 GB/s while it streams, about 100-105 on
  average over a cycle; the two clusters at 52.8 and 45.8), with the GPU idle;** a GPU drafter and a balanced verify are worth about 5-8% and up
  to 7%. Decode does not heat-throttle (7 W).
- **Data types:** the Neural Engine's FFN in INT8 x INT8 (WL3A8, the existing 3-bit indices) is 13-16% faster per FFN
  but more bandwidth-hungry; the GPU should keep its 4-bit weight bytes, which win 1.25-1.3x under contention.

## What the served build does today

| phase | GPU | Neural Engine | DRAM | limits seen |
|---|---|---|---|---|
| prefill 2K | 87% active | FFN, 12.5-13 W | 114 GB/s (74% of 155) | |
| prefill 8K-16K | 97-100% active, 1734 MHz | 11.7-13.4 W | 123-124 GB/s (79-80%) | GPU memory latency 330-342 cycles (100-220 alone) |
| prefill 32K-64K | 99-100% active; 1618-1712 MHz after about 50 s (`PMULOOP`) | falls to 5-7 W as the pipeline slows | 108-124 GB/s | the GPU's attention grows with the context |
| decode 2K-32K | idle (1-4% active) | everything, 6.6-6.8 W | ANE 98-104 GB/s on average (58% of samples at the 128 GB/s top bin) | the Neural Engine's weight stream, at 128+ GB/s while it streams |
| decode 64K | 24% active (split-verify attention, 28 ms of a 134 ms step) | 5.3 W | ANE 75 + GPU about 16 | |

Measured on cold requests to the served server (`results/18nov/coupling/v8bench/`, `coupling/prefill1/`).

## Prefill

A trace of the served pipeline ([`power/v8_trace.py`](pipe/unit/power/v8_trace.py): a copy of the server's V8 engine
that records every GPU job's GPU start and end and every Neural Engine call; same build, same environment, same
prompt text; `results/18nov/coupling/trace1/`):

| prompt | rate | GPU busy | ANE busy | both / GPU only / ANE only | GPU ms per 128-row call: DeltaNet; attention, first to last tenth | sets the pace |
|---|---|---|---|---|---|---|
| 2K | 687-696 tok/s | 89% | 99% | 88 / 0.6 / 11% | 2.58; 2.4 to 3.0 | the Neural Engine |
| 8K | 684 tok/s | 96% | 95% | 92 / 4.7 / 3.4% | 2.58; 2.5 to 4.8 | both |
| 32K | 543 tok/s | 99% | 79% | 78 / 21 / 0.9% | 2.57; 3.0 to 10.7 | the GPU (attention) |
| 64K | 420 tok/s | 99.5% | 64% | 63 / 36 / 0.5% | 2.54; 3.5 to 18.8 | the GPU (attention: 93 of its 155 s) |

- **2K is Neural-Engine-bound.** The Neural Engine is busy 99% of the time and the GPU waits for it 11% of the time.
  Its FFN pair takes 2.88 ms of hardware time per 128 rows, against 2.42 ms alone and cool (+19%, the contention);
  64 layers x 2.88 ms per 128 tokens gives the measured 695 tokens per second exactly. A faster FFN is a faster 2K.
- **8K is balanced:** both units are 95-96% busy.
- **From 32K the GPU sets the pace, on attention.** Attention per 128-row call grows linearly with the position (64K:
  3.5 ms for the first tenth of the prompt, 18.8 for the last) and takes 60% of the GPU's time at 64K.
- **DeltaNet costs 2.54-2.59 ms per 128-row call at every length** (48 of 64 layers). The same kernel alone takes
  1.94 ms looping one layer and 2.12 ms cycling all 48 layers (cold caches, as in the pipeline): about a third of the
  gap is cache coldness, two thirds (+20%) the contention with the Neural Engine's FFN stream.

### Attention at long context: where its bytes go

At 60K a 128-row attention call on the served engine (unfused: Q.K scores to DRAM, softmax in place, P.V) takes 16.8 ms
alone and moves about 1.4 GB, while its keys and values are 0.19-0.25 GB (served: INT8 K and an FP16 V copy; the
standalone test: FP16 K and V). The rest is the
score matrix: 3,072 rows x 2,048 keys x 2 bytes written, read and written by the softmax, read by P.V, about 50 MB per
2,048-key tile. Next to the FFN the call takes 24.1 ms (+43%).

The engine's opt-in flash forms keep the scores on chip but read the keys and values again for every query-row tile
(`results/18nov/coupling/attn_*`, each against the FFN):

| attention form at 60K | alone | DRAM per call | next to the FFN | the FFN next to it |
|---|---|---|---|---|
| unfused (served) | 16.8 ms | 1.4 GB | 24.1 ms (+43%) | +16.7% |
| `FA2=16_32_mm` | 24.8 ms | 2.0 GB | 48.0 ms (+92%) | +4.6% |
| `FLASH=1` | 30.8 ms | 3.0 GB | 44.9 ms (+46%) | +2.2% |
| `FA2=16_32` | 35.5 ms | 2.3 GB | 40.9 ms (+15%) | +3.0% |
| `FA2=s_16_16_8_mm` | 53.0 ms | 1.7 GB | 54.3 ms (+2.4%) | +2.3% |

The unfused kernel stays the fastest under contention. What would help is a kernel that does both: scores on chip and
the keys and values read once for all 768 query rows that share a key/value head (6 heads x 128 rows), about 0.25 GB
per call. Its compute floor at 60K is about 10 ms (193 GFLOP: Q.K in INT8, P.V in FP16), against 24 ms today under
contention; attention is 60% of the GPU's time in a 64K prefill. Cheaper steps on the way: 8-bit scores and
probabilities (half the score traffic) and the softmax folded into P.V (one of the four passes). A per-kernel profile
at 60K (`gdn_prof.py` with `PROFILE=1`, each dispatch in its own command buffer, so shares only; the standalone FP16
K/V path) gives the three score-matrix passes 86% of attention's time: P.V 39%, Q.K 31%, softmax 15.7%, combine 3.3%,
the projections about 9%.

### A time model of the prefill, and 96K

The trace fits a two-unit model per 128 prompt rows ([`power/prefill_model.py`](pipe/unit/power/prefill_model.py),
`results/18nov/coupling/trace1/prefill_model.txt`). On the GPU, 48 DeltaNet calls take 2.57 ms each at every length,
and 16 attention calls take 2.95 ms + 0.255 ms per 1K keys of position. The attention term is fitted on the 32K and
64K deciles, with a largest residual of 0.23 ms. The Neural Engine runs 64 FFN pairs at 2.88 ms. The slower unit sets
the rate. The model reproduces the traced 8K / 32K / 64K rates within 1% (683 / 539 / 420). For 95K, which it was not
fitted on, it predicts 350, against the 347-358 measured on 16-17 Nov (`cmp_bench.py`, 95K prompts, 96K context entry).

| prompt | today (model) | attention, share of GPU time | Neural Engine busy | 8-bit S/P + softmax fold (score part -25%) | new kernel, 75% of peak | new kernel at peak | new kernel + INT8 P.V |
|---|---|---|---|---|---|---|---|
| 8K | 683 | 34% | 98% | 694 (+2%) | 694 (+2%) | 694 | 694 |
| 32K | 539 | 48% | 78% | 580 (+8%) | 623 (+16%) | 658 (+22%) | 672 (+25%) |
| 64K | 420 | 59% | 61% | 472 (+12%) | 514 (+22%) | 563 (+34%) | 585 (+39%) |
| 95K | 350 | 66% | 50% | 404 (+15%) | 443 (+26%) | 498 (+42%) | 523 (+49%) |

How the new kernel is modelled:

- **Fixed part:** 2.4 ms per call for the projections, gate and norms (an attention call at the start of a prompt).
- **Score part, at its compute floor:** 193 GFLOP per 128-row call at 60K, with Q.K in INT8 (39.5 TOPS) and P.V in
  FP16 (19.2 TOPS), takes 7.5 ms at peak.
- **INT8 P.V** needs 8-bit probabilities and an INT8 V (the served cache keeps an FP16 V copy for P.V).
- **DeltaNet stays at 2.57 ms.** Its cost does not change while attention's traffic triples over a 64K prompt, so less
  attention traffic is not expected to speed it.

What the model says about 96K:

- **The attention lever is largest at 96K:** attention is 66% of the GPU's time at a 95K prompt, so the gain is +26 to
  49%. The rate then still falls with length, but more slowly.
- **The Neural Engine levers do nothing for 96K prefill.** WL3A8 and cluster balance don't help, because the Neural
  Engine is idle half the time. Moving mixer work onto it was tried on 16 Nov: 16 DeltaNet mixers on the Neural
  Engine (M16) gained +1.9% at 96K.
- **GPU heat:** a 95K prefill takes about 270 s. The `PMULOOP` clock drop starts at about 50 s at 64K and is inside the
  fitted slope, which the 95K check covers.
- **Not traced at 96K:** the 96K rows are model estimates.
- **Decode at 96K was not measured with the new counters today.** That cycle is 149 ms with SPLIT+GPU (17 Nov), and its
  long-context part is the split verify's GPU attention.

An earlier version of this doc estimated 64K at 420 -> 600 and 32K at 543 -> 680, above this model (which keeps the
projections' fixed cost and puts the score part at its compute floor); corrected.

### What the counters add

- **The pipeline runs in the contended state the whole time.** From 8K up the GPU is 97-100% busy, total DRAM is at
  77-80% of the peak and the GPU's memory latency is 2-3 times its alone value. Every byte the GPU or the Neural
  Engine reads per token slows the other.
- **The Neural Engine's DRAM traffic tracks the prefill rate:** its FFN weights are read once per 64-row block, about
  100 MB per token for 64 layers, so ANE GB/s divided by 0.1 GB is tokens per second (544 vs the server's 537 at 32K).
- **The GPU's bytes per token triple over a 64K prompt** (about 78 to 242 MB), all in attention: the unfused
  attention writes its score matrix to DRAM and reads it back twice (about 50 MB per 2048-key tile against 6 MB of
  keys and values).
- **Heat:** the GPU's clock falls 3 to 7% in the second half of 32K-64K prefills (`PMULOOP`); the Neural Engine's heat
  throttle is hidden behind the GPU-bound pipeline.
- **The Neural Engine's two clusters:** the second carries 9% less traffic in every prefill, built into the FFN's
  8192 + 9216 split (the 1024-wide mid Hadamard leaves 17 blocks, split 8 + 9).

## Decode

Cold requests to the server with 256 decode tokens each (`results/18nov/coupling/v8bench/`, the server log's
per-request cycle breakdown, telemetry over each decode window; the GPU's DRAM taken from the DRAM total because its
own counter reads the rate while active):

| context | decode | ms / cycle (draft, verify) | tokens / verify | ANE DRAM | GPU DRAM | ANE W | ANE cluster balance |
|---|---|---|---|---|---|---|---|
| 2K | 31.4-35.2 tok/s | 109 (13.3, 94.5) | 3.4-3.8 | 104 GB/s | 0 | 6.8 | 0.868 |
| 8K | 28.8 | 113 (13.1, 98.7) | 3.2 | 103 | 0 | 6.8 | 0.867 |
| 16K | 32.0 | | | 101 | 0 | 6.8 | 0.876 |
| 32K | 23.0 | | | 98 | 0 | 6.6 | 0.883 |
| 64K | 22.4 | verify 134 (GPU attention 28) | | 75 | about 16 | 5.3 | 0.881 |

A long decode (2,500 tokens from a short prompt, `results/18nov/coupling/longdec/`) holds the same state for 83 s: 30.2
tokens per second on average (24-35 per 10 s, following the drafter's acceptance), 109 ms per cycle (draft 13.5,
verify 94.5), ANE 6.88-6.95 W, ANE DRAM 103.4-103.9 GB/s, cluster balance 0.867-0.868, temperature 74 to 88 C, no
throttling: at 7 W the Neural Engine does not heat up the way it does in prefill (13-18 W).

- **Bound by the Neural Engine's weight stream.** A cycle (109-113 ms at 2K-8K: draft 13, verify 95-99) moves at least
  10.6 GB (the drafter's 0.7 GB and the target's 9.9 GB of weights): at least 97 GB/s on average, and the counter reads
  98-104 with 58% of its samples in the 128 GB/s top bin, so while the Neural Engine streams it runs faster than 128
  GB/s; the average is held down by the cycle's other parts (the draft, sampling, the gaps between the verify's
  calls). The GPU is idle; the Neural Engine draws only 6.8 W: it waits for bytes, not for compute.
- **The second cluster carries 12-13% less traffic in decode** (52.8 vs 45.8 GB/s, the same in every decode window).
  If the two clusters stream their shares in parallel, the busier one sets the verify's streaming time; an even split
  would cut it by up to about 7%. Prefill's imbalance (0.91) matches the FFN's 8 + 9 block split; decode's is larger, so the verify's other
  entries (the mixers run on the Neural Engine in decode) add to it.
- At 64K the split verify moves attention to the GPU (28 ms of a 134 ms step).

## Levers, ranked by measured potential

Estimates are marked as such; each comes from the measurements above.

| # | lever | where | measured basis | estimated gain | cost |
|---|---|---|---|---|---|
| 1 | **Attention kernel with on-chip scores and keys / values read once per key/value head** | prefill 32K+ (GPU-bound) | 1.4 GB per call at 60K, about 90% score traffic; existing flash forms re-read K / V (1.7-3.0 GB); compute floor about 10 ms vs 24 ms under contention; attention is 49% of the GPU's time at 32K, 60% at 64K | time model below: 32K 539 -> 623-672 tok/s; 64K 420 -> 514-585; 95K 350 -> 443-523 (the high end needs INT8 P.V) | a new kernel |
| 1a | 8-bit scores and probabilities; softmax folded into P.V | prefill 32K+ | score traffic is 4 passes x 2 bytes per score; the three score passes are 86% of attention's time at 60K (softmax alone 16%) | if the score part gets 25% cheaper: +8% at 32K, +12% at 64K, +15% at 95K (time model). **Fold measured 18 Nov ([results/18nov/fold/](results/18nov/fold/README.md)): exact, but 3-5% slower alone; next to the FFN 2-4% faster with the FFN slowed half as much; in the V8 prefill -1% to +1% at 2K-95K (a wash: DeltaNet -4 to -6% late in long prompts, attention +2% there)** | the folded P.V must match the plain P.V's speed (wide and online forms drafted) |
| 1b | **FP8 P.V** (e4m3 probabilities x an FP8 V copy; Q.K INT8 and fp16 scores unchanged) | prefill 32K+ | measured 18 Nov ([results/18nov/f8/](results/18nov/f8/README.md)): P.V 2x faster (47 -> 23 ms per 8 calls at 60K), attention per call -12 to -15% alone, -11 to -13% next to the FFN | **measured: 32K +4.5%, 64K +8.2%, 95K +9.4%** (interleaved repeats), 2K-8K within noise | real-text teacher-forced KL first (last-token KL 5e-6 to 4.6e-3); the V copy halves (1.6 GB at 96K) |
| 2 | **FFN as WL3A8 on the Neural Engine** (the 3-bit indices dequantized to INT8 codes, INT8 activations; no second copy of the weights) | prefill 2K-8K (Neural-Engine-bound at 2K) | FFN pair 2.88 ms at 2K under contention; WL3A8 at 64 rows: -16% alone, -13% next to a 4-bit GPU stream (5.16 vs 5.91 ms per chained call) | 2K: about 695 -> 775 tok/s (the GPU's own 2.64 s per 2K then sets the pace); 8K: smaller (the pipeline is balanced, and WL3A8 reads more bytes per second) | real-text KL and an end-to-end A/B (the 512-row form checks out: 1.2% after 4 passes) |
| 3 | **Drafter on the GPU** | decode | the drafter takes 13.5 of 109 ms per cycle on the Neural Engine (0.7 GB of weights); the GPU is idle in decode and streams up to 155 GB/s alone | about -5 to -9 ms per cycle (+5 to 8%; 0.7 GB at up to 150 GB/s is 4.6 ms, plus the GPU's start after idle) | port the drafter's GEMVs to the GPU engine |
| 3b | Verify layers on the GPU (bandwidth) | decode | at 16-32 rows the GPU streams weights at 134-145 GB/s in any format (`gpu_wa_gemm`); the Neural Engine streams at 128+ GB/s while busy (its counter clips there), so the margin is small; splash's GPU-only decode (33.7 vs our 31.8 tok/s, first article) agrees | small; first measure the Neural Engine's streaming rate per verify call | a GPU decode path for whole layers, or more layers on the GPU in the split verify (the FFNs next to the 16 attention layers: about -4 ms per verify, no extra hand-over) |
| 4 | **Balance the Neural Engine's clusters in the verify** | decode | 52.8 vs 45.8 GB/s in every decode window; the busier cluster sets the streaming time | up to about 7% | find which verify entries load one cluster: the HWX parser does not show it (the served chunk's 110 streams come in pairs with about 2:1 task counts; the tasks' `ENE` field takes values 0-5, not a cluster index) |
| 5 | Larger Neural Engine blocks (512 rows) | prefill | weights read about twice per 512 rows instead of once per 64: about 3.5-4x fewer bytes per token; WL3A8 at 512 rows 14.7 us per token and only 0.3-4% contention | FFN time per token -25%; DeltaNet's +20% contention mostly gone | a new pipeline schedule (4 GPU pair calls per Neural Engine call) |
| 6 | Keep the GPU's 4-bit weight bytes | prefill | W8A8 is 17% faster alone but 1.25-1.3x slower next to the Neural Engine than W4A8 | (protects today's speed) | none: the served GPU projections are WL4A8 |
| 7 | Heat in long prefills | prefill 32K+ | GPU clock -3 to -7% after about 50 s at 95 C (`PMULOOP`) | a few % | cooling, or fewer bytes (levers 1, 5) |

Not levers: FP8 on the GPU (the same rate as INT8), FP4 x FP4 (no faster than FP16), FP16 activations with 4- or
8-bit weights (the FP16 rate either way), an even FFN chunk split (impossible: the 1024-wide mid Hadamard leaves 17
blocks), the existing flash attention forms (slower than the unfused kernel even under contention).

## Isolated tests run for this analysis

All on the M6, forge stopped where a test needed the Neural Engine (one model at a time), swap watchdog on every
process, every Core AI load journaled; tools in [pipe/unit/power/](pipe/unit/power/), logs in
[results/18nov/](results/18nov/README.md).

| test | what it answered | result |
|---|---|---|
| traced V8 prefill (`v8_trace.py`, 2K-64K) | which unit sets the pace, per-call costs | table above |
| V8 benchmark under telemetry (`bench_tel.py` + `srv_bench.py`) | per-request prefill and decode state | tables above |
| long decode (`long_decode.py`, 2,500 tokens) | does decode heat-throttle | no: steady 104 GB/s, 7 W, 83 s |
| DeltaNet alone, one layer vs all 48 (`gpu_mix_win.py gdn / gdnall`) | contention vs cache coldness | 1.94 / 2.12 ms vs 2.55 in the pipeline |
| attention variants vs the FFN (`coupling_run.py attn_*`) | do the flash forms reduce traffic | no: 1.7-3.0 GB per call vs 1.4 unfused; the unfused stays fastest |
| GPU compute types vs the Neural Engine's FFN forms (set `wa`) | which data types survive contention | GPU W4A8 wins 1.25-1.3x; WL3A8 needs a light partner |
| GPU weight streaming at 16-32 rows (`gpu_wa_gemm`) | GPU bandwidth at decode-like batches | 134-145 GB/s in every format |
| the Neural Engine's on-chip memory (`ane_cache_probe.py`) | what stays on chip | under 1 MB of reused weights; activations stream |
| counter calibration (`cal_analyze.py`) | are the counters scaled | no; the GPU's per-agent counter reads the rate while active |
| even FFN chunk split | balance the two clusters in prefill | not possible: 17 blocks of 1024 (the mid Hadamard) |
| WL3A8 at 512 rows vs a host reference (`w8a8/ref512.py`, `check512.py`) | is the fast 512-row INT8 form correct | yes: after four chained FFN passes WL3A8 is within 1.16% of the served FFN on the host (0.89% of its own host reference); WL3A16 0.13% |
