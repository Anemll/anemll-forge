# Two accelerators, one memory: what the GPU and the Neural Engine really compete for on the M6

*ANEMLL Forge research notes, 18 November 2026. A follow-up to step 7 of
[One model, two accelerators](ARTICLE_ANE_GPU_M6.md). Every number here comes from the raw logs in [results/18nov/](results/18nov/README.md).*

## Summary

Our hybrid runs Qwen3.8-27B on the M6's GPU and Neural Engine (ANE) at the same time, and the two units slow each
other down: the GPU's attention call at 60K takes 18.7 ms alone and 27 ms while the Neural Engine runs its FFN. The
first article named two candidates: a shared power or clock limit, or contention in the system cache. We measured
both, with telemetry that needs no root on macOS 27.3 plus `powermetrics` for the Neural Engine's power. Neither is
the cause.

- **It is DRAM.** Attention at 60K reads about 93 GB/s from DRAM, the FFN about 85 GB/s. Together they ask for about
  180 GB/s from a memory that delivers about 155; they get about 134, and the GPU's memory latency doubles.
- **Not power, not clock.** The GPU stays at its top clock (1735 MHz) in every pair. The pairs that draw the most
  power slow down the least; a memory stream that draws 2 W slows its partner more than matrix work at 21 W.
- **Not the system cache.** It is about 8 MB, far smaller than the weights and the key/value cache these loads read.
- **The Neural Engine usually wins.** Next to the synthetic GPU loads the FFN slows 0 to 6%, while the memory-heavy
  GPU loads lose 23 to 78%. Against attention at full clock the two split the bandwidth about evenly.
- **The Neural Engine also slows down when it gets warm,** by 30 to 45% after about four minutes of continuous work;
  the GPU does not. In real prefill this stays hidden: the pipeline is GPU-bound from 8K up.
- **Real V8 prefill uses about three quarters of the memory's bandwidth** with the GPU's memory latency about doubled,
  and the served FFN loads the Neural Engine's second cluster about 9% less than the first.

This is also why the split works: at short context the mixers on the GPU and the FFN on the Neural Engine together
stay mostly within the memory's bandwidth, and the coupling is only the queueing cost. At long context attention's
key/value reads push the pair over the limit.

## The question

The first article measured the coupling with stress loads and reasoned from operation types: a 153 GB/s GPU memory
stream slowed the Neural Engine 7%, an INT8 DeltaNet loop slowed it 20%, and a Neural Engine program "with few
weights but heavy compute" slowed the GPU as much as the weight-heavy FFN. That pattern pointed away from bandwidth
and towards the matrix engines sharing a power or clock budget. It stayed a hypothesis: there were no power, clock
or bandwidth samples.

## How we measured

**Telemetry without root.** `powermetrics` reads a private interface, IOReport, that any process can read. On this M6
(macOS 27.3) it exposes about 8,000 channels; [`power/iorep_sample.c`](pipe/unit/power/iorep_sample.c) records the
useful ones every 100 ms:

| channels | what they tell us |
|---|---|
| GPU performance states, power-controller states, limit reasons, throttle counters | the GPU clock, and whether a power or thermal limit holds it down |
| PMP DCS and AF bandwidth per agent (GPU, the two Neural Engine clusters, CPU clusters) | DRAM traffic and fabric traffic of each unit |
| PMP AFI DCS | the Neural Engine's own DRAM traffic |
| PMP AGX memory latency | the GPU's memory latency in cycles |
| SoC events DCS_F2 to DCS_F10, DCS floors per agent | the DRAM clock state, and which unit asked for it |
| SoC events *_THROTTLE_*, SOCHOT | throttle and over-current triggers |
| PMP voltage-temperature histograms | temperatures per cluster |

The Neural Engine's power is not among them (the energy group lists only the GPU), so a small root service,
[`power/pm_service.sh`](pipe/unit/power/pm_service.sh), runs `powermetrics` at 100 ms on request: it only ever runs
`powermetrics`, and a client starts and stops recordings by writing files.

Two limits: the bandwidth histograms top out at 128 GB/s per agent, so readings near that are underestimates (we
report the share of clipped samples); and the system-cache hit counters read zero on this OS, so cache traffic is
taken as fabric traffic minus DRAM traffic.

**Loads.**

- Neural Engine: `ffn2_L0` of the served build (two 64-row FFN blocks per call), and `hot`, one 2048 x 2048
  convolution (8 MB of weights) applied eight times to 512 rows.
- GPU: attention at 60K (layer 3, 128 rows per call, FP16 keys and values) and the DeltaNet layer in FP16 and INT8,
  from our engine (the first article's test engine `libgpu_v8`, and the fused engine for the INT8 paths; the server
  runs `libgpu_v8_srv3`, whose attention reads the Neural Engine's INT8 key/value cache and cannot run alone);
  synthetic loads in [`power/gpu_load.swift`](pipe/unit/power/gpu_load.swift): an FMA chain in registers, a memory
  stream of a chosen size, and INT8 and FP16 matrix loops on the GPU's matrix units whose weight footprint and weight
  reuse are set independently (22.1 TOPS at a 4 MB footprint and at 128 MB).
- CPU: a register-only spin on eight cores (about 10 W).

**Method.** [`power/coupling_run.py`](pipe/unit/power/coupling_run.py) starts every load once and pauses it
(SIGSTOP); each phase resumes some loads for 8 to 14 s. Every pair phase sits between alone phases of both loads, a
slowdown is the pair against those neighbours, and every set is followed by 5 minutes idle. A full scan recorded all
8,000 channels at idle, under each load and under each pair, to find counters we did not know to look for.

## Result 1: the GPU clock does not drop, and power does not predict the slowdown

| pair | slower than alone | GPU clock | GPU W | ANE W | SoC W |
|---|---|---|---|---|---|
| FFN + attention@60K | FFN +15 to +32%, attention +42 to +46% | 1735 MHz | 11.6 to 12.3 | 11.7 to 17.1 | 25 to 30 |
| FFN + INT8 matrix loop (128 MB) | FFN +4 to +8%, GPU +3 to +5% | 1735 MHz | 20.4 to 20.8 | 10.4 to 15.0 | 32 to 37 |
| FFN + FMA chain | FFN +0.5%, GPU 0% | 1735 MHz | 16.5 | 15.7 | 33.5 |
| FFN + memory stream | FFN +6 to +16%, GPU +76 to +78% | 1266 MHz | 1.1 | 14.0 to 15.6 | 16 to 18 |

The GPU stays at its top clock whenever its work keeps it busy, and its power controller never enters a power or
thermal limit state. In the attention pair the GPU draws less power than alone (about 12 W against 16): it is
waiting, not throttled. The pair that draws the most power barely slows; the memory stream, at 2 W, slows its
partner and itself the most (it runs at 1266 MHz only because the GPU's controller lowers the clock for
memory-bound work, alone as well).

Adding about 10 W of CPU power (the spin loop) did not change attention at all (+0.1%, -0.0%). It slowed the
Neural Engine by 4 to 10%, which fits its heat sensitivity (result 5) or a delay on its submitting CPU thread; we
cannot separate the two yet.

## Result 2: the coupling follows DRAM traffic

| GPU load | alone: GPU W, DRAM GB/s | GPU slower next to the FFN | next to hot |
|---|---|---|---|
| FMA chain (no memory) | 16.5, 2 | 0% | 0% |
| INT8 matrix loop, weights in cache | 21.3, 3 | +2 to +3% | +16 to +18% |
| INT8 matrix loop, 128 MB of weights | 21.1, 22 | +2 to +3% | +10.5 to +11% |
| INT8 matrix loop, low weight reuse | 16.2, 104 | +47 to +51% | +66 to +67% |
| FP16 matrix loop (DRAM-bound) | 19.9, 114 | +23 to +45% | +50 to +53% |
| memory stream, 1 GB | 1.9, 123+ | +64 to +78% | +75% |
| DeltaNet FP16 / INT8 (our engine) | 20 / 17, 39 / 43 | +12 / +12 to +14% | +15 / +16.5% |
| attention@60K (our engine) | 16.1, 93 | +42 to +46% | +41 to +42% |

The same INT8 matrix work costs 3% or 51% depending only on how many bytes it takes from DRAM. Every GPU load that
reads 100 GB/s or more loses 23 to 78% next to the Neural Engine; matrix work served from cache loses a few percent.
The INT8 and FP16 rows come from two sets run in opposite orders (`types`, `types_rev`); the spread is the Neural
Engine's heat: warmer, it asks for less bandwidth.

The bytes per call explain the earlier puzzles:

- **Attention@60K is not compute-bound.** It reads about 1.76 GB per call, about seven times the 252 MB of FP16
  keys and values at 60K: each of the six query heads that share a key/value head reads it again.
- **The FFN streams its weights once per 64-row block.** About 206 MB per `ffn2_L0` call, twice the layer's 3-bit
  FFN weights (about 100 MB).
- **`hot` was never weight-light in traffic.** Its 8 MB of weights do not stay on the Neural Engine; it reads more
  than 100 GB/s from DRAM (60% of samples in the 128 GB/s top bin), more than the FFN. That is why it slowed the GPU
  as much as the FFN did.

The full scan found one counter that changes in a pair by more than the sum of the two loads: the GPU's memory
latency. Attention alone: 190 to 221 cycles; with the FFN: 382 to 429. The memory stream: 267 alone, 436 to 443 with
the FFN. The INT8 matrix loop in cache: about unchanged (92 to 103). Even when the two units' total stays below the limit (DeltaNet and
FFN together use 110 to 120 GB/s), the latency doubles and the GPU loses 12 to 16%: its kernels wait on memory.

## Result 3: the Neural Engine usually wins the memory

Against the synthetic loads the FFN slows 0 to 6% while the GPU slows up to 76%; next to a memory stream the GPU
falls from 153 to 87 GB/s and the Neural Engine keeps most of its traffic. Attention is the exception: running at its
top clock it takes an even share (about 67 and 67 GB/s), and the FFN slows 15 to 32%. The bandwidth-hungrier `hot`
program slows 9 to 28% next to the DRAM-heavy matrix loops and the memory stream, and 44 to 68% next to attention.

## Result 4: the system cache is small and not the bottleneck

| GPU streaming reads of | rate | from DRAM |
|---|---|---|
| 1 MB | 371 GB/s | none (inside the GPU) |
| 2 MB | 549 GB/s | none |
| 4 to 8 MB | 307 to 317 GB/s | 2 to 26 GB/s (the system cache) |
| 12 MB and more | 153 to 172 GB/s | all |

The GPU's own cache holds about 1 to 2 MB and the system cache about 8 MB, both far smaller than the FFN's weights,
attention's keys and values or the DeltaNet weights. One cache effect does exist: a GPU stream served by the system
cache loses 34 to 44% next to the Neural Engine without any extra DRAM traffic, so the fabric between them is shared
too. Our attention and DeltaNet kernels read from DRAM, so it is not what slows them.

## Result 5: the Neural Engine slows down when it gets warm

Within each set the Neural Engine's alone phases got slower: the FFN went from 2.42 to 3.49 ms per call and its power
from 18 to 9 W for the same work, while the CPU cluster's temperature reading rose from 75 to 95 C. Five minutes idle
restored it. `powermetrics` reported thermal pressure "Nominal" throughout, and the GPU's clock and times did not
change.

The DRAM clock follows: each unit votes for a DRAM frequency, the GPU always for the top state (F10), the Neural
Engine for F10 when cool and F7 to F8 when warm. Keeping DRAM at F10 with a GPU load did not speed the warm Neural
Engine (3.107 ms against 3.119 alone), so its own speed drops and its vote follows. The first article put thermal
drift at 5 to 10%; for continuous Neural Engine work it is several times that.

## Result 6: what the Neural Engine keeps on chip

[`power/ane_cache_probe.py`](pipe/unit/power/ane_cache_probe.py) measures DRAM bytes per call for small FP16 programs:
one C x C convolution applied 8 times (weights W = 0.5 to 16 MB) at 64 and 512 rows, 8 distinct convolutions, and a
small convolution on 64 to 16,384 rows (activations 32 KB to 8 MB).

| program | DRAM per call | reading |
|---|---|---|
| 0.5 MB weights, 8 times, 64 rows | 1.1 MB (2 x W) | kept on chip across the 8 ops |
| 1 MB weights, 8 times, 64 rows | 7.6 MB (8 x W) | reloaded for every op |
| 2 to 16 MB weights, 8 times, 64 rows | 6.9 to 8.0 x W | once per op |
| 1 to 16 MB weights, 8 times, 512 rows | 8.4 to 17 x W | about twice per op (at 2 and 4 MB the second read comes from the system cache) |
| 0.12 MB weights, activations 0.5 to 4 MB, 8 ops | input + output + weights | intermediate activations never reach DRAM |

The traffic is always split evenly between this M6's two Neural Engine clusters (ANE0 and ANE1, two links each). At 64
rows the clusters share an op's weights, each streaming half; at 512 rows each streams all of them for its half of the
rows, which is why `hot` (8 MB, 512 rows) reads its weights twice per op. The compiler keeps less than 1 MB of reused
weights on chip and runs row tiles through a whole chain of ops, so the activations stay local. This measures what
the compiler keeps, not the size of the on-chip memory: if each cluster has a few MB, they go to activation tiles and
streaming buffers, not to caching weights.

For the FFN this sets the bytes: today's 64-row blocks read the layer's weights once per block (about 100 MB); a
512-row block reads them about twice, so weight traffic per token falls about 3.5 to 4 times, not 8.

## Result 7: real V8 prefill, measured

The same telemetry during cold requests to the served build ([`power/prefill_tel.py`](pipe/unit/power/prefill_tel.py),
a different document slice per request, a minute idle before each):

| prompt | prefill (server) | GPU active, clock | ANE W, start to end | DRAM GPU / ANE GB/s, start to end | DRAM total | GPU memory latency | P-cluster temperature |
|---|---|---|---|---|---|---|---|
| 8K | 658 tok/s | 98%, 1732 MHz | 14.9 | 51 / 72 | 114 GB/s (73% of peak) | 340 cycles | 82 C |
| 32K | 537 tok/s | 96 to 100%, 1735 MHz | 14.3 to 7.5 | 51 / 70 to 68 / 46 | 118 (76%) | 341 to 258 | 84 to 96 C |
| 64K | 420 tok/s | 95 to 100%, 1735 then 1664 to 1690 MHz | 12.8 to 5.4 | 50 / 69 to 75 / 31 | 114 (74%) | 336 to 252 | 86 to 96 C |

(The first request after the server's restart, 2K, took 10 s, a warm-up cost; it is left out.)

- **Prefill is GPU-bound from 8K up.** The GPU is busy 96 to 100% of the time; the memory system carries about three
  quarters of its peak and the GPU's memory latency stays at 250 to 340 cycles (100 to 220 alone): the pipeline runs
  in the contended state the whole time.
- **The Neural Engine's DRAM traffic is a token-rate meter.** It reads its FFN weights once per 64-row block, about
  100 MB per token across 64 layers, so its traffic divided by 100 MB gives the prefill rate: 544 tok/s at 32K
  (server: 537), 444 at 64K (420). Within the 64K request it falls from about 640 to about 310 tokens per second as
  the GPU's attention grows; the GPU's DRAM bytes per token triple (78 to 242 MB) over the same request.
- **So the Neural Engine's falling power during a long prefill is mostly the pipeline slowing down,** not its heat
  throttle: the GPU sets the pace, and the Neural Engine's slower calls stay hidden behind it.
- **The GPU does get clocked down in long prefills.** After about 50 s of the 64K request its clock falls to 1664 to
  1690 MHz with the power controller in a state not seen before (`PMULOOP`, 49% of the time) and 425 ms of throttle
  events; the SoC reads 95 C. The only GPU limit in all the tests, and it comes from sustained heat, not from the
  coupling.
- **The two Neural Engine clusters are not evenly loaded in the served FFN.** In every prefill and every FFN loop the
  second cluster moves 9% less traffic than the first (balance 0.90 to 0.92); the `hot` probe and the cache probes are
  balanced (0.999). The chunked FFN splits its intermediate layer into 8,192 and 9,216 channels, and 8,192 / 9,216 =
  0.89: if each cluster takes one chunk, the second waits about a tenth of every FFN call. An even split (8,704 +
  8,704) has not been tried; the 17 November split search tried [9216, 8192] (-10%), more chunks and [16384, 1024].

## Why the split works anyway

| pair, each unit flat out | DRAM demand alone (GPU + ANE) | coupling |
|---|---|---|
| DeltaNet mixer + FFN (prefill at short context) | 40 + 85 = 125 GB/s | GPU +12 to +14%, FFN +3 to +6% |
| attention@60K + FFN (long context) | 93 + 85 = 178 GB/s | attention +42 to +46%, FFN +15 to +32% |

The split puts the largest steady DRAM stream, the FFN weights, on the unit that wins the arbitration, and the
mixers, lighter on memory and heavy on compute, on the GPU. At short context their combined demand mostly fits, and
the coupling is the queueing cost: about 12 to 16% on the GPU, against a prefill about twice as fast as the Neural
Engine alone (576 to 668 against about 300 tokens per second at 2K to 16K, first article). At long context attention's key/value reads grow with the context and push the pair past the memory's limit;
that is where the first article saw the split pay off less.

## What to change

The levers are bytes, not watts:

1. **Read keys and values once per call in GPU attention.** Today each query head re-reads its key/value head
   (1.76 GB per call at 60K). A kernel that serves all six query heads and all 128 rows from one pass would read
   about 0.25 to 0.3 GB per call instead (an estimate: the keys and values once, plus queries and outputs), speeding
   attention alone and removing most of the long-context coupling.
2. **Larger FFN blocks on the Neural Engine.** The weights are read once per 64 rows; a 512-row block reads them
   about twice (result 6), so about 3.5 to 4 times fewer weight bytes per token. The INT8 FFN row sweep of
   17 November measured 14.7 us per token at 512 rows against 19.7 for today's 64-row blocks (its 512-row output was
   not checked against a reference yet). The cost is a coarser pipeline.
3. **Larger DeltaNet blocks on the GPU,** for the same reason (about 100 MB of DRAM traffic per 128-row call).
4. **Balance the two Neural Engine clusters.** The served FFN's chunks (8,192 and 9,216) leave one cluster about 9%
   less loaded; an even split is a one-line builder change to test.
5. **Watch the heat.** In real prefill the GPU sets the pace, so the Neural Engine's heat throttle stays hidden; the
   GPU itself loses 3 to 4% of its clock after about a minute. Where the Neural Engine sets the pace (decode, the
   Neural Engine alone), its 30 to 45% drop after minutes of work matters directly.

## Next: faster compute modes per block

The other half of the budget is compute. The GPU's matrix units run 8-bit by 8-bit at twice the 16-bit rate (below),
and the Neural Engine's INT8 rate is about twice what our FP16-activation programs reach. A read of the served
pipeline's code (`results/18nov/datatype_inventory.md`) gives the operand types today:

| block | unit | product today | share of the model's multiply-adds per token |
|---|---|---|---|
| DeltaNet qkv, z, out; attention q + gate, o | GPU | INT8 x INT8 (4-bit palette weights re-quantized to INT8 in the kernel, INT8 activations per 1024-channel block) | about 27% |
| attention k, v | GPU | INT8 x INT8 (INT8 export weights) | under 1% |
| attention Q.K | GPU | INT8 x INT8 | grows with context |
| attention P.V | GPU | FP16 x FP16 | grows with context |
| low-rank corrections, DeltaNet a / b | GPU | FP16 x FP16 | about 3M per layer |
| FFN gate, up, down and its Hadamards | Neural Engine | FP16 x FP16 (3-bit palette expanded to FP16) | about 71% |
| DeltaNet recurrence, conv, norms, softmax | GPU / ANE | FP32 / FP16 vector work | not matrix work |

So the GPU side is already almost all 8-bit by 8-bit, and the FFN, about 71% of the work, is all 16-bit. The
follow-up: the FFN as INT8 x INT8 on the Neural Engine (the palette-dequant form of 17 November compiles without a
second copy of the weights, 16% faster at 64 rows), P.V in 8 bits at long context, and INT4 where the units take it
(on the GPU only as the weight side: INT8 x INT4 runs at 32.3 TOPS, 82% of INT8; FP16 x INT4 at the FP16 rate).

## Side result: compute rates by mode, both units

Names give the weight storage and the activation width; the compute type follows: WL3A16 = 3-bit palette weights
expanded to FP16, FP16 activations, FP16 x FP16; WL3A8 = the palette dequantized to INT8 codes, INT8 activations,
INT8 x INT8; W4A8 = linear INT4 weights, INT8 activations (the GPU's matrix op widens INT4 to INT8).

**GPU, `matmul2d` peak** (128-row tiles, data in cache, [`power/gpu_mm_peak.swift`](pipe/unit/power/gpu_mm_peak.swift)):

| compute | rate |
|---|---|
| INT8 x INT8 -> INT32 (W8A8) | 39.5 TOPS |
| FP8 e4m3 x e4m3 -> half / float | 36.9 / 35.8 TFLOPS |
| INT8 x INT4 -> INT32 (W4A8) | 32.3 TOPS |
| FP16 x FP8 -> float | 20.7 TFLOPS |
| FP16 x INT4 -> float (W4A16) | 19.5 TFLOPS |
| BF16 x BF16 / FP16 x FP16 -> float | 19.3 / 19.2 TFLOPS |
| FP4 e2m1 x e2m1 -> float | 17.7 TFLOPS |

**GPU at prefill shapes** (weights streamed from DRAM once per call, 16,384 x 8,192, [`power/gpu_wa_gemm.swift`](pipe/unit/power/gpu_wa_gemm.swift)):

| compute | 128 tokens per call | 64 tokens per call |
|---|---|---|
| W8A8 | 32.2 TOPS (126 GB/s of weights) | 16.7 TOPS (130 GB/s: DRAM-bound) |
| W4A8 | 27.8 TOPS (54 GB/s) | 31.1 TOPS (121 GB/s) |
| WF8AF8 (FP8 x FP8) | 32.0 TFLOPS (125 GB/s) | |
| W16A16 / BF16 | 12.7 / 12.6 TFLOPS (99 GB/s) | 8.9 TFLOPS (139 GB/s: DRAM-bound) |
| W8A16 / W4A16 / WF8A16 / WF4A16 | 18.0 / 18.6 / 19.8 / 18.3 TFLOPS | |
| FP4 x FP4 | 16.8 TFLOPS | |

With FP16 activations the weight format hardly matters (all at the FP16 compute rate); with INT8 activations, 4-bit
weights win where the call is DRAM-bound (64 tokens: 1.87 times W8A8) and lose a little where it is not (128 tokens).

**Under contention** (each GPU form at 128 tokens next to layer 36's FFN on the Neural Engine; ANE / GPU slowdown
against their adjacent alone phases, all warm):

| ANE FFN (layer 36, chain of 4) | GPU W16A16 | GPU W8A8 | GPU FP8 x FP8 | GPU W4A8 | GPU W4A16 |
|---|---|---|---|---|---|
| WL3A16, 64 rows: ANE / GPU slower by | +7.5% / +62% | +3.0% / +74% | +2.3% / +54% | +4.0% / +10.5% | +1.6% / +4% |
| WL3A8, 64 rows | +36% / +56% | +31% / +70% | +24% / +68% | +11% / +20% | +4.6% / +11% |
| WL3A16, 512 rows | (first-use compile) | +2.0% / +34% | +3.2% / +49% | +2.6% / +8% | +0.7% / +4% |
| WL3A8, 512 rows | +3.1% / +51% | +1.9% / +68% | +0.6% / +67% | +4.2% / +13% | +0.3% / +8% |

GPU W4A8 wins every pairing: 17% slower than W8A8 alone, 1.25 to 1.3 times faster next to the Neural Engine, because
it asks for half the DRAM. The Neural Engine's WL3A8 is faster alone but more bandwidth-bound; next to an 8- or 16-bit
GPU stream it loses 24 to 36% and ends slower than WL3A16, next to 4-bit GPU weights it keeps its lead. 512-row blocks
on the Neural Engine read their weights only about twice per 512 rows and barely feel the GPU at all.
Our served GPU projections are WL4A8 (4-bit palette indices widened to INT8 tiles in the kernel); their LUT GEMMs run
at about 25.5 TOPS, and a whole DeltaNet layer at 15.6 TOPS of projection work per 128-row call (1.95 ms, with the
recurrence and norms).

**Neural Engine, layer 36's served FFN** (17 November row sweep, 290M multiply-adds per token, four passes per call):

| rows per call | WL3A16 (served) | WL3A16, INT8 palette codes | WL3A8 gate / up only | WL3A8 everywhere |
|---|---|---|---|---|
| 8 | 6.3 TFLOPS | 6.3 | 5.6 TOPS | 5.5 TOPS |
| 32 | 24.3 | 23.8 | 21.0 | 20.2 |
| 64 | 29.4 | 30.7 | 31.5 | 35.1 |
| 128 (one call) | 19.6 | 20.3 | 18.2 | 18.1 |
| 256 | 25.5 | 26.2 | 19.1 | 29.7 |
| 512 | 25.6 | 26.3 | 29.8 | 39.4 |

The Neural Engine's FP16 FFN tops out near 29 TFLOPS at 64 rows; INT8 x INT8 reaches 35 at 64 rows and 39 at 512
(14.7 us per token against 19.7), the same as the GPU's INT8 peak. Below 64 rows the INT8 activation quantization
costs more than it saves. Single 128-row calls are slow in every mode.

## Follow-ups done the same day

- **Counter calibration** at controlled rates: no scale factor; the GPU's own DRAM counter reports the rate while it
  is active (`pipe/unit/power/IOREPORT_COUNTERS.md`, "Calibration").
- **The V8 pipeline traced and benchmarked under the same telemetry,** prefill and decode, with every lever this
  suggests ranked by measured potential: [ANALYSIS_V8_M6.md](ANALYSIS_V8_M6.md).

## Corrections to the first article

- Step 7 named a shared power or clock limit as the best explanation. The coupling is DRAM bandwidth and latency.
- "A Neural Engine program with few weights but heavy compute": `hot` reads more DRAM than the FFN.
- "A 153 GB/s memory stream slowed the Neural Engine only 7%": true, but the stream itself lost 44 to 76%; the
  Neural Engine wins the arbitration.
- "Thermal drift costs 5 to 10%": the Neural Engine loses 30 to 45% after minutes of continuous work.

## Reproduce

All tools in [pipe/unit/power/](pipe/unit/power/), results and the index in [results/18nov/](results/18nov/README.md).

```
# once, as root (only runs powermetrics on request)
sudo -v && sudo nohup pipe/unit/power/pm_service.sh >/dev/null 2>&1 &
# build the C and Swift tools
clang -O2 -o iorep_sample iorep_sample.c -framework CoreFoundation -lIOReport
swiftc -O -o gpu_load gpu_load.swift; swiftc -O -o gpu_mm_peak gpu_mm_peak.swift; clang -O2 -o cpu_spin cpu_spin.c
# the campaign (forge stopped: one model at a time), about an hour
campaign.sh OUT_ROOT slc,real,types,gdn,cpu,types_rev
# a full IOReport scan, and its diff
coupling_run.py scan OUT; scan_analyze.py OUT
```
