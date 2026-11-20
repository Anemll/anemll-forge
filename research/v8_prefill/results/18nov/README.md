# Results of 18 Nov 2026

All on the M6 (32 GB, macOS 27.3, build 26C5035r). Tools in [../../pipe/unit/power/](../../pipe/unit/power/).

## GPU / Neural Engine coupling (article step 7)

The question: why the two units slow each other down (attention@60K 18.7 ms alone, 27 ms with the ANE FFN). The
article's two candidates were a shared power or clock limit, or contention in the system cache. Measured here with
telemetry during the article's load pairs and new synthetic loads.

### Telemetry, no root and with root

- `iorep_sample.c`: IOReport (the private API `powermetrics` reads), no root, 100 ms deltas as JSON lines. Records the
  GPU P-state residency and the GPU power controller's state (what limits its clock), power histograms, DRAM (DCS)
  and fabric (AF) bandwidth histograms per agent (GPU = AGX, the two ANE clusters, CPU clusters), the ANE's own fabric
  and DRAM traffic (AFI), SoC voltage floors, throttle and SOCHOT events, DRAM clock state, GPU memory latency,
  voltage-temperature histograms. `IOREP_ALL=1` records every channel (the full scan).
- `pm_service.sh`: a root service the user starts once (`sudo -v && sudo nohup .../pm_service.sh >/dev/null 2>&1 &`);
  it runs only `powermetrics` (cpu_power, gpu_power, ane_power, thermal; plist at 100 ms) when a client drops a start
  file in its control folder. Gives the ANE power, which IOReport does not expose without root on this OS (its
  "Energy Model" group has only the GPU).
- `telemetry.py`: window summaries of both; `coupling_analyze.py`: per-phase table; `scan_analyze.py`: full-scan diff.
- Limits: the PMP bandwidth histograms top out at 128 GB/s per agent and for the total (AMCC), so totals above that
  are clipped (the `clip` column gives the share of samples in the top bin); the system-cache DSID hit / miss
  counters read zero on this OS; cache traffic is estimated as fabric minus DRAM per agent.

### Loads

- `ane_win.py` (ANE, journaled): `ffn2_L0` of the served build (`v8_dec8/chunk_L00-07`), and `hot` (one 2048 x 2048
  conv, 8 MB, 8 times on 512 rows; rebuilt today with `v8/share/tiny_ane_probe.py build_hot` because the 15 Nov copy
  no longer loads: `nilError` at `function("t")`).
- `gpu_mix_win.py` (GPU engine): attention layer 3 at 60K (gpu_steady.py's engine and tables; `attq`: the fused engine
  with INT8 Q.K), DeltaNet layer 0 fp16 / INT8 (`gdn16`, `gdn8`).
- `gpu_load.swift`: `alu` (FMA in registers), `bw` (streaming reads of an MB-sized buffer), `mm8` / `mm16` (INT8 /
  FP16 matmul2d on the GPU's matrix units; each threadgroup copies its weight tile into threadgroup memory once and
  reuses it REP times, so the weight footprint changes without changing the compute: 22.1 TOPS at 4 MB and 128 MB,
  FP16 12.8 TFLOPS at 4 MB and 256 MB); duty 0.5 variants halve the average power. `cpu_spin.c`: CPU power only.
- `coupling_run.py`: starts each load once, pauses it with SIGSTOP, then runs the phases by SIGCONT / SIGSTOP; every
  pair phase sits between alone phases of both loads; 5 min idle between sets (`campaign.sh`).

### Files

| dir | what |
|---|---|
| `coupling/iorep_channels_idle.txt` | every IOReport channel on this OS (8027) with one idle sample |
| `coupling/smoke1/` | first pass: attention + FFN, INT8 matmul + FFN (`summary.txt`) |
| `coupling/scan2/` | full scan: every channel at 500 ms, idle / each load / each pair; `scan.txt` = counters that respond to each unit and those that are not additive in the pairs; `summary.txt` = the per-phase table |

### Results so far (smoke1, scan2)

| phase | slower vs alone | GPU MHz | GPU W | ANE W | SoC W | DRAM GPU / ANE GB/s | GPU mem latency (cycles) |
|---|---|---|---|---|---|---|---|
| FFN alone | | | | 18.6 | 19.8 | 6 / 84 | |
| attention@60K alone | | 1735 | 16.1 | | 17.3 | 93 / 2 | 197 |
| FFN + attention | FFN +32%, attention +46% | 1735 | 11.6 | 17.1 | 30.4 | 68 / 66 | 429 |
| hot alone | | | | 16.4 | 18.2 | 5 / 99-101 | |
| hot + attention | hot +68%, attention +41% | 1735 | 12.0 | 13.8 | 27.5 | 68 / 65 | 407 |
| INT8 matmul 128 MB alone | | 1735 | 21.3 | | 22.4 | 22 / 2 (fabric 123) | 96 |
| FFN + INT8 matmul | FFN +8%, matmul +5% | 1734 | 20.4 | 15.0 | 36.7 | 21 / 79 | 103 |
| DRAM stream alone | | 1403 | 1.9 | | 3.1 | 123+ (clipped) / 2 | 267 |
| FFN + DRAM stream | FFN +16%, stream 153 -> 86 GB/s | 1266 | 1.1 | 15.6 | 18.3 | 79 / 72 | 443 |

- The GPU clock does not drop when the ANE works (1735 MHz, the top state, in every pair with attention or the
  matmul); the GPU power controller shows no power or PPM limit states. The DRAM stream runs below the top clock
  alone already (`CORE_EFF`: the controller lowers the clock of memory-bound work).
- The slowdown follows DRAM demand, not power: the pair with the most power (INT8 matmul + FFN, 36.7 W) slows 5-8%,
  the pair with the least (DRAM stream + FFN, 18.3 W) slows the FFN 16% and the stream 44%.
- Attention@60K is not compute-bound: it pulls 93 GB/s from DRAM (more in the clipped samples), about 1.75 GB per
  call, about 7 times the K / V it reads. With the FFN (84 GB/s) the demand is about 180 GB/s; the pair gets about
  134 GB/s, and the GPU's memory latency doubles.
- `hot` is not weight-light in traffic: its 8 MB weights do not stay on the ANE; it reads 100+ GB/s from DRAM
  (60% of samples in the 128 GB/s top bin), more than the FFN. That is why it slowed the GPU as much as the FFN in the
  15 Nov test, which the article read as "not bandwidth".
- The full scan's only counter that changes in a pair beyond the sum of the two loads is the GPU memory latency
  (attention: -26 cycles alone, +173 with the FFN; stream: +12 alone, +188 with the FFN; matmul: no change).

### Campaign `coupling/c18/` (`campaign.sh`: sets slc, real, types, gdn, cpu, types_rev; 5 min idle between sets)

Each set dir: `driver.log`, one log per load (timestamped 0.5 s windows), `phases.json`, `summary.{txt,json}`
(`coupling_analyze.py`: per phase the load medians, the slowdown against the adjacent alone phases of the same load,
IOReport and powermetrics means), `io.jsonl.gz` (IOReport), `pm.plist.gz` (powermetrics).

**slc** (GPU only): streaming reads 371 GB/s at 1 MB (inside the GPU), 549 at 2 MB, 307-317 at 4-8 MB (fabric about
120 GB/s, DRAM 2-26: the system cache), 153-172 GB/s from 12 MB up (DRAM). GPU L2 about 1-2 MB, system cache about
8 MB. INT8 matmul 22.2 TOPS at every footprint 1-128 MB; DRAM 2-4.5 GB/s up to 4 MB, 21 GB/s from 8 MB.

**real** (the article's pairs, two orders): FFN + attention@60K: FFN +14 to +27%, attention +43 to +45%; hot +
attention: hot +44 to +56%, attention +35 to +42%; fused attention with INT8 Q.K (16.7 ms alone, 82 GB/s DRAM): +35 to
+43%. GPU at 1735 MHz in every pair; pair DRAM about 67 / 67 GB/s (GPU / ANE).

**types** (slower vs adjacent alone; GPU load alone: W, DRAM GB/s):

| GPU load | alone | FFN | GPU vs FFN | hot | GPU vs hot |
|---|---|---|---|---|---|
| FMA (alu) | 16.5 W, 2 | +0.5% | 0% | +1.8% | 0% |
| INT8 matmul 4 MB / 128 MB | 21.3 W, 3 / 21.1 W, 22 | +4.9 / +3.8% | +2.8 / +2.9% | -0.3 / +2.3% | +16 / +10.5% |
| INT8 matmul low reuse (128 ops / B) | 16.2 W, 104 | +2.3% | +51% | +3.5% | +67% |
| FP16 matmul (8 MB activations: DRAM-bound) | 19.9 W, 114 | +0.9% | +45% | +18% | +53% |
| DRAM stream 1 GB | 1.9 W, 123+ | +5.6% | +76% | +28% | +75% |
| cache-resident stream 8 MB | 4.0 W, 17 (fabric 120) | 0% | +34% | -0.8% | +44% |

**gdn** (the article's DeltaNet loops): DeltaNet FP16 2.58 ms, 39 GB/s DRAM; INT8 1.95 ms, 43 GB/s. With the FFN:
DeltaNet +12 to +14%, FFN +3 to +6%; with hot: DeltaNet +15 to +16.5%, hot +4.5 to +10%. GPU memory latency 101-120
-> 224-251 cycles in the pairs, with total DRAM only 110-120 GB/s.

**Heat**: the ANE alone slows across every set (FFN 2.42 -> 3.49 ms, ANE power 18 -> 9 W at the same work, CPU-cluster
temperature bin 75 -> 95 C), recovered by the 5 min idle; `powermetrics` thermal pressure stays Nominal; the GPU's
times and clock do not change. The DRAM clock (DCS) follows the ANE's own floor vote (F10 when cool, F7-F8 when warm),
but holding DRAM at F10 with a GPU load did not speed the warm ANE (`ffn+mm8l` 3.107 ms at F10 vs 3.119 alone), so the
ANE's own speed drops and its DRAM vote follows.

### GPU matrix-unit peak by element type (`power/gpu_mm_peak.swift`, TM 128, data in cache, 1.2 s each)

| A x B -> C | T(FL)OPS |
|---|---|
| INT8 x INT8 -> INT32 | 39.5 |
| FP8 e4m3 x e4m3 -> half / float | 36.9 / 35.8 |
| INT8 x INT4 -> INT32 | 32.3 |
| FP16 x FP8 e4m3 -> float | 20.7 |
| FP16 x INT4 -> float | 19.5 |
| BF16 x BF16 -> float | 19.3 |
| FP16 x FP16 -> float | 19.2 |
| FP4 e2m1 x e2m1 -> float | 17.7 |

Throughput only (constant 1.0 inputs, no accuracy check). TM 64: INT8 33.9, FP16 9.5. Packed formats (FP8, FP4,
INT4) need `typename T::data_handle_type` (a `uchar` pointer) and contiguous tiles.

### types_rev (the load-type set in reverse order)

Confirms `types`; the GPU's slowdowns vary with the ANE's heat (a warmer ANE asks for less bandwidth): FP16 matrix
loop next to the FFN +45% forward, +23% reverse; low-reuse INT8 +51 / +47%; memory stream +76 / +64%; cache-resident
loads +0.5 to +3% both ways. `coupling/c18/types_rev/`.

### ANE on-chip memory (`coupling/anecache/`, `power/ane_cache_probe.py`)

DRAM bytes per call (`anecache.txt`; per cluster and link in the article's result 6): a 0.5 MB weight reused by 8 ops
is read about twice per call (kept on chip), a 1 MB one every op (8 x W); 2 to 16 MB weights once per op at 64 rows,
about twice per op at 512 rows; 8 distinct weight sets read once each (2.07 / 4.02 / 7.28 / 16.19 MB for 2 / 4 / 8 /
16 MB); intermediate activations never reach DRAM up to 4-8 MB. Traffic always split evenly over the two clusters
(four links) in these probes.

### Real V8 prefill telemetry (`coupling/prefill1/`, `power/prefill_tel.py`)

Cold requests to the served build (forge 8765, dec8), a different doc slice each, 60 s idle before each:

| prompt | prefill (server log) | GPU active | ANE W | DRAM GPU / ANE | DRAM total | GPU latency | notes |
|---|---|---|---|---|---|---|---|
| 2K | 204 tok/s | 32% | 5.0 | 45 / 24 | | 329 | first request after the restart: warm-up, not valid |
| 8K | 658 tok/s | 98% | 14.9 | 51 / 72 | 114 (73% of 155) | 340 | |
| 32K | 537 tok/s | 96-100% | 14.3 -> 7.5 | 51 / 70 -> 68 / 46 | 118 (76%) | 341 -> 258 | |
| 64K | 420 tok/s | 95-100% | 12.8 -> 5.4 | 50 / 69 -> 75 / 31 | 114 (74%) | 336 -> 252 | GPU 1664-1690 MHz after 50 s (PMULOOP 49%), 425 ms throttle events |

ANE DRAM traffic / 100 MB per token = the prefill rate (32K: 544 vs 537; 64K: 444 vs 420), so the ANE's decline
within a request follows the GPU-bound pipeline's rate. ANE cluster balance (ANE1 / ANE0 DRAM): 0.909-0.918 in every
prefill, 0.90-0.91 in every FFN loop, 0.999 for `hot` and the cache probes: the served FFN (chunks 8192 + 9216)
loads the second cluster about 9% less.

### GPU compute types at prefill shapes, GPU alone (`power/gpu_wa_gemm.swift`, weights streamed from DRAM)

| compute | 128 tokens: TOPS, weight GB/s | 64 tokens: TOPS, weight GB/s |
|---|---|---|
| W8A8 (INT8 x INT8) | 32.2, 126 | 16.7, 130 (DRAM-bound) |
| W4A8 (INT4 weights, INT8 compute) | 27.8, 54 | 31.1, 121 |
| WF8AF8 (FP8 x FP8) | 32.0, 125 | |
| W16A16 / BF16 | 12.7 / 12.6, 99 | 8.9, 139 (DRAM-bound) |
| W8A16 / W4A16 / WF8A16 / WF4A16 | 18.0 / 18.6 / 19.8 / 18.3 | |
| FP4 x FP4 | 16.8 | |

### Other files

- `datatype_inventory.md`: operand types of every step of the served prefill (from the code): GPU projections
  WL4A8 (INT8 x INT8), Q.K INT8, P.V FP16, low-rank FP16; ANE FFN WL3A16 (FP16 x FP16), about 71% of the work.
- [../../pipe/unit/power/IOREPORT_COUNTERS.md](../../pipe/unit/power/IOREPORT_COUNTERS.md): the IOReport counter
  reference (GPU, ANE, DRAM, SoC), decoding, what each read here, what is missing on this OS.

### Counter calibration (`coupling/cal/`, `power/cal_analyze.py`)

Known bytes at controlled rates (`cal.txt`): ANE `AFI DCS` = expected within 7% at 18-68 GB/s (0.93-1.01), about +2
GB/s at low rates (its 0-4 GB/s bin counts as 2); AMCC total tracks the average up to its 128 GB/s clip; the GPU's
`DCS AGX` reports the rate while the GPU is active (a 1 GiB stream at duty 0.75 / 0.5 / 0.25 reads 104-105 GB/s every
time), right for continuous work (INT8 matmul 0.93-1.03); the per-ANE-link histograms are right only for continuous
ANE work (0.30-0.36 with sleeps between calls). No constant scale factor.

### Compute types under contention (`coupling/wa/`)

GPU `gpu_wa_gemm` at 128 tokens (W16A16, W8A8, W4A8, W4A16, FP8 x FP8) against layer 36's FFN on the ANE (17 Nov
LUT3 packages: WL3A16 `dec_fp16_L36`, WL3A8 `dec_i8gud_L36`, entries r64 / r512, chain of 4). ANE / GPU slowdown vs
adjacent alone phases (`summary.txt`; the WL3A16 r512 entry compiled on its first call, 335 s, so its first pair is
invalid):

| ANE FFN | W16A16 | W8A8 | FP8 x FP8 | W4A8 | W4A16 |
|---|---|---|---|---|---|
| WL3A16 r64 | +7.5 / +62% | +3.0 / +74% | +2.3 / +54% | +4.0 / +10.5% | +1.6 / +4% |
| WL3A8 r64 | +36 / +56% | +31 / +70% | +24 / +68% | +11 / +20% | +4.6 / +11% |
| WL3A16 r512 | invalid | +2.0 / +34% | +3.2 / +49% | +2.6 / +8% | +0.7 / +4% |
| WL3A8 r512 | +3.1 / +51% | +1.9 / +68% | +0.6 / +67% | +4.2 / +13% | +0.3 / +8% |

### V8 benchmark under full telemetry (`coupling/v8bench/`, `power/bench_tel.py` around `srv_bench.py`)

Cold requests to the served build (2K twice, 8K, 16K, 32K, 64K; 256 decode tokens each; 60 s idle between;
`bench_tel.txt` per request: prefill and decode windows, prefill in 10 s segments; `server_log_excerpt.log`):

| prompt | prefill tok/s (client) | decode tok/s | decode ms / cycle (draft, verify) | decode: ANE DRAM, GPU DRAM (AMCC-based), ANE W, cluster balance |
|---|---|---|---|---|
| 2K | 553 / 562 | 35.2 / 31.4 | 109 (13.3, 94.5) | 104 GB/s, 0, 6.8 W, 0.868 |
| 8K | 650 | 28.8 | 113 (13.1, 98.7) | 103, 0, 6.8, 0.867 |
| 16K | 613 | 32.0 | | 101, 0, 6.8, 0.876 |
| 32K | 535 | 23.0 | | 98, 0, 6.6, 0.883 |
| 64K | 413 | 22.4 | verify 133.7 (GPU attention 28.4) | 75, about 16 (GPU 24% active), 5.3, 0.881 |

Decode moves at least 10.6 GB per cycle (the drafter's 0.7 GB + the target's 9.9 GB of weights), 98-104 GB/s on
average with 58% of the ANE counter's samples in its 128 GB/s top bin (the ANE streams faster than 128 GB/s while
busy; the cycle's draft, sampling and call gaps hold the average down), at only 6.8 W of ANE power. The second ANE cluster moves 12-13% less than the first in decode (prefill: 9%). Prefill
from 8K up: GPU 97-100% active, DRAM 119-124 GB/s (77-80%); in the 32K and 64K requests the GPU clock falls to
1618-1712 MHz after about 50 s (`PMULOOP` 37-99%).

### Traced V8 prefill (`coupling/trace1/`, `power/v8_trace.py` + `power/v8_engine_trace.py`, `power/trace_analyze.py`)

A copy of the server's V8 engine that records every GPU job (CPU submit, GPU start / end) and every ANE call; the
served build (dec8, srv3, w_dec8 tables) with the server's environment, the srv_bench prompt text, 45 s idle between
runs, forge stopped. `trace_summary.txt`:

| prompt | rate | GPU busy | ANE busy | both / GPU only / ANE only | GPU ms / 128 rows: DeltaNet; attention first to last tenth |
|---|---|---|---|---|---|
| 2K (x2) | 687 / 696 tok/s | 89% | 99% | 88 / 0.6 / 11% | 2.58; 2.4 -> 3.0 |
| 8K | 684 | 96% | 95% | 92 / 4.7 / 3.4% | 2.58; 2.5 -> 4.8 |
| 32K | 543 | 99% | 79% | 78 / 21 / 0.9% | 2.57; 3.0 -> 10.7 |
| 64K | 420 | 99.5% | 64% | 63 / 36 / 0.5% | 2.54; 3.5 -> 18.8 |

ANE hardware time per FFN pair at 2K: 2.88 ms (99% busy x 2.98 s / 1024 calls) vs 2.42 alone and cool. DeltaNet alone
(`gpu_mix_win.py gdn` / `gdnall`, srv3 + w_dec8): 1.94 ms looping layer 0, 2.12 ms cycling all 48 DeltaNet layers.

Time model from this trace (`power/prefill_model.py trace1`, `trace1/prefill_model.txt`): attention per 128-row call
2.95 ms + 0.255 ms per 1K keys of position, DeltaNet 2.57 ms, ANE 184 ms per 128 rows. It gives 683 / 539 / 420 tok/s
at 8K / 32K / 64K (traced: 684 / 543 / 420), and 350 at a 95K prompt, against 347-358 measured on 16-17 Nov (not in
the fit). With an attention kernel at its compute floor it gives 64K 514-585 and 95K 443-523 (see
`ANALYSIS_V8_M6.md`, "A time model of the prefill, and 96K").

Server against engine (`coupling/v8bench/server_log_excerpt.log`): from 8K up the server's V8 call matches the traced
engine within 1% per row. At 2K it takes 3.41 s for 2,176 rows (2,087 prompt + 89 pad), against 3.13 s at the traced
rate, plus 173 ms of hand-over / drafter context and 50 ms of setup, about 0.5 s outside the pipeline's throughput (not
yet explained).

### Attention-core variants at 60K vs the FFN (`coupling/attn_*`)

Served engine srv3 (w_dec8 tables, FP16 KV path, QK8, TKV 2048), each variant in its own set against `ffn2_L0`:
unfused 16.8 ms alone (1.4 GB per call) / 24.1 with the FFN (+43%, FFN +16.7%); `FA2=16_32_mm` 24.8 (2.0 GB) / 48.0
(+92%, FFN +4.6%); `FLASH=1` 30.8 (3.0 GB) / 44.9 (+46%, FFN +2.2%); `FA2=16_32` 35.5 (2.3 GB) / 40.9 (+15%, FFN +3.0%);
`FA2=s_16_16_8_mm` 53.0 (1.7 GB) / 54.3 (+2.4%, FFN +2.3%). `attn_fa2_64` is invalid (a GPU check ran during it by
mistake: `INVALID.txt`); its alone check: 42.2 ms.

### Long decode (`coupling/longdec/`, `power/long_decode.py` under `bench_tel.py`)

2,500 tokens from an 81-token prompt: 30.2 tok/s (24-35 per 10 s, the drafter's acceptance), 109 ms per cycle (draft
13.5, verify 94.5), 3.29 tokens per verify; ANE 6.88-6.95 W, ANE DRAM 103.4-103.9 GB/s, clusters 52.8 / 45.8 GB/s
(balance 0.868) in every 10 s window, 74 -> 88 C: no throttling.

### GPU weight streaming at decode-like batches (`power/gpu_wa_gemm.swift`, GPU alone)

16-32 tokens per call, weights from DRAM: W4A16 135-139 GB/s, W8A16 134, W4A8 138-145, W8A8 141, W16A16 144: about
90% of the DRAM peak in every format, against the ANE's ceiling of about 104 GB/s.

### Attention kernel profile at 60K (`coupling/attn60K_profile.log`, `pipe/unit/gdn_prof.py` with PROFILE=1)

srv3 engine, standalone FP16 K/V path, layer 3 at position 61440, each dispatch in its own command buffer (shares
only): P.V 39%, Q.K 31%, softmax 15.7%, combine 3.3%, projections and the rest about 9%.

### WL3A8 at 512 rows vs a host reference (`coupling/check512/`, `w8a8/ref512.py`, `w8a8/check512.py`)

CPU references (about 10 min each) of the served FFN and the LUT3 FP16 / INT8 modules on the 17 Nov inputs, written to
a new `a8probe/lut3_rows/ref512.npz`; the ANE packages' r512 entries (four chained FFN passes): WL3A16 0.131% from the
served FFN (and from its own reference), WL3A8 1.159% from the served FFN, 0.893% from its own reference. Correct.

### Softmax fold on the prefill attention (`fold/`)

A prototype engine (`libgpu_v8_fold.dylib`: srv3 + `FOLD`, `FOLD_MIN`): exact (output within fp16 rounding of plain).
It is 3-5% slower alone from 8K keys up (the folded P.V costs 69 ms against 65 for P.V + softmax at 60K). Next to the
FFN it is 2-4% faster, and the FFN is slowed +3 to +6% instead of +7 to +14%. In the traced V8 prefill it is -1% to +1%
at 2K-95K, single runs. Details and the per-decile table: `fold/README.md`.


## GPU clock for the decode split's short attention calls (`gpu_clock/`)

Between the split verify's ~5 ms ANE calls, the GPU idles and its power controller keeps the 16 attention calls at a
low clock: 502 MHz GPU-only, ~860 MHz in the server.
- **Fix:** keep one simdgroup spinning on a second command queue (20 us serial dispatches, 250 per command buffer)
  while the decode runs. Pre-warms, periodic micro-dispatches, low-duty bursts, event-gated commits, QoS,
  latencyCritical, and the private deadline / priority calls do not work: the controller needs 10-20 ms of continuous
  busy time.
- **GPU-only:** 3.21 to 1.06 ms per call at 64K, 4.47 to 1.54 at 95K, with no slowdown of the attention kernel.
- **Server, 66-70K tokens, 4 on / off pairs:** split verify step 138.2 to 122.4 ms, decode cycle 152.5 to 137 ms;
  ANE side unchanged; +2.1 W combined.
- **Integration:** prototype dylib `pipe/engine/dec_gpu_clk.swift` (`DEC_KA_HOLD_MS`). Details, all tables and the 95K
  server check: `gpu_clock/README.md`.

### FP8 P.V on the prefill attention (`f8/`)

`libgpu_v8_f8.dylib` (`F8=c|v|w|x`): e4m3 probabilities x an FP8 V copy, FP8 x FP8 matmul2d. P.V is 2x faster, and
attention per call is 12-15% faster alone and 11-13% faster next to the FFN. In the traced V8 prefill (interleaved
repeats), `F8=x` gives +4.5% at 32K, +8.2% at 64K, +9.4% at 95K, and noise at 2K-8K. Last-token KL to plain is 5e-6 to 4.6e-3, with
the same argmax and top-10. A teacher-forced KL is still needed. Details: `f8/README.md`.

## Evening additions (18 Nov 2026)

- `f8/README.md`, section "Teacher-forced, every position" (`f8/tf_kl/`): FP8 x vs plain over all 32,768 / 95,232
  positions: KL mean 1.2e-3 / 9.9e-4, p99 6.8e-3 / 7.7e-3, top-1 98.7%, top-10 overlap 9.8, NLL +0.07%.
- `gpu_clock/README.md`, section "Deployed in the server" (`gpu_clock/e2e_deploy/`): keep-alive in dec_split
  (libdec_gpu_clk, 60 ms hold), 4 on / 4 off at 66-70K: verify 136.1 -> 119.1 ms, cycle 150 -> 133 ms, +2.66 W.
- `f8/README.md`, section "95K repeat, 4K / 16K / 48K, fill path, and serving": 95K +9.9% (repeat), 4K +0.3%,
  16K +2.4%, 48K +6.2%; the FP8 refill path checked; FP8 x served on 8765 from 20:27 (v8_server_dec8.sh defaults).
- `f8/README.md`, section "F8=x with the softmax folded into P.V": correct, 7-9% slower alone at 32K+, -1.5% end to end at 64K vs x; not adopted.
- `splash_cmp/README.md`: Splash 1.6.1 (ANE split, UD-IQ3_XXS, Oct 4 model) vs forge (FP8 x + keep-alive), 8K-64K: forge prefill +26 to +42%, decode +1 to +13% at ~24 W vs ~42 W.
