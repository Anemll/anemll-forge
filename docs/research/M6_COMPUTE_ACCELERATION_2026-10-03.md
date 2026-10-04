# M6 ANE compute acceleration: DeltaNet core and attention tiles

- **Date:** 3 October 2026, Apple M6 (`Mac18,5`), macOS 27.0.1 (26A434), Core AI with the Swift bridge, bonded compile mode 2.
- **Starting point:** [HANDOFF_M6_ANE_COMPUTE.md](HANDOFF_M6_ANE_COMPUTE.md). That work modeled bytes on a Linux VM and asked for the attention share to be measured on an M6 first.
- **Status:** measured on device. Two exact graph changes speed up the full Core AI target without changing the model's arithmetic beyond FP16 rounding order. The builder now defaults to them: V8 KV cache only, `GDN_FAST=1`, `ATT_BLOCK=2048`, `ATT_BLOCK_PREFILL=4096` (see [Cold compile cost](#cold-compile-cost-measured)). `GDN_FAST=0 ATT_BLOCK=16384` rebuilds the release graph; `--kv-cache-dtype both` the selectable FP16 / V8 packages.

Labels: **Measured** (this session, on the M6), **Modeled** (arithmetic from measurements), **Prior** (earlier records in this repository).

## Answer

The target is not bound where the byte model assumed. On the M6 ANE, two context-dependent and context-independent pieces were far more expensive than their arithmetic:

1. **Gated DeltaNet core** (48 of 64 layers). Tiny arithmetic, but about a third of a 64-row prefill chunk call. Most of it was the row-by-row triangular solve. `GDN_FAST=1` replaces it with an exact Neumann-product inverse, writes the causal conv1d as one native depthwise conv, and merges two state matmuls per prefill sub-chunk.
2. **Full-attention history** (16 layers). Not bandwidth bound: at 64K it reads K/V at about 36 GB/s. Smaller score tiles fit the ANE better. `ATT_BLOCK=2048` (release 16384) cuts the attention core 14 to 38%.

With the builder defaults (V8 cache only, 2048-wide verify tiles, 4096-wide prefill tiles), the complete target verifies **19% faster at 8K to 31% at 64K** and runs a 64-row prefill call **29 to 35% faster** (225 to 316 rows/s at 8K, 106 to 162 at 64K). It compiles cold in about 23 minutes, like the release graph, and a 64K long-context check matches the release (perplexity 5.0686 to 5.0673). Everything stays on the ANE.

The first full candidate (both KV formats, 2048-wide tiles everywhere) carried the server and quality measurements: with DFlash2, cold prefill **+26% (8K) to +41% (64K)** tokens/s and decode **+21 to +25% at 32K to 64K**, with identical replies and acceptance at 48K and 64K; compiled KL-512 against BF16 unchanged (mean 0.18427 to 0.18417), direct KL between release and candidate 3.4e-5 nats. Its narrow prefill tiles made the cold compile 5 times longer, which `ATT_BLOCK_PREFILL=4096` removed without losing speed. Neither change is a 2x end-to-end. Details in [Full model](#full-model).

## Where the time goes (Measured)

### Context sweep of the release target

All 16 chunks plus the head chained as one plan, random inputs, release `_kvselect` export, V8 format ([`scripts/m6_entry_sweep.py`](../../scripts/m6_entry_sweep.py) `--chain`). Medians of 15 calls.

| Entry | Verify, 8 rows | Prefill, 64 rows |
| --- | ---: | ---: |
| 8K | 113.1 ms | 284.5 ms |
| 16K | 123.5 ms | 351.5 ms |
| 32K | 144.8 ms | 412.6 ms |
| 48K | 168.2 ms | 497.7 ms |
| 64K | 190.3 ms | 603.4 ms |

A least-squares fit `time = fixed + slope x context` gives verify **101.5 ms + 1.38 ms per 1K** and prefill **248 ms + 5.42 ms per 1K**. The context-dependent part (attention over history) is 10% / 18% / 31% / 47% of verify and 15% / 25% / 42% / 57% of prefill at 8K / 16K / 32K / 64K. The byte model in the handoff put the KV share of a 64K verify at 23%. Measured time is about twice that, because attention on the ANE is compute and op bound, not KV-bandwidth bound.

The same sweep with FP16 V gives verify 99.3 ms + 1.66 ms/K and prefill 215 ms + 8.59 ms/K.

### Inside one chunk

Chunk L00-03 (three GDN layers, one attention layer, 2-bit vector-LUT weights) at the 8K entry, rebuilt from the real export with one part removed at a time ([`scripts/m6_layer_ablation.py`](../../scripts/m6_layer_ablation.py)). Ablated programs are timing probes; their outputs are wrong by construction.

| Removed | Verify, 8 rows (5.25 ms) | Prefill, 64 rows (16.79 ms) |
| --- | ---: | ---: |
| MLP, four layers | -2.29 ms | -5.72 ms |
| GDN core, three layers | -0.88 ms | -5.61 ms |
| Attention core | -0.30 ms | -1.88 ms |
| Hadamard rotations | -0.30 ms | -0.44 ms |
| Low-rank FP16 corrections | +0.16 ms (noise) | -0.10 ms |
| GDN core, attention core and MLP (left: projections, norms, call) | 1.44 ms remain | 3.15 ms remain |

The MLP at 64 rows runs at about 24 TFLOPS of FP16 work, close to compute bound. The GDN core costs as much as an MLP layer at 64 rows with almost no arithmetic: it was op and latency bound.

An Instruments `Core AI` trace (`xctrace`) shows per-call ANE intervals (4.9 to 5.0 ms for this verify call, 0.3 ms below wall time) but no per-op split. anemll-profile on a Core ML conversion of the GDN core ranked layout transposes first. A direct test (all large GDN transposes replaced by reshapes) saved only 3% at 64 rows and was slower at 8 rows, so that cost-model ranking does not hold for the Core AI compile.

## Gated DeltaNet core (Measured)

[`scripts/m6_gdn_bench.py`](../../scripts/m6_gdn_bench.py): three GDN cores with real conv taps, `A_log`, `dt_bias` and norm weights, between their projections. `ref` is the builder's own `GDNW` code. Every variant matched `ref` on the host in FP32 (largest relative difference 3e-7) before timing; the ANE error column is relative RMSE of the layer output against FP32.

| Variant, 3 layers | Placement | Verify, T=8 | Prefill, T=64 | Output error |
| --- | --- | ---: | ---: | --- |
| Release (`fwd_sub` row by row) | ANE | 1.18 ms | 6.82 ms | 3.5e-3 / 2.8e-3 |
| Neumann inverse as broadcast multiply-reduce, one matmul with the right-hand side | ANE | 0.64 ms | 4.15 ms | same |
| plus native depthwise conv1d | ANE | 0.66 ms | 3.75 ms | same |
| plus one state matmul per prefill sub-chunk (**`GDN_FAST=1`**) | ANE | **0.64 to 0.69 ms** | **3.42 to 3.49 ms** | same |
| Neumann inverse as a chain of matmuls | ANE compile fails, GPU | 1.16 ms | 2.53 ms | rejected |
| Sub-chunks of 16 / 32 / 64 rows (with the new inverse) | ANE | | 4.26 / 7.03 / 15.08 ms | slower |
| All large layout transposes as reshapes (timing only) | ANE | 1.30 ms | 4.02 ms | not a gain |

The target uses the chunkwise form of the gated delta rule: within each 8-row block a WY representation whose UT transform needs `T = (I - A)^-1` for a strictly lower-triangular `A`, followed by one state update per block. [Songlin Yang, "DeltaNet Explained (Part II)"](https://sustcsonglin.github.io/blog/2024/deltanet-2/) derives this form, computes `T` by forward substitution (the release graph's `fwd_sub`), and reads `(I - A)^-1` as a path sum: entry `[i, j]` sums the weights of all paths from `j` to `i`. In an 8-row block no path is longer than 7, so the sum ends at `A^7`, and grouping path lengths by binary digits gives `(I + A)(I + A^2)(I + A^4)`. In the builder's notation the solve is `(I + N) X = rhs` with `N = -A` (8 x 8 per head and sub-chunk), so `N^8 = 0` and `(I + N)^-1 = (I - N)(I + N^2)(I + N^4)` exactly. Written with matmuls between computed tensors, ANEC aborts with an internal error and the program falls back to the GPU (the earlier "doubling inverse" failure). Written as broadcast multiply plus reduce for the 8 x 8 products, it stays on the ANE.

After `GDN_FAST`, the remaining GDN prefill time per layer is roughly: sequential state loop 0.4 ms, triangular solve 0.3 ms, conv1d 0.13 ms, the rest 0.4 ms. Pending-row commit and the gated norm are negligible.

### Why the gain comes from the 8-row block

The triangular solve exists only because several tokens are processed as one block: inside a block of `T` tokens, every token's delta-rule update depends on all earlier ones, and the chunkwise form resolves that dependency with the `T x T` UT-transform solve. The release solve takes `T - 1` dependent row updates; the closed form takes `log2(T)` product steps and one matmul. The larger the block, the more serial work the rewrite removes:

| Call | Block structure | Release solve | `GDN_FAST` solve | GDN core, 3 layers |
| --- | --- | --- | --- | --- |
| Plain decode, 1 token | one recurrent step | none | none | no change (not the release path) |
| 3- or 4-row verify | 3 or 4 rows | 2 or 3 serial steps | 1 or 2 product steps | not measured separately |
| 8-row verify (release) | 8 rows | 7 serial steps | 3 product steps | 1.18 to 0.64 ms (-45%) |
| 64-row prefill | 8 sub-blocks of 8 rows, solved together | 7 serial steps | 3 product steps | 6.82 to 3.42 ms (-49%) |

At 8 rows the release solve was more than half of the GDN verify core: dropping it entirely (timing only) brought the three layers from 1.18 to 0.55 ms. The depthwise conv1d helps any block size, mostly prefill; the merged state matmul applies only to prefill, which chains its 8 sub-blocks within a call.

This is also why the 8-row verifier now costs about the same as a 3- or 4-row one ([../verifier_len.md](../verifier_len.md)): after the rewrite, the extra rows add little GDN work. On the default build at 8K the full verify takes 91.5 ms for 8 rows against 94.5 to 95.3 ms for 3 or 4 rows, so the 8-row verifier gets its extra drafts almost free. Before it, the solve's serial steps grew with the block (7 at 8 rows against 3 at 4 rows); that difference was not measured on full builds. A rough estimate from the 3-layer timing (0.63 ms for the 7-step solve) puts the extra cost of 8 rows over 4 in the release graph at about 6 ms per verify across the 48 GDN layers.

Unit test: [tests/test_gdn_fast.py](../../tests/test_gdn_fast.py) checks the inverse, release versus fast graphs, and that verify blocks of 8, 4 and 3 rows with partial acceptances reproduce a token-by-token gated delta rule in FP64.

## Attention history tiles (Measured)

[`scripts/m6_attn_bench.py`](../../scripts/m6_attn_bench.py): one attention core (real q/k norm weights) over V8 history, `ref` being the builder's `AttnW.forward`. Relative RMSE against FP32 is 1.9e-3 to 2.2e-3 for every exact variant.

| History | Verify T=8, 16K tiles to 2K tiles | Prefill T=64, 16K tiles to 2K tiles |
| --- | --- | --- |
| 8K | 0.713 to 0.614 ms (-14%) | 2.756 to 2.211 ms (-20%) |
| 16K | 1.421 to 0.920 ms (-35%) | 5.296 to 3.613 ms (-32%) |
| 32K | 2.624 to 1.638 ms (-38%) | 10.28 to 6.88 ms (-33%) |
| 64K | 5.22 to 3.47 ms (-34%) | 22.47 to 16.18 ms (-28%) |

At 64K: 8K tiles -15 to -17%, 4K tiles -13 to -26%, 1K tiles slower again, 32K tiles slower than release. Not useful at 64K: folding `hd^-0.5` into q (no change), denominators as a matmul (+10% at T=64), dropping the mask add or the running max (timing-only bounds, -2 to -12%). Dequantizing V with its per-token scale inside the program falls back to the GPU.

**Cost:** first-load (cold) ANE compilation grows with the number of tiles. Chunk L00-03 with all 20 entries took 425 s to specialize versus about 75 s for the release chunk. It is a one-time cost per OS build and cache. Wired memory did not grow (1.07 GB versus 1.20 GB added for the chunk's entries).

## One real chunk, release versus candidate (Measured)

Chunk L00-03 rebuilt with `GDN_FAST=1 ATT_BLOCK=2048`, all 20 entries, compared with the release chunk on identical random inputs ([`scripts/m6_chunk_ab.py`](../../scripts/m6_chunk_ab.py), rounds interleaved, timed while other jobs used the CPU):

| Entry | V8 verify | V8 prefill | FP16-V verify | FP16-V prefill |
| --- | ---: | ---: | ---: | ---: |
| 8K | -11.6% | -21.9% | -12.1% | -20.5% |
| 16K | -10.2% | -22.5% | -23.0% | -29.8% |
| 32K | -19.3% | -24.3% | -27.1% | -29.2% |
| 48K | -22.9% | -21.1% | -34.5% | -30.7% |
| 64K | -23.1% | -25.5% | -33.8% | -34.0% |

Every output stayed finite and within relative RMSE 2e-3 of the release chunk (the release chunk itself is about 3e-3 from FP32). With 2K tiles the FP16-V entries run as fast as V8 at the chunk level (64K verify 7.50 versus 7.57 ms).

## Full model

Candidate: the complete 16-chunk selectable export rebuilt with `GDN_FAST=1 ATT_BLOCK=2048` from the same quantized export (`mix25in_mixr_lr64mix`), both KV formats, 8K to 64K; the release head package is reused unchanged.

### Placement (Measured)

`coreai/inspect_coreai_cache.py --strict` on the cached specializations: all 16 chunks and the head fully on the ANE with no GPU regions, bonded compile mode 2, for both the V8 and FP16-V functions.

### Verify and prefill forward (Measured)

All chunks and the head chained as one plan (`m6_entry_sweep.py --chain`), random inputs, medians of 15 calls. The candidate sweep ran while other packages were compiling on the CPU, so its numbers are if anything pessimistic.

V8 cache (the release default):

| Entry | Verify, release to candidate | Prefill call, release to candidate | Prefill rows/s |
| --- | --- | --- | --- |
| 8K | 113.1 to 99.4 ms (-12.1%) | 284.5 to 240.0 ms (-15.6%) | 225 to 267 |
| 16K | 123.5 to 104.2 ms (-15.6%) | 351.5 to 272.7 ms (-22.4%) | 182 to 235 |
| 32K | 144.8 to 115.4 ms (-20.4%) | 412.6 to 353.5 ms (-14.3%) | 155 to 181 |
| 48K | 168.2 to 126.9 ms (-24.6%) | 497.7 to 385.9 ms (-22.5%) | 129 to 166 |
| 64K | 190.3 to 140.2 ms (-26.3%) | 603.4 to 435.3 ms (-27.9%) | 106 to 147 |

Fits: verify `101.5 ms + 1.385 ms/K` to `92.8 ms + 0.727 ms/K`; prefill `248 ms + 5.42 ms/K` to `221 ms + 3.47 ms/K`. The context slope (attention history) fell 48% in verify and 36% in prefill; the fixed part (mostly the GDN core) fell 9 to 11%.

FP16 V: verify -12.1 / -17.3 / -21.9 / -22.3 / -27.1% and prefill -18.5 / -27.8 / -29.7 / -29.4 / -33.1% at 8K / 16K / 32K / 48K / 64K (prefill 217 to 266 rows/s at 8K, 80 to 119 at 64K). With the new tiles, FP16 V prefill is as fast as or faster than V8 up to 32K, while V8 keeps a verify advantage at 48K and 64K (127 versus 140 ms, 140 versus 149 ms).

### Full server with DFlash2 (Measured)

[`scripts/m6_server_bench.py`](../../scripts/m6_server_bench.py): one owned server per build and context, the V8 study's decode fixture (public synthetic testing notes plus a small Python task), greedy, thinking off, 256-token cap, the tested Core AI DFlash2 drafter with a 3 ms draft gap, V8 cache, bonded mode 2. A tiny warmup, then one cold prompt (its prefill is the prefill figure; the timer covers all target layers, head, host cache writes, context growth and drafter ingestion) and three cached requests (median decode). Identical prompt token IDs in both builds. Every case passed the strict cached-graph audit and within-build reply equality.

| Entry | Prompt tokens | Cold prefill, release to candidate | Decode, release to candidate | Verify per cycle | Draft acceptance | Same reply |
| --- | ---: | --- | --- | --- | --- | --- |
| 8K | 7,673 | 222.7 to 279.8 tok/s (+25.6%) | 55.77 to 55.12 tok/s (-1.2%) | 109.0 to 97.8 ms (-10.3%) | 88.5% to 79.3% | no |
| 16K | 15,865 | 207.5 to 266.5 (+28.4%) | 53.49 to 57.64 (+7.8%) | 118.2 to 104.5 ms (-11.6%) | 91.4% to 88.5% | yes |
| 32K | 32,249 | 177.6 to 230.9 (+30.0%) | 44.68 to 54.12 (+21.1%) | 141.1 to 116.5 ms (-17.4%) | 88.5% to 91.4% | yes |
| 48K | 48,633 | 152.0 to 207.2 (+36.3%) | 38.59 to 47.27 (+22.5%) | 160.9 to 127.8 ms (-20.6%) | 84.6% both | yes |
| 64K | 64,953 | 132.2 to 185.8 (+40.5%) | 36.53 to 45.78 (+25.3%) | 182.9 to 141.2 ms (-22.8%) | 91.4% both | yes |

The release 8K row reproduces the V8 study (55.37 tok/s and 88.49% acceptance there). At 48K and 64K acceptance and tokens per cycle are identical between builds, so those decode gains are the speedup itself. At 8K the greedy reply diverged (both valid answers), acceptance fell, and the faster cycle (127.5 to 116.6 ms) did not show in tokens per second. At 16K and 32K the replies are identical but acceptance still shifts a few points: the drafter reads target features that differ at FP16 rounding level. A 64,953-token cold prompt now prefills in 350 s instead of 492 s. Server prefill gains exceed the chain-sweep gains because a cold prompt walks the whole context ladder.

Scope: one synthetic workload per context, one cold prefill and three cached repeats; repeat ranges within a build were 0.1 to 0.5 tok/s. Not a general performance estimate.

### Quality (Measured)

Compiled KL-512 on the V8 study's 64-sequence BF16 trace ([`scripts/m6_kl512_eval.py`](../../scripts/m6_kl512_eval.py), 8K verify entry, T=8 teacher forcing, 40,023 positions, same reference file `84d25d99...`). Rerunning the release export with this harness reproduced the V8 study's numbers bit for bit.

| Metric | Release (V8) | Candidate (V8) |
| --- | ---: | ---: |
| Mean KL to BF16 | 0.184270 | 0.184168 |
| Median KL | 0.030238 | 0.030266 |
| p99 KL | 2.274433 | 2.261292 |
| Top-1 agreement with BF16 | 86.0205% | 86.0280% |
| Trace perplexity | 2.412098 | 2.413053 |

Direct KL(release || candidate) on the same top-512 partition plus tail: mean **3.39e-5**, median 5.5e-6, p99 3.1e-4, max 0.029 nats; the top token agrees at 99.82% of positions. For scale, the V8 study measured a mean of 7.37e-5 between the FP16-V and V8 caches. The change is FP16 rounding order, not a quality shift.

This trace covers the verify path on sequences up to 857 tokens. Long context, through the prefill path ([`scripts/m6_long_ctx_eval.py`](../../scripts/m6_long_ctx_eval.py)): the first 64,000 tokens of the public WikiText-2 test text prefilled through the whole context ladder (64-row prefill entries, KV growth 8K to 64K), then the next 1,024 tokens teacher-forced in 8-row verify calls at positions 64,000 to 65,024 in the 64K entry.

| Metric | Release (V8) | Candidate (V8) |
| --- | ---: | ---: |
| Perplexity on the 1,024 positions | 5.0686 | 5.0696 |
| Prefill, 64,000 tokens through the ladder | 495 s (129 tok/s) | 334 s (192 tok/s) |

Direct full-vocabulary KL(release || candidate) at those positions: mean 5.2e-5, median 1.9e-5, p99 4.7e-4, max 3.3e-3 nats; the top token agrees at 99.3% of positions. The long-context state built by the new prefill path matches the release to FP16 rounding.

### Cold compile cost (Measured)

The first load of each package specializes it for the ANE. With 2K tiles the candidate took about 6.2 minutes of compiler throughput per chunk (16 chunks: about 1 h 40 min, measured while other builds also used the CPU), versus about 21 minutes for the whole release export (Prior, 20.6 minutes for the V8 export). Running several packages at once does not speed this up: the system compiler's total throughput stayed the same. It is a one-time cost per OS build and cache; later loads take 1 to 3 seconds.

Tile width trades that cost against speed. Chunk L00-03 with `GDN_FAST=1`, all 20 entries, cold first load on an otherwise idle machine, and the speed change against the release chunk (V8 entries, interleaved A/B):

| `ATT_BLOCK` | Cold compile, one chunk | Verify 8K / 16K / 32K / 48K / 64K | Prefill 8K / 16K / 32K / 48K / 64K |
| --- | ---: | --- | --- |
| 16384 (release) | about 77 s (Prior) | baseline | baseline |
| 8192 | 95 s | -9.9 / -11.4 / -12.1 / -12.4 / -13.7% | -16.0 / -17.3 / -17.4 / -17.2 / -16.0% |
| 4096 | 159 s | -10.2 / -12.4 / -13.1 / -14.8 / -13.1% | -18.7 / -21.5 / -23.5 / -24.5 / -25.6% |
| 2048 | about 425 s | -12.2 / -16.6 / -21.0 / -23.7 / -24.5% | -20.6 / -22.1 / -22.9 / -23.9 / -26.0% |

`ATT_BLOCK=4096` keeps nearly all of the prefill gain at about twice the release compile time; 2048 adds about ten points of long-context verify for about 5.5 times the compile.

**Where the compile time goes, and the fix (Measured).** Deleting a package's Core AI cache entry only redoes the MPSGraph stage: the system ANE service also caches compiled programs it has seen, so true cold times were measured on packages made new with a negligible constant change (`GDN_SQ=16.001`). One chunk (L00-03, `GDN_FAST=1` except the release rows), cold first load, speed against the release chunk on V8 entries:

| Configuration | KV formats | Tiles, verify / prefill | Cold compile | Full model (x16) | Verify 8K / 64K | Prefill 8K / 64K |
| --- | --- | --- | ---: | ---: | --- | --- |
| Release | both | 16384 / 16384 | 82.9 s | about 22 min | baseline | baseline |
| Release | V8 only | 16384 / 16384 | 45.9 s | about 12 min | baseline | baseline |
| Candidate above | both | 2048 / 2048 | 407.6 s | about 1 h 50 min | -11.6 / -24.6% | -20.6 / -26.3% |
| `ATT_BLOCK_PREFILL=4096` | both | 2048 / 4096 | 135.9 s | about 36 min | -12.2 / -24.5% | -19.7 / -25.1% |
| V8 only | V8 | 2048 / 2048 | 243.8 s | about 65 min | -11.8 / -26.0% | -20.5 / -27.4% |
| V8 only, `ATT_BLOCK_PREFILL=4096` | V8 | 2048 / 4096 | 90.3 s | about 24 min | -12.4 / -26.6% | -20.9 / -26.1% |
| V8 only, 4096 everywhere | V8 | 4096 / 4096 | 80.7 s | about 22 min | -10.0 / -16.9% | -20.2 / -26.0% |

About three quarters of the candidate's compile time came from the 64-row prefill entries with 2048-wide tiles, which run no faster than with 4096. `ATT_BLOCK_PREFILL` (builder switch, default `ATT_BLOCK`; recorded in the manifest and the server's startup line) sets the prefill tile width separately: `ATT_BLOCK=2048 ATT_BLOCK_PREFILL=4096` keeps the full speed of the candidate at 1.6 times the release compile with both KV formats, or about the release compile with V8 only.

**The builder default (Measured).** A plain `qwen38_coreai_build.py all` now produces V8-only packages with `GDN_FAST=1 ATT_BLOCK=2048 ATT_BLOCK_PREFILL=4096`. The full 16-chunk target built that way (23 minutes) compiled cold through `python forge.py compile` in **22 min 48 s** (head and drafter already cached), against about 22 minutes for the release graph. Full chain on an idle machine, release (both formats, V8 functions) to the default build:

| Entry | Verify | Prefill call | Prefill rows/s |
| --- | --- | --- | --- |
| 8K | 113.1 to 91.5 ms (-19.1%) | 284.5 to 202.8 ms (-28.7%) | 225 to 316 |
| 16K | 123.5 to 97.4 ms (-21.1%) | 351.5 to 230.9 ms (-34.3%) | 182 to 277 |
| 32K | 144.8 to 108.1 ms (-25.4%) | 412.6 to 285.1 ms (-30.9%) | 155 to 225 |
| 48K | 168.2 to 119.5 ms (-28.9%) | 497.7 to 338.1 ms (-32.1%) | 129 to 189 |
| 64K | 190.3 to 131.5 ms (-30.9%) | 603.4 to 394.1 ms (-34.7%) | 106 to 162 |

Fits: verify `85.8 ms + 0.710 ms/K`, prefill `176 ms + 3.40 ms/K`. These beat the two-format candidate above; part of the difference is one format per package and part that the candidate's sweep ran while other packages compiled. Long-context check on the default build: perplexity 5.0673 (release 5.0686) on the 1,024 positions after a 64,000-token prefill, direct full-vocabulary KL to release mean 5.0e-5 (p99 3.7e-4), top token agreeing at 99.7% of positions; prefilling the 64,000 tokens through the ladder took 196 tokens/s against 129.

**First start guidance.** The runtime (and `python forge.py compile --build <dir>`) prints `[ANE compile]` lines from [`scripts/coreai_compile_guide.py`](../../scripts/coreai_compile_guide.py): how many packages are not yet compiled for this macOS build and Python, an estimate from the build's tile settings (fit to the table above, then re-scaled by measured package times), a heartbeat with the time left every 30 s, and only the build options that would compile faster. A cold package compiles in a worker thread, so Ctrl-C exits at once; finished packages stay cached and the next start resumes. `scripts/qwen38_server.sh start` shows these lines live and Ctrl-C there only stops watching.

Compile cost grows faster than linearly with the size of one program. The history attention has no weights, so a single shared attention program could serve all 16 attention layers: compiled alone, the complete 2048-tile history attention for all five contexts and both row counts took 21.0 s (3.5 s with 16384-wide tiles), against roughly 325 s per chunk when embedded. Splitting each chunk around a shared attention program would cost about 16 extra calls per forward (about 0.25 ms each) and a runtime restructure; it is not implemented. Writing the tiles as one batched op does not help: the ANE compiler expands it into the same per-tile work (16.4 s versus 16.2 s for one 64K layer) and it ran slower.

## Rejected or not pursued

- **Kronecker Hadamard** (`H_1024 = H_32 x H_32`, two 32 x 32 grouped convs and a channel transpose): exact but slower than the release 1-bit grouped conv (0.47 versus 0.39 ms for four 17408-channel rotations at T=8).
- **Core AI composite ops.** `coreai_torch` ships `GatedDeltaUpdate` and `SDPA` composites. The DeltaNet composite is an FP32 while-loop over tokens; Apple's authoring notes place native SDPA on the GPU path and recommend per-head attention on the ANE. Not tested further.
- **INT8 / FP8 attention MACs.** Not built. The measured breakdown shows the history path is op bound with large elementwise passes over scores (tile width alone moved it 14 to 38%), so a narrower MAC alone would not remove the dominant cost.
- **W8A8 for the MLP.** Closed for this model with the current toolchain (Measured). Four 1x1 convs 5120 to 17408 to 5120 (357M weights), Core AI, idle ANE:

  | Weights x activations | Placement | T=8 | T=64 |
  | --- | --- | ---: | ---: |
  | FP16 dense x FP16 | ANE | 6.23 ms | 7.26 ms |
  | INT8 dense x INT8 (coreai-opt `Quantizer`, per-channel weights, per-tensor activations) | ANE | 2.42 ms | 2.41 ms |
  | 2-bit vector LUT (release form) x FP16 | ANE | 0.81 ms | 2.03 ms |
  | 2-bit vector LUT with INT8 values x INT8 | ANEC compile fails, GPU | 26.0 ms | 26.5 ms |

  The only form that fits beside the LUT weights in 32 GB (LUT weights times INT8 activations) does not compile for the ANE. Dense W8A8 runs on the ANE but is weight-bandwidth bound even at 64 rows (about 147 GB/s) and slower than the release LUT path, which already runs the 64-row MLP at about 22.5 TFLOPS. A first attempt of this probe failed with `ANEProgramProcessRequestDirect() ... Request cancelled` while another job held the ANE; the rerun on an idle ANE is the table above.

## Reproduce

```sh
CAI=<Core AI conversion venv>/bin/python     # coreai-torch 0.4.2, coreai-opt 0.2.1, torch 2.11
python scripts/m6_entry_sweep.py --build <build> --format v8 --chunks all --chain
$CAI scripts/m6_gdn_bench.py check --variants ref,fast_b
$CAI scripts/m6_gdn_bench.py build --variants ref,fast_b --out DIR && $CAI scripts/m6_gdn_bench.py time --variants ref,fast_b --out DIR
$CAI scripts/m6_attn_bench.py build --variants ref,b2k --ctx 65472 --out DIR && $CAI scripts/m6_attn_bench.py time --variants ref,b2k --ctx 65472 --out DIR
EXPORT_DIR=<export> GDN_FAST=1 ATT_BLOCK=2048 $CAI coreai/qwen38_coreai_build.py all --kv-cache-dtype both \
  --ctx 8192,16384,32768,49152,65536 --pctx 8192,16384,32768,49152,65536
```

Qwen3.8-27B is developed by the Qwen Team (Alibaba Cloud, Apache 2.0). ANEMLL's work here is independent conversion and ANE research.
