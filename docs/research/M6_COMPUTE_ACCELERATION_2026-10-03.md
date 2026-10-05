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

Follow-up measurements from 4 October (INT8 keys, contexts above 64K, softmax forms, larger prefill calls, INT8 compute on the ANE, and a comparison with a GPU engine) are in [Follow-up, 4 October](#follow-up-4-october-measured).

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

## Follow-up, 4 October (Measured)

Same M6 and OS. Every long run had a swap watchdog (stop the job if swap grows more than 1 GB). Timings of single cores and chunks come from idle, same-session runs unless noted; whole-machine power is mactop's `total_power` (the SMC's system total, not its `system_power` field, which excludes the SoC).

### Against a GPU engine: Splash on the same model

[Splash](https://github.com/incoai/splash) 1.2.0 (Apache-2.0) serving Unsloth's `Qwen3.8-27B-GGUF:UD-IQ3_XXS` with its DFlash2 drafter on the Metal GPU, against our default build (faster graph, V8 cache, DFlash2), each server alone on the machine. Same harness ([`scripts/m6_compare_bench.py`](../../scripts/m6_compare_bench.py)): one cold prompt filling the context entry minus 512 tokens (a per-run nonce defeats prefix caching), then three identical greedy requests with thinking off and 256 tokens; client-side stream timing; power sampled once a second.

| Context | Splash prefill | Ours prefill | Splash decode | Ours decode |
| --- | --- | --- | --- | --- |
| 8K | 311 tok/s, 45.9 W, 0.148 J/tok | 288 tok/s, 31.1 W, 0.108 J/tok | 54.2 tok/s, 46.5 W, 0.86 J/tok | 63.6 tok/s, 27.2 W, 0.43 J/tok |
| 16K | 286, 45.0 W, 0.157 | 270, 32.4 W, 0.120 | 55.7, 46.2 W, 0.83 | 57.8, 28.1 W, 0.49 |
| 32K | 272, 46.2 W, 0.169 | 240, 32.2 W, 0.134 | 52.0, 47.1 W, 0.91 | 52.7, 27.6 W, 0.52 |
| 48K | 257, 46.8 W, 0.182 | 215, 31.3 W, 0.146 | 48.4, 46.2 W, 0.96 | 49.4, 26.9 W, 0.55 |
| 64K | 231, 46.9 W, 0.203 | 195, 31.1 W, 0.160 | 48.7, 46.7 W, 0.96 | 45.7, 27.7 W, 0.61 |

The machine idles at 8.8 to 9.1 W with either model loaded. Splash prefills 6 to 16% faster; the ANE uses 20 to 27% less energy per prompt token and 37 to 50% less per generated token, with decode level from 16K to 48K. Not matched: Splash's cache stores INT8 keys and values (ours FP16 keys, INT8 values), the models differ (10.9 GB target and 3.85 GB drafter against 9.9 GB and 1.8 GB), and the synthetic coding prompt gives both drafters very high acceptance.

From Splash's source and compiled kernels: weight matmuls dequantize the GGUF codes to FP16 tiles in threadgroup memory and run Metal `matmul2d` tensor ops (the GPU neural accelerators) on BF16 activations with FP32 accumulation, prefill in 128-row tiles; attention over the INT8 cache feeds the INT8 operand directly to the tensor op (BF16 x INT8) with key scales on the scores and value scales on the probabilities (the algebra of our V8), in one pass with an online softmax; the DeltaNet prefill is a token-by-token FP32 scan, verify a fused 8-row kernel with a separate commit. No kernel multiplies INT8 by INT8.

### INT8 keys (`kv8`)

New cache format `kv8` (builder `--kv-cache-dtype kv8`, server `--kv-cache-dtype kv8`): INT8 keys and values with FP16 scales per token and KV head. Key scales multiply the scores of each history tile, so the graph computes `q . (codes x scale)` exactly; 32.25 KiB per history position against 48.125 for V8 (2.0 GiB against 3.0 at 64K).

- **Exactness:** host FP64 test equal to the FP16 graph fed the dequantized cache to 1e-15 ([`tests/test_kv8_attention.py`](../../tests/test_kv8_attention.py)); compiled chunk 0 against the V8 chunk fed the same dequantized keys: everything before attention bit-identical, chunk output within 2.3 to 2.7e-4 (8K and 64K, verify and prefill).
- **Build:** 16 chunks, all 17 packages fully on the ANE, cold compile 29 min 23 s (V8 22 min 48 s; seven chunks took about 2.5 instead of 1.4 minutes).
- **Quality (KL-512, 64 sequences, 40,023 positions):** mean KL to BF16 0.183815 (V8 0.184168), top-1 agreement 86.008% (86.028%), trace perplexity 2.4135 (2.4131); direct KL between V8 and `kv8` mean 6.6e-5 nats, top token the same at 99.75% of positions. Four greedy smoke replies identical to V8.
- **Speed (server, same harness as above):** prefill 288 to 283, 240 to 236, 195 to 194 tok/s and decode 63.6 to 60.2, 52.7 to 51.7, 45.7 to 45.3 tok/s at 8K, 32K, 64K. Same speed within 2%, a third less cache.

Why INT8 caches stopped paying (chunk 0 A/B, 3 October): in the release graph's 16K-wide tiles V8 cut a 64K call against FP16 by 13% (verify, 11.33 to 9.83 ms) and 18% (prefill, 47.86 to 39.22 ms); in the faster graph's 2K tiles by 0% (7.50 to 7.56 ms) and 7% (31.57 to 29.22 ms). Small tiles removed most of the memory traffic INT8 saved.

### Contexts above 64K

The 65,472-row cap came from the old single-softmax graph, which concatenated `[history | block]` along one axis. The tiled graph never forms that tensor, so `kv_len` now keeps the whole context for entries above 65,536 rows (64K stays at 65,472). One `kv8` attention core of the production graph ([`scripts/m6_long_ctx_attn.py`](../../scripts/m6_long_ctx_attn.py)):

| Context | Cold compile | Verify (8 rows) | Prefill (64 rows) | Error against FP32 |
| --- | ---: | ---: | ---: | ---: |
| 32K | 5.5 s | 1.88 ms | 7.52 ms | 2.1e-3 |
| 64K (65,472) | 16.8 s | 3.44 ms | 14.79 ms | 2.2e-3 |
| 80K | 34.8 s | 4.32 ms | 18.47 ms | 2.2e-3 |
| 100K | 33.3 s | 5.30 ms | 23.56 ms | 2.3e-3 |

All fully on the ANE, no layout change needed. Memory decides what runs on 32 GB. The cache allocation was checked (one generation per context, dense, verify and prefill entries sharing identical layouts); the difference is what each package wires on first use:

| One chunk, first call | Wired |
| --- | ---: |
| 8K to 64K package | +0.35 GB |
| 80K-only package | +0.39 GB |
| 80K + 100K package | +0.67 GB |

- **80K + 100K build:** 28.5 GB wired at the 80K entry; swap grew and the watchdog stopped the server twice.
- **80K-only build:** cold compile 27 min 44 s, 25.7 GB wired, no swap growth; four smoke replies correct; an 81,401-token cold prefill at 135 tok/s and 27.3 W (pessimistic: the whole prompt runs in the 80K entry) and decode 41.6 to 41.9 tok/s at about 26 W.
- **100K** did not fit next to 80K in the same packages; a 100K-only package is untested.

### Softmax forms

Builder switches `ATT_SOFTMAX` / `ATT_SOFTMAX_PREFILL` (default `two_pass`: global max over all tiles first), all exact against the default in FP64 for the three cache formats. `online` keeps a running max from tile to tile; `split` gives every tile its own max, sum and output and combines them at the end (flash decoding).

| One `kv8` layer | Two-pass prefill / verify | Online | Split |
| --- | --- | --- | --- |
| 32K | 7.53 / 1.87 ms | -7% / +3% | -10% / +3% |
| 64K | 14.86 / 3.46 ms | -7% / +2% | -10% / +4% |
| 100K | 22.67 / 5.29 ms | -4% / +1% | -10% / +6% |

On a full chunk (chunk 0, `kv8`, interleaved A/B) the gain mostly disappears: `split` for prefill only gives +0.7%, +0.4% and -3.4% at 8K, 32K and 64K with verify bit-identical; `split` everywhere adds +0.2 to +1.5% to verify; `online` everywhere is within about 1% both ways. A control that recomputes the scores timed the same as two-pass: the compiler merges the duplicate. With 128-row prefill calls the forms still differ by about 1%. The two-pass softmax is not a bottleneck.

### Larger prefill calls

Chunk 0, `kv8`, 128-row prefill entries (`TPS=128`): 36.3, 46.1 and 60.1 ms at 8K, 32K and 64K against 12.5, 17.7 and 24.8 ms for 64 rows, so 45%, 30% and 21% slower per row. The 64-row call is already the efficient size on the ANE.

### INT8 compute on the ANE

Localized probe ([`scripts/m6_int8_probe.py`](../../scripts/m6_int8_probe.py)): one op type, 8 layers with distinct weights, 4096 x 4096, dense random data that neither clips nor contains runs of zeros, native `coreai.quantize` / `dequantize` with scales on both operands and the output. Everything below ran fully on the ANE.

| Rows per call | FP16 | W8A8, per-channel weight scale | W8A8, shared weight scale | A8A8, runtime INT8 operand |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 13.3 TOPS | 34.2 (conv) / 31.3 (matmul) | 34.4 (conv) | 53.2 (matmul) |
| 1,024 | 18.6 (conv) | 38.4 (conv) | **52.8 (conv)** | 53.2 (matmul) |

- **INT8 x INT8 engages in Core AI on the M6 ANE**, for constant weights and for a runtime second operand. The ceiling is about **53 TOPS**: A8A8 stays there from 256 to 1,024 rows (a compute limit, not bandwidth), the FP8 peak measured earlier. Constant-weight W8A8 reaches it with a **shared weight scale** (2.8x FP16 at 1,024 rows); per-channel scales cost about 28%.
- **Weights must be compile-time INT8 constants:** the `quantize_weights` pass, or `coreai.constexpr_blockwise_shift_scale` written in torch, both lowering to `coreai.blockwise_shift_scale` on a constant that feeds the conv or matmul directly (a reshape in between makes the pass skip it). A runtime `coreai.dequantize` of a constant is rejected by ANEC for conv. INT8 weights with FP16 activations: 23.0 TOPS.
- **Output-channel splits** with separate constants per branch help at 256 rows (shared scale: 34.4 to 43.3 TOPS with TP2, 40.7 with TP4) and not at 1,024 (already at the ceiling); FP16 is unchanged. The 1 MiB-per-core weight window (K 4096 against 4032) moved matmul about 9% and conv not at all.
- **Zeros:** 90% zeros in the inputs ran 2 to 13% faster, so timing data must be dense. mactop's per-cluster ANE fields read 100% even at idle, so cluster use could not be observed; ANE bandwidth during these runs was 113 to 128 GB/s.

Applied to the history attention (research switch `ATT_INT8MM`, one `kv8` layer, wrong numerics accepted): INT8 QK with the keys as the untransposed operand and a requantized output cut verify by 13 to 15%; INT8 QK and PV with requantized outputs cut the core by 12 to 17%. A timing control that replaces both history matmuls with reductions reading every key and value (`nomm`) removes only 7 to 10% of prefill and nothing from verify: **the matmuls are under a tenth of the attention core**. The INT8 variants saved more than that because requantized scores and weights shrink the score-sized intermediates (and the PV variants zeroed many small weights). The long-context lever is the bytes and passes over the score tensors (scaling, mask, max, exp, sum), not multiply-add speed.

Where INT8 x INT8 can matter is the weight matmuls of prefill, which are compute-bound. INT8 weights for the whole model (about 27 GB) do not fit beside the rest on 32 GB, so the candidate is our 2/4-bit LUT weights with INT8 activations, built the way that works here (compile-time weights, shared scales, output splits) and measured at the 64-row prefill size. An earlier LUT x INT8-activation probe that failed ANEC predates these corrections.

## Follow-up, 5 October: compiled ANE programs and 8-bit attention (Measured)

Workflow and tools from this round are written up for reuse in [ANE HWX and MLIR analysis](../ANE_HWX_MLIR_ANALYSIS.md). The 4 October INT8 attention variants were judged by timing alone. This round reads the compiled ANE program of each package to see what the hardware actually runs, then rebuilds the 8-bit attention variants with correct scales, one part at a time.

### Reading the compiled program (HWX)

- **Where it is.** A package's Core AI cache manifest names its ANE regions by `ANERegionsHash` (`<a>_<b>` for `h18g`); aned keeps the compiled program at `/Library/Caches/com.apple.aned/<OS build>/ModelAssetsCache/-_unsigned/<a>/<b>/model.hwx` (root-owned; read here through a read ACL granted by the user). The Core AI cache itself holds only the MLIR graph with its ANE regions.
- **Tools.** `hwx_parsing` from [freedomtan/coreml_to_ane_hwx](https://github.com/freedomtan/coreml_to_ane_hwx) decodes M6 programs (`h18g`, CPU subtype 0xb, ISA v24). [`scripts/m6_hwx_inspect.py`](../../scripts/m6_hwx_inspect.py) maps a package to its HWX and summarizes the tasks: `summary [--ne-dims]` (engine, operand and kernel formats, whether results go to DRAM or stay in L2), `pseudo` (one line per task), `roles` (attention matmuls by role). Each task is a register block for a fixed pipeline (DMA in, the NE multiply-add array or the PE elementwise / reduction engine, L2 or DMA out), so there is no instruction code to decompile. Cycle counts are the compiler's static estimates (`ExeCycles`), not measurements; for `pvtu` below they predicted the measured change within a point (-12.4% against -13%).
- **MLIR.** `AIModelAsset.load(pkg).program` prints a package's Core AI program as MLIR text (`coreai` dialect). The text parses back (`ir.Module.parse`, then the private `AIProgram._from_mlir_module`) and saves as a package that compiles to a byte-identical HWX, so the graph can be edited or written by hand at this level. The next stage down (MPSGraph's `mps` dialect, where the ANE regions are formed) cannot be printed with Core AI's bindings.
- **Cache pitfall.** The Core AI cache does not key on `MPSGRAPH_ANE_BONDED_COMPILE_MODE`: a package first loaded in mode 0 keeps that program in later mode-2 loads. For one `kv8` core the mode-0 program had 2.2x the static cycles (25,059 against 11,506). Our scripts and the server set mode 2 on the M6; ad-hoc loads must too, or recompile with `compile --force`.

### What the history attention compiles to

One `kv8` attention core at 32K (verify tiles of 2,048 tokens and 48 score rows per KV head, prefill tiles of 4,096 and 384 rows):

- **The INT8 cache is never converted to FP16.** No task turns INT8 into FP16; the QK and PV multiply-add tasks read the INT8 key and value tiles directly (INT8 input, FP16 query or probability operand). Dequantizing the whole history first or each tile next to its matmul (`ATT_TILE_DEQUANT`) compiles to the identical program (806 tasks), which is why the two timed the same.
- **Keys are transposed on every tile.** The cache stores keys as (token, 256 dims) and QK needs (256 dims, token), so an INT8 NE pass with an identity kernel and output transpose rewrites each key tile. In prefill it writes the transposed tile to DRAM (43 cycles against 95 for the QK that follows); in verify it stays in L2 but costs more than the QK itself (9 against 5). Values need no transpose. A key cache stored transposed would remove these passes.
- **Score tensors cross DRAM several times per tile:** written by QK, then read and written again by each softmax pass on the PE.

Where the cycles go (static estimates, the same shares at 32K and 64K):

| Part | Prefill (64 rows) | Verify (8 rows) |
| --- | ---: | ---: |
| Softmax elementwise (scale, mask, exp, value-scale fold) | 50% | 22% |
| Softmax reductions (max, sum) | 15% | 13% |
| History PV | 14% | 25% |
| History QK | 14% | 10% |
| Key transposes | 6% | 19% |
| New-block attention, other | 0% | 10% |

The measured `nomm` control (both history matmuls replaced by reductions, 7 to 10% of prefill) is smaller than the static matmul share, as matmuls partly overlap with data movement; both say the softmax passes dominate prefill. Verify has no single dominant part.

Two weight findings from the same programs: compile-time INT8 K/V projection weights run as INT8 kernels with no FP16 copy (`QCONV_INT8`, now the builder default: 40 MB instead of 50 MB per attention layer, the 64-row projection task 272 to 152 cycles); the vector-LUT Q/O weights run as a hardware palette with FP16 entries (`Fmt=fp16 Pal=1(4bit)`, about 2 bits per weight in the program), decoded on the fly. A LUT with INT8 entries (`palettize_weights(lut_dtype=INT8)`) fails ANEC for both vector LUT 2x16 and scalar LUT4 weights.

### Quantize / dequantize: the API and when the ANE fuses it

The API, from coreai-torch 0.4.2 (`coreai_torch/_compression/custom_layers.py`, the same on GitHub `main` apart from FP8 handling in `utils.py`):

```
coreai::quantize(Tensor input, Tensor scale, ScalarType output_dtype, Tensor? zero_point=None, Tensor? minval=None, SymInt axis=0)
coreai::dequantize(Tensor input, Tensor scale, Tensor? zero_point=None, Tensor? minval=None, SymInt axis=0,
                   ScalarType? input_dtype=None, ScalarType? output_dtype=None)
```

- Output types: int8, uint8, int4, uint4, `fp8_e4m3fn`, `fp8_e5m2`. Integer: `clip(round(x / scale) + zero_point)`, or with `minval`, `clip(round((x - minval) / scale) + qmin)`; FP8: `cast(x / scale)`, with neither offset.
- A scalar scale is per tensor; a vector scale with `axis` is per channel (`scale.numel()` equal to `input.shape[axis]`, and the same rank as the input when it is more than 1-D). `zero_point` and `minval` are mutually exclusive. `ActivationQuantizeModule` / `ActivationDequantizeModule` wrap the pair.
- Both lower to `coreai.quantize` / `coreai.dequantize` (in the dialect a scale plus two offsets, so zero point and `minval` are one op). Apple's own W8A8 path in coreai-optimization emits exactly this pair: its default is INT8 per-channel weights with INT8 per-tensor symmetric activations, and `finalize(ExportBackend.CoreAI)` replaces the fake quantization with these ops.
- Apple documents no rule for when the ANE compiler fuses the pair into an 8-bit multiply-add. The rules below come from the HWX of the variants in the next section.

Rule we follow: **every operand of an 8-bit matmul is written as `quantize` then `dequantize` in the graph, FP8 exactly as INT8, and an operand that arrives as INT8 (the cache codes) too** (`dequantize`, `quantize`, `dequantize`; the quantize of `codes / 128` is exact). With coreai-torch 0.4.2 on the M6 the extra pair on an INT8 input compiles to the identical program (same tasks, cycles and formats; `pvtu` timed the same), so it costs nothing, matches the form Apple's W8A8 export emits for every quantized operand, and does not depend on the compiler treating a bare `dequantize` of an input as the second half of a pair. The pair that decides the result is the one on the FP16 operand (the queries for QK, the probabilities for PV):

| Quantize / dequantize pair on the FP16 operand of a history matmul | Compiled program (HWX) | Effect |
| --- | --- | --- |
| None (production) | INT8 x FP16; the INT8 cache read directly | baseline |
| INT8 or UINT8, zero point 0, constant scale | INT8 x INT8 or INT8 x UINT8; the producing task writes the 8-bit codes | fused |
| INT8 with zero point -128, or `minval` mode | no fusion: explicit dequantize tasks write both operands in FP16 to DRAM (the INT8 value tiles too), then FP16 x FP16 | slower than none |
| Runtime per-axis (per-row) scale | ANEC rejects the region; it runs on the GPU | unusable |
| FP8 e4m3, constant scale | INT8 x e4m3 at FP16 multiply-add cycles; the producer writes 1 byte | fused, no MAC gain |
| FP8 e5m2, constant scale | compiles; the call crashes | unusable |

The scale must fill the code range. The 4 October PV variants quantized the value-scaled exp weights (about 0 to 3) with a step of 0.0625, so most codes were zero: 0.33 output error, and a speedup that came from zeros.

### 8-bit attention, part by part

Correctly scaled forms (`ATT_INT8MM`, research switch, forms combine as a comma list):

- `pvtu`: per tile, `exp(s - m_t)` with `m_t` the tile's row maximum (already computed for the global maximum) times the value scales divided by their tile maximum per head, so the weights lie in [0, 1]; UINT8 codes at step 1/255; the value tiles as `dequantize`, `quantize`, `dequantize` (INT8, step 1/128); the corrections `exp(m_t - m)` and the scale maximum multiply the small PV output. No extra pass over the scores. `pvt` is the same with INT8 at 1/127, `pvf8` with FP8 e4m3.
- `pvn` reaches [0, 1] by dividing by each row's maximum of the folded weights: an extra pass over the scores.
- `qkn` quantizes the queries per row to [-1, 1] and rescales the scores per row; `qkf` puts that per-row scale inside the pair instead (per-axis scale).

One `kv8` core, random data (scores of a flat distribution over 32K are near the worst case for 8-bit weights), error of the attention output against an FP32 host evaluation; the device matched a host simulation of the same rounding:

| Form | Prefill 32K | Verify 32K | Prefill 64K | Verify 64K | Error | Zero weight codes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Production (INT8 x FP16) | 7.55 ms | 1.89 | 14.83 | 3.46 to 3.53 | 2e-3 | |
| `pvtu` (INT8 x UINT8) | **6.53 (-13.5%)** | 1.88 | **12.81 (-13.6%)** | 3.50 | 3.6e-2 to 4.4e-2 | 12 to 16% |
| `pvf8` (INT8 x FP8 e4m3) | 6.69 to 6.74 (-11%) | 1.88 | 13.04 (-12%) | 3.49 to 3.54 | **2.7e-2** | 0% |
| `pvt` (INT8 x INT8) | 6.54 (-13%) | 1.87 | 12.69 to 13.00 | 3.48 to 3.50 | 7e-2 to 8.5e-2 | 26 to 33% |
| `pvn` (extra pass) | +2% | +12% | +10% | +11% | 4.6e-2 to 5.5e-2 | 12 to 16% |
| `pvta` / `pvtm` (offset codes) | 8.87 (+17%) | 2.67 | 17.27 | 4.95 | 3.6e-2 to 4.4e-2 | |

Half of the `pvtu` gain is not the multiply-adds: the quantize fuses into the PE pass that writes the probabilities, which now writes 1-byte codes (2,064 to 1,456 cycles), while PV drops from 1,600 to 960. `pvf8` keeps the first half only (its multiply-adds run at FP16 cycles) but is the most accurate.

In the real layer 3 (production projections, residual included, which dilutes the error), each part alone and together, with the HWX confirming 8-bit multiply-adds where intended:

| Layer 3 | HWX of the converted part | Verify 8K | Prefill 8K | Verify 64K | Prefill 64K |
| --- | --- | ---: | ---: | ---: | ---: |
| Production | QK, PV: INT8 x FP16 | 0.954 ms | 2.510 | 3.777 | 15.075 |
| K/V projections W8A8 | INT8 x INT8, 278 to 260 cycles | 0.926 | 2.448 | 3.775 | 15.061 |
| History QK, per-row scale (`qkn`) | INT8 x INT8, 3,776 to 3,688 cycles, plus a rescale pass (+2,170) | 0.936 | 2.610 (+4%) | 3.822 | 15.770 (+4.6%) |
| History QK, scale inside the pair (`qkf`) | ANEC fails, GPU (about 330 ms) | | | | |
| History PV (`pvtu`) | INT8 x UINT8, 4,458 to 2,880 cycles | 0.931 | **2.352 (-6.3%)** | 3.740 | **13.653 (-9.4%)** |
| QK, PV and K/V projections | all three 8-bit | 0.942 | 2.538 | 3.802 | 14.147 |

- **Only PV pays.** QK's multiply-adds barely shrink (2%): it is bound by reading key tiles and writing scores. A fixed query step (4 October `qk`) gained nothing, a per-row step costs a pass, and a per-row scale inside the pair does not compile. The K/V projections are under 1% of the layer.
- **Gains are prefill only:** about 9% of a full-attention layer's prefill at 64K and 6% at 8K, nothing for verify. A chunk holds one attention layer among four, so the model-level gain is smaller; it needs a chunk A/B and KL-512 (the 3.6% to 4.4% attention error above is on random data).

### Softmax: native op and fused quantization

One prefill tile in isolation (4 KV heads x 384 rows x 4,096 tokens, dense data; [`scripts/m6_softmax_probe.py`](../../scripts/m6_softmax_probe.py)):

| Tile | Time | Tasks | Static cycles |
| --- | ---: | ---: | ---: |
| `exp(s - max)` and its sum (our tile pass) | 0.662 ms | 8 | 366 |
| Hand-written softmax (max, exp, sum, divide) | 0.645 ms | 20 | 740 |
| `torch.softmax` (lowers to `coreai.softmax`) | 0.647 ms | 20 | 740 |
| `exp(s - max) @ V` (production PV form) | 0.778 ms | 12 | 716 |
| UINT8 pair on `exp(s - max)`, then `@ V` | **0.634 ms (-18.5%)** | 12 | 464 |
| `softmax(s) @ V` | 0.909 ms | 14 | 826 |
| UINT8 pair on `softmax(s)` rescaled to [0, 1], then `@ V` | 1.006 ms | 30 | 1,169 |

- **`coreai.softmax` is not a hardware softmax.** It compiles to the identical program as the hand-written passes; `torch.softmax` and SDPA both lower to it.
- **A quantize / dequantize pair fuses into the task that computes the exp:** that task writes UINT8 directly and PV reads it as INT8 x UINT8. This is the `pvtu` mechanism in isolation.
- At verify size (48 rows x 2,048 tokens) every variant takes 0.21 to 0.23 ms: call overhead.
- The dialect also has `coreai.symmetric_quantization_statistics` / `asymmetric_quantization_statistics` (per-axis scales computed at run time). No torch op emits them; using them means writing MLIR.

### 8-bit softmax

The softmax passes take 8-bit input and produce 8-bit output when the scores and the weights carry quantize / dequantize pairs: QK writes INT8 or FP8 scores, the max, `s - max` and exp passes read them directly (the dequantize folded into the pass's scale), and the exp pass writes UINT8 or FP8 weights. One prefill history tile in isolation (4 KV heads x 384 rows x 4,096 tokens; [`scripts/m6_softmax_probe.py`](../../scripts/m6_softmax_probe.py), the HWX confirming 8-bit at each step):

| Softmax of one tile | Prefill tile | Verify tile (48 x 2,048) | Error (random data) |
| --- | ---: | ---: | ---: |
| FP16 (production) | 0.822 to 0.899 ms | 0.280 to 0.290 ms | 1.8e-3 |
| INT8 scores in, UINT8 weights out, FP16 inside | -10 to -14% | 0% | 3.3e-2 |
| INT8 scores, INT8 `s - max`, UINT8 weights, sum over the UINT8 weights | -24 to -28% | -13% | 1.1e-1 |
| **INT8 scores, INT8 `s - max`, FP8 e4m3 weights, sum over the FP8 weights** | **-22 to -25%** | **-15%** | **3.0e-2** |
| FP8 e4m3 throughout | -21 to -24% | -12% | 2.1e-1 |

The UINT8 sum is biased (small weights round to zero); FP8 weights do not underflow, so the 8-bit sum stays accurate. FP8 scores are unusable (3 mantissa bits on logits).

In the attention core the per-token key scales multiply the scores and the per-token value scales are folded into the weights, so the forms are placed around those: `s8b` pairs the scores after the key scales and the mask (the tensor the max and exp passes read), `sm8` quantizes the exp output to FP8 and takes the sum from it before the value-scale fold, `t8` pairs `s - m_t`. A pair right after QK (`s8`) only saves the first pass: the key-scale pass reads INT8 but writes FP16 again. One `kv8` core, idle machine, three interleaved runs (medians):

| Attention core | Prefill 32K | Prefill 64K | Verify 32K | Verify 64K | Chips |
| --- | ---: | ---: | ---: | ---: | --- |
| FP16 softmax (production) | 7.55 ms | 14.91 | 1.88 | 3.46 | |
| `pvtu` | -13.4% | -14.6% | +0.5% | +0.9% | M5, M6 |
| `s8` + `pvtu` | -15.9% | -17.0% | +3.7% | +4.6% | M5, M6 |
| **`s8b` + `pvtu`** | **-20.7%** | **-21.5%** | **-4.8%** | **-6.1%** | M5, M6 |
| **`s8b` + `sm8` + `pvtu`** (FP8 softmax sum) | **-27.5%** | **-28.2%** | -6.4% | -6.9% | M6 only |
| `s8b` + `t8` + `sm8` + `pvtu` | -27.9% | -29.3% | -6.9% | -7.2% | M6 only |

FP8 forms need an option enabled per chip (the M5 ANE has no FP8). The core is about 60% of a chunk's prefill time at 64K, 43% at 32K and 20% at 8K, so the estimated model prefill gain is about 13%, 9% and 4% for `s8b` + `pvtu` and about 17%, 12% and 6% with `sm8`; verify gains about 3% at 64K. Chunk A/B and KL-512 have not been run.

**Real score ranges and 8-bit error** ([`scripts/m6_attn_logit_stats.py`](../../scripts/m6_attn_logit_stats.py), the streamed host reference with the deployed export, 46,640 tokens of KL-trace chats and wikitext up to 16K; error of the attention output against FP32):

- True scores reach 31.8 (p99.99 16.9); 13% of rows have a maximum above 16 (layers 35 to 51 most). The raw QK output before the key scales stays within 6.5, but a constant step there is multiplied by each key's scale (up to 23): 3% error at 1/16, up to 20% at 1/8. So the pair belongs after the key scales (`s8b`).
- `s8b` at a **symmetric step of 1/4 (+-32): 1.16% overall, 0.55% to 2.64% per layer**, no clipping in practice. A step of 1/8 needs zero point 64 to cover the range (0.64%), which does not fuse; +-16 clips 13% of row maxima (6.3%).
- FP8 e4m3 scores: 5.1% (2.4% to 11.8% per layer), about 4x worse than INT8 at 1/4.
- `pvtu` weights: **6.1% overall** (3.7% to 9.2% per layer, more at 4K to 16K); without the value-scale fold 2.7%. The fold, not the UINT8 codes, is most of the 8-bit attention error.
- `kv8` keys alone: 0.45%.

One value scale per tile (or per layer) instead of per token would remove the fold pass and its error, and the key-scale pass with it; FP8 weights (`sm8`) are the M6 alternative. Accuracy losses here may also be recoverable by retraining the rank-64 low-rank corrections with the 8-bit attention simulated.

These errors are pooled over layers (root of summed squared errors over summed squared references), which weights layer 63 most; the layer means below are the fairer measure (INT8 scores at 1/4: 2.3% as a layer mean).

### 8-bit attention on V8 (FP16 keys)

On V8 the keys are FP16, so there is no key-scale pass and QK writes the true scores: the INT8 pair sits directly on the QK output and again after the mask (`s8` and `s8b` on the same scores, step 1/4). The forms tested, each a full graph with every 8-bit operand as a quantize / dequantize pair:

- A: INT8 scores, UINT8 PV weights (`s8,s8b,pvtu`); M5 and M6.
- B: A plus the FP8 softmax sum (`s8,s8b,sm8,pvtu`); M6.
- C: INT8 scores, FP8 softmax sum, FP8 PV weights (`s8,s8b,sm8,pvf8`); M6.

**Accuracy on real text** (host simulation, 26,160 tokens of KL-trace chats and wikitext, error of the attention output against FP32, mean over the 16 attention layers; the V8 value codes alone cost 1.06%):

| Form | Error | Weight codes that round to zero (share of softmax mass) |
| --- | ---: | --- |
| INT8 scores alone | 2.33% | |
| A | 5.41% (7.2% on wikitext) | 84% of entries (6.95% of the mass) |
| B | 5.47% | the same |
| C | 2.81% | 22% (0.02%) |
| FP8 PV weights, exact sum | 2.77% | 22% (0.02%) |
| Per-tile value scales, UINT8 weights (cache-format change) | 3.74% | 73% (2.31%) |

Folding the per-token value scales into the weights makes most UINT8 codes zero and drops about 7% of the softmax mass; FP8 weights keep it. Per-tile value scales double the value error by themselves (2.38% against 1.06%) and only help an INT8-only path.

**KL-512** (8K entry, 64 sequences, 40,023 positions; full builds with only these forms changed, `QCONV_INT8=0`):

| Build | Mean KL to BF16 | Top-1 agreement | Perplexity | Direct KL vs V8 | Same top token as V8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| V8 (faster graph) | 0.184168 | 86.03% | 2.4131 | | |
| A | 0.186473 | 85.89% | 2.4328 | 0.0113 | 96.4% |
| B | 0.187369 | 85.92% | 2.4349 | 0.0113 | 96.3% |
| **C** | **0.184021** | **86.02%** | **2.4135** | **0.00037** | **99.3%** |

C matches V8 (for scale, `kv8` against V8 measured 0.00007). A and B move the model measurably; the INT8-only path needs better PV weights before it is usable.

**Chunk 0** (3 DeltaNet layers and one attention layer) against the V8 chunk, idle machine, two interleaved runs:

| Form | Prefill 8K | Prefill 32K | Prefill 64K | Verify 8K | Verify 32K | Verify 64K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | -2.6 to -3.3% | -6.8 to -7.8% | -9.6 to -9.8% | -1.0% | -3.2% | -6.2 to -6.6% |
| B | -3.2% | -9.9 to -11.3% | -14.7 to -15.4% | -1.5 to -1.8% | -3.9 to -4.6% | -5.9 to -6.3% |
| **C** | **-2.5 to -2.8%** | **-8.1 to -8.8%** | **-13.9 to -14.1%** | **-1.5%** | **-3.8%** | **-6.2%** |

C is the M6 candidate: about 14% off a chunk's 64K prefill and 6% off its 64K verify at V8 quality. The full build is in the next section.

### Form C at long context: the FP8 range fault, and C2

The full build of C matched V8 on KL-512 but failed the long-context evals, which KL-512 cannot see (its sequences fit in one history tile). Long-context evals (`scripts/m6_long_ctx_eval.py`, target only), against the V8 build:

| Build | 8K (7,600 + 512): KL vs V8, top-1, perplexity | 64K (64,000 + 1,024): KL vs V8, top-1, perplexity |
| --- | --- | --- |
| V8 | perplexity 8.0952 | perplexity 5.0673 |
| C (FP8 scale 1/256) | 0.0345, 93.6%, 8.3335 | 0.0626, 92.2%, 5.3652 |
| Cr (C plus `s8r`, shift 8) | 0.0451, 92.8%, 8.4143 | 0.0676, 92.6%, 5.3995 |
| **C2 (FP8 scale 1/64)** | **0.00021, 98.8%, 8.0957** | **0.00048, 98.9%, 5.0685** |

Cr tested whether the fixed INT8 score range clipped (`s8r` re-centres each row before the pairs); it was no better. Bisection with hybrid builds (`scripts/m6_hybrid_build.py`: the forms in some chunks, V8 in the rest) and real captured attention inputs (`scripts/m6_capture_attn_inputs.py`, `scripts/m6_attn_core_check.py`: device against host simulation, per head) found one head, layer 63 head 19, whose FP8 PV operand reached about 200 at scale 1/256. The M6 ANE computed that INT8 x FP8 PV wrongly, although e4m3 holds 448, and matched the host at 100 and below. C2 is C with FP8 scale 1/64 (`ATT_PF8_UNIT`, now the default): softmax weights in [0, 1] become at most 64.

C2 quality against V8: verify path (64 + 4,096) KL 0.00028, top-1 99.3%, perplexity 6.6452 against 6.6441; KL-512 0.183763 to BF16 (V8 0.184168), top-1 86.02%, perplexity 2.4144, direct KL to V8 0.00039, same top token 99.3%. Prefill in the long-context evals (target only, no drafter): 8K 292.5 to 310.8 tok/s (+6.3%), 64K 199.6 to 223.9 tok/s (+12.2%).

**Whole server against Splash** (`scripts/m6_compare_bench.py`: greedy, thinking off, 256-token cap, one cold prompt filling each context to capacity - 512, decode the median of three requests). C2 on 5 October with current code (windowed drafter ingestion: 2.3 to 2.5% of a cold prefill above 16K, identical output); Splash (UD IQ3_XXS on the GPU) recorded on 4 October. Energy is whole-machine (mactop `total_power`) for both: the Splash record's stored joules per token use `system_power`, which excludes the SoC (at 64K prefill 21.0 W against 46.9 W total), so every value here is recomputed.

| Context | Prompt | Prefill tok/s, C2 / Splash | J per prompt token | Decode tok/s, C2 / Splash | J per generated token |
| --- | ---: | --- | --- | --- | --- |
| 8K | 7,673 | 297.0 / 310.7 | 0.107 / 0.148 | 61.2 / 54.2 | 0.474 / 0.857 |
| 16K | 15,865 | 283.9 / 286.5 | 0.118 / 0.157 | 51.7 / 55.7 | 0.567 / 0.831 |
| 32K | 32,249 | 263.8 / 272.4 | 0.126 / 0.169 | 55.7 / 52.0 | 0.516 / 0.901 |
| 48K | 48,633 | 241.8 / 256.5 | 0.135 / 0.182 | 48.8 / 48.4 | 0.580 / 0.960 |
| 64K | 64,953 | 224.3 / 231.1 | 0.143 / 0.203 | 47.8 / 48.7 | 0.587 / 0.960 |

C2 prefills 1 to 6% slower than Splash at 25 to 29% less energy per prompt token (about 32 W against 46 W for the whole machine). Decode is within -7% to +13% (one fixture, three requests; the sign changes with the context, so most of it is acceptance noise) at 32 to 45% less energy per generated token (about 29 W against 46.5 W). Net of idle (9 to 10 W): prefill 0.073 to 0.103 J per token against 0.119 to 0.165, decode 0.31 to 0.40 against 0.67 to 0.78. Raw: `compare/ane_c2_window.json`, `ane_64k_C2.json`, `splash_ud_iq3xxs.json`, summary `c2_vs_splash_total_power.json` (research volume).

### Transposed key cache (C2T)

V8 keys are FP16 and stored (token, 256) while QK reads (256, token), so C2's program transposes every key tile before its QK; in chunk 0's HWX those output-transpose tasks are 13.7% of the cycles and grow with the context. `KV_KEYS_T=1` stores keys as (KV head, 256, token): the builder records `kv_cache.key_layout = dim_token`, the runtime writes accepted and prefill key rows transposed and checks the bound buffer shapes. In the HWX the transpose cycles stop growing with the context (only the new-block and projection transposes remain). Chunk 0, interleaved: verify -0.8 / -1.9 / -4.2% and prefill -1.6 / -4.5 / -7.1% at 8K / 32K / 64K.

Full C2T build (C2 plus `KV_KEYS_T`, full ladder; the export and the ANE compile overlapped, 27.5 min for both): logits identical to C2 (KL 0.0, same perplexity at 8K and 64K). Whole server against C2 and Splash (same benchmark and whole-machine power as the table above):

| Context | Prefill tok/s, C2T / C2 / Splash | J per prompt token, C2T / C2 / Splash | Decode tok/s, C2T / C2 / Splash | J per generated token, C2T / C2 / Splash |
| --- | --- | --- | --- | --- |
| 8K | 306.3 / 297.0 / 310.7 | 0.102 / 0.107 / 0.148 | 60.8 / 61.2 / 54.2 | 0.458 / 0.474 / 0.857 |
| 16K | 296.6 / 283.9 / 286.5 | 0.108 / 0.118 / 0.157 | 60.3 / 51.7 / 55.7 | 0.465 / 0.567 / 0.831 |
| 32K | 274.7 / 263.8 / 272.4 | 0.120 / 0.126 / 0.169 | 54.5 / 55.7 / 52.0 | 0.519 / 0.516 / 0.901 |
| 48K | 253.2 / 241.8 / 256.5 | 0.130 / 0.135 / 0.182 | 46.1 / 48.8 / 48.4 | 0.590 / 0.580 / 0.960 |
| 64K | 236.6 / 224.3 / 231.1 | 0.136 / 0.143 / 0.203 | 47.9 / 47.8 / 48.7 | 0.569 / 0.587 / 0.960 |

C2T prefills 3.1 to 5.5% faster than C2 at the same power, within -1.4% to +3.6% of Splash, at 31 to 33% less energy per prompt token. Decode stays within the fixture's acceptance noise. Each run first measures its idle floor (whole machine: ANEMLL 8.3 to 8.7 W, Splash 8.8 W). The joules above include it; net of the floor, ANEMLL (C2T) uses 36 to 39% less energy than Splash per prompt token (0.073 to 0.100 J against 0.119 to 0.165) and 48 to 55% less per generated token (0.315 to 0.400 J against 0.673 to 0.778); gross, 29 to 33% and 39 to 47% less. The floor is about a third of ANEMLL's 27 to 33 W and a fifth of Splash's 46 W, so the net savings are larger. Raw: `compare/ane_c2t_window.json`, `ane_64k_C2T.json`, summary `c2t_c2_splash_total_power.json`.

### Model-file quality: Splash's GGUF on the same KL-512

Splash's API returns no logits, so its model file was scored through llama.cpp (llama-cpp-python 0.3.36, M5 Max) with `scripts/kl512_gguf.py`: the same 64 sequences, 40,023 positions and metric as `m6_kl512_eval.py`. Splash reports 99.3 to 99.45% same top token as llama.cpp on this model. Mean KL to the BF16 teacher (reference perplexity 2.1355):

| Model file | Stored bits per layer weight | Mean KL | Top-1 agreement | Perplexity |
| --- | ---: | ---: | ---: | ---: |
| UD-Q5_K_XL, FP16 / q8_0 cache | | 0.00328 / 0.00329 | 97.70 / 97.78% | 2.1437 / 2.1428 |
| UD-IQ3_XXS (Splash's), FP16 / q8_0 cache | 3.16 | 0.16253 / 0.16232 | 89.40 / 89.41% | 2.2531 / 2.2533 |
| C2 (ANE) | about 3.2 (package size, with FP16 corrections) | 0.18376 | 86.02% | 2.4144 |
| V8 (ANE) | | 0.18417 | 86.03% | 2.4131 |

(Flash attention on in both runs of each pair: llama.cpp needs it for a quantized V cache.) The 8-bit KV cache costs nothing measurable even on the near-lossless Q5_K_XL. Splash's own INT8 cache uses one scale per token and head (256 values) where q8_0 uses one per 32; our `kv8` has Splash's granularity and measured a direct KL of 0.00007 against V8. UD-IQ3_XXS is a mix (layers: 31.5% IQ3_XXS, 24.4% IQ3_S, 10.9% IQ4_XS, 22% IQ2 types, small shares from IQ1_S to Q8_0; LM head Q4_K, embeddings Q2_K). At about the same stored size per layer weight, it is closer to BF16 than the VQ builds (KL 0.163 against 0.184; direct KL between it and C2 0.29, same top token 84.5%). Raw: `kl512_splash/` on the research volume.

### Review of coreai-torch `_compression` for attention

| Item | Use for attention |
| --- | --- |
| FP8 activations (`fp8_e4m3fn`, `fp8_e5m2`; upstream adds `e8m0fnu` power-of-two scales) | FP8 softmax weights (`sm8`, `pvf8`, measured above); FP8 scores are too coarse |
| `minval` mode | Takes the no-fusion path, like zero point -128 |
| Per-axis scales | Constant scales only; a runtime per-row scale fails ANEC |
| int4 / uint4 activations | Too coarse for scores or weights |
| `lut_to_dense`, `constexpr_blockwise_shift_scale` (including FP4 e2m1 block-scaled data), `sparse_to_dense` | Weights only |

### Next

- `s8b` (step 1/4) + `pvtu` on one chunk, and with `sm8` on M6, A/B at 8K to 64K, then a full build and KL-512.
- Value (and key) scales per tile instead of per token: removes the fold and key-scale passes and most of the `pvtu` error.
- A transposed key cache: removes the per-tile key transposes (19% of verify).
- Retrain the rank-64 corrections with the 8-bit attention simulated if KL-512 moves.

## Rejected or not pursued

- **Kronecker Hadamard** (`H_1024 = H_32 x H_32`, two 32 x 32 grouped convs and a channel transpose): exact but slower than the release 1-bit grouped conv (0.47 versus 0.39 ms for four 17408-channel rotations at T=8).
- **Core AI composite ops.** `coreai_torch` ships `GatedDeltaUpdate` and `SDPA` composites. The DeltaNet composite is an FP32 while-loop over tokens; Apple's authoring notes place native SDPA on the GPU path and recommend per-head attention on the ANE. Not tested further.
- **INT8 / FP8 attention MACs.** Built and measured on 5 October (above): only the history PV pays, and half of that gain is the 1-byte probability write, not the narrower MAC. The history path stays dominated by elementwise passes over scores.
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

Follow-up, 4 October:

```sh
# kv8 build (8K to 64K) and an 80K-only build; contexts above 64K keep their whole history
EXPORT_DIR=<export> $CAI coreai/qwen38_coreai_build.py all --kv-cache-dtype kv8 --ctx 8192,16384,32768,49152,65536 --pctx 8192,16384,32768,49152,65536
EXPORT_DIR=<export> $CAI coreai/qwen38_coreai_build.py all --kv-cache-dtype kv8 --ctx 81920 --pctx 81920
python scripts/m6_kl512_eval.py run --build <kv8 build> --format kv8 --ref-dir <reference> --out DIR
# any OpenAI-compatible server, with whole-machine power (mactop)
python scripts/m6_compare_bench.py --url http://127.0.0.1:8765/v1 --model <id> --tokenizer <model dir> --ctx 8192,32768,65536 --label L --out L.json
# one kv8 attention core: softmax forms and INT8 matmul research variants
$CAI scripts/m6_long_ctx_attn.py build --variant split --ctx 32768,65472,102400 --out DIR && $CAI scripts/m6_long_ctx_attn.py time --variant split --ctx 32768,65472,102400 --out DIR
$CAI scripts/m6_long_ctx_attn.py build --int8mm nomm --ctx 32768,65472 --out DIR && $CAI scripts/m6_long_ctx_attn.py time --int8mm nomm --ctx 32768,65472 --out DIR
# localized INT8 x INT8 probe
$CAI scripts/m6_int8_probe.py --out DIR --form conv,matmul --variants fp16,xw8,xw8a8 --wscale tensor --n 1024 [--tp 2]
```

Follow-up, 5 October:

```sh
# compiled ANE program of a loaded package (aned cache read access, hwx_parsing from freedomtan/coreml_to_ane_hwx)
HWX_PARSING=<path>/hwx_parsing python scripts/m6_hwx_inspect.py roles <package>.aimodel
HWX_PARSING=<path>/hwx_parsing python scripts/m6_hwx_inspect.py summary --ne-dims <package>.aimodel
# 8-bit history PV in one kv8 core: time, error against FP32, host simulation, zero codes
$CAI scripts/m6_long_ctx_attn.py build --int8mm pvtu --ctx 32768,65472 --out DIR && $CAI scripts/m6_long_ctx_attn.py time --int8mm pvtu --ctx 32768,65472 --out DIR
# each part in the real layer 3, and together
EXPORT_DIR=<export> $CAI scripts/m6_attn_layer_int8.py build --variants base,kvw8a8,qkn,pvtu,all8 --out DIR
EXPORT_DIR=<export> $CAI scripts/m6_attn_layer_int8.py time --variants base,kvw8a8,qkn,pvtu,all8 --out DIR
# native softmax against the hand-written passes; 8-bit softmax tiles
$CAI scripts/m6_softmax_probe.py --out DIR --rows 384 --tokens 4096
$CAI scripts/m6_softmax_probe.py --out DIR --rows 384 --tokens 4096 --variants tile_base,tile_s8p8,tile_i8sm,tile_i8pf8sm
# 8-bit softmax in one kv8 core (steps for the synthetic data; real data: ATT_S8B_UNIT=0.25)
ATT_S8B_UNIT=0.0625 $CAI scripts/m6_long_ctx_attn.py build --int8mm s8b,sm8,pvtu --ctx 32768,65472 --out DIR
ATT_S8B_UNIT=0.0625 $CAI scripts/m6_long_ctx_attn.py time --int8mm s8b,sm8,pvtu --ctx 32768,65472 --out DIR
# real score ranges and 8-bit error on the host (CPU, streamed layers)
EXPORT_DIR=<export> MODEL=<Qwen3.8-27B> python scripts/m6_attn_logit_stats.py run --out DIR --pass "kl=4,25" --pass "wiki=4096x1;stride=4"
```

Qwen3.8-27B is developed by the Qwen Team (Alibaba Cloud, Apache 2.0). ANEMLL's work here is independent conversion and ANE research.
