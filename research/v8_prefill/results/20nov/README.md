# 20 Nov 2026: overnight YOLO plan (tree verify 8 / 16 / 32, then JetSpec++)

Plan: `PLAN.md` (Anemll, 19 Nov 22:32). Status line: `/tmp/m6_yolo_status.txt`. Ops log: `ops.log`.

## Summary (number-first)

(written at the end)

## Layout

| path | what |
|---|---|
| `PLAN.md` | the plan as received |
| `ops.log` | server stops / starts, processes killed, builds |
| `bench/` | realistic-set decode benches (`v8_work/bench/tree/tree_bench.py`), one JSON + log per arm |
| `tree_dyn_real/` | offline tree simulation on the realistic-set logs |
| `decomp/` | where the ANE's wide-verify cost goes (scratch probe, deleted after) |

## A0. Setup (22:33)

- Killed an orphaned `iorep_sample` (pid 78825, started 07:24 for 19 Nov's g256 e2e trace, PPID 1, output idle since
  08:03, 25% CPU): it would have perturbed every bench tonight.

## A1. Realistic prompt set and the chain baseline

**Prompt set** (`v8_work/bench/tree/make_prompts.py` -> `realistic_v1.json`, sha256 693e3097d942e5a5): per rung
(2K, 8K, 32K, 64K, 96K) nine tasks, the first user turn of the JetSpec bench prompt files: code = humaneval 0, 1 and
mbpp 0; math = gsm8k 0, 1 and math500 0; chat = mt-bench 0, 1, 2. Above 2K the system message carries a reference
document of real, distinct files from this checkout (scripts/, coreai/, docs, top-level *.md; sorted, each file once)
cut to fill the rung, so a rung's nine tasks share one cold prefill through the server's 'system' snapshot. Greedy,
thinking off, 320 tokens max. Prompt tokens: 2K 33-279, 8K 7,453-7,699, 32K 32,029-32,275, 64K 64,797-65,043,
96K 97,565-97,811.

**Bench** (`tree_bench.py`): client-side stream timing per request, the server's own line (tok/verify, accept
histogram, ms/cycle), mactop power over the decode window, sha256 of each reply. The one-line banner (V8 variant and
dylib sha, dec8 BANNER, tree mode) is printed before anything is timed and checked with `--expect` (abort on mismatch).

**Chain baseline** (`bench/chain_v1.json`, `.log`; forge restarted 22:43 in the served config with `DRAFT_PROF=1
DRAFT_TREE_LOG` for the tree logs):

| rung | tok/s (token-weighted) | tok/verify | code | math | chat |
|---|---|---|---|---|---|
| 2K | 40.99 | 4.567 | 6.0-7.8 | 6.3-6.8 | 2.5-4.7 |
| 8K | 45.98 | 5.086 | 6.3-7.8 | 5.8-6.3 | 2.2-4.7 |
| 32K | 44.05 | 5.164 | 4.6-7.8 | 6.0-6.8 | 2.5-4.7 |
| 64K | 38.38 | 5.040 | 5.4-7.8 | 5.9-6.8 | 2.1-3.0 |
| 96K | 34.03 | 4.722 | 6.7-7.6 | 6.5-6.8 | 3.0 |

Code and math accept most drafts (often all 7: the depth cap of DFlash2's 8-token block); chat accepts 2.1-2.8 (mt-bench
0 stops after 9-14 tokens). Mean accepted drafts per cycle: 3.6 (2K), 4.1 (8K), 4.2 (32K), 4.0 (64K), 3.7 (96K).

## A1b. Tree verify on the realistic set, offline (`tree_dyn_real/`, `pipe/unit/power/tree_dyn2.py`)

The 1,948 logged cycles (`DRAFT_TREE_LOG`, the chain run above, all five rungs). Per cycle: the chain's realized
acceptance and the best-first dynamic tree (served selector score, top-4 children per node), its first 7 / 15 / 31
nodes = the 8 / 16 / 32-row trees. Costs: the rung's measured served cycle and verify (bench), with the verify scaled
by the 19 Nov measured ANE group ratios (tree8 +5%, tree16 +26-39%, tree32 +87-116%, linear in the entry's KV rows).
Expected acceptance for the dynamic rule: tau fit 2-fold by request (1.2 / 1.5).

| rung | tok/verify chain / tree8 / 16 / 32 | chain tok/s | tree8 always | tree8 when branched | dynamic (chain / 8 / 16 / 32) | oracle |
|---|---|---|---|---|---|---|
| 2K | 4.61 / 4.84 / 5.18 / 5.41 | 42.06 | 42.28 (+0.5%) | 43.16 (+2.6%) | 43.23 (+2.8%) | 45.60 |
| 8K | 5.17 / 5.31 / 5.74 / 6.02 | 47.66 | 46.77 (-1.9%) | 48.00 (+0.7%) | 48.81 (+2.4%) | 50.98 |
| 32K | 5.23 / 5.39 / 5.77 / 6.01 | 43.30 | 42.80 (-1.2%) | 43.86 (+1.3%) | 44.15 (+2.0%) | 45.88 |
| 64K | 5.11 / 5.22 / 5.61 / 5.85 | 37.64 | 36.98 (-1.8%) | 37.82 (+0.5%) | 37.70 (+0.2%) | 39.52 |
| 96K | 4.77 / 4.90 / 5.28 / 5.51 | 33.24 | 32.78 (-1.4%) | 33.43 (+0.6%) | 33.72 (+1.4%) | 35.12 |

By category: chat chain 2.40 tok/verify, tree8 2.69, tree32 3.15; tree8 +7.4% tok/s, dynamic +8.2%. Code (7.06) and
math (6.46) gain nothing: they already accept close to the 7-draft cap, and the top-7 tree is the chain in 85-86% of
their cycles (chat: 5.5%). Fixed 16 / 32-row trees lose 8-19% / 33-44% at the ANE's measured costs.

## A2. Where the ANE's wide-verify cost goes (`decomp/`, scratch probe, deleted)

`v8/build_tree_decomp.py`: one target group (layers 0-3: three DeltaNet layers + the attention layer, the chunked FFN,
the served weight constants), split into parts, each a verify entry of T = P rows; one package per P = 8 / 16 / 32;
compiled with forge.py compile (1m11s), timed with `pipe/unit/power/tree_decomp_time.py` (B.Plan.run, median of 30
after 5 warm-up; banner in `decomp/timing.log`, packages sha 69aa3f41fb610203 / ...). Forge was down.

Per group, ms (8 / 16 / 32 rows):

| part | 8 | 16 | 32 |
|---|---|---|---|
| 4 FFNs | 2.98 | 3.04 (+2%) | 3.10 (+4%) |
| 3 DeltaNet mixers (projections, conv, UT transform, lazy commit) | 1.76 | 2.23 (+27%) | 4.04 (+130%) |
| attention mixer, 8K of KV | 0.94 | 1.13 | 1.47 |
| attention mixer, 64K of KV | 2.70 | 4.15 | 6.43 |
| whole group (served form), 8K | 5.06 | 5.79 (+14%) | 7.90 (+56%) |

- **The FFN is nearly free at 16 / 32 rows** (the weight stream dominates). The wide-verify cost is the DeltaNet
  mixers' row-dependent work (the T x T / T^3 UT transform) and the attention (rows x KV).
- **Chain form vs tree form:** the whole group in the chain form costs +14% / +56% at 16 / 32 rows, against 19 Nov's
  tree16 / tree32 (+26% / +87%): the conv-tap selection matmul adds the rest.

Where tree8's +5% over chain8 comes from (8K, 8 rows):

| form | ms | vs served | output vs served (chain masks) |
|---|---|---|---|
| served chain8 (`full8_8k`) | 5.06 | | |
| ancestor mask input only (`treeA8`) | 5.09 | +0.5% | bit-identical |
| conv taps by one-hot matmul only (`treeC8`) | 5.38 | +6.4% | rel 1.8e-3 |
| taps gathered + served conv kernel at stride 4 (`treeG8`) | 5.52 | +9.0% | rel 2.1e-4 |
| both (19 Nov's tree8, `tree8_8k`) | 5.35 | +5.7% | rel 1.8e-3 |

The masks cost nothing and change nothing; the whole overhead and the numeric difference are the conv taps.

**Tree shape (offline, the same 1,948 cycles, top-7 tree in 8 rows; `tree_verify.build_tree`):** tok/verify by
children per node (topc) and ordering temperature (tau); chain 4.953.

| topc \ tau | 0.7 | 1.0 | 1.5 | 2.5 |
|---|---|---|---|---|
| 2 | 5.047 | 5.090 | 5.095 | 5.002 |
| 3 | 5.056 | 5.108 | 5.119 | 4.990 |
| 4 | 5.057 | 5.111 | 5.123 | 4.985 |
| 8 | 5.057 | 5.113 | 5.129 | 4.973 |

tau 1.5 (the calibrated value) with 4 children per node is the served default (+3.4% tok/verify; chat 2.40 -> 2.73,
code / math +0.8% / +1.1%).

## A3. Served tree verify in the 8 rows: build and runtime

**Build** (`v8/build_dec8t_all.sh` -> `v8_dec8t`, 23:08-00:1x, forge down): build_dec8_chunk.py's packages with every
served entry unchanged plus, on the same weight constants, `g0_t8_<ctx>k` / `g1_t8_<ctx>k` (8K / 16K / 32K / 64K: the
contexts the server runs without the split) and `tdva` / `tdvba` (the split verify's ANE parts; `dvb` has no DeltaNet
layer and serves both), all with the ancestor-mask (`anc`) and conv-tap (`ctap`) inputs (19 Nov's tree8 form). Each
package is +3 MB over the served one (1,189 / 1,409 MB): the weights are shared. Package 0's compiled Core AI entry is
1.19 GB like the served one (no expanded-weight fallback).

**Runtime** (`pipe/tree_verify.py`, `TREE=1`; launcher `pipe/v8_server_tree.sh`):
- **Tree:** best-first by path probability under the drafter's own selector score (top-4 children per node, tau 1.5),
  the anchor + 7 nodes in pop order (topological). The drafter's chain is the tree of argmax children
  (unit test: 186 / 186 cycles identical to the offline simulator's tree; the drafter's path is always the chain).
- **Policy:** `branched` (default): the tree entry only when the 7 nodes are not the chain, else the served chain
  entry, so a chain-shaped cycle is the served computation. `always`, `dink` (per-cycle throughput rule) for research.
- **Verify:** non-split `t8_<ctx>k` (RoPE at p0 + depth); split (96K entry) `tdva` / `tdvba` + the GPU attention with
  per-row ancestor masks (`engine/dec_kernels_tree.metal`, `libdec_gpu_tree.dylib`: `dec_set_tree`; the mask (0, 0)
  is the served visibility, so chain steps are unchanged).
- **Accept:** greedy, from the root the child whose token is the row's argmax; the next token is the last accepted
  row's argmax; stop tokens as in the chain.
- **Commit:** the accepted rows are moved to rows 0..k in place (every DeltaNet layer's pending tape rows and cum
  columns and conv rows; the new K / V rows: the entry outputs before accept(), or the cache rows the GPU wrote on the
  split path; the drafter's tap features), so the rest of the runtime sees a chain of k + 1 rows (snapshots, the next
  call's commit, V8 continuation prefill).

**Smoke tests** (00:25-00:29, `bench/smoke_always.*`, `bench/smoke_topc1.*`; banner checked: tree BANNER
`mode=tree8 policy=always topc=4 tau=1.5 t8 contexts [8, 16, 32, 64]K | split tree on`, `libdec_gpu_tree.dylib`
sha256 dfe059c68dce376a, `dec_kernels_tree.metal` 674918fb1e682a2c):
- `TREE_POLICY=always` (every cycle through the tree entry): humaneval 1 token-identical to the served chain (sha
  1681b92d273f64ad, 7.80 tok/verify; the tree verify 98.8 ms against the chain's 94.5, +4.6%). mt-bench 2 differs from
  char 142 on, into an equally fluent continuation, 2.71 tok/verify against 2.50, 50 of 118 cycles accepted a path off
  the chain rows. The commit (row moves) works: no degeneration over 320 tokens.
- `TREE_TOPC=1` (the tree is always the chain, but computed by the tree entry): mt-bench 2 diverges at the same char
  142, mt-bench 1 at char 173; humaneval 1 and mt-bench 0 stay identical. So the divergence is the tree entry's conv-tap
  numerics (rel 1.8e-3 per group, A2) flipping near-ties, not the tree logic. A conv form exact on chain-tap rows
  (`TREE_FORM=corr`: the served conv kernel + a correction that is exactly 0 on rows whose taps are the 3 rows before
  them) is in `build_dec8t_chunk.py` and in the next probe.

**Realistic bench, tree8 `branched`** (00:29-00:41, `bench/tree8_branched_v1.*`; `pipe/v8_server_tree.sh`, build
v8_dec8t, `TREE_POLICY=branched`; banner as above). Compared with `bench/compare.py` (v8_work/bench/tree):

| rung | chain (served) tok/s, tok/verify, ms/cycle, W | tree8 branched | tok/s | replies identical |
|---|---|---|---|---|
| 2K | 40.99, 4.567, 110, 21.5 | 44.52, 4.895, 112, 20.6 | +8.6% | 2 / 9 |
| 8K | 45.98, 5.086, 108, 23.1 | 50.92, 5.527, 112, 22.8 | +10.7% | 2 / 9 |
| 32K | 44.05, 5.164, 121, 24.0 | 46.84, 5.613, 124, 23.2 | +6.3% | 3 / 9 |
| 64K | 38.38, 5.040, 136, 23.5 | 38.79, 5.257, 140, 23.8 | +1.1% | 7 / 9 |
| 96K | 34.03, 4.722, 144, 27.9 | 35.21, 4.962, 145, 27.3 | +3.5% | 8 / 9 |

- 1,824 cycles: 941 (52%) through the tree entry (3.24 tok/verify), 883 chain-shaped on the chain entry (7.35); in 380
  tree cycles the accepted path left the chain rows.
- **Not like-for-like:** the replies diverge (fluent near-tie flips from the conv-tap numerics, as in the smoke test;
  e.g. 8K mt-bench 2 from char 2), so the texts and their acceptance differ. The offline estimate on the chain's own
  texts is +3.4% tok/verify (A1b). The exact-on-chain conv form is needed for a fair A/B and for identity.
