# 20 Nov 2026: overnight YOLO plan (tree verify 8 / 16 / 32, then JetSpec++)

Plan: `PLAN.md` (Anemll, 19 Nov 22:32). Status line: `/tmp/m6_yolo_status.txt`. Ops log: `ops.log`.

## Summary (number-first)

**Phase A, tree verify. Built, served behind a flag (`pipe/v8_server_tree.sh`, `TREE=1`); not the default.**
- **Realistic set, served chain** (code / math / chat at 2K-96K, `bench/chain_v1`): 41.0 / 46.0 / 44.1 / 38.4 /
  34.0 tok/s, 4.57 / 5.09 / 5.16 / 5.04 / 4.72 tok/verify, so 3.6-4.2 accepted drafts per cycle. The ~4 accepted target
  is already met at 8K-64K. Code 7.1 and math 6.5 tok/verify sit near DFlash2's 7-draft cap; chat is at 2.4.
- **Served tree8** (dynamic top-7 tree in the 8 rows; the tree entry only when the tree is not the chain):
  - **Bench:** +8.6 / +10.7 / +6.3 / +1.1 / +3.5% tok/s at 2K / 8K / 32K / 64K / 96K. Only 22 of 45 replies were
    token-identical to the chain, so this is not like for like.
  - **Like for like** (offline on the chain's own texts, measured tree verify cost x1.033-1.050): +0.5 to +3.0% by rung;
    chat +8%, code / math +0.3 / +0.6%.
- **Why it is not the default:**
  - **Not token-identical:** the tree entry's conv taps differ from the served conv by rel 1.8e-3 per group and flip
    near-ties. The same flips happen with chain-shaped trees run through the tree entry.
  - **The exact conv form costs too much:** it is bit-identical in the probe, but costs x1.096 per tree verify, which
    projects to about break-even (build and identity check: A8).
- **16 / 32 rows on the ANE:**
  - **Where the cost is:** the FFN is nearly free (+2 / +4%). The cost is the DeltaNet UT solve (+27 / +130% for 3
    layers; block forward substitution cuts 32 rows by 40%), the tree conv taps (+0.66 / +1.58 ms per group, now the
    largest term) and attention at long context.
  - **What is reachable:** tree16 about +23%, tree32 about +58% per verify. A dynamic 8 / 16 / 32 policy then adds
    about what tree8 adds (+0.4-2.9%).
  - **What is missing:** a cheap exact tree conv. 707 distinct tree shapes occur, so fixed shapes don't work.

**Phase B, JetSpec++ (JetX), by replay of our served model's own greedy streams (lossless on every trace):**
- **JetSpec++ tokens per round vs our DFlash2 chain on the same prompts:**
  - gsm8k 0-31: 9.50 vs 5.75 (x1.65)
  - humaneval: 13.20 vs 7.30 (x1.81)
  - mbpp: 10.55 vs 5.89 (x1.79)
  - mt-bench: 4.94 vs 3.48 (x1.42)
  - The whole B16 block is accepted in 33% of gsm8k rounds. This is the lever for more accepted tokens per cycle,
    not wider trees.
- **Its verify fits this target:** a 16-row CHAIN verify costs x1.085 at the 8K entry (block solve; the 16-row head
  is free), x1.17 at 32K and x1.27 at 64K. Projected at 2K: about 72 tok/s with a 30 ms drafter, against 53.4
  (+34%).
- **Drafter at 4 bits:** RTN 4-bit adapter + head loses 9.3% (9.28 vs 10.23 tok/round, still x1.53 over DFlash2);
  3 bits breaks it.
- **Not integrated:** the drafter is 6.2B parameters (12 GB bf16; about 3.4 GB and about 28 ms per round at 4 bits on
  the ANE, estimated). Next steps: an ANE port of the B16 head and ctxkv adapter, and a 16-row chain verify entry
  with the block solve.

**Served state at the end:** see A8 / the last ops.log lines (forge restored on 8765 in the served configuration).

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

## A5. DeltaNet solve and tree conv forms at 8 / 16 / 32 rows (`gdnwy/`, scratch probe, deleted)

`v8/build_gdn_wy_probe.py`: the 3 DeltaNet mixers of layers 0-2 (served weight constants) as verify entries, one
package per P; compiled with forge.py compile (29 s), timed with `tree_decomp_time.py` (median of 30). Forge down.
ms per 3 layers; "exact" = bit-identical to the served form under chain masks:

| form | 8 rows | 16 rows | 32 rows | exact |
|---|---|---|---|---|
| served (`inv`: log-depth Neumann inverse over T rows) | 1.74 | 2.21 | 4.03 | (reference) |
| no solve (`none`: timing floor, wrong output) | 1.75 | 1.74 | 1.92 | |
| block inverse, 8-row blocks (`blk`) | | 1.98 | 2.73 | no (rel 1.8e-3 / 2.1e-3) |
| block forward substitution (`bfs`) | | 1.98 | 2.44 | no (rel 1.7e-3 / 2.0e-3) |
| tree conv taps, one-hot matmul (19 Nov form, `tc`) | 2.03 | 2.87 | 5.61 | no (rel 3.0e-3 / 2.5e-3 / 2.4e-3) |
| tree conv taps channel-major (`tcT`) | 2.36 | 3.39 | 6.62 | no |
| served conv + correction exact on chain-tap rows (`tcC`) | 2.28 | 3.22 | 5.97 | yes at 8 and 32 (16: rel 3.2e-4) |
| channel-major taps + block inverse (`tcTb`) | | 3.14 | 5.32 | no |

- **The UT solve is the DeltaNet's whole row cost:** without it the 3 layers cost the same at 8, 16 and 32 rows.
  Block forward substitution takes 32 rows from 4.03 to 2.44 ms (16 rows: 2.21 -> 1.98).
- **The tree's conv taps are now the main wide-tree cost** (+0.29 / +0.66 / +1.58 ms per 3 layers at 8 / 16 / 32
  rows, x 16 groups); the channel-major layout is worse.
- **Exact tree8:** the correction form is bit-identical to the served form on chain-tap rows but costs +0.54 ms per
  group at 8 rows (+9% of a verify) against +0.29 (+5%) for the inexact taps.

## A6. Can 16 / 32 rows be cheap on the ANE? The cost build-up (analysis)

Per target group at 8K (ms; the served chain8 group = 5.06), each term measured in A2 / A5:

| term | 16 rows | 32 rows |
|---|---|---|
| FFN | +0.06 | +0.12 |
| DeltaNet solve, best form (block forward substitution) | +0.24 | +0.70 |
| tree conv taps (one-hot matmul; the exact form costs more) | +0.66 | +1.58 |
| attention at 8K of KV (at 64K: +1.45 / +3.73) | +0.19 | +0.53 |
| group total | +1.15 (+23%) | +2.93 (+58%) |
| head: two / four 8-row head calls (+5.5 / +16.5 ms per verify), or a wider head entry (FFN-like, about +0.2) | | |

Projected verify cost with the best forms: 16 rows about +23% (+29% with two head calls), 32 rows about +58%; at 64K
the attention adds +25% / +63% more unless the split path's GPU attention takes 16 / 32 rows (its kernel is 8-row).
With the realistic-set simulator (A1b), a dynamic chain / 8 / 16 / 32 policy gains +2.9-5.2% over the chain if 16 / 32
rows cost +14% / +30%, +4-7% at +10% / +20%, +8-10.5% at +5% / +10%. At +23% / +58% it gains about what tree8 alone
gains. **What would make wide trees pay here:** a cheap tree conv (the taps are now the largest term: a form that
gathers only the 3 rows before each branch, or a conv-friendly tree layout), the block solve, a 16 / 32-row head
entry, and for long contexts the split verify's GPU attention at 16 / 32 rows. Each is a build plus runtime work; not
started tonight (the projected gain over tree8 is 2-4% at the reachable costs).

## B. JetSpec++ (JetX) by replay on our target (`jetspec/`)

**Capture** (00:48-00:53, forge down; `v8_work/jetspec/phaseB.sh capture` -> `jsp_capture_server.sh` on port 8771, the
served configuration of v8_server_dec8_qk.sh with `pipe/v8_server_tapdump.py`; banner in
`/Volumes/SSD4TB/anemll-forge-research/jetspec_pp/capture_gsm8k32_20nov/banner.txt`: v8_server.py f0be4385, dylib
ea591258, libdec_gpu_clk 2b64e979, build v8_dec8 manifest b59b6319, ka on): gsm8k prompts 0-31 of
`27b-prompts_gsm8k128.json` (sha 2ac6748d), greedy, thinking off, presence 0, max 2,112 tokens. 32 traces (915 MB),
14,984 generated tokens (one capped at 2,112). The served DFlash2 chain on these: 6.4-7.3 tok/verify per request
(`jetspec/capture_decode_lines.txt`).

**Replay memory (fixed tonight):** the full replay had never run on the real artifacts. It peaked above the free
25 GB (the swap watchdog stopped it twice at +1.5 GB): Metal allocations do not show in RSS. Changes in
`jsp_replay.py` (copies `jsp_replay.pre_resume.py`, `jsp_replay.pre_mem.py` kept): the B16 drafter head is loaded
before the target (its CPU build + CPU state, 6.4 GB transient, no longer stacks on the target); the hybrid decoder's
verification read of the tail tensors (3.5 GB, dropped by the reference) checks the pins, names and header shapes only;
the 2.5 GB CPU embedding table is memory-mapped from the checkpoint; resume from the output file. Then: MPS 16.6 GB
after load, swap +2.2 GB during load and back to +0 after (watchdog limit +3 GB for the replay). Later (02:24), with
10 GB of file cache left by the captures, two replays were stopped again at the load (+3.5 / +4 GB): the adapter is now
preloaded before the target as well (its 8.3 GB file streams through the cache while the most memory is free); then
swap does not grow at all during the load.

**Check** (1 trace x 64 tokens): `--rows all` and `--rows outcome` give the same accept lengths
[7, 4, 6, 1, 3, 16, 3, 16, 7] (9 / 9 hits, lossless), 13.77 against 1.97 s per round.

## A7. Tree8 like for like (offline, the chain's own texts, measured costs)

The served tree8 verify cost, measured in the branched run (server `tree:` lines, per context entry): 8K entry 99.9 /
95.5 ms (x1.046), 32K 112.8 / 108.8 (x1.036), 64K 127.5 / 123.5 (x1.033), 96K split 133.4 / 127.0 (x1.050). With
these ratios in the simulator (A1b; `tree_dyn_real/meas_*.json`), tok/s against the chain:

| rung | tree8 always | tree8 when branched (served default) | dynamic chain / 8 / 16 / 32 (16 / 32 at +29% / +58%) | tree8 when branched, exact conv form (x1.096) |
|---|---|---|---|---|
| 2K | +1.2% | +3.0% | +2.9% | +0.7% |
| 8K | -1.2% | +1.0% | +2.2% | -0.8% |
| 32K | -0.8% | +1.5% | +2.0% | -0.4% |
| 64K | -1.7% | +0.5% | +0.4% | -1.4% |
| 96K | -1.3% | +0.6% | +1.8% | -1.6% |
| chat / code / math | +7.8 / -3.0 / -2.7% | +8.0 / +0.3 / +0.6% | +9.5 / +0.0 / +1.1% | +3.8 / -0.3 / -0.1% |

So the bench's +1-11% (A3) is mostly the replies changing; like for like, tree8 adds +0.5-3% (chat +8%) with the
inexact conv taps, and about nothing with the exact form.

**Fixed tree shapes** (so the conv taps would be compile-time slices, exact and nearly free): 1,990 cycles have 707
distinct top-7 shapes (the chain 45%, no other above 0.8%). "Chain of L + leaf siblings" layouts reach 4.57-4.61
tok/verify alone (chain 4.953) and 5.09 / 5.14 / 5.18 with an oracle choice among the best 2 / 3 / 4 shapes per cycle,
against 5.12 for the dynamic tree without an oracle. Not pursued.

**Replay, J3-R3 over the 32 traces** (01:15-01:57, `jetspec/replay_j3r3.json` / `.log`, `compare_j3r3.txt`;
banner: JetSpec 41a30357, rows outcome, dataset gsm8k, release pins checked, bf16 on MPS, max 2,048 new tokens):
- **Lossless on all 32** (output = the captured stream), 1,519 rounds, 1.50 s per round (the reference on MPS).
- **JetSpec++ 9.50 tokens per round against our DFlash2 chain's 5.75 tok/verify on the same prompts: x1.65.**
  Per trace 6.9-14.2 against 4.6-7.3. Accept lengths: 16 (the whole B16 block) in 33% of rounds; 1-3 in 23%.
- **Projection** (served 2K cycle: draft 13.2 + verify 94.5 ms; tok/s = tok/round / (draft + verify x ratio)): with a
  16-row chain verify at x1.14 (A2's chain form) and a 30 ms drafter, 68.9 tok/s against DFlash2's 53.4 (+29%); at
  x1.26 and 60 ms, break-even.
- **Integration is not possible tonight:** JetSpec++'s drafter is the B16 head (1.6B parameters, 3.2 GB bf16) plus the
  adapter (4.47B, 8.3 GB) plus the anchor decoder (the target's layers 62 / 63 on the tap-61 features): about 12 GB in
  bf16 next to our 10 GB target on a 32 GB Mac, and the reference needs 1.5 s per round on the GPU. A served version
  needs a quantized ANE port of the head and adapter (about 3 GB at 4 bits; one pass per round after the verify) and a
  16-row chain verify entry (next section).

**What a B16 block costs our verify** (`chain16/`, `v8/build_chain16_probe.py`, scratch, deleted; forge.py compile
59 s; `tree_decomp_time.py`, median of 30). One target group in the served verify form, and the LUT4 head with 8- and
16-row entries in one package (one weight copy):

| | 8 rows | 16 rows, served solve | 16 rows, block forward substitution |
|---|---|---|---|
| group, 8K of KV | 5.05 ms | 5.76 (+14%) | 5.50 (+8.9%) |
| group, 32K / 64K | 5.80 / 6.75 | | 6.83 (+18%) / 8.63 (+28%) |
| head (final norm + lm_head) | 4.29 | 4.30 (+0.4%) | |

- A whole 16-row CHAIN verify (16 groups + head) is x1.085 at the 8K entry, x1.17 at 32K, x1.27 at 64K (there the
  split verify's GPU attention would take the rows instead). A chain needs no tree conv taps, so B16 chains suit
  this target much better than 16-row trees (+23%, A6).
- **JetSpec++ projection** with these costs (2K: tok/s = 9.50 / (drafter + 94.5 x 1.085)): 82 / 72 / 59 tok/s with
  a 13 / 30 / 60 ms drafter, against DFlash2's 53.4: +54% / +34% / +10%. The drafter's cost on the M6 decides it.

**What the JetSpec++ drafter would cost here (from its artifacts):** the adapter (`adapter/head.safetensors`, 4.47B
parameters) is a "ctxkv" head: 5 streams of target-sized layers (stream depths 1 / 1 / 1 / 4 / 5 = 12 layers, each
an attention + a 17,408-wide SwiGLU initialized from the target) that turn the target's tap features of each newly
committed token into the drafter's context K / V. So per round it runs on about 9.5 new tokens, not on the 16 draft
positions. At 4 bits on the ANE (bandwidth-bound, about 120 GB/s): adapter 2.2 GB, about 18 ms; B16 head (1.6B,
0.8 GB), about 7 ms; anchor decoder (target layers 62 / 63 on the tap-61 feature, about 0.4 GB), about 3 ms. That is
about 28 ms per round, the projection's 30 ms row (+34% at 2K), if 4-bit weights keep its acceptance. Memory: about
3.4 GB more beside the served stack.

**Beyond gsm8k** (02:09-03:11; `make_pinned.py`: the first user turn of humaneval-t0 / mbpp-t0 / mt-bench-t0 0-7
re-rendered in the 27b files' no-think form and tokenized with our tokenizer, the rendering checked against
27b-prompts_gsm8k128's ids; capture as above; replay with each set's own calibration entry `--dataset`; mt-bench 0 is
9 tokens long, shorter than the reader's 16-token margin, so it is skipped):

| set (prompts) | JetSpec++ J3-R3 tok/round | our DFlash2 chain tok/verify | ratio | rounds | lossless |
|---|---|---|---|---|---|
| gsm8k (0-31) | 9.50 | 5.75 | x1.65 | 1,519 | 32 / 32 |
| humaneval (0-7) | 13.20 | 7.30 | x1.81 | 190 | 8 / 8 |
| mbpp (0-7) | 10.55 | 5.89 | x1.79 | 100 | 8 / 8 |
| mt-bench (1-7) | 4.94 | 3.48 | x1.42 | 710 | 7 / 7 |

JetSpec++'s B16 blocks give x1.4-1.8 more tokens per verify on code, math and chat. Compare tree8 (A7): x1.03-1.08.

**Does the JetSpec++ drafter survive an ANE port's bit width?** (`jsp_replay.py --quant`: round-to-nearest symmetric
fake quantization of every Linear weight of the adapter and the B16 head, per output row and 128-input group, in place;
gsm8k traces 0-7, J3-R3, rows outcome; `jetspec/replay_q_*.json` / `.log`):

| drafter weights | tok/round, traces 0-7 | vs bf16 | lossless |
|---|---|---|---|
| bf16 (reference) | 10.23 | | yes |
| adapter 4 bits, head 4 bits (6.2B parameters) | 9.28 | -9.3% | 8 / 8 |
| adapter 3 bits, head 4 bits (traces 0-1 only, stopped) | 4.04 (bf16 on those: 11.58) | -65% | 2 / 2 |
| our DFlash2 chain on the same 8 prompts | 6.08 tok/verify | | |

Plain 4-bit RTN keeps x1.53 over our DFlash2 chain; 3 bits breaks the adapter (a calibrated or LUT quantization, as
our target uses, may do better; not tested). So an ANE port needs the adapter at 4 bits or more: about 3.4 GB.

**Projected decode at 2K per set** (DFlash2: its tok/verify / 107.7 ms, the served cycle; JetSpec++: tok/round /
(30 ms drafter + 94.5 ms x 1.085 16-row chain verify); the 4-bit column applies the -9.3% of the 4-bit drafter):

| set | DFlash2 chain tok/s | JetSpec++ bf16 drafter | JetSpec++ 4-bit drafter |
|---|---|---|---|
| gsm8k (0-31) | 53.4 | 71.7 (+34%) | 65.0 (+22%) |
| humaneval (0-7) | 67.8 | 99.6 (+47%) | 90.3 (+33%) |
| mbpp (0-7) | 54.7 | 79.6 (+46%) | 72.2 (+32%) |
| mt-bench (1-7) | 32.3 | 37.3 (+15%) | 33.8 (+5%) |

These are projections (the drafter's 30 ms is a bandwidth estimate, not measured on the M6).

## A8. Exact-conv tree8 build: identity, control, branched bench (`bench/tree8c_*`, `bench/chain_v2_ctl`, `tree_exact/`)

**Build** (03:32-04:47, forge down): `build_dec8t_all.sh` with `TREE_FORM=corr` -> `v8_dec8t` + `c` = `v8_dec8tc`
(9.9 GB, GPU tables `pipe/w_dec8tc`; packages +3 MB each, Core AI entry 1.19 GB as served). Launch:
`BUILD=.../v8_dec8tc GM=pipe/w_dec8tc/manifest_best.json PORT=8765 pipe/v8_server_tree.sh`.

**Controls:**
- **The served chain is deterministic across restarts:** `chain_v2_ctl` (served launcher, 2K + 8K) = `chain_v1` in 18
  of 18 replies and in tok/verify; its tok/s is +4.2% / +4.6% (same texts), so run-to-run tok/s noise is about 4%.
- **The new build's chain path is the served one:** v8_dec8tc with `TREE=0`, 2K: 9 / 9 identical (same output
  checksum 2b08f021 as chain_v2).

**Identity of the tree entries** (`TREE_POLICY=always TREE_TOPC=1`: every cycle through the tree entry, the tree
always the chain): 3 / 9, 3 / 9, 4 / 9, 5 / 9, 8 / 9 replies identical at 2K-96K. The direct check on package 0
(`pipe/unit/power/tree_entry_check.py`: `g*_v8_8k` and `g*_t8_8k` on the same random inputs, chain masks):
- exact build: y, taps, conv states and the new K / V rows bit-identical; only the DeltaNet pending tapes `pend1_out`
  / `pend2_out` differ (rel 1-3e-7). The in-block cumulative decay is a matmul with the ancestor-mask INPUT where
  the chain entry multiplies by the constant triangle; the tiny tape differences enter the next call's commit and
  later flip near-ties. Fix (not built: another 1.3 h build): the constant cumsum + a correction exactly 0 for
  chain rows, as for the conv.
- 19 Nov form (`v8_dec8t`): y rel 1.8-2.1e-3, every state output differs.

**Branched bench, exact-conv build** (`tree8c_branched`):

| rung | chain v1 tok/s, tok/verify | chain v2 (control) | tree8 19 Nov form | tree8 exact conv | identical (19 Nov / exact) |
|---|---|---|---|---|---|
| 2K | 40.99, 4.567 | 42.72, 4.567 | 44.52, 4.895 | 44.64, 5.008 | 2 / 4 of 9 |
| 8K | 45.98, 5.086 | 48.09, 5.086 | 50.92, 5.527 | 48.29, 5.357 | 2 / 3 |
| 32K | 44.05, 5.164 | | 46.84, 5.613 | 47.20, 5.726 | 3 / 3 |
| 64K | 38.38, 5.040 | | 38.79, 5.257 | 38.59, 5.299 | 7 / 6 |
| 96K | 34.03, 4.722 | | 35.21, 4.962 | 33.59, 4.885 | 8 / 7 |

Chain-shaped cycles through the exact tree entries (`tree8c_topc1`) cost +7% per cycle at 2K-32K (118 / 117 / 130 ms
against 110 / 108 / 121), so with the 4% tok/s noise and the diverging texts, the bench gains are mostly not tree
gains; the like-for-like estimate stays A7's. **Decision:** tree8 stays behind the flag (`TREE=1`), not the default:
it is not token-identical, and like for like it adds about 0-3%.

## B7. JetSpec++ quant ladder and resident memory (`b7/`; requested 05:0x)

**Resident memory, measured.** In the replay, MPS allocation after each load step (`MEM` lines, every run):

| piece (bf16 in the replay) | GPU allocated | note |
|---|---|---|
| JetSpec++ B16 head | 3.5 GB (driver 4.3) | |
| JetSpec++ ctxkv adapter | +9.0 GB (12.5 total; driver 12.9) | |
| replay target: layers 62 / 63, final norm, lm_head | +4.1 GB (16.6 total; driver 17.6) | not the served target |
| replay target embedding (2.5 GB table) | UNMEASURED | memory-mapped from the checkpoint (file pages) |
| process RSS (CPU) | 7.3 GB peak | the head's CPU build during load; not steady state |

The fake quantization below rewrites the bf16 weights in place: the resident sizes stay the bf16 ones, and the packed
size of each quantized form is UNMEASURED.

**DFlash2 drafter alone** (`pipe/unit/power/drafter_mem.py`, the served package
`drafter/all3/dflash2_lut4_gptq_nohead.aimodel`, main.hash 7284b3cf1ef40529, Core AI bridge on the ANE; forge down,
second run on a quiet machine, `b7/dflash2_mem2.json`):

| step | process RSS / phys_footprint | system wired | free | anonymous / file-backed |
|---|---|---|---|---|
| before | 0.029 / 0.019 GB | 2.271 GB | 13.458 | 1.931 / 15.707 |
| package loaded (functions not bound) | 0.034 / 0.021 | 2.271 | 13.456 | 1.932 / 15.707 |
| draft + ctx64 bound and run twice | 0.085 / 0.069 | 3.108 (+0.837) | 13.358 | 1.934 / 14.967 |
| + selector tables (2 x 127 MB bf16, here as copies) | 0.502 / 0.199 | 3.095 | 13.222 | 2.083 / 14.967 |

So the served DFlash2 holds about 0.84 GB wired (its ANE program and weights) plus about 0.05 GB of buffers in the
process; its selector tables add about 0.25 GB (mapped from the file in the server: file pages). It drafts through
the target's own head, which stays. Dropping it saves about 0.84 GB wired + 0.25 GB of file pages (measured pieces;
the served process's total was not re-measured with and without it).
