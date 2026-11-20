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
  - **The exact conv form costs too much:** it was built and served (A8). It matches the chain entry bit for bit
    except the DeltaNet pending tape (rel 1-3e-7, from the in-block decay matmul on the mask input), and that is
    still enough to flip near-ties: 3-8 of 9 replies identical per rung. Its chain-shaped cycles cost +7%. The served
    chain is deterministic across restarts (18 / 18 identical), with about 4% tok/s noise between runs.
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
  is free), x1.17 at 32K and x1.27 at 64K. The +34% projected earlier assumed a single 30 ms drafter pass per
  round. The replay counters show several passes per round (B8), so that projection is withdrawn.
- **Drafter bit width (B7):** int8 per channel and 6-bit g128 (adapter + head) are lossless (within 1%). 4-bit RTN
  g128 loses 9%, and a per-row LUT4 on the adapter loses 7-17% (no rotation or calibration: not our pipeline). 3 bits
  collapses.
- **But it does not pay on this M6 as a serial ANE drafter (B8):**
  - **Drafter cost:** the reference schedule J3-R3 runs per round about 4.5 B16-head passes, 4.5 adapter passes and
    3.4 decoder passes. The adapter alone at LUT6 on the ANE measured 27.2 ms per pass (3.35 GB at 123 GB/s; 16
    rows, 8K of KV) and holds 3.7 GB wired.
  - **Projection** (8K entry, measured pieces, head / decoder passes projected; LUT6 timing; LUT6 acceptance -2..7% at J3-R3, about 0 at J0-R0): 14-45% slower than our DFlash2 at
    every schedule. Even J0-R0, which keeps DFlash2's acceptance or beats it by up to 18%, drafts in 43 ms against
    DFlash2's 13 ms.
  - **What would make it pay:** an adapter 2-3x cheaper per pass, or drafting that overlaps the verify.
- **Memory:** DFlash2 measured 0.84 GB wired (+0.25 GB selector file pages); the LUT6 adapter alone 3.7 GB wired.

**Batch D (11:45-15:20; sections D1-D4):**
- **D1, the 09:02 forge exit:** forge's own swap watchdog (`watchdog_v8_server.log`: swap 1071 -> 3173 MB > its +1536
  MB limit) while the C3 replay held 17.6 GB of GPU memory next to forge's 19 GB wired. Contention only. Forge alone,
  cold Pi-shaped prefills of 7,009 / 7,009 / 6,213 tokens: 675 tok/s, swap flat at 846.75 MB. Fix in the tooling:
  `jsp_replay.py` refuses to start while 8765 serves.
- **D2, LUT4 (Hadamard + GPTQ) adapter on the ANE, quiet:** 22.2 ms per pass at 8K, 36.9 ms at 32K (16 rows; LUT6
  27.2 / 44.8); 2.55 GB wired for the 8K build (LUT6 3.67). J0-R0 acceptance at LUT4 6.07 / 8.53 / 6.59 (bf16 6.12 /
  8.59 / 6.64). Jet Flash serial: 8K J0-R0 43.1 / 59.6 / 46.4 tok/s vs DFlash2 56.5 / 67.8 / 54.7 (-12 to -24%), 32K
  -22 to -32%, J3-R3 -33 to -46%.
- **D3, the same adapter on the GPU** (Metal LUT4 kernels, 7 designs): best 38.2 ms per pass without attention (59
  GB/s of 155 GB/s raw read), 39.9-51.3 ms with it (ANE 22.2 / 36.9). Even at the GPU's read-bandwidth floor
  (projected) Jet Flash stays 7-34% below DFlash2: the 102-125 ms 16-row verify is most of the round.
- **D4, cheap exact tree conv: found.** `TREE_FORM=corrx0` (the served conv + one channel-major matmul with D = ctap -
  a constant chain shift) is bit-identical to the chain entry under chain masks at +4.5% per group (A8's corr +11.2%).
  A8's inexactness was the ANE's one-hot gather, not the decay matmul.
  - **Identity:** full build `v8_dec8tx`; every cycle through the tree entry with chain trees: 45 / 45 replies
    byte-identical to the served chain (2K-96K).
  - **Control:** the chain itself with only a different cycle split keeps 20 / 45. So branched tree8 (23 / 45) cannot
    be token-identical to the chain, and diverges no more than a split change does.
  - **Cost and gain:** tree verify x1.024-1.039; like for like +0.9 to +3.5% tok/s (chat +9%).
  - **Not made the default:** needs a decision on the identity criterion.
- **Served at the end:** forge on 8765 (15:11, pid 50586, `v8_server_dec8_qk.sh`), smoke byte-identical.

**Batch C, Jet Flash estimates (08:40-11:45; sections C1-C5):**
- **C1, hybrid split** (adapter attention on the GPU's `dec_attn_r4`, projections and MLPs on the ANE at LUT6, 16
  rows): one 12-block adapter pass sums to 30.7-32.4 ms at 8K (all on the ANE: 27.2 measured), 35.6-42.1 at 32K
  (44.8), 41.9-54.7 at 64K (64.7), 48.4-67.8 at 96K (85.8). The handoffs are UNMEASURED. It pays from 32K up only.
- **C2, overlap** (two processes: Splash 1.7.0 GPU-only 16-row verify + an ANE LUT6 adapter-pass loop): Splash
  decode 58.7 -> 38.2 tok/s (-35%), ANE pass 29.5 -> 45.0 ms (+53%). They contend for DRAM. Best case, J0-R0 hidden
  under the slowed verify: about 35 tok/s vs DFlash2 56.5 (gsm8k, 8K).
- **C3, a thinner adapter: no.** Dropping one block costs -35% tok/round (gsm8k J1-R1 8.54 -> 5.55); dropping the 7
  extra layers collapses it (1.1). The MLPs are not the target layer + a low-rank delta: |delta| / |W| 0.86-0.97,
  90% of the delta's energy needs rank 2,494-3,615 of 5,120.
- **C4, LUT4 adapter with Hadamard + GPTQ: lossless for acceptance.** J3-R3 tok/round 10.17 / 13.55 / 10.45
  (gsm8k / humaneval / mbpp) vs bf16 10.23 / 13.19 / 10.55 (-0.6 / +2.7 / -1.0%), against -13 / -10 / -22% for plain
  per-tensor LUT4. Calibrated on held-out gsm8k traces 8-15. On the ANE: 22.1 ms per adapter pass (LUT6 27.2,
  -19%), 3 x 780 MB packages (LUT6 3.35 GB); resident size UNMEASURED.
- **C5, Jet Flash projection with all of it** (adapter LUT4 measured, head LUT6 and decoder projected, 16-row
  verify measured ratio, serial ANE): still 12-38% below the served DFlash2 chain (best: humaneval J0-R0 59.9 vs
  67.8 tok/s). The verify is 72-80% of a cheap round.
- **Saved queue (item 5):** nothing left to run. WL3A8 temp build, items 9 / 11 / 12 / 13 and the QKE=tg deploy were
  done on 19 Nov (19 Nov README); item 10 (hybrid GPU + ANE drafter split) is covered by C1.
- **Caveat:** forge stayed loaded, idle, on 8765 during C1, C2 and the first C3 replay. The ANE and Splash timings
  ran beside an idle resident forge. At 09:02:49 a Pi request (5,939 tokens, thinking medium) reached it while the C3
  replay held 17.6 GB of GPU memory, and forge exited 6 s into the prefill (no kernel jetsam kill logged; its stdout
  was /dev/null, so the cause is UNKNOWN). Restored at 11:37, see below.

**Served state at the end** (08:3x, after the JetSpec++ work as Anemll asked):
- **Restored again at 11:37** (pid 50952, `pipe/v8_server_dec8_qk.sh`), after the 09:02:55 exit above. Smoke
  `bench/final_served2`: banner matched (qk=attn_qk8isk_tg, dec8=libdec_gpu_clk.dylib), and humaneval 1 / mt-bench 2
  replies byte-identical to 08:34's (sha 1681b92d273f64ad / 3b060108f4af10b0), 71.2 / 22.8 tok/s.
- **Kept on the SSD from batch C:** `jetspec_pp/adapter_lut4h` and `adapter_lut4hg` (LUT4 tensors), `calib_inputs`
  (4.5 GB of captured adapter inputs; delete when no more GPTQ runs are planned), `calib_gsm8k_8_15` (symlinks). The C1
  and C4 scratch ANE builds are deleted (Core AI entries first).
- **Forge:** on 8765 in the served configuration (`pipe/v8_server_dec8_qk.sh`, build v8_dec8, DFlash2 chain). Smoke
  check: banner matched, and humaneval 1 / mt-bench 2 replies byte-identical to the night's chain baseline.
- **Tree verify:** stays opt-in. `PORT=8765 pipe/v8_server_tree.sh` (build v8_dec8t, 19 Nov conv form), or with
  `BUILD=/Volumes/SSD4TB/anemll-forge-research/v8_dec8tc GM=pipe/w_dec8tc/manifest_best.json` (exact-conv build).
- **Disk:** both tree builds are kept on the SSD (9.9 GB each) with their Core AI entries on the internal disk (about
  10.7 GB each, aned's programs about the same again; internal free 53 GB). To drop one:
  `scripts/coreai_cache_delete.py --delete <build>`, then delete the files.
- **Kept on the SSD:** the JetSpec++ captures (`jetspec_pp/capture_*_20nov`, about 1.2 GB) and the LUT6 adapter
  tensors (`jetspec_pp/adapter_lut6`, 4.2 GB).
- **Deleted:** every scratch probe build (decomposition, GDN WY, chain16, the ANE adapter), Core AI entries first.
- **Commits:** local, on `v8-gpu-ane-prefill` (not pushed).

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
about 28 ms per adapter pass (measured later at LUT6: 27.2 ms, B8). The schedule runs several passes per round
(B8), so this is not the round cost. Memory: about 3.4 GB more beside the served stack (LUT6 measured: 3.7 GB wired).

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

**Correction (B8, 07:5x):** that table assumed one 30 ms drafter pass per round. The replay's own counters show
J3-R3 running about 4.5 head + 4.5 adapter + 3.4 decoder passes per round, and the adapter pass alone measured
27.2 ms at LUT6 on the ANE. With those, every schedule projects 14-42% slower than DFlash2 (B8). The table above
is superseded.

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

**Quant ladder** (05:29-06:49; `jetspec/b7_ladder.sh` -> `jsp_replay.py --multi gsm8k=...,humaneval=...,mbpp=... --limit 8 --quant ...`: one process per setting, the models loaded once, the drafter rebuilt per set with its calibration entry; fake quantization in place on MPS; banner per set in `b7/b7_*_<set>.json`; swap flat in every run). tok/round on the same traces:

| drafter weights | gsm8k tok/round (vs bf16) | humaneval tok/round (vs bf16) | mbpp tok/round (vs bf16) | lossless |
|---|---|---|---|---|
| bf16 | 10.23 | 13.19 | 10.55 | yes |
| int8 per channel, adapter + head | 10.17 (-0.6%) | 13.26 (+0.5%) | 10.66 (+1.0%) | yes |
| 6-bit g128, adapter + head | 10.20 (-0.3%) | 13.19 (+0.0%) | 10.45 (-1.0%) | yes |
| LUT4 per row, adapter (head bf16) | 9.05 (-11.5%) | 12.23 (-7.3%) | 8.72 (-17.4%) | yes |
| LUT3 per row, adapter (head bf16) | 2.67 (bf16 on the same 2: 11.58, -77%) | not run | not run | yes |
| our DFlash2 chain (tok/verify, same prompts) | 6.08 | 7.30 | 5.89 | |

- **8-bit per channel and 6-bit g128 (adapter + head) are lossless for acceptance** (within +-1%).
- **Per-row LUT (one 2^b palette per output row, k-means, no rotation, no calibration):** 4 bits loses 7-17%; 3 bits
  collapses (2.6-2.7 tok/round). This is not our quantization pipeline (the target uses Hadamard rotations, GPTQ
  and vector LUTs); a rotated and calibrated 4-bit form is the open test. 4-bit RTN g128 (adapter + head, gsm8k) lost
  9.3% (B above).
- The LUT3 run was stopped after 2 gsm8k traces (clear collapse, about 2 min per trace).

## B8. JetSpec++ 6-bit / LUT6 baseline on the ANE (requested 06:4x; forge stays down per Anemll)

**What runs per round (measured, replay counters).** J<m>-R<n> runs J0, then n refreshes (R: one adapter predict +
one anchor-decoder pass each) interleaved with m more B16 head passes (J), and one final adapter predict per round.
bf16, gsm8k / humaneval / mbpp 0-7 (`jetspec/b8_sched.sh`, `b8/b8_sched_*`, `jetspec/b8_passes.py`; every run
lossless):

| schedule | set | tok/round | head passes / round | adapter passes / round | decoder passes / round |
|---|---|---|---|---|---|
| J3-R3 | gsm8k | 10.23 | 4.29 | 4.29 | 3.21 |
| J3-R3 | humaneval | 13.19 | 4.65 | 4.65 | 3.49 |
| J3-R3 | mbpp | 10.55 | 4.80 | 4.80 | 3.60 |
| J0-R0 | gsm8k | 6.12 | 1.01 | 1.01 | 0.00 |
| J0-R0 | humaneval | 8.59 | 1.08 | 1.08 | 0.00 |
| J0-R0 | mbpp | 6.64 | 1.05 | 1.05 | 0.00 |
| J1-R0 | gsm8k | 6.98 | 2.11 | 1.06 | 0.00 |
| J1-R0 | humaneval | 9.79 | 2.25 | 1.12 | 0.00 |
| J1-R0 | mbpp | 7.64 | 2.13 | 1.07 | 0.00 |
| J1-R1 | gsm8k | 8.54 | 2.16 | 2.16 | 1.08 |
| J1-R1 | humaneval | 11.94 | 2.40 | 2.40 | 1.20 |
| J1-R1 | mbpp | 8.72 | 2.30 | 2.30 | 1.15 |

The DFlash2 chain on the same prompts: 6.08 / 7.30 / 5.89 tok/verify. So J0-R0 (one head pass + one adapter
predict per round) already matches or beats our DFlash2 on these sets (+1% / +18% / +13%), J1-R1 adds
+40% / +64% / +48%, J3-R3 +68% / +81% / +79%, at about 1 / 2.2 / 4.5 heavy drafter passes per round.

**LUT6 in the ANE build's form** (`jetspec/lut_tensor.py`, `jetspec/quant_adapter_lut6.py` ->
`/Volumes/SSD4TB/anemll-forge-research/jetspec_pp/adapter_lut6`, 4.2 GB): every adapter matrix as one 64-entry palette
per tensor (Lloyd on a histogram of w / max|w_row|) + a per-output-channel scale, the form `QConv` takes. 84
matrices, weight error rel 3.97% mean (5.6% max); 6-bit g128 RTN (lossless in B7) is about 2.4%, LUT4 in this form
about 10%. Row normalization by RMS or a Hadamard rotation first changed the error by less than 0.2 points (the
adapter's weights are already near-Gaussian).

**The adapter on the ANE at LUT6, measured** (`v8/build_jsp_adapter_ane.py`, `v8/build_jsp_adapter_all.sh`; a scratch
latency build, deleted after): the 12 blocks built as the target builder's own attention + MLP layers (target
attention indices 3-47, nothing of the target read) on the LUT6 files above, 3 packages of 4 blocks, 1,118 MB each
(0.75 byte per weight: 6-bit indices), forge.py compile 1m22s, Core AI entries 1.12 GB each (no fallback). Entries:
16 query rows (the B16 window) against an INT8 KV history, per block rms -> attention -> + -> rms -> MLP -> +. The
16 rows' own keys / values come from the block input and the kv_norm is skipped (the real stream reads them from its
tap's prefix): the weight traffic, which sets the time, is the real one. `tree_decomp_time.py`, median of 30
(banner in `b8/ane/timing_8k.log`, packages sha ad831782afd41262 / ...):

| piece | 8K of KV | 32K of KV |
|---|---|---|
| 4 blocks (one package) | 9.08 / 9.07 / 9.09 ms | the 32K entry was dropped at compile ("no procedureInfo entry"): UNMEASURED |
| one adapter pass (12 blocks) | **27.2 ms** (3.35 GB in 27 ms: 123 GB/s, the ANE's bandwidth) | UNMEASURED |
| resident when loaded and run (`pipe/unit/power/ane_pkg_mem.py`, `b8/ane/adapter_mem.json`) | wired +3.67 GB (2.34 -> 6.01), process +0.20 GB (KV inputs); +1.16 GB stayed wired after release | |

Not built: the B16 head (1.73B parameters, a DFlash architecture the target builder does not have) and the anchor
decoder. Their pass times below are PROJECTED from measured rates (head: its LUT6 bytes at the adapter's 123 GB/s,
10.5 ms, + the target head on 16 rows, 4.30 ms measured; decoder: target layers 62 / 63 at 16 rows from the
measured 16-row group, 2.75 ms, + 4.30).

**Projected decode at the 8K entry** (`jetspec/b8_project.py`; serial: draft, then a 16-row chain verify at 102.5 ms;
bf16 tok/round; DFlash2: the served 107.7 ms cycle and its tok/verify on the same prompts):

| schedule | set | drafter ms / round at LUT6 on the ANE (adapter + head + decoder) | round ms | acceptance, tok/round: bf16 replay (LUT6 replay) | JetSpec++ tok/s (LUT6 timing; LUT6 acceptance where measured) | DFlash2 tok/s | vs DFlash2 |
|---|---|---|---|---|---|---|---|
| J0-R0 | gsm8k | 43 (28 + 15 + 0) | 146 | 6.12 (6.21) | 42.5 | 56.5 | -25% |
| J0-R0 | humaneval | 45 (29 + 16 + 0) | 149 | 8.59 (8.53) | 57.2 | 67.8 | -16% |
| J0-R0 | mbpp | 44 (29 + 16 + 0) | 148 | 6.64 (6.51) | 44.1 | 54.7 | -19% |
| J1-R0 | gsm8k | 60 (29 + 31 + 0) | 163 | 6.98 | 42.7 | 56.5 | -24% |
| J1-R0 | humaneval | 64 (31 + 33 + 0) | 167 | 9.79 | 58.5 | 67.8 | -14% |
| J1-R0 | mbpp | 61 (29 + 32 + 0) | 164 | 7.64 | 46.6 | 54.7 | -15% |
| J1-R1 | gsm8k | 98 (59 + 32 + 8) | 202 | 8.54 | 42.3 | 56.5 | -25% |
| J1-R1 | humaneval | 109 (65 + 36 + 9) | 213 | 11.94 | 56.1 | 67.8 | -17% |
| J1-R1 | mbpp | 105 (62 + 34 + 8) | 208 | 8.72 | 41.9 | 54.7 | -23% |
| J3-R3 | gsm8k | 203 (117 + 63 + 23) | 306 | 10.23 (9.98) | 32.6 | 56.5 | -42% |
| J3-R3 | humaneval | 220 (127 + 69 + 25) | 324 | 13.19 (12.99) | 40.1 | 67.8 | -41% |
| J3-R3 | mbpp | 227 (131 + 71 + 26) | 331 | 10.55 (9.86) | 29.8 | 54.7 | -45% |

Every row is timed at LUT6 on the ANE (the measured 27.2 ms adapter pass; no bf16 ANE timing exists, and bf16 would
move 2.7x the bytes). The bf16 / LUT6 numbers are acceptance only, from the GPU replays. Where the LUT6 replay ran
(J0-R0, J3-R3), tok/s uses its acceptance.

**LUT6 acceptance** (adapter from the ANE build's own LUT6 files, head in the same form; `--quant adapter:lut6f,head:lut6t`,
`b8/b8_lut6t_*`; every trace lossless): J3-R3 9.98 / 12.99 / 9.86 tok/round (bf16 10.23 / 13.19 / 10.55: -2.4 / -1.5 /
-6.6%); J0-R0 6.21 / 8.53 / 6.51 (bf16 6.12 / 8.59 / 6.64: +1.4 / -0.7 / -2.0%).

**So a serial ANE JetSpec++ is 14-45% slower than our DFlash2 chain at every schedule.** Even J0-R0 (one head + one
adapter pass) drafts in 43-45 ms against DFlash2's 13 ms; J1-R1 would need drafting at 48 ms or less per round to
break even (it takes 98-109). Measured end-to-end tok/s: UNMEASURED (no served JetSpec++ path: it needs the B16
head port, the adapter's real attention and its own 12-layer context KV cache, a 16-row chain verify entry, and the
round driver in the server). What could change it: an adapter 2-3x cheaper per pass (it is 72% of the parameters;
at 4 bits per-row LUT loses 7-17% acceptance, B7), or drafting that overlaps the ANE verify.

## C. Jet Flash estimates (batch of 20 Nov ~08:40; forge down, Splash briefly up on 8011)

### C1. Hybrid split of the JetSpec++ adapter: MLPs (and projections) on the ANE, attention on the GPU (`c1/`)

**GPU attention core** (the served decode kernel `dec_attn_r4`: prep + attention + combine per layer;
`pipe/unit/dec_gpu_steady.py`, libdec_gpu_clk.dylib 2b64e979, dec_kernels.metal 00f351b5, keep-alive on; the adapter's
attention has the target's head layout, 24 q / 4 kv x 256, over an INT8 KV): 8 query rows, one layer, one call at a
time:

| KV keys | 8K | 32K | 64K | 96K |
|---|---|---|---|---|
| ms per layer, 8 rows | 0.141 | 0.546 | 1.071 | 1.613 (122-125 GB/s of K / V) |

16 rows lie between 1x (the K / V read is shared, as in the 16-row kernel Splash runs) and 2x (two 8-row calls with
this kernel; there is no 16-row variant of ours).

**ANE pieces of one block** (`v8/build_jsp_adapter_ane.py` with `JSP_PARTS=1`: streams.0 at LUT6, 16 rows; scratch,
deleted; forge.py compile 48 s; `tree_decomp_time.py`, median of 30, package sha d1497c4ad7221174):

| entry | ms |
|---|---|
| whole block on the ANE, 8K / 32K / 64K / 96K of KV | 2.48 / 3.73 / 5.39 / 7.15 |
| part A: rms + q (gate) / k / v projections + q / k norm + RoPE | 0.70 |
| part B: o projection + residual + MLP | 1.72 |
| MLP alone | 1.57 |

So the ANE's attention core per block costs about 0.06 / 1.31 / 2.97 / 4.73 ms at 8K / 32K / 64K / 96K (block minus
A minus B), against the GPU's 0.14-0.28 / 0.55-1.09 / 1.07-2.14 / 1.61-3.23.

**12 blocks (one adapter pass), sum of measured pieces; the ANE <-> GPU handoffs (12 GPU calls + 13 ANE calls if
B(i) + A(i+1) are fused as in the dec8 split) are not included, UNMEASURED:**

| KV keys | all on the ANE (12 single-block calls) | split: ANE A + B, GPU attention | change |
|---|---|---|---|
| 8K | 29.8 ms (27.2 measured with 4-block packages) | 30.7-32.4 | +3 to +9% |
| 32K | 44.8 | 35.6-42.1 | -6 to -21% |
| 64K | 64.7 | 41.9-54.7 | -15 to -35% |
| 96K | 85.8 | 48.4-67.8 | -21 to -44% |

The split pays from 32K up (where the adapter's 12 attention layers read the long context), not at 2K-8K, where the
adapter is 97% weight traffic: MLPs 57%, projections the rest.

### C2. Can the ANE draft overlap a GPU 16-row verify? (`c2/`, `pipe/unit/power/overlap_run.py`, `ane_pass_loop.py`)

**Setup:** our target has no GPU verify engine. The GPU 16-row verify of a 27B model here is Splash 1.7.0's, run
GPU-only (`--disable-ane`; git 61f6d516, binary sha 4c20b1d0, UD-IQ3_XXS rev 4ca72078, DFlash2 015e7956,
`--verify-rows auto`, 16K). Its own logged step costs: 2 / 8 / 15 / 16 rows 95.1 / 111.8 / 112.5 / 112.8 ms. The ANE
load is one adapter pass in a loop: 12 back-to-back calls of the LUT6 block (16 rows, 8K), each re-reading its 280 MB,
3.35 GB per pass like the real 12 blocks. Two processes. Prompts: realistic_v1 2K humaneval 1, gsm8k 0, math500 0;
256 tokens; 3 repeats each; greedy, thinking off.

| | alone | both running at once | change |
|---|---|---|---|
| Splash decode, tok/s (median of 9) | 58.7 | 38.2 | -35% |
| ANE adapter pass | 29.48 ms (1,016 passes) | 45.0 ms (median per request, 142-169 passes each) | +53% |

**They contend for DRAM** (both are bandwidth-bound): overlapping the two adds throughput (about 140 GB/s together)
but slows each. **Projection vs DFlash2's 107.7 ms cycle (8K entry, gsm8k):**
- GPU verify at Splash's 16-row step x 1.54 = 174 ms, with the J0-R0 draft (adapter 29.5 x 1.53 + head 14.8 x 1.53 =
  68 ms) fully hidden under it: 6.12 tok / 174 ms = 35 tok/s.
- J3-R3: the draft (about 203 ms serial, about 310 contended) does not fit under the verify: about 33 tok/s.
- DFlash2 chain: 56.5 tok/s.

Even with perfect overlap and no outcome misses, the contention leaves it 38-40% below the served chain. And our own
verify is on the ANE, so drafting on the ANE cannot overlap it at all.

### C3. A cheaper adapter (`c3/`)

**MLP as target layer + low-rank delta: no** (`jetspec/mlp_delta_svd.py`, `c3/mlp_delta_svd.json`; CPU float32
singular values, 722 s). Each block's gate / up / down minus the bf16 target layer its FFN was initialized from
(`resolve_ffn_layer_map`: 5, 19, 33, 37, 40, 44, 47, 50, 53, 56, 59, 61), 36 matrices:
- |delta| / |W| is 0.86-0.97 (median 0.92): training moved the MLPs almost as far as their own size from the init.
- 90% of a delta's energy needs rank 2,494-3,615 of 5,120 (50%: 578-1,327).
- W_target + the rank-r delta still misses W by 0.79-0.88 (r = 256), 0.57-0.72 (r = 1,024), 0.38-0.54 (r = 2,048),
  relative to |W|. A rank that small would not save anything (rank 2,048 is 40% of the dense matrix).

**Fewer adapter blocks (replay, bf16; skipped blocks pass their input through: `JSP_SKIP_BLOCKS` in jsp_replay.py):**

| adapter | schedule | gsm8k tok/round | vs the whole adapter |
|---|---|---|---|
| all 12 blocks | J1-R1 | 8.54 | |
| drop the last block (`extra_stream_layers.4.3`; 11 left) | J1-R1 | 5.55 | -35% |
| drop all 7 extra layers (5 left) | J1-R1 | 1.10-1.16 on traces 1-4 (run stopped) | collapse |

The adapter cannot be thinned without retraining: one block less already costs more acceptance than the 8% of the
pass it saves. The depth schedules (B8: J0-R0 to J3-R3) are the cheap knob that exists. Runs `c3/c3_skip_*`; the
two-block drop and the all-7 J0-R0 runs were stopped as uninformative.

### C4. LUT4 on the adapter with a Hadamard rotation and GPTQ (`c4/`; head kept bf16)

**Calibration:** the replay's capture hook (`JSP_CAPTURE` in jsp_replay.py) sampled the inputs of all 84 adapter
Linears, up to 48 rows per call and 4,096 per Linear (344,064 rows, 4.5 GB), on held-out gsm8k traces 8-15 (J3-R3,
bf16; that run itself 9.27 tok/round, lossless).

**Forms** (all the ANE build's: one 16-entry palette per tensor + a per-output-channel scale):
- `lut4t`: on the weights as they are, nearest entry.
- `lut4h`: on Hadamard-rotated weights (x' = R x, R = blockdiag(H_1024) diag(sign), the target builder's rotation
  form), nearest entry (`jetspec/quant_adapter_gptq.py ... rtn` -> `jetspec_pp/adapter_lut4h`). Weight error 10.0%.
- `lut4hg`: the same rotation, then GPTQ (blocks of 128 columns, damp 1%, nearest palette entry x row scale, error
  fed forward with the Cholesky of H'^-1) on the captured inputs (`... gptq` -> `jetspec_pp/adapter_lut4hg`). Weight
  error 11.5% mean, output error on the calibration rows 0.71% mean (1.73% max); unit check on a correlated random
  matrix: RTN 9.9% -> GPTQ 6.5% output error.

The replay reads the rotated forms back as W = (lut[idx] x scale) R (spec `lut4rf`; the round trip is exact to 6e-6).

**Acceptance** (replay, J3-R3, adapter at LUT4 in each form, head bf16, gsm8k / humaneval / mbpp 0-7, every trace
lossless; `c4/c4_*_J3-R3`):

| adapter weights | gsm8k | humaneval | mbpp |
|---|---|---|---|
| bf16 | 10.23 | 13.19 | 10.55 |
| LUT4 per row, k-means (B7) | 9.05 (-11.5%) | 12.23 (-7.3%) | 8.72 (-17.4%) |
| `lut4t`: LUT4 per tensor + row scale | 8.91 (-12.9%) | 11.83 (-10.3%) | 8.18 (-22.5%) |
| `lut4h`: + Hadamard | 9.36 (-8.5%) | 12.47 (-5.4%) | 8.72 (-17.4%) |
| **`lut4hg`: + Hadamard + GPTQ** | **10.17 (-0.6%)** | **13.55 (+2.7%)** | **10.45 (-1.0%)** |

**With the Hadamard rotation and GPTQ, LUT4 on the adapter is lossless for acceptance**, on two sets it was not
calibrated on (humaneval, mbpp) too. The rotation alone recovers about a third of the loss; GPTQ the rest.

**LUT4 adapter on the ANE, measured** (`build_jsp_adapter_all.sh` with `LUT=adapter_lut4hg JSP_ROT=1 CTXS=8192`: the
12 blocks with their 4 activation Hadamards each in the graph (q / k / v input, o input, MLP input, down input; the
timing form: the graph's signs are not the quantizer's), 3 packages x 4 blocks, 780 MB each (0.53 byte per weight);
compile 26 s; scratch, deleted; banner sha 651a9fe54918b551): 7.41 / 7.37 / 7.30 ms per package, **22.1 ms per
adapter pass** at 16 rows / 8K, against LUT6's 27.2 (-19%; bytes -33%: the Hadamards and per-call work take the
rest).

## C5. Jet Flash estimate with everything measured so far (8K entry, serial ANE)

Adapter at LUT4 GPTQ (22.1 ms measured), B16 head at LUT6 (14.8 ms, projected), decoder 7.1 ms (projected), 16-row
chain verify 102.5 ms (measured ratio):

| schedule | set | drafter ms / round | round ms | tok/round | Jet Flash tok/s | DFlash2 tok/s | vs DFlash2 |
|---|---|---|---|---|---|---|---|
| J0-R0 | gsm8k / humaneval / mbpp | 37 / 40 / 39 | 141 / 143 / 142 | 6.12 / 8.59 / 6.64 (bf16) | 43.5 / 59.9 / 46.6 | 56.5 / 67.8 / 54.7 | -23% / -12% / -15% |
| J1-R1 | gsm8k / humaneval / mbpp | 87 / 97 / 93 | 191 / 201 / 196 | 8.54 / 11.94 / 8.72 (bf16) | 44.7 / 59.5 / 44.4 | | -21% / -12% / -19% |
| J3-R3 | gsm8k / humaneval / mbpp | 181 / 196 / 203 | 284 / 300 / 306 | 10.17 / 13.55 / 10.45 (LUT4 GPTQ) | 35.8 / 45.2 / 34.1 | | -37% / -33% / -38% |

**Still 12-38% below the served DFlash2 chain.** Where it stands:
- The verify is 72-80% of the cheap schedules' round; JetSpec++'s extra tokens come only with extra drafter passes.
- The split (C1) helps the adapter only from 32K up.
- Overlap with a GPU verify (C2) loses more to DRAM contention than it hides.
- The adapter cannot be thinned without retraining (C3).
- To beat DFlash2 at J1-R1 on gsm8k, the drafting would need to fit in about 48 ms per round, against 87 now, and
  no remaining knob found tonight closes that gap.

## D. Next batch (20 Nov ~11:45; forge down for D2-D4 measurements, restored after)

### D1. The 09:02 forge exit: forge's own swap watchdog, from memory contention (`d1/`)

**Cause, from the logs:** `coreai_seg_logs/watchdog_v8_server.log`: `09:02:55 WATCHDOG: swap grew 1071 -> 3173 MB;
killing pid 62989 and children`. The server launcher (`pipe/v8_server_dec8.sh`) arms `swap_watchdog.sh` on forge with
+1,536 MB over the system swap at start (08:34, 1,071 MB). The C3 replay (started 08:57, 17.6 GB of GPU memory) beside
forge's 19 GB of wired memory pushed system swap to 3,173 MB (the replay's own watchdog log has the same peak), and
the Pi request's prefill began at 09:02:49. The watchdog counts system swap, so the replay's pressure is charged to
forge. Not a forge bug.

**Reproduction on a quiet machine, forge alone** (`bench/d1/prefill_mem.py`: a Pi-shaped request: 4 tools, a long
system prompt, reasoning_effort medium, temperature 1.0, top_p 0.95, cold prefix, max_tokens 768; swap and vm_stat
every 0.5 s; banner checked qk=attn_qk8isk_tg, dec8=libdec_gpu_clk.dylib):

| prompt tokens | prefill | swap start -> peak -> end | free memory min | forge RSS | wired peak |
|---|---|---|---|---|---|
| 7,009 (cold) | 10.4 s, 675 tok/s | 846.75 -> 846.75 -> 846.75 MB | 0.20 GB | 4.56 -> 4.69 GB | 19.03 GB |
| 7,009 (cold, again) | 10.4 s, 676 tok/s | 846.75 -> 846.75 -> 846.75 MB | 0.13 GB | 4.69 -> 4.99 GB | 19.07 GB |
| 6,213 (cold) | 9.2 s, 675 tok/s | 846.75 -> 846.75 -> 846.75 MB | 0.13 GB | 4.99 -> 5.22 GB | 19.05 GB |

Forge's phys_footprint after these: 5.06 GB (of it 2.65 GB "neural", the ANE programs); inactive (reclaimable) pages
stayed above 5.9 GB. **Contention only:** forge alone does not grow swap on 6-7K prefills. Forge (19 GB wired) plus a
17.6 GB replay does not fit in 32 GB.

**Fix (tooling, not forge):** `jetspec/jsp_replay.py` now refuses to start while 8765 serves ("forge is serving on 8765:
stop it first (one model at a time), or set JSP_ALLOW_FORGE=1"). The ANE build drivers already refuse. Not changed:
the server watchdog itself (system-wide swap against a baseline taken at start; a slow creep from other apps over a
long session counts too. A per-process or re-baselined guard is possible; not done without asking).

### D2. The LUT4 (Hadamard + GPTQ) adapter on the ANE, quiet machine (`d2/`)

Builds `build_jsp_adapter_all.sh` with `LUT=adapter_lut4hg JSP_ROT=1` (the 4 activation Hadamards per block in the
graph), forge down, nothing else running; 16 rows; scratch, deleted after:

| build | 8K pass | 32K pass | packages | wired after a run, per package |
|---|---|---|---|---|
| 3 x 4 blocks, 8K entry (`d2/b4`) | 7.28 + 7.33 + 7.33 = **21.9 ms** | (not built: in B8 the 4-block 32K entry failed at compile, 4 blocks x 3 KV segments pass the ~51 bound-tensor limit) | 3 x 780 MB | +0.83 / 0.86 / 0.86 GB (2.55 GB) |
| 4 x 3 blocks, 8K + 32K entries (`d2/b3`) | 5.55 + 5.53 + 5.56 + 5.55 = **22.2 ms** | 9.24 + 9.22 + 9.24 + 9.24 = **36.9 ms** | 4 x 585 MB | +0.89 GB each (3.57 GB; both entries' buffers bound) |

LUT6, for comparison: 27.2 ms at 8K, 3.67 GB wired (B8); 44.8 ms at 32K (C1, single-block packages). The 11:32
measurement (22.1 ms) stands.

**J0-R0 acceptance with the LUT4 adapter** (replay, head bf16, lossless; `d2/d2_lut4hg_J0-R0_*`): 6.07 / 8.53 / 6.59
tok/round on gsm8k / humaneval / mbpp, against bf16 6.12 / 8.59 / 6.64 (-0.9 / -0.7 / -0.7%). J3-R3: C4.

**Jet Flash, serial ANE, measured LUT4 adapter and acceptance** (`jetspec/d2_project.py`; head 14.8 ms and decoder
7.1 ms PROJECTED as in B8; 16-row verify = the served 8-row verify (chain_v1: 94.1 ms at 8K, 106.5 at 32K) x the
measured 1.085 / 1.17; DFlash2 cycle 107.7 ms at 8K, 121 ms at 32K; acceptance from short-context traces at both
entries, so tok/round at 32K of context is UNMEASURED; `d2/projection_ANE.md`):

| entry | schedule | gsm8k | humaneval | mbpp |
|---|---|---|---|---|
| 8K | J0-R0 | 43.1 vs 56.5 (-24%) | 59.6 vs 67.8 (-12%) | 46.4 vs 54.7 (-15%) |
| 8K | J3-R3 | 35.7 vs 56.5 (-37%) | 45.2 vs 67.8 (-33%) | 34.1 vs 54.7 (-38%) |
| 32K | J0-R0 | 34.1 vs 50.2 (-32%) | 47.0 vs 60.3 (-22%) | 36.7 vs 48.7 (-25%) |
| 32K | J3-R3 | 27.5 vs 50.2 (-45%) | 34.7 vs 60.3 (-43%) | 26.2 vs 48.7 (-46%) |

### D3. The same LUT4 adapter on the GPU (`d3/`, `jetspec/gpu_lut4_pass.py`)

Metal kernels through `torch.mps.compile_shader` on the same LUT4 tensors (packed 2 indices per byte, the per-tensor
16-entry palette, per-row scale; 2.233 GB resident): a 16-row LUT4 matmul, the block-1024 Hadamard of the rotated
inputs, RMSNorm, SwiGLU. Checked against float32 torch on all 7 shapes: rel 2.1e-4 (fp16 output rounding); Hadamard
rel 2.1e-4 vs the quantizer's rotation. One pass = 12 blocks without the attention core.

| matmul kernel | 12-block pass, no attention | weights read |
|---|---|---|
| v1: scalar dot products, x tile in threadgroup memory | 69.6 ms | 32 GB/s |
| v2: dequantized tile in threadgroup memory + 8x8 simdgroup matrices (26 KB per threadgroup) | 71.3 ms | 31 GB/s |
| v3: v2 with the palette by simd_shuffle | 79.3 ms | 28 GB/s |
| v4: indices straight into matrix fragments (registers spill) | 128-218 ms | 10-18 GB/s |
| v5 / **v6 (4 simdgroups x 64-column chunks, 10 KB: about 3 threadgroups per core)** | 38.6 / **38.2 ms** | **58-59 GB/s** |
| v7: v6 with W as the untransposed operand | 39.3-48.1 ms (sweep, matmuls only) | 46-57 GB/s |

Reference points on this M6 (12-core GPU), same harness: plain read of 2.23 GB 14.4-14.9 ms (150-155 GB/s); torch
fp16 matmul at 16 rows 115-118 GB/s of weights; fp16 matmul peak 9.3 TFMA/s (M = 256-2048). So the 16-row LUT4 work
(71.5 G multiply-adds) is not compute-bound; the best kernel reaches 38% of the read bandwidth. Occupancy was the
lever (v2 -> v5: same work, 1.8x).

**GPU adapter pass** = 38.2 ms measured + the adapter's 12 attention layers on the GPU (C1's measured `dec_attn_r4`,
8 rows: 0.141 ms per layer at 8K, 0.546 at 32K; x1 to x2 for 16 rows) = **39.9-41.6 ms at 8K, 44.8-51.3 ms at 32K**
(q / k norm, RoPE and the gate: UNMEASURED, small). The ANE does it in 22.2 / 36.9 ms.

**Jet Flash with the GPU adapter** (serial; head and decoder as above; the lower attention bound;
`d3/projection_GPU.md`): 8K J0-R0 38.3 / 52.5 / 41.1 tok/s (-32 / -22 / -25% vs DFlash2), J3-R3 -48 to -51%. **Even
at the GPU's read-bandwidth floor** (14.5 ms + attention, PROJECTED; `d3/projection_GPUfloor.md`): 8K J0-R0 45.1 /
62.4 / 48.6 (-20 / -8 / -11%), 32K -14 to -25%. A GPU drafter does not change the conclusion: the 16-row verify (102 /
125 ms) is most of every round.

### D4. A cheap exact tree conv for tree8 (`d4/`)

**Probe** (`v8/build_tree_corrx_probe.py`: one target group, layers 0-3, 8 rows, 8K of KV, the served weight constants;
one package per form, each with the served chain entry `full8_8k` and the tree entry `tree8_8k`;
`pipe/unit/power/d4_probe_check.py`: every output compared under chain masks, then cost per group, median of 30;
scratch, deleted). Every form adds to the served conv a correction meant to be exactly 0 on rows whose taps are the
chain's:

| form | correction | bit-identical outputs (chain masks) | per group, chain masks | branched tree |
|---|---|---|---|---|
| served chain entry | | | 5.04-5.14 ms | |
| `corr` (A8's build) | sum_t w_t (ctap_t @ rows - rows shifted by t) | 10 / 12 (pend1, pend2 differ) | +11.2% | +10.2% |
| `corrc` | corr + cum as g @ tri + g @ (anc - tri) | 10 / 12 (pend1, pend2 differ) | +10.2% | +10.4% |
| `corrd0` | sum_t w_t (D_t @ rows), D = ctap - the chain shift | **12 / 12** | +11.4% | +12.1% |
| **`corrx0`** | **[rows w_0 .. rows w_3] (cdim x 44) @ D (44 x 8), channel-major** | **12 / 12** | **+4.5%** | **+4.0%** |
| `corrb` / `corrdb` / `corrxb` | corr / corrd0 / corrx0 + a row blend (served silu on chain rows) | 12 / 12 | +12.7 / +14.8 / +5.3% | +12.8 / +13.0 / +5.6% |

(19 Nov's inexact tree8 form, A2: +5.7%.)

**What the A8 inexactness was** (`pipe/unit/power/d4_pend_diff.py` on `corrc`): the cum row of the pending tapes is
identical. The differences are 1-ulp changes in the keys (36 of 49,152 elements in layer 1, 57 in layer 2) and the
`u` rows that follow from them. **A8's attribution to the in-block decay matmul was wrong; the one-hot gather
`ctap @ rows` is not exact on the ANE**, so `taps - shifted` is not exactly 0 everywhere. With D = ctap - shift (a
CONSTANT shift; 0 / 1 values, exact) a chain row's D row is zero and its correction is exactly 0 whatever the matmul's
rounding. No blend is needed.

Two build pitfalls, both fixed: a shift built by in-place indexing inside forward traces as ops the ANE does not run
(the first probe's corrd / corrx / corrx2 packages, and the first full corrx package 0, ran off the ANE or failed to
compile: "MLIR MPS to ANEC conversion failed"); built from numpy it is a constant.

**Full build** (`TREE_FORM=corrx0 build_dec8t_all.sh` -> `v8_dec8tx`, 13:10-14:29, forge down; 9.9 GB on the SSD, GPU
tables `pipe/w_dec8tx`, 8 HWX referenced, none missing). Package 0 checked with `tree_entry_check.py`: `g0` 12 / 12,
`g1` 13 / 13 outputs bit-identical between the tree and the chain entries (A8's v8_dec8tc: pend1 / pend2 differed).

**Benches** (`run_arm.sh` with `pipe/v8_server_tree.sh` on v8_dec8tx, realistic_v1 9 prompts x 2K-96K, greedy, 320
tokens; banner checked: `tree=policy=<arm> tree=topc=<n> v8=qk=attn_qk8isk_tg dec8=libdec_gpu_tree.dylib`; compared
with `bench/compare.py` against `chain_v1`; `d4/compare_*.txt`):

| arm | what changes | identical to chain_v1 (2K / 8K / 32K / 64K / 96K) | tok/verify | ms / cycle |
|---|---|---|---|---|
| served chain (`chain_v1`) | | | 4.567 / 5.086 / 5.164 / 5.040 / 4.722 | 110 / 108 / 121 / 136 / 144 |
| `d4_cap3` (control) | chain entries only, at most 3 drafts accepted per cycle | 3 / 2 / 3 / 5 / 7 of 9 (**20 / 45**) | 3.224 / 3.386 / 3.427 / 3.374 / 3.197 | 109 / 109 / 120 / 135 / 142 |
| `d4_topc1` | every cycle through the corrx0 tree entry, the tree = the chain | **9 / 9 at every rung (45 / 45)** | identical to the chain | 112 / 112 / 124 / 139 / 146 |
| `d4_branched` | tree8 as it would serve (tree entry when the tree branches) | 3 / 3 / 4 / 5 / 8 of 9 (23 / 45) | 4.862 / 5.512 / 5.522 / 5.275 / 4.980 | 112 / 111 / 123 / 139 / 145 |

- **The tree entry is exact end to end:** 45 of 45 replies byte-identical when the tree is the chain, the 96K split
  path (tdva / tdvba + the GPU tree mask) included. A8's exact build: 3-8 of 9 per rung.
- **The chain itself is not split-invariant** (`d4_cap3`): the same served chain entries with only a different cycle
  split (3 drafts per cycle at most) keep 20 of 45 replies. The DeltaNet block solve and the pending tape depend on
  which tokens share a verify call, and fp16 near-ties flip. A branched tree changes the split, so **no tree form can
  be token-identical to the chain**. The branched arm's 23 / 45 is the control's level: the remaining divergence is the
  split, not the tree numerics.
- **Cost:** the tree verify, from the branched run's server lines, costs x1.038 / 1.039 / 1.037 / 1.033 / 1.024 of
  the chain verify per cycle (98.5 vs 94.9 ms at 2K, ..., 132.0 vs 128.9 at 96K). 19 Nov's inexact form: x1.033-1.050;
  A8's exact form: x1.096.
- **Like for like** (offline on the chain's own texts, `pipe/unit/power/tree_dyn2.py` with 8=1.034;
  `d4/sim_corrx0_8_1_034.*`): tree8 when branched +3.5 / +1.4 / +1.9 / +0.9 / +1.1% tok/s at 2K / 8K / 32K / 64K /
  96K; chat +9.0%, code +0.4%, math +0.7% (A7 with the inexact form: +3.0 / +1.0 / +1.5 / +0.5 / +0.6%).
- The bench's own tok/s (+8.4 / +11.1 / +4.8 / +2.0 / +4.3%) is on different texts, not like for like.

**Served state:** unchanged. Tree8 stays behind the flag. Making it the default needs a decision: the plan's
"token-identical to the chain" cannot be met by any tree (the chain fails it against itself under a different split);
what corrx0 does meet is "bit-identical to the chain whenever the split is the same". Launch:
`BUILD=/Volumes/SSD4TB/anemll-forge-research/v8_dec8tx GM=pipe/w_dec8tx/manifest_best.json PORT=8765 pipe/v8_server_tree.sh`.
Forge restored at 15:11 (pid 50586, `v8_server_dec8_qk.sh`, build v8_dec8): `bench/final_served3` banner matched,
humaneval 1 / mt-bench 2 byte-identical to 08:34's (sha 1681b92d273f64ad / 3b060108f4af10b0).

**Disk:** v8_dec8tx is kept (9.9 GB on the SSD; Core AI entry about 10.7 GB + aned's programs on the internal disk). With
v8_dec8t (19 Nov ctap form) and v8_dec8tc (A8 corr form) also kept, three tree builds sit on the internal disk; the two
older ones are superseded by v8_dec8tx (not deleted: needs Anemll's OK). Internal disk free after the build: 30 GB.
The D-batch scratch builds are deleted.
