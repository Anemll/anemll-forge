# YOLO batch, 21 Nov 2026 11:13 to 12:50 (Anemll: proceed in YOLO mode)

**Result, number first.**
- **Review fixes deployed** (11:13): smoke 29aef2ec9db18a3f on every launch, 0 errors in 9 server logs.
- **Sampled tree8 holds in both run orders:** +8.5% (chain first) and **+10.2%** (tree first, new seeds). The agent
  set loses both times (−1.6%, −3.9%).
- **Prompt lookup** (NGRAM_DRAFT=6), off/on/on/off, both orders agree within 0.4 points:
  - greedy code +94.9%
  - **sampled code +26.0%**
  - **sampled agent +6.6%**
  - greedy agent +0.3%
- **Verify widths 8-16:** every width built and checked bit-exact. Cost rises steadily with no cliff (8K: +1.4% at 9
  rows ... +10.6% at 16). On greedy texts no width beats served tree8 by more than +1.6% (8K), and every width loses at
  32K.
- **At 8 rows the loss is in the selector,** not the candidate list:
  - 30-34% of cycles lose the true path because the 7-node budget drops a top-4 child.
  - After the first fork the tree keeps the truth only 14-24% of the time.
  - A scalar tau retune gains nothing (1.5 is already best).
- **Forge** up on the served config, **pid 24075** (12:45), smoke 29aef2ec9db18a3f.

All arms: fresh launches of the standard launcher (`pipe/v8_server_dec8_qk.sh`) with the banner checked. No Pi traffic
in any arm (checked per server log). Machine: M6, build v8_dec8tx.

## 1. Review fixes, deployed (commit d5d9f2e; `review_fixes/README.md`)

- **Snapshot restore:** all or nothing, plus the empty-drafter-ring crash. `pipe/v8_server.py` `_restore_cold`.
- **Split tree mask:** the plan is built before the mask is set, checked setter calls, the chain mask restored in
  `finally`, profiling counters charged. `pipe/tree_verify.py`.
- **Capped sampled chain:** uses `drafts[:k]` in the history.
- **CPU checks:** the reviewer's cache_checks (v8 gate), restore_fault_unit 4/4, tree_mask_unit 3/3 at -O0 and -O1,
  tree_sampled_unit 3/3.
- **Live from the 11:13 launch.** Every later launch gave the fresh smoke 29aef2ec9db18a3f, and the 9 server logs of
  this batch have 0 Tracebacks.

## 2. Sampled tree8, both run orders (`yolo/sampled_rev_compare.json`; first order `tree_sampled/`)

Thinking on, temperature 1.0, 384 tokens, the same prompt and seed in both arms of a block. Values are tok/s change
(tree8 vs chain for sampled cycles).

| set | block 1: chain first, seeds 2100/2200 (10:22) | block 2: tree first, seeds 3100/3200 (11:13) | mean |
|---|---|---|---|
| 8K | +9.0% (7/9 faster) | +5.7% (6/9) | +7.4% |
| 32K | +11.6% (8/9) | +19.3% (7/9) | +15.4% |
| code | +10.9% (8/9) | +14.9% (8/9) | +12.9% |
| agent | −1.6% (4/9) | **−3.9%** (6/9 paired, pooled lower) | −2.8% |
| all | +8.5% (27/36) | **+10.2%** (27/36) | +9.4% |

- **Keep TREE_SAMPLED=1.** tok/verify rises +8-11%; ms per cycle changes −1 to +2.4%.
- **Order effect:** the second arm of a block runs slower (block 2's chain 32K cycle 134.6 ms vs block 1's 125.1). So
  block 1 understates and block 2 overstates the tree. The mean of the two orders is the estimate.
- **Agent:** it already accepts about 5 tokens per verify, so the tree adds no tokens and costs about 2 ms a cycle.
  It is reproducibly negative, and block 2 passes the review's 2% bar. A cost-aware fallback (chain when the
  expected tree gain is small, TREE_POLICY=dink exists) is the candidate fix.
- **A caveat on block 2's 8K:** an offline CPU job of mine ran at nice 19 for about 2 minutes during it.

## 3. Prompt lookup, off / on / on / off (`yolo/lookup_compare.json`)

The served config plus NGRAM_DRAFT=6 when on. The code and agent sets, greedy (320 tokens) and sampled (seed 4100).

| stratum | off tok/s | on tok/s | change | off1 -> on1 | off2 -> on2 | tok/verify | lookup cycles | faster |
|---|---|---|---|---|---|---|---|---|
| greedy code | 40.83 | 79.58 | **+94.9%** | +94.9% | +95.0% | 4.49 -> 7.60 | 92% | 18/18 |
| greedy agent | 54.07 | 54.23 | +0.3% | +0.3% | +0.3% | 5.56 -> 5.20 | 39% | 8/18 |
| sampled code | 37.54 | 47.30 | **+26.0%** | +26.2% | +25.8% | 4.28 -> 5.18 | 33% | 18/18 |
| sampled agent | 45.96 | 49.00 | **+6.6%** | +6.8% | +6.5% | 5.14 -> 5.35 | 28% | 13/18 |

- **Lookup pays on sampled Pi-like traffic too,** not only on greedy full-file copies. Sampled lookup cycles use the
  release chain's lossless rule.
- **Greedy texts:** 16/18 identical on code, 14/18 on agent.
- **Not tested here:** lookup on the realistic (chat / math) set with sampling. Greedy there it was −2.5..+0.8%
  (20 Nov U1: tree8lk vs tree8 at 8K −2.5%). That check is needed before turning it on by default (Anemll's call).

## 4. Verify widths 8-16 (`widths/`; review addendum)

**How it was built:**
- Probe: one target group (layers 0-3), chain and tree (corrx0) entries at 8K and 32K. Widths above 8 use the 8-row
  block solve with a partial last block (`v8/gdn_partial_solve.py`), plus a full-inverse chain at 8K as the on-device
  reference.
- Toolchain: `v8/build_treew_probe.py` -> `journal_compile.py` (forge.py compile, journaled) ->
  `pipe/unit/power/tree_width_time.py`. Two timing blocks per width (median of 30 each); width 8 rebuilt at the end
  as the drift check.
- Cleanup: every package's cache entries deleted (coreai_cache_delete.py) and the package removed. Forge was down
  12:14-12:45.

**Checks, at every width:**
- tree<w> under chain masks = chain<w>, bit-identical at 8K and 32K.
- Block solve vs full inverse: relative 4.6-8.2e-4, the fp16 level of the CPU check.
- Drift of tree8 over the sweep: within 1% (8K 5.249-5.267 -> 5.270-5.313 ms; 32K 6.005-6.039 -> 5.986-5.989).

Per-group ms (tree entries with the branched mask, mean of two blocks), change vs width 8:

| width | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|
| tree 8K ms | 5.258 | 5.334 | 5.403 | 5.446 | 5.540 | 5.627 | 5.701 | 5.821 | 5.814 |
| vs 8 | | +1.4% | +2.8% | +3.6% | +5.4% | +7.0% | +8.4% | +10.7% | +10.6% |
| tree 32K ms | 6.022 | 6.168 | 6.267 | 6.409 | 6.515 | 6.694 | 6.750 | 7.089 | 7.096 |
| vs 8 | | +2.4% | +4.1% | +6.4% | +8.2% | +11.2% | +12.1% | +17.7% | +17.8% |

The cost rises steadily, with no cliff at row 9. There is a step at 15, and 15 -> 16 adds nothing. The largest block
spread is 0.068 ms.

**What the rows buy** (`serve_t/tree_widths_sim.py`, `widths/sim_widths.log`):
- Method: the served tree builder (top-4, tau 1.5) cut to w - 1 nodes, on the chain's own greedy texts (chainlog_v1).
  Fixed anchors, not a trajectory replay.
- Policy: branched, i.e. chain8 when the top-7 tree is the chain, as served.
- Cost: cycle = served cycle8 + verify8 x (measured ratio - 1).

| rung | tok/verify w8 -> w12 -> w16 | best width vs served tree8 | widths 9-12 | w16 |
|---|---|---|---|---|
| 2K | 4.866 -> 5.127 -> 5.254 | w11 +1.4% | +0.5..+1.4% | +0.4% |
| 8K | 5.299 -> 5.640 -> 5.821 | w16 +1.6% (w13 +1.5%) | +0.6..+1.4% | +1.6% |
| 32K | 5.407 -> 5.650 -> 5.816 | none (w9 −0.2%) | −0.2..−0.8% | −2.9% |

**Conclusion:** on greedy texts no width earns a build. The review's +5% gate would need about 3x the yield per row.
Sampled requests gained more from the tree (+9.4% vs +2-6% greedy), so a sampled per-width yield curve could differ,
but measuring it needs on-device logging of sampled acceptance per node (not done).

## 5. Where the 8-row tree loses tokens (`widths/coverage_inventory.log`, `widths/tau_sweep.log`)

Per cycle, along the true continuation of the chain's greedy texts (all rungs 2K-96K, 10,778 depth steps):

| | before any fork | one step after the first fork | later |
|---|---|---|---|
| true token among the 16 candidates | 96-98% | 69-82% | 75-82% |
| in the top 4 children, given a candidate | 97-98% | 92-95% | 93-94% |
| kept by the 7-node tree, given top 4 | 93-95% | **14-24%** | **2-7%** |

Where each cycle first loses the true path:
- **top-4 but dropped by the 7-node budget: 30-34%**
- not among the 16 candidates: 13-21%
- outside the top 4: 10-14%
- all 7 depths kept: 34-43%

So the budget's allocation, not candidate recall, is the largest loss. That matches the review's literature
(DForkTree: post-fork scores are poorly calibrated). A scalar change does not fix it: tau 0.75-6.0 for the best-first
order, 2-fold by request, held-out gain vs tau 1.5 is 0.00% to −0.56% at every rung. The remaining selector test is a
fork-aware edge-acceptance table (rank, score gap, depth, distance from the fork), fit on half the requests and
scored on the other half. It is offline and cheap, and not run yet.

## 6. State

- **Forge:** pid 24075 on 127.0.0.1:8765 since 12:45.
- **Banner:** v8 F8=x KV8 QK8 qk=attn_qk8isk_tg libgpu_v8_f8q; dec8 libdec_gpu_tree.dylib sha256 dfe059c68dce376a,
  kernels dec_kernels_tree.metal 674918fb1e682a2c, keep-alive 60 ms; tree8 policy branched topc 4 tau 1.5
  sampled=on, lookup off; drafter ring snapshot fix; ctx 98304; model id qwen38-27b-ane. Smoke 29aef2ec9db18a3f.
- **Not changed:** lookup default (off), COLD_SCRUB (off), the release qwen38_server.py (Anemll's edits, not staged).
- **Cleanup:** all probe packages and their cache entries deleted; 29 GB free on the internal disk.
- **Commits:** local only, branch v8-gpu-ane-prefill, no push.

## 7. Candidates for the next tasks

1. **Lookup on by default?** It is the largest measured gain on Pi-like traffic (sampled code +26%, agent +6.6%).
   First a sampled lookup off/on check on the realistic 8k / 32k chat and math rungs (about 25 min), because greedy
   there was −2.5..+0.8%.
2. **Agent-set regression of sampled tree8** (−1.6% / −3.9%): test TREE_POLICY=dink (cost-aware tree vs chain per
   cycle) against branched on the code / agent sets, sampled, both orders.
3. **Fork-aware selector at 8 rows** (offline first): it targets the 30-34% of cycles lost to the 7-node budget.
4. **Sampled per-width yield:** log per-node sampled acceptance on device before any wider build is reconsidered.
5. **Stale KV rows:** serve COLD_SCRUB=kv, or localize the leaking kernel first (a one-layer probe).

## Files

- **Drivers:** `serve_t/yolo_benches.sh`, `serve_t/width_sweep.sh`.
- **Analysis:** `serve_t/{tree_sampled_compare,lookup_compare,tree_widths_sim,coverage_inventory,tau_sweep}.py`.
- **Probe:** `v8/build_treew_probe.py`, `v8/gdn_partial_solve.py`, `pipe/unit/power/tree_width_time.py`.
- **Results:** `results/21nov/yolo/` (logs, server logs without prompt text, compare JSONs), `results/21nov/widths/`
  (build / compile / HWX / timing per width, `widths_summary.json`, sim and inventory),
  `results/21nov/bench/{tree_s2,chain_s2,lk_*,served_after_widths}*`.
- **Review inputs:** `results/22nov_review/` (REVIEW.md, TREE_LITERATURE.md, README_m5_1111.md from m5.local).
