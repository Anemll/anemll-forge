# 22 Nov 2026: paper sweep for untried techniques (tree verify, sampled acceptance, hybrid verify, overlap)

Three parallel literature reviews (primary sources, arXiv HTML / PDF), each briefed with our constraints (fixed 8-row
ANE verify, DFlash2 drafter about 11% of a cycle, sampled T=1.0 Pi workload, hybrid GDN target) and the list of
tried-and-failed items. Every "untried" claim was then checked against results/19nov-22nov. No forge run; estimates
are labeled. Papers read: DDTree 2604.12989, DForkTree 2611.12356, D3Tree 2611.11371, TAPS 2606.00487, CaDDTree
2606.01813, TreeSpark 2609.22098, DominoTree 2607.08642, DFlow 2609.06498, Bastion 2605.29727, Traversal Verification
2505.12398, RSD 2402.14160, SpecInfer 2305.09781, Hu et al. 2502.18779, RheoSampling 2609.21827, HSD 2601.05724,
Block Verification 2403.10444 (abstract), SpecTr 2310.15141 (abstract), TreeWY 2608.20961, LumoTree 2609.23900,
MFD 2611.13017, SpecLA 2607.16673, Twig 2608.01651, Weaver 2607.06763, HybridSpec 2611.04055, STree 2505.14969,
Mirror-SD 2510.13161, JetSpec++ 2610.19792.

## Result

**One large untried idea: foresight drafting on the GPU during the ANE verify.** Everything else is already done,
proven not to apply, or worth about 0-3%.

## A. Untried and applicable

| # | technique (paper) | why untried here | expected gain (estimate) | cheapest test |
|---|---|---|---|---|
| A1 | **Foresight drafting on the GPU during the ANE verify** (JetSpec++ §4.1-4.2; Mirror-SD §3.1/3.4): draft the next block for the most likely verify outcomes (accepted length k, bonus token) while the current verify runs, keep the hit | 19 Nov measured the contention, never the foresight: a GPU drafter twin run ONCE per cycle during the ANE verify costs the verify only +1.6% at 2K / 32K (95.8 vs 94.3; 123.2 vs 121.3 ms) and +5% at 64K, while hiding a 13.3-14.5 ms draft; back-to-back it costs +25-27% (+68-73% at 64K). The 21 Nov drafter-placement analysis (max 2.0%) did not consider drafting ahead. | **+3 to +8%** below 64K: hides about 12 of ~110 ms per cycle on a hit, minus +1.6% verify, minus misses. Hit rate unknown: DFlash2 is conditioned on target tap features, which for the k newly accepted tokens do not exist until the verify ends | (1) offline: top-B outcome hit rate from TREE_LOG (accepted length + bonus per cycle); (2) offline ANE replay: DFlash2 acceptance when its context features stop at the anchor (forge stopped); (3) only then a GPU DFlash2 port |
| A2 | **Better post-fork candidates: trigram PMI rescoring** (DForkTree §3, §4.2 Eq. 4; their drafter is 89% right on its top-1 path, 18% just after a fork) | 21 Nov forkrank re-ranked the same candidates by rank / score / depth / fork distance (no gain) and concluded better post-fork CANDIDATES are the lever; PMI from a trigram table adds context two tokens back, a different signal | 0 to +3% | offline on TREE_LOG_DIST (depth-1 alternatives exact from the anchor row; deeper via the replay simulator); trigram table from our own Pi outputs |
| A3 | **Oracle 7-node tree gap** (ceiling for any selection change; suggested by the sampled-acceptance review) | not computed (forkrank and the tau sweep compared rules, not the ceiling) | 0 (a measurement); decides whether A2 / A5 are worth anything | offline: best 7 nodes among logged nodes under the logged target law vs served |
| A4 | **One stochastic slot** (RheoSampling 2609.21827 §3.2-3.3): top-m deterministic children plus one child sampled from the renormalized drafter tail, verified with min(1, p/q) then recursive rejection; lossless (Thm 1) | not tried | 0 to +2% (their +4.2-4.4% is at 60 nodes) | slot acceptance exact offline from TREE_LOG_DIST + softmax(s/1.5) over the top-16; subtrees below a sampled node are not logged (lower bound 0) |
| A5 | **DFlow verifier-to-drafter relay** (2609.06498 §4.2): the target's features of the rejected suffix + bonus embedding fill reusable drafter positions next round | needs drafter retraining (80K samples, 6 epochs) and a new drafter ANE build | +6 to +10% claimed (chain, block 16, Qwen3 1.7-8B, A800, sampled T=1 +11-12%) | not scoreable offline; a training project |
| A6 | **Fold the commit into the next verify** (SpecLA §5.2) | not tried | about 1% | profile the commit share of the served cycle first |
| A7 | **Weaver adapter** (2607.06763 §3.2-3.3): a small AR adapter re-scores top-512 DFlash candidates | needs training; at budget 16 its overhead lost speed in the paper | 0 to +3% | offline replay once trained |

## B. Already done here (no new gain)

| technique (paper) | our result |
|---|---|
| Log-depth Neumann / doubling inverse for the UT solve (Twig §IV-A Lemma 3, Weaver §3.4.2) | SERVED since Oct (`GDN_FAST=1`, M6_COMPUTE_ACCELERATION_2026-10-03); 20 Nov A5 also tested block inverse and block forward substitution (best at 16 / 32 rows) |
| Calibrated edge scorer / target-calibrated reach (TAPS §4.2-4.4, D3Tree §III-C, TreeSpark §3.4) | 21 Nov forkrank: rank / score bucket / depth / fork-distance bins, fit on half, scored on the sampled target law: -0.09 to -0.13% sampled, -0.08 to -0.50% greedy; tau sweep: served 1.5 best |
| Fork-aware trees (DForkTree's ranking part) | same forkrank run; DForkTree's PMI part is A2 |
| Path-local GDN verify + accepted-path commit (TreeWY, LumoTree, MFD App. C, SpecLA, Twig, STree) | our served pending tape + compaction is the same math; their speedups come from GPU batching / tile kernels |
| GPU drafter instead of the ANE drafter | 19 Nov: 13.3-14.5 ms against the ANE verify vs 12.3 ms ANE; no gain as a placement change (A1 is the overlap, not the placement) |
| Cost-aware tree budget / per-round length (Bastion, CaDDTree, TreeSpark ladder, HybridSpec) | 19 Nov per-cycle size 8/16/32 +0.3-1.6% offline; 22 Nov Dinkelbach tree-vs-chain -5.9%; HybridSpec's gain is MoE-specific (verify +71% from 4 to 16 rows there) |
| JetSpec++ drafter serially | 20 Nov: 12-46% slower end to end (projection from measured parts) |

## C. Shown not to apply

- **Every verification-only rule** (Traversal Verification, Block Verification, HSD, SpecInfer multi-step, RSD as a
  verifier, SpecTr, Hu et al. optimal multi-draft): with deterministic drafts and a fixed tree, no lossless rule can
  accept more than sum over nodes of the product of target probabilities along the path, and our served child-by-child
  rule reaches that bound (its acceptance equals "sample from the target, follow the matching child"). HSD §6.2 states
  the same for top-K drafts. Gain 0%; this is also a correctness confirmation of the served rule.
- **D3Tree pruning, DominoTree, MFD modes, TreeWY / Twig / LumoTree kernels**: gains come from batch > 1, budgets of
  15-512 nodes, GPU kernels or a retrained multi-mode drafter. DominoTree's one 27B hybrid point (Qwen3.6-27B, H200,
  budget 15) gains +20% over the chain at 8K but +2.7% at 32K, matching our long-context trend.

## Recommended order (needs Anemll's go for anything on forge or the ANE)

1. A3 and the A1 hit-rate count: offline, from existing TREE_LOG files, about 1 hour, no forge.
2. A1 step 2 (DFlash2 with features stopping at the anchor): an offline ANE replay with forge stopped, about 1-2 hours.
   Go / no-go for a GPU DFlash2 port.
3. A2 and A4 offline on TREE_LOG_DIST, only if A3 shows a gap of more than about 3%.
4. A5 (DFlow) is the only roughly 10% item that needs drafter training; a separate project.

## Probe results (22 Nov, offline, forge untouched)

`serve_t/probe_tree_ceiling.py` (A3 + A1), `serve_t/probe_rescore.py` (A2 + A4); raw `results/22nov/probes/`. Data:
`treelog_s21` (1,887 sampled cycles, realistic 2K / 8K / 32K, 21 Nov) and `treelog_lgo21` (2,642 sampled cycles with
distributions, long Pi-like session). Metric: expected tok/verify = 1 + sum of target path probabilities (served
reproduces 21 Nov forkrank: 4.083 / 4.075).

| probe | realistic 2K-32K | long session | verdict |
|---|---|---|---|
| A3 served tok/verify | 4.083 | 4.075 | reference |
| A3 best 7 nodes from the same drafter candidates (lower bound) | 4.435 (**+8.6%**) | 4.490 (**+10.2%**) | real headroom |
| A3 best 7 nodes, perfect proposer (lower bound) | 4.526 (+10.8%) | 4.612 (+13.2%) | |
| A1 foresight hit, (endpoint, bonus) in top 1 / 2 / 4 / 8 | 26.8 / 35.4 / 46.8 / 57.0% | 23.2 / 31.6 / 41.5 / 52.8% | weak |
| A1 endpoint only in top 1 / 2 / 4 | 38.6 / 59.0 / 82.1% | 35.5 / 56.4 / 79.9% | |
| A2 trigram PMI, lam (spine, fork) = (0.15, 0.3) / (0.3, 0.6) / (0.6, 1.2) | -0.22 / -1.43 / -4.28% | -0.15 / -1.11 / -3.65% | no gain |
| A4 one sampled anchor slot replacing the weakest node | -4.36% (slot accepted 0.046, dropped node 0.224) | -3.80% (0.037 vs 0.192) | no gain |

- **A3:** the same 16 candidates per depth hold about +9-10% more expected tokens per verify than the served
  selection. But the 21 Nov calibrated ranking (drafter-side features) captured none of it and the PMI table makes it
  worse: the information that picks the better nodes is the target's own distribution at the verified rows, which
  only exists after the verify. That points at verifier-to-drafter relays (DFlow, A5), not at host-side rescoring.
  The rebuilt trees are scored as lower bounds (subtrees under an unverified node count 0), which biases A2 down a
  little; the oracle uses the same metric and still shows +9-10%, so the metric can show a gain.
- **A1:** to hide the drafter, the next block must be drafted for the right (endpoint, bonus). Even 8 guesses hit only
  53-57%. A rough ceiling (estimate): hit 0.55 x about 12 ms hidden of about 110 ms = 6%, before the cost of 8 GPU
  drafts per cycle (64 rows) against the ANE verify and the unknown acceptance loss from features that stop at the
  anchor. Likely 0 to +4%. Step 2 (an ANE drafter replay with stale features, forge stopped) decides it.
- **A2, A4:** dead on this target and budget.

## Engine use per decode cycle: where could a draft-ahead run? (22 Nov, from logs, forge untouched)

Live served decode (tree8 + sliced entries), medians of the server's 10 s decode lines over the last 8 server logs
(41 samples):

| prompt | cycle ms | ANE draft | ANE verify | host (sample + ctx + host) | ANE busy share | ANE idle ms |
|---|---|---|---|---|---|---|
| < 16K | 110.0 | 14.4 | 93.7 | 1.5 | 98.3% | 1.9 |
| 16-32K | 122.0 | 14.5 | 106.3 | 1.5 | 99.0% | 1.2 |
| > 64K | 151.0 | 14.4 | 135.2 (incl. GPU attention, split) | 1.6 | 99.1% (ANE waits during GPU attention) | 1.4 |

18 Nov decode telemetry (results/18nov, chain build): GPU 0% active in decode up to 32K, 24% at 64K (split attention);
DRAM 98-104 GB/s on average, the ANE counter in its 128+ GB/s top bin 58% of the time while it streams; both engines
together reach 164 GB/s. So while the ANE verifies, about 36 GB/s of DRAM and the whole GPU are spare.

- **The ANE has no room:** 98-99% busy; the idle ~1.5 ms per cycle is host work. Only in the split (> 64K) does the
  ANE wait for GPU attention (18 Nov: 28 ms per step at 64K), a slot for an ANE draft-ahead at long context only.
- **The GPU has room:** idle below 64K. A drafter pass reads about 0.7 GB, about 19 ms at the 36 GB/s spare, inside a
  94-135 ms verify. Measured 19 Nov: one GPU drafter pass per cycle beside the ANE verify costs the verify +1.6% (2K,
  32K) / +5% (64K). A batch of B guesses reads the weights once (1 -> 16 rows +16-21%, 20 Nov E1), so B = 4-8 costs
  about one pass of DRAM.
- **Revised A1 estimate (with probe A1's hit rates):** hidden draft = hit x 14.4 ms: B = 4 (hit 0.42-0.47) saves
  6.0-6.8 ms, B = 8 (0.53-0.57) saves 7.6-8.2 ms of a 110-122 ms cycle (+5.5 to +7%), minus +1.6-2% verify:
  **about +3 to +5%**, if features that stop at the anchor do not lower acceptance (unknown; that is the step 2
  replay). Needs a working GPU DFlash2 (the 19 Nov twin is a cost model with the real shapes, not a drafter).

## A6 commit fold (SpecLA §5.2): closed

The commit (tree_verify `compact` + `mb.accept` + the drafter's feature queue) is the server's `ctx` phase, t3 -> t4 in
`Tree.cycle` / `_chain`: 1.1-1.4 ms of a 110-127 ms cycle (live decode lines, 22 Nov), about 1%. Folding it into the
next verify saves at most part of that: under the 2% bar, not pursued.
