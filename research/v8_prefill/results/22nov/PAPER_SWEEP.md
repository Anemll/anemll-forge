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
