# Tree Speculative Decoding on a Split ANE + GPU Target: Qwen3.8-27B on Apple M6

ANEMLL forge research notes, 20 to 22 November 2026. M6 runs are archived under `research/v8_prefill/results/{17,18,19,20,21,22}nov/`. Replay estimates and literature comparisons are identified separately.

**Short version:** [Tree verification, simply](TREE_VERIFY_SIMPLE.html) (plain language, illustrated).

**Jump to:** [What's new](#whats-new) · [M6-specific findings](#m6-specific-findings) · [Related work](#8-related-work) ·
[References](#references)

## Abstract

We serve Qwen3.8-27B (64 layers: 48 Gated DeltaNet, 16 full attention; 3-bit MLP weights, INT8 KV) on an Apple M6 with
the feed-forward networks on the Neural Engine (ANE) and the token mixers split between the ANE and the GPU. Decoding
uses a DFlash2 block drafter that proposes 7 tokens per cycle on the ANE, verified by an 8-row target call. We use the
drafter's candidate distributions to build a dynamic tree of seven proposal nodes inside the same 8 verify rows, keeping the physical row budget fixed. Each row uses its own ancestry and depth. A correction to the served DeltaNet short convolution (`corrx0`) preserved all 45 greedy replies when the tree entry was forced to run chains. After a branch is accepted, the runtime packs its pending state and gathers its drafter features in path order. Sampled requests use child-by-child rejection.
On two 36-request sampled benchmark blocks, tree8 delivered +8.5% and +10.2% pooled client tok/s over chain8. The tool-agent subset lost 1.6% and 3.9%. Greedy gains ranged from +1.9% to +6.5% on the 2K–64K benchmark rungs. Four-layer timing probes and fixed-anchor greedy replay put the best wider tree only 1.6% ahead at 8K. On a separate 19-turn Pi-like workload, adding four static KV widths to served tree8 gave +3.40% pooled, or +2.49% with one anomalously slow baseline seed excluded. All 152 paired replies matched.

## What's new

The six contributions below describe what we implemented or measured for this model on the M6. The published Gated DeltaNet (GDN) tree algorithms establish path-local verification, compact speculative state and accepted-path commit [24–27, 31, 32]. The comparison below uses the primary texts available on 22 November 2026. The [claim audit](results/22nov_review/tree_verify_figures/CLAIM_AUDIT.md) gives paper sections, benchmark records and the limits of each comparison.

1. **A tree that competes with the chain in eight ANE rows.** DDTree builds a tree from one block-drafter pass under a node budget [10]. LumoTree pads trees to reusable shapes [26]. TensorFold allocates Qwen3.8-27B draft rows using measured Metal costs [23]. Forge gives both entries precisely one anchor and seven candidate slots, sharing weights across precompiled ANE programs. It dispatches a one-path draft to the served chain entry and pays the extra mask and convolution cost only on a fork. Sections 3.1–3.2 and 4.3.
2. **An ANE short-convolution correction with measured chain parity.** Multi-Future Drafting specifies the correct convolution window through parent pointers [27, Appendix C]. Our first implementation of that window, a one-hot matmul, differed from the served chain on the M6. `corrx0` keeps the served convolution and adds a channel-major matmul against the difference between the tree tap map and the constant chain shift. On chain masks, 12/12 group outputs matched bit for bit in the form comparison and 45/45 full greedy replies matched. The group overhead was 4.5% in that probe. Arbitrary branched-path bit parity remains open. Sections 3.3 and 6.
3. **An accepted path compatible with Forge's existing chain commit.** TreeWY reconstructs the selected recurrent state from saved factors [24]. Multi-Future Drafting updates it directly without replay and retains the corresponding convolution history and KV [27]. LumoTree already coordinates recurrent state, convolution, KV remapping and the next drafter input [26, Algorithm 1]. Forge packs the selected pending DeltaNet operands and convolution rows into the existing buffers, compacts the new KV, gathers the selected tap features in path order, and calls the existing chain accept routine. This is a runtime integration contribution. Section 3.5.
4. **Tree attention inside a split ANE + GPU target call.** Mirror-SD maps the drafter to NPUs and the target to GPUs [30]. In Forge's 96K decode entry, the target itself crosses engines twice per eight-layer package for GPU attention. Metal applies the ancestor mask and writes the new rows directly into shared INT8 KV storage. The KV-independent ANE functions on this split path can be reused at different key lengths. Serving a larger context also requires adequate cache allocation and prefill support. Sections 2.1 and 3.2.
5. **Static KV widths that reuse the resident cache.** Decode context-length bucketing, including room for speculative writes, has documented precedent [33]. Forge's 24/40/48/56K entries bind whole KV segments from the resident larger entry and use the smallest compiled width covering all eight rows. The bucket may contain masked spare keys. On the checked 471 calls the logits matched exactly. On the long-session comparison the four widths improved throughput at identical replies. Sections 3.6 and 4.4.
6. **An M6 breakdown of the cost of wider verification.** Sequoia profiles hardware to choose its tree [7]. DDTree explicitly measures increasing verification latency and a throughput optimum [10]. On the M6, four FFNs grew only 2% / 4% at 16 / 32 rows, while three DeltaNet mixers grew 27% / 130% in the original decomposition. The solve ablation, convolution taps and long-context attention explain much of the difference. The improved 8–16 sweep found its largest 32K step at row 15. The full-model throughput consequences are projections. Section 5.1.

### M6-specific findings

These results apply to the recorded 32 GB M6, macOS 27.2/27.3, build and workload. The archive identifies the route, measurement and baseline for each.

- **Shared DRAM bandwidth.** The GPU stream reached 156 GB/s, the ANE split-call stream 121 GB/s, and the two together 164 GB/s against a nominal 170 GB/s. Concurrent streaming slowed both engines even with a steady GPU clock. This supports bandwidth contention in those runs.
- **Drafter placement.** A synthetic GPU cost twin took 8.9 ms alone and 13.3–14.5 ms in selected overlapping arms. Its matrices used synthetic indices and did not implement the full draft path. The served ANE draft phase took about 13.3 ms including 12.3 ms of ANE work. These figures and the concurrent verify slowdown supported retaining the ANE drafter.
- **Engine occupancy in decode.** The served decode keeps the ANE 98-99% busy (draft 14.4 ms + verify 94-135 ms of a
  110-151 ms cycle; the rest is host work), while the GPU is idle below 64K. A batched GPU draft of B candidate
  outcomes took 8.64 / 8.97 / 9.61 / 11.70 ms alone at B = 1 / 2 / 4 / 8 (synthetic-index twin), and run once per
  cycle beside the served ANE verify it slowed the verify by +1.8-2.7% at 2K and 32K. The GPU therefore has room for
  draft-ahead; the drafter's inputs do not (Section 5.3).
- **Memory at a context-entry switch.** On fresh launches, the 32K -> 64K switch added 1.06 GB of KV buffers
  (IOSurface) on both the sliced and the unsliced build, with swap flat; forge peaked at 6.6 GB over a 55.5K session.
  The sliced build costs about 75 MB more. A 22 Nov watchdog kill (14.3 GB after 93 requests on one process) was a leak, not
  the switch: the GPU engines kept the Metal wraps of KV segments freed at every context reset (+1.8 GB per cycle).
  Fixed the same day; the footprint stays flat over alternating cold 2K / 56K requests (`results/22nov/memleak`).
- **GPU clock hold on split decode.** Between short attention bursts the GPU ran at 502 MHz in the GPU-only test and about 860 MHz in the server. Keeping one SIMD group busy held 1735 MHz and took the split verify from 138.2 to 122.4 ms over four on/off pairs at 66–70K tokens, at +2.1 W combined. The served policy used a 60 ms hold. The ordinary ANE-only verify does not use this route.
- **Cross-engine ordering.** A serial GPU round trip added about 0.65 ms beyond the kernel in the recorded probe. The tested Core AI compute stream on 27.3 failed to enforce the required cross-engine ordering, so the split runtime orders those operations on the host. A separate probe at drafter shapes stalled before producing a usable join timing.
- **Concurrent calls to one function.** In the experimental prefill schedule, overlapping the same `ffn2_L` function with different bindings corrupted outputs. A per-function lock restored exact output in that test. Other functions and runtime releases require separate tests.
- **Bound tensors and cache storage.** A four-block 32K adapter failed compilation at approximately 51 bound tensors. The tested binding path exposed whole IOSurface segments. The sliced verifier therefore reuses whole segments and narrows the attention width inside the program.
- **Numerical behavior.** In one four-layer assay, the ancestor-mask-only entry matched the chain while one-hot convolution taps gave relative error 1.8 × 10⁻³. An earlier correction still differed in two pending outputs. The final `corrx0` form passed the chain controls. A traced in-place `index_put` prevented ANE execution in that build. Section 3.3.
- **Verifier width.** At 32K, a four-layer branched probe took 6.750 ms at width 14, 7.089 at 15 and 7.096 at 16. A fixed-anchor replay using the measured costs projected a loss for every width above eight at 32K. Section 5.1.
- **JetSpec++ on the serial ANE schedule.** The 16-row LUT4 adapter took 22.2 ms at 8K and streamed about 110 GB/s. W4A8 changed adapter time little. Combining component timings with oracle replay **projected 12–46% lower throughput** than DFlash2 across the tested settings. Section 5.2.
- **Repeatability and state hygiene.** Two fresh chain runs gave identical outputs on 18/18 matched requests but differed by 4.4% in pooled client tok/s. This is one control pair. The slicing comparison used ABBA; the first sampled tree comparison was one A/B followed by a reversed block with different seeds. A separate used-server test recovered 9/9 fresh replies with a cold KV scrub, versus 1/9 without it after the specified long request. Sections 4 and 6.

## 1. Introduction

Speculative decoding [1, 2] hides the cost of a large target model behind a cheap drafter: the drafter proposes k
tokens, the target scores all of them in one call, and an acceptance rule keeps a prefix. On GPU configurations with spare capacity in verification, extra rows can be inexpensive. Several systems use that capacity to verify a tree of candidates
[3, 4, 5, 6, 7], including trees built from a block diffusion drafter's per-position distributions [10, 11].
Our target runs largely on the ANE, which executes precompiled programs of fixed shape, and whose cost per extra row is
not small for this hybrid model. The questions were:

- Can a tree help when the verify width cannot grow?
- What does a wider verify really cost on the ANE, and where does the cost go?
- Can the tree be exact (same outputs as the chain when the tree is a chain), so that served replies stay reproducible?

## 2. System

### 2.1 Target model and placement (V8)

- **Model:** Qwen3.8-27B, 64 layers (48 Gated DeltaNet, 16 full attention at layers 3, 7, ..., 63), hidden 5,120,
  vocabulary 248,320. Release 0.2 weights: 3-bit vector-quantized MLPs, rank-64 FP16 residual factors on the mixer
  projections; an INT8 KV cache (KV8) shared by the ANE and the GPU.
- **Prefill (V8):** every FFN on the ANE, every mixer on the GPU (Metal), with a shared INT8 KV cache.
- **Decode verify (dec8 build):** 8 packages of 8 layers. Below 64K the verify is one ANE program per package. In the
  96K entry the verify is split per package: ANE `dva` (up to the attention projections), GPU attention (INT8 Q.K,
  online softmax, FP16 x INT8 P.V, the 8 new K/V rows written into the ANE's own KV segments), ANE `dvba`, GPU
  attention, ANE `dvb`. The split ANE programs bind no KV, so changing the key length does not require recompiling these split ANE functions.
- **Memory bandwidth** (the shared resource): GPU streaming 156 GB/s, ANE split calls 121, both at once 164 of a
  170 GB/s peak.

![Neural Engine and GPU sharing the INT8 KV cache](results/22nov_review/tree_verify_figures/01-hardware.png)

**Figure 1. Hardware placement.** DFlash2 and the target feed-forward layers run on the ANE. The GPU handles token mixing during prefill and the attention operations of the split 96K decode verifier.

### 2.2 Drafter (DFlash2)

The drafter is DFlash2, a DFlash-style block drafter [9] conditioned on target features, derived from a public
checkpoint [19] and converted to Core AI for the ANE.

- 5 layers, 32 heads, conditioned on 5 target tap features (layers 5, 19, 33, 47, 61), drafting a block of 7 tokens
  per pass. LUT4 weights (Hadamard + GPTQ), on the ANE.
- A **candidate selector** (predecessor and successor codebooks, rank 256, top-16 candidates) scores each candidate
  token given its parent. The chain is the argmax child at each depth.
- Draft phase per cycle: 13.3 ms, of which 12.3 ms is ANE work (body 6.86, head 4.41, context 0.99 ms) and about
  1.0 ms host. The GPU cost twin slowed under concurrent verify load. The serving configuration keeps the drafter on the ANE.

### 2.3 Baseline: the 8-row chain

On a realistic prompt set (code, math, chat, 2K to 96K context) the served chain accepts 4.57 to 5.16 tokens per
verify. Code and math sit near the 7-draft cap (7.1 and 6.5 tok/verify); chat is at 2.4. The gap to the cap is where a
tree can help.

## 3. Method

### 3.1 Tree construction in 8 rows

One DFlash2 forward produces 16 candidates at each of seven depths, their unary scores and one selector vector per depth. For a child token `c` and parent token `p`, the score is the unary score plus the dot product of the successor code for `c` with the elementwise product of the predecessor code for `p` and the depth’s selector vector. Changing the parent reranks that same shortlist. There is no additional drafter forward on a branch.

Per cycle, from the anchor token:

1. Expand best-first by **path probability** (as in dynamic draft trees [5, 10]) using the product of conditional probabilities along each path. At each parent, normalize all 16 selector scores with `softmax(score / tau)` and enqueue the four highest-scoring children (`tau = 1.5` in the served build).
2. Stop at the anchor + 7 nodes. Rows are filled in pop order, which is topological.
3. If the 7 nodes form the chain, the cycle runs the served chain entry (policy `branched`); otherwise the tree entry.

The dynamic builder produced 707 distinct shapes on the realistic set.

### 3.2 Tree verify entries

The tree entries (`t8_<ctx>k`, and `tdva` / `tdvba` in the split) are the served 8-row programs with two extra inputs
on the same weight constants, so they add no weight memory:

- **Ancestor mask:** each row attends to the KV history plus its ancestors only (in the split, the GPU attention
  applies the same per-row mask).
- **Depth positions:** RoPE at `p0 + depth`, not `p0 + row`.

### 3.3 DeltaNet in a tree

The 48 DeltaNet layers are recurrent, so a tree needs two things a chain gets for free.

- **Pending tape, not a state per row.** Gated DeltaNet [13] is computed chunkwise with the UT / WY form of the delta
  rule [14]. The verify keeps the block's DeltaNet rows (k / u / wk) and cumulative decay
  columns as a pending tape, and the in-block recurrence is a matmul over a mask input. With the ancestor mask in place
  of the causal one, each row sees only its own path, and no state is copied per branch. (Path-local verification and compact speculative state are also described in [18, 24, 25, 27, 31, 32].)
- **Exact tree convolution (`corrx0`).** DeltaNet's short causal conv must read each row's ancestors, not the
  previous rows.
  - A one-hot gather (`ctap @ rows`) was the first form. It is not exact on the ANE (rel 1.8e-3 per group) and flips
    near-ties.
  - The exact form keeps the served conv and adds one correction matmul: `[rows * w_t] (cdim x 44) @ D (44 x 8)`, with
    `D = ctap - shift` and the chain shift built in numpy as a constant.
  - With chain masks, all 12 group outputs matched bit for bit in the form probe and all 45 greedy replies matched at 2K–96K. The probe cost +4.5% per group. The measured tree/chain verify cost ratio used for replay was 1.024–1.039.
  - Pitfall: an in-place `index_put` inside the traced forward produces ops the ANE will not run (the package falls
    off the ANE).

### 3.4 Acceptance

- **Greedy:** from the root, take the child whose token is the row's argmax, until none matches. The next token is the
  last accepted row's argmax.
- **Sampled (`TREE_SAMPLED=1`):** at each node, the chain's own rule child by child: accept child `c` with probability
  `r(c)`, where `r` is the row's sampling distribution with the rejected siblings removed and renormalized; if all are
  rejected, sample the next token from `r`. This is multi-candidate recursive rejection sampling [3, 8], so the output
  has the distribution of plain sampling. A one-child node makes exactly the chain's random draws. Checked against the release loop, including
  the RNG state.
- **No better lossless rule exists for these drafts.** The drafts are deterministic (top-k from the selector, not
  sampled). For a fixed tree, any lossless rule can accept through a node at most with the target probability of its
  path, so expected tokens per verify are bounded by 1 + the sum of those path probabilities. The child-by-child rule
  reaches the bound (its acceptance equals "sample from the target, follow the matching child"). Rules that gain by
  coupling sampled drafts to their proposal, such as traversal verification, block verification, SpecTr and optimal
  multi-draft acceptance [34, 35], reduce to it for point-mass drafts. Section 5.3.

**Stopping.** A matched stop-token ID ends the walk before that child enters the committed path. User-supplied stop strings are checked against detokenized text in the outer generation loop. A string crossing a token boundary can trim the returned text after the cycle has already committed model state. The paper's parity tests do not establish rollback to that text boundary or retraction of previously streamed characters.

### 3.5 Commit: back to a chain

Let the accepted path, including the anchor, have k + 1 rows. If its row IDs already equal 0..k, buffer compaction is skipped. Otherwise the runtime packs each DeltaNet layer's pending k/u/wk rows, cumulative-decay columns and convolution rows into those positions. For non-split verify it packs the entry's new K/V outputs. For split verify it moves accepted rows and their quantization scales within the shared KV cache. In either case it gathers the drafter features in path order, appends them at consecutive positions, and calls `mb.accept(k + 1)`. The recurrent matrix is advanced through the ordinary chain mechanism. The object being rearranged here is the **pending tape**, and the feature gather has its own destination.

![An eight-row tree, its ancestor mask, and compaction of rows 0, 2, 4 and 5](results/22nov_review/tree_verify_figures/02-tree8-method.svg)

**Figure 2. One tree verification cycle.** Row 0 holds the anchor. In this example the accepted path is 0 → 2 → 4 → 5. Compaction places those four rows in slots 0 through 3. The mask also governs the in-block DeltaNet recurrence. Parent-path convolution and coordinated accepted-state publication are documented in [26, 27]. The M6 implementation packs pending operands for its existing chain commit.

### 3.6 Sliced verify entries (22 Nov)

A verify at position `p0` in the 64K entry still reads 64K keys. Sliced entries (`v8s_<w>k`, `t8s_<w>k`) bind the
same KV segment buffers but attend only to the first `w` keys. A call uses the smallest `w` with `p0 + 8 <= w`. The
served set is 24/40/48/56K, binding the 32K and 64K entries' KV. In the group probe, the intermediate widths fell close to linear interpolation between neighboring entries. Sliced calls were bit-exact against the bigger entry
with the keys at or above `w` masked: 471 calls, max |logit diff| 0, KL 0.

## 4. Evaluation

### 4.1 Setup

- Apple M6, macOS 27.2 / 27.3. One model process on port 8765.
- **Prompt sets:** a realistic set (code, math, chat at 2K to 96K); an agent set (Pi-like code-agent turns); and a long
  Pi-like session (19 turns, 8.6K to 55.5K tokens, 512-token replies), greedy plus three sampling seeds at
  temperature 1.0 with thinking on.
- **Metric:** client-side decode tok/s, token-weighted. The 18-request same-output restart control moved from 43.20 to 45.09 tok/s (+4.4%). Section 4.3 reports two blocks with opposite run orders and different seeds. Section 4.4 reports a fresh-launch ABBA experiment. Fixed-anchor replay estimates the cost and yield on the chain's recorded texts.

**Origin of the +4.5–5% projection.** On 19 Nov, `tree_sim_report2.json` covered 12 greedy requests and 964 fixed anchors. The simulated chain reproduced the logged acceptance on all 964. Dynamic top-7 raised tokens per verify from 3.064 to 3.348 (+9.3% before verifier cost). Its mean maximum depth was 4.98. Zero-accept cycles fell from 248 to 153, but seven-accept cycles also fell from 51 to 41. Applying a measured 4.6–5.3% tree-graph overhead from layers 0–3 to the served verification time projected +4.5% to +4.9% decode throughput. The 32K and 64K projections reused the same 64K cost curve and assumed the split attention scaled with the ANE. They provide no independent split-kernel timing. The later +4.5% in §3.3 is a *latency cost* for the corrected convolution. The served measurements below use separate runs.

### 4.2 Tree8 vs chain, greedy

| measurement | result |
|---|---|
| Realistic set, end to end (2K-64K rungs) | +1.9% to +6.5% tok/s |
| Like for like (chain's own texts, measured tree cost x1.024-1.039) | +0.9% to +3.5% by rung; chat +9% |
| Served 21 Nov: 2K / code / agent | 45.80 / 40.66 / 54.04 vs chain 42.74 / 39.46 / 54.48 tok/s |
| 2K, fresh-launch like for like | +5.1% (44.91 vs 42.74) |

The realistic set includes high-acceptance code and math. On the D4 fixed-anchor code replay, chain8 projects to 57.43 tok/s, always taking the tree entry to 56.24, and dispatching it only on forks to 57.68. That last change is about +0.4%, compared with the much larger gain on low-acceptance chat. Section 5.4 tests the reviewer’s alternative shape.

### 4.3 Tree8 vs chain, sampled with Pi settings

Before `TREE_SAMPLED`, every temperature > 0 cycle went to the chain, so Pi (temperature 1.0) never used the tree.

| set | change |
|---|---|
| overall | **+8.5%** |
| 8K / 32K | +9.0% / +11.6% |
| code / agent | +10.9% / -1.6% |
| tok/verify | 3.91 -> 4.35 |
| ms per cycle | +2.4% |

The table covers the first 36 benchmark requests. A second 36-request block reversed the launch order with new seeds and gave +10.2% pooled and −3.9% on tool-agent. Brief CPU contention affected its 8K arm. A separate long Pi-like benchmark on 22 Nov gave +10.0% for branched tree8 over chain8 across both orders and three seeds. Production Pi traffic has no matched baseline here.

On these benchmark aggregates sampled gains exceed greedy ones. Our reading, not separately measured: sampled chains reject earlier, so a second
candidate at an early depth is worth more.

![Sampled throughput gains by cohort alongside measured verifier-group latency at widths 8 through 16](results/22nov_review/tree_verify_figures/03-tree8-speed-and-width.svg)

**Figure 3. Throughput and width cost.** Left, the two 36-request blocks compare tree8 with chain8. The overall gain and the tool-agent regression both recur when launch order is reversed and seeds change. Right, each point times a four-layer hardware probe. At 32K the step from 14 to 15 rows costs 0.34 ms per group. The wider full-model throughput estimates come from replay using these measured costs.

### 4.4 Sliced entries on the long session (22 Nov)

Served tree8 vs tree8 + sliced 24/40/48/56K, ABBA, 4 sets (greedy + seeds 7100/7200/7300), 152 replies:

| band | served | sliced | change (pooled) |
|---|---|---|---|
| 8-16K | 39.60 | 40.09 | +1.24% |
| 16-32K | 36.15 | 37.13 | +2.72% |
| 32-57K | 29.64 | 31.33 | +5.69% |
| all | 34.65 | 35.82 | +3.40% |

One served set ran about 11% slow on its first 7 turns, below any sliced entry. Without that seed the overall gain is
**+2.49%** (order a +2.11%, order b +2.86%). All 152 replies are identical. Tree cycle 111.4 -> 108.2 ms, chain cycle
107.0 -> 104.4 ms. Served since 22 Nov 11:15.

![Comparison of 64K and 40K KV reads and sliced-verifier gains by context band](results/22nov_review/tree_verify_figures/04-sliced-kv.svg)

**Figure 4. Slicing the KV prefix.** At a 33K prefix, the 40K entry covers all eight new rows. The right panel compares the four-entry sliced build with the full-width tree8 build. The pooled speedup is 3.40%, or 2.49% after excluding seed 7100, whose baseline had seven unusually slow early turns. All 152 paired replies matched.

## 5. What did not work

### 5.1 Wider trees on the ANE

Earlier layer-group probes, relative to the eight-row chain (8K–64K entries). The tree rows below use the original tree implementation; the later partial-block sweep uses `corrx0` and a different baseline (Figure 3).

| width | cost |
|---|---|
| tree8 | +4.6% to +5.3% |
| tree16 | +26% to +39% |
| tree32 | +87% to +116% |
| 16-row chain, measured group with block solve | +8.9% at 8K, +18% at 32K, +28% at 64K |

A separate group-plus-head projection for a full 16-row chain verify gave 1.085× / 1.17× / 1.27× the eight-row chain cost at 8K / 32K / 64K, respectively.

Where a wider verify's cost goes:

- **FFN:** nearly free (+2% / +4% at 16 / 32 rows).
- **Three DeltaNet mixers:** +27% / +130% at 16 / 32 rows. The separate solve ablation largely removed the width dependence. At 32 rows block forward substitution reduced the three-mixer time from 4.03 to 2.44 ms.
- **Tree convolution taps:** the original one-hot form added 0.66 / 1.58 ms per group at 16 / 32 rows. At 64K the attention increment was larger.
- **Attention at long context:** the score passes.

In the later sweep the measured 32K group cost rose 5.0% from row 14 to 15, then 0.1% from 15 to 16. Fixed-anchor greedy replay using those costs projected at most +1.6% over tree8 at 8K; all wider entries lost at 32K. Full-model serving at widths 9–16 was not timed in this sweep. Splash 1.7.0 [18] measured its one-lane step at 111.8 ms for eight rows and 112.8 ms for sixteen (+0.9%) on this M6. That within-engine timing shows how cheaply its verifier could widen. End-to-end differences also include the quantizer, lookup policy and generated tokens, so its acceptance advantage cannot be attributed solely to tree width.

### 5.2 Other policies and drafters

| idea | result |
|---|---|
| Per-cycle tree / chain choice by a cost model (`TREE_POLICY=dink`, a Dinkelbach-style ratio rule [21]) | long session -5.9% (-10.8% at 32-57K): the running-mean chain estimate is inflated by confident cycles |
| Per-cycle size selection 8/16/32 (offline) | +0.3% to +1.6%; oracle only +5-9% |
| Fork-aware ranking, tau sweeps | no gain; served tau 1.5 already best |
| Prompt lookup (n-gram) drafts [17] | full-file code edits +101%; gated (`yield:1.25`) code +92% greedy / +27% sampled; long Pi session: no gate gains at 3 seeds. Off for Pi |
| Lookup + drafter tree combined | loses: it pays for both the drafter and the tree verify |
| JetSpec++ [12] (B16 drafter) by replay | x1.42 to x1.81 tokens per round, but component timings and oracle replay project 12–46% lower throughput on the serial ANE schedule. Parked until drafting can overlap the verify |
| Drafter on the GPU | synthetic GPU twin slows under concurrent ANE verify load (§2.2) |

### 5.3 Selection, acceptance and draft-ahead (22 Nov paper sweep)

A sweep of 27 recent papers (tree selection for block-diffusion drafters, sampled acceptance, hybrid-model verify,
drafter overlap) was checked against the logs; untried and applicable ideas were measured offline on 4,529 logged
sampled cycles (`results/22nov/PAPER_SWEEP.md`). The metric is expected tok/verify, 1 + the sum of target path
probabilities, exact for the served tree (4.08).

| probe | realistic 2K-32K | long Pi-like session |
|---|---|---|
| ceiling: best 7 nodes from the same 16 drafter candidates per depth (lower bound) | +8.6% | +10.2% |
| ceiling: perfect proposer (lower bound) | +10.8% | +13.2% |
| calibrated edge scorer (rank, score, depth, fork distance; TAPS / D3Tree / TreeSpark style [38]) | -0.1% | -0.1% |
| trigram PMI rescoring from Pi's own outputs (DForkTree [11]) | -0.2% to -4.3% | -0.2% to -3.7% |
| one sampled child at the anchor (RheoSampling [36]) | -4.4% | -3.8% |
| verification-only rules [34, 35] | 0 (Section 3.4) | 0 |

- **The ceiling is real but not reachable from the host.** The drafter already proposes the better nodes; choosing
  them needs the target's distribution at the verified rows, which exists only after the verify. Relaying verifier
  information into the drafter (DFlow [37]) or training the drafter's scores for path acceptance (VSD [39]) target
  this gap; both need drafter training (Section 7).
- **Draft-ahead fails on the drafter's inputs, not on the hardware.** Drafting cycle c + 1 during cycle c's verify
  (JetSpec++ foresight [12], Mirror-SD [30]) would hide up to about 7 ms per cycle on the idle GPU (hit rate of the
  verify outcome among 8 predicted (endpoint, bonus) pairs: 53-57%). But DFlash2 then lacks the target features of
  the tokens it continues. Replaying the 20 Nov tap captures with the drafter's context stopping at the previous
  anchor cut tree8 tok/verify from 7.30 to 2.67 (humaneval, 353 cycles, -63%), 6.22 to 2.50 (mbpp, -60%) and 3.84 to
  2.00 (mt-bench, 935 cycles, -48%); even a one-position gap cost mt-bench 23%. Draft-ahead needs a trained feature
  predictor, as JetSpec++'s adapter provides.
- **Committing inside the next verify** (SpecLA [25]) would save part of a 1.1-1.4 ms commit, about 1% of a cycle.

### 5.4 A six-token spine with one root sibling

The review proposed keeping six consecutive top-1 draft tokens and spending the seventh slot on the runner-up at depth one. Relative to chain8 this recovers one draft token if the target takes that runner-up and gives up one if the target follows the entire seven-token chain. The root and total row budget stay fixed.

| Greedy fixed-anchor replay | Cycles | Chain8 | Dynamic top-7 | Six-token spine + sibling |
|---|---:|---:|---:|---:|
| Original low-acceptance set | 964 | 3.064 | 3.348 | 3.113 |
| Later realistic set, all | 1,948 | 4.953 | 5.123 | 4.614 |
| Code subset | 423 | 7.057 | 7.111 | 6.314 |
| Math subset | 740 | 6.459 | 6.530 | 5.888 |
| Chat subset | 785 | 2.399 | 2.726 | 2.497 |

Entries are tokens per verify, including the next target token. Each greedy comparison uses the same logged anchors and output tokens for every policy. We used the original selector temperature of 1 on the 19 Nov trace and the served value of 1.5 on the later trace. The simulated chain reproduced all logged accepted lengths in both sets. The fixed shape helps on the original log and chat, while its lost seventh token costs more on code and math.

On **2,642 sampled cycles from the long Pi-like benchmark**, the logged policy gave 4.075 expected tokens per verify. The same fixed spine gave **3.683–3.727**. To bound unobserved paths, we allowed their unknown next-token probabilities to range from zero to one and allowed unlisted tokens at an observed parent to use its remaining probability mass. Even the upper bound falls below the logged policy. These are fixed-anchor expectations from the benchmark, separate from production Pi traffic and wall-clock throughput.

We also tried the proposed **early-confidence fallback** at one threshold chosen before replay. A gap of at least 0.5 between the two highest root probabilities sent the cycle to the chain; otherwise it used the existing topology-based dispatch. The realistic greedy set gave 5.051 tokens per verify overall and 7.054 on code, with tree dispatch on 26.3% and 6.9% of cycles respectively. This is an untuned yield comparison. A throughput claim for the gate would need whole-cycle timing on held-out requests. Replay inputs, implementation and aggregates are archived in `results/22nov/reviewer_spine/`.

## 6. Correctness and reproducibility

- **Identity criterion.** A tree can change the cycle split and subsequently the floating-point decisions. A
  chain-only control with a different cycle split keeps only 20 of 45 replies identical; branched tree8 keeps 23 of
  45. The forced-chain greedy control matched 45/45 replies. The sampling rule targets the verifier's conditional distribution. Host tests covered 5,040 topological parent arrays and 128 increasing paths in each compaction mode. The unresolved device test starts a branched verify and two ordinary chain verifies from the same snapshot and positions. Give siblings the same depth and RoPE position. Compare each path’s full-vocabulary logits, argmax and top-two margin, then commit an off-prefix leaf and compare the next fixed anchor’s logits, pending tape, convolution, KV/scales and drafter features. Vary the unused sibling, repeat a chain control, and test the ANE-only and 96K split entries separately. The existing controls establish forced-chain parity only.
- **Used vs fresh server.** Replies on a used server differed from a fresh one. The cause was the drafter's context
  ring, which snapshots restored by position only: a cold request rewrote those slots, and the next restore drafted
  from the wrong rows. Fix (`DRAFT_RING_SNAP=fix`, served): copy the visible ring slots into the snapshot.
- **Leftover.** After a long request, 1/9 cold-start replies matched the fresh reference. Clearing leftover KV rows (`COLD_SCRUB=kv`, 2–8 ms) restored 9/9 in that experiment. This optional scrub covers cold starts. Partial restores and continuations have separate state lifetimes.
- **ANE pitfall.** The concurrent `ffn2_L` prefill test required a per-function lock. The outside review also corrected the order in which the split tree mask was installed relative to GPU buffer binding.

## 7. Limitations and next steps

- The tree's gain is capped by the 8 rows. More tokens per cycle need a better drafter, not a wider tree, unless the
  ANE's per-row DeltaNet cost comes down. The measured selection ceiling (+8.6-10.2% from the same candidates) needs
  verifier information in the drafter (DFlow [37], VSD [39]); a DFlow prototype plan is in
  `results/22nov/DFLOW_PLAN.md`. Draft-ahead needs a feature predictor (Section 5.3).
- Tree verify adds 2-4% per cycle. The cost is the conv correction and the mask inputs, not the FFN.
- Gains are measured on one machine and one model pair. A same-output restart control differed by 4.4%. Launch order, seed, output trajectory and workload all matter when interpreting small gains.

## 8. Related work

- **Speculative decoding.** Draft-then-verify with a lossless acceptance rule [1, 2].
- **Tree verification.**
  - SpecInfer [3] introduced token-tree verification with tree attention (our ancestor masks).
  - Medusa [4], EAGLE / EAGLE-2 / EAGLE-3 [5, 6] and Sequoia [7] build static or dynamic draft trees for GPU verifiers.
  - Recursive speculative decoding [8] studies sampling without replacement in trees.
  - Our setting differs in one constraint: the verify width is fixed at 8 rows by a precompiled ANE program, so the
    tree replaces the chain inside the same rows.
- **Block diffusion drafters and their trees.** Our drafter is a DFlash-style block drafter [9]. The closest prior art
  to our tree construction is DDTree [10], which builds a draft tree from one block diffusion pass and verifies it in
  one target pass. DForkTree [11] makes that tree fork-aware; our fork-aware ranking experiment (section 5.2) gave no
  gain in 8 rows. We implemented the tree independently on DFlash2's candidate selector and did not use code from [10]
  or [11]. What we add is listed in [What's new](#whats-new).
- **Parallel drafting.** JetSpec [12] and JetSpec++ [12] draft larger blocks or trees with a learned head and
  self-refinement. We replayed JetSpec++'s published Qwen3.8-27B drafter on our target (section 5.2) and measured its
  cost on the ANE.
- **Linear attention in a tree.** TreeWY extends the chain WY solve to strict ancestors and reconstructs one accepted GDN state [24, §3]. SpecLA uses topology-aware verification and factor-buffer state recovery [25, §§4–5]. LumoTree publishes recurrent, convolution, attention and drafter state along the accepted path [26, Alg. 1]. Multi-Future Drafting walks parent pointers for short convolution and commits a rank-|P| update from accepted-node factors [27, Appendix C]. Weaver also derives a rollback-free GDN tree verifier [31], and Twig stores token-level factors [32]. STree is earlier tree verification for structured state-space models and SSM/attention hybrids [29]. HybridSpec adapts **chain length** for GDN-MoE and measures its cost [28]. These precedents delimit Forge's ANE implementation contributions.
- **NPU + GPU speculative decoding.** Mirror-SD maps drafting to NPUs and target verification to GPUs [30, §3 and experimental setup]. Forge additionally splits attention inside the target's 96K verify call, using the shared KV cache.

- **DFlash2 trees on the same target (Metal).** TensorFold [23] (v1.3.2) runs DFlash2 trees for Qwen3.8-27B on Apple
  GPUs. It measures a row-cost table, keeps ancestor-closed node sets, and allocates rows by expected accepted tokens
  per measured millisecond; its documented legacy serial window is 16 rows. That allocator is the same idea as our cost-model policy
  (`dink`, section 5.2), which lost on our target. It also reports that verifying siblings in separate slots, each
  duplicating the pending row, gave no gain on DeepSeek-V4.1 Metal; our shared-prefix tree avoids that duplication.
  We learned of this comparison from the outside review (21 Nov) after tree8 was built and served, and used no
  TensorFold code.
- **Apple silicon engines.** Splash [18] (Inco AI) is the GPU + ANE reference we compared against throughout; its
  16-row tree motivated the width study, and its DeltaNet tree handling matches ours.
- **Quantization of the drafter.** GPTQ [15] with a Hadamard rotation [16].
- **Attention kernels.** The split verify's GPU attention is a flash-decoding style kernel [20].

## Acknowledgements

- The Qwen Team (Alibaba Cloud) for Qwen3.8-27B [22].
- The DFlash2 drafter is derived from ProCreations/Ternary-Bonsai-2-27B-DFlash2 [19] (Apache 2.0, upstream NOTICE
  preserved), with selector codebooks extracted from it; drafter parity uses the vendored DFlash reference code
  (MIT) [9].
- The JetSpec++ drafter, adapter and calibration used in the replay are the authors' published artifacts [12].
- Inco AI's Splash [18] served as the measured GPU + ANE reference (versions 1.3.x to 1.7.0).
- An outside review of the 21 Nov code found the empty-ring restore crash, the non-atomic 'sysk' restore and the
  split tree mask ordering (section 6), and corrected the lookup replay's row budget.
- TensorFold [23]: its Qwen3.8 Gated DeltaNet Metal recipe was part of a 15 Nov survey of public DeltaNet kernels
  (prefill, not decoding; nothing adopted), and the outside review used its tree path as a comparison.
- The M6 ANE bot proposed several of the experiments in section 5 (fork-aware ranking, the cost-model policy, the
  lookup gates).

## References

1. Y. Leviathan, M. Kalman, Y. Matias. Fast Inference from Transformers via Speculative Decoding. ICML 2023.
   arXiv:2211.17192.
2. C. Chen, S. Borgeaud, G. Irving, J.-B. Lespiau, L. Sifre, J. Jumper. Accelerating Large Language Model Decoding
   with Speculative Sampling. arXiv:2302.01318, 2023.
3. X. Miao et al. SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and
   Verification. ASPLOS 2024. arXiv:2305.09781.
4. T. Cai et al. Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads. ICML 2024.
   arXiv:2401.10774.
5. Y. Li, F. Wei, C. Zhang, H. Zhang. EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (ICML 2024,
   arXiv:2401.15077) and EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees (EMNLP 2024,
   arXiv:2406.16858).
6. Y. Li, F. Wei, C. Zhang, H. Zhang. EAGLE-3: Scaling up Inference Acceleration of Large Language Models via
   Training-Time Test. arXiv:2503.01840, 2025.
7. Z. Chen et al. Sequoia: Scalable and Robust Speculative Decoding. [arXiv:2402.12374](https://arxiv.org/abs/2402.12374), 2024.
8. W. Jeon, M. Gagrani, R. Goel, J. Park, M. Lee, C. Lott. Recursive Speculative Decoding: Accelerating LLM Inference
   via Sampling Without Replacement. arXiv:2402.14160, 2024.
9. J. Chen, Y. Liang, Z. Liu. DFlash: Block Diffusion for Flash Speculative Decoding. ICML 2026 (PMLR 306).
   arXiv:2602.06036.
10. L. Ringel, Y. Romano. Accelerating Speculative Decoding with Block Diffusion Draft Trees (DDTree).
    [arXiv:2604.12989](https://arxiv.org/abs/2604.12989), 2026.
11. DForkTree: Accelerating Speculative Decoding with Fork-Aware Block Diffusion Draft Trees. arXiv:2611.12356, 2026.
12. JetSpec: Breaking the Scaling Ceiling of Speculative Decoding with Parallel Tree Drafting (arXiv:2606.18394,
    2026), and Z. Feng et al., JetSpec++: Foresight Parallel Drafting with Self Refinements (arXiv:2610.19792, 2026).
    Code: github.com/hao-ai-lab/JetSpec.
13. S. Yang, J. Kautz, A. Hatamizadeh. Gated Delta Networks: Improving Mamba2 with Delta Rule. ICLR 2025.
    arXiv:2412.06464.
14. S. Yang, B. Wang, Y. Zhang, Y. Shen, Y. Kim. Parallelizing Linear Transformers with the Delta Rule over Sequence
    Length. NeurIPS 2024. arXiv:2406.06484.
15. E. Frantar, S. Ashkboos, T. Hoefler, D. Alistarh. GPTQ: Accurate Post-Training Quantization for Generative
    Pre-trained Transformers. ICLR 2023. arXiv:2210.17323.
16. S. Ashkboos et al. QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs. arXiv:2404.00456, 2024.
17. A. Saxena. Prompt Lookup Decoding. github.com/apoorvumang/prompt-lookup-decoding, 2023.
18. Inco AI. Splash: a local inference engine for Apple silicon. github.com/incoai/splash.
19. ProCreations. Ternary-Bonsai-2-27B-DFlash2. huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2, revision
    4cfb6ad0.
20. T. Dao, D. Haziza, F. Massa, G. Sizov. Flash-Decoding for long-context inference. 2023.
21. W. Dinkelbach. On Nonlinear Fractional Programming. Management Science 13(7), 1967.
22. Qwen Team. Qwen3.8-27B. huggingface.co/Qwen/Qwen3.8-27B.
23. TensorFold. [Qwen3.8-27B recipe, v1.3.2](https://github.com/ashhart/TensorFold/blob/v1.3.2/docs/recipes/qwen3.8-27b.md). [DeepSeek sibling-slot experiment, PR 3326](https://github.com/ashhart/TensorFold/pull/3326).
24. [TreeWY: Speculative Verification for Gated DeltaNet Hybrids](https://arxiv.org/html/2608.20961v2).
25. [SpecLA: Efficient Speculative Decoding for Linear-Attention Models](https://arxiv.org/html/2607.16673v1).
26. [LumoTree: Path-Parallel Speculative Verification for Hybrid Language Models](https://arxiv.org/html/2609.23900v3).
27. [Multi-Future Drafting: Mixture Block Drafters for Tree-Verified Speculative Decoding](https://arxiv.org/html/2611.13017v1).
28. [HybridSpec: Throughput-Aware Speculative Decoding for Hybrid GDN-MoE Models](https://arxiv.org/html/2611.04055v1).
29. Y. Wu, Z. Qin, A. Wong, S. Soatto. [STree: Speculative Tree Decoding for Hybrid State-Space Models](https://arxiv.org/abs/2505.14969v2). NeurIPS 2025.
30. N. Bhendawade et al. [Mirror Speculative Decoding: Breaking the Serial Barrier in LLM Inference](https://arxiv.org/html/2510.13161v2).
31. Y. Oda et al. [Trees from Marginals: Autoregressive drafting with factorized priors](https://arxiv.org/abs/2607.06763v3).
32. L. Wang et al. [Twig: Parallel and Memory-Efficient Tree Speculative Decoding for Hybrid-Attention LLMs](https://arxiv.org/abs/2608.01651v2).
33. AWS Neuron. [Decode context-length bucketing](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.33.1/vllm-neuron/docs/design/vllm/decode-context-length-bucketing.html), v2.33.1 design document.
34. Hu et al. Optimal multi-draft speculative decoding. [arXiv:2502.18779](https://arxiv.org/abs/2502.18779); with
    Z. Sun et al., SpecTr (arXiv:2310.15141), Block Verification (arXiv:2403.10444) and Hierarchical Speculative
    Decoding (arXiv:2601.05724).
35. [Traversal Verification for Speculative Tree Decoding](https://arxiv.org/abs/2505.12398), 2025.
36. [RheoSampling](https://arxiv.org/abs/2609.21827), arXiv:2609.21827, 2026.
37. [DFlow: Enabling Verifier Information Flow in Block Diffusion Speculative Decoding](https://arxiv.org/abs/2609.06498), 2026.
38. TAPS: Target-Aware Prefix Tree Selection (arXiv:2606.00487); D3Tree: Learning to Prune Block-Diffusion Draft
    Trees (arXiv:2611.11371); TreeSpark: Calibrated, Load-Adaptive Draft Trees (arXiv:2609.22098). 2026.
39. [Variational Speculative Decoding](https://arxiv.org/abs/2602.05774), arXiv:2602.05774, 2026.

## Appendix A: served configuration (22 Nov 2026)

- Launcher `pipe/v8_server_dec8_qk.sh`, which execs `v8_server_tree.sh`.
- Build `v8_dec8ts` (the served entries + tree entries + sliced entries, one set of weights), GPU tables `pipe/w_dec8ts`.
- Environment: `TREE=1`, `TREE_POLICY=branched`, `TREE_TOPC=4`, `TREE_TAU=1.5`, `TREE_SAMPLED=1`,
  `RUNG_SLICE=24,40,48,56`, `DRAFT_RING_SNAP=fix`, `QKE=tg`, `NGRAM_DRAFT=0`.
- GPU engines: `libgpu_v8_f8q.dylib` (prefill mixers and attention) and `libdec_gpu_tree.dylib` (split-verify
  attention with ancestor masks). Their KV-wrap memory fix (`libgpu_v8_f8q_uw.dylib`, `libdec_gpu_tree_uw.dylib`,
  built by `pipe/engine/build_engines.sh uw`) has served since 22 Nov 18:20 through `V8_SRV_DYLIB` / `DEC_DYLIB`
  (replies identical; `results/22nov/memleak`). Forge's own memory watchdog (`tools/forge_watchdog.sh`, 12,288 MB).
- Rollbacks: `SERVED_SLICE=0` (tree8 on `v8_dec8tx` without sliced entries), `SERVED_TREE=0` (the 19-20 Nov chain).
- On another Mac: [hf/DEPLOY.md](hf/DEPLOY.md) (build, deploy, test; the portable launcher `hf/serve_v8_96k.sh` and
  the GPU-table kit) and [hf/AGENTS.md](hf/AGENTS.md) (the same as staged steps for coding agents).

## Appendix B: code

- `pipe/tree_verify.py`: tree construction, policies, greedy and sampled acceptance, commit.
- `pipe/rung_slice.py`: sliced entries.
- `pipe/dec_split.py`: the ANE + GPU split verify.
- `v8/build_dec8t_chunk.py`: tree and sliced entry builder (`TREE_FORM=corrx0`, `TREE_SLICED`).
- `bench_tree/`: prompt sets (including `smoke_2k_v1.json`), `tree_bench.py`, `run_arm.sh`, compare scripts.
- `pipe/gpu_tables_kit.py`, `pipe/build_gpu_tables.sh`, `pipe/engine/build_engines.sh`, `hf/`: deployment.
- `pipe/unit/power/tree_dyn2.py`, `t4_replay_sim.py`: offline tree and lookup replay.
