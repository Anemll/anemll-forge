# Temperature sampling with DFlash2: test and implementation plan

- **Status:** plan only, written 3 October 2026. Nothing here is implemented or measured yet unless marked **Measured**.
- **Goal:** keep the target's exact sampling distribution at temperature 0.7 to 1.0 while accepting more DFlash2 drafts per verifier call.
- **Related:** [M6_COMPUTE_ACCELERATION_2026-10-03.md](M6_COMPUTE_ACCELERATION_2026-10-03.md) (verifier cost), [../verifier_len.md](../verifier_len.md) (why the verifier stays at 8 rows), [../SPECULATIVE_DECODING.md](../SPECULATIVE_DECODING.md) (current rules).

## Current behavior (source-verified)

- **Defaults** ([`scripts/qwen38_server.py`](../../scripts/qwen38_server.py), request handling): temperature 1.0 with thinking, 0.7 without; top_p 0.95 / 0.8; top_k 20; `presence_penalty` from `--presence` (default 0). They follow the Qwen3.8 model card except the presence penalty (the card recommends 1.5 without thinking, 0 with thinking).
- **Target distribution** (`Engine.dist`): top-k, then presence penalty (`logit - presence` for every token already generated in this reply), DRY penalty, then temperature and top-p. Greedy (`temperature 0`) returns the argmax and ignores both penalties.
- **Drafter** ([`scripts/dflash2_coreai_drafter.py`](../../scripts/dflash2_coreai_drafter.py), `propose`): one forward gives logits for 7 positions; a chain selector scores the top 16 candidates of each position conditioned on the previous token (`unary + sc[cand] @ (pc[pred] * hp)`) and takes the argmax. The path is deterministic. The drafter does not know the temperature, top-p or penalties.
- **Acceptance** (`Engine.draft_cycle`): greedy accepts a draft equal to the target argmax. Sampling accepts draft `d` with probability `p(d)` (the target sampling probability) and, at the first rejection, samples from `p` with `d` removed. This is exact speculative sampling for a point-mass proposal. Its acceptance per position is `p(d)`, so a target split 50/50 between two good tokens accepts even a perfect draft half the time, and the losses compound over 7 positions.
- **Verifier cost is independent of temperature.** Temperature only changes how many drafts are accepted. **Measured** on the default build: about 2 ms per extra verifier row at 8K (8-row call 91.5 ms, 64-row call 202.8 ms).

## Step 1: measure the baseline

Extend [`scripts/m6_server_bench.py`](../../scripts/m6_server_bench.py) with `--temperature`, `--top-p`, `--top-k`, `--presence` and `--seed`, and skip the reply-equality assertion for sampled runs (replies depend on the seed).

Matrix, default build, V8, thinking off and on:

| Axis | Values |
| --- | --- |
| temperature | 0 (control), 0.7, 0.9, 1.0 |
| presence_penalty | 0, 1.5 |
| context entry | 8K, 32K, 64K |
| seeds | 5 per cell, 3 cached repeats each |
| workloads | the synthetic coding fixture; a few public multi-turn agent-style prompts |

Record per request: tokens per cycle, the accepted-draft histogram (`draft_accept_histogram`), ms per cycle by phase (draft, verify, sample, ctx, host) and decode tokens/s. Report medians and ranges across seeds. **Measured** reference points: greedy fixture 6.4 to 7.3 tokens per cycle; the owner's coding-agent session 3.31 tokens per cycle (its sampling settings were not logged).

## Step 2: exact speculative sampling with the drafter distribution

The textbook rule (Leviathan et al. 2023; Chen et al. 2023): sample the draft from a proposal `q`, accept with `min(1, p(d) / q(d))`, and on rejection sample from `normalize(max(p - q, 0))`. The output distribution equals `p` for any `q`. Acceptance per position becomes `sum_x min(p(x), q(x))` instead of `p(d)`.

DFlash2 already has the right `q`: at each position the chain score `s` over 16 candidates, conditioned on the previous token, which after an acceptance is the accepted prefix.

1. **Drafter** (`propose(..., sample=True, temp_d, seen, presence)`): at step `i`, `s_i` as today; subtract `presence` from candidates already in the reply (`seen`) so `q` sees the same penalty as `p`; `q_i = softmax(s_i / temp_d)`, optionally trimmed by the target's top-p; sample `pred ~ q_i`; return the path and `(cand_i, q_i)` per step. Keep the argmax path when `sample=False`.
2. **Server** (`draft_cycle`): for position `k`, `p_k = self.dist(...)` as today; accept with probability `min(1, p_k(d) / q_k(d))`; on rejection sample from `max(p_k - q_k, 0)` over `p_k`'s support (tokens outside the 16 candidates have `q = 0`); with all 7 accepted, sample the bonus token from row 7 as today. Update `seen` and the DRY history with accepted drafts only (already done).
3. **Correctness conditions:** the draft must be sampled from exactly the `q` used in the test; `q_k` may depend only on the accepted prefix (true for the chain selector); `p_k` must include penalties computed on that same prefix (true today). Greedy stays deterministic and unchanged.
4. **Draft temperature:** start at `temp_d = temperature`; sweep 0.5x, 0.7x, 1.0x and 1.3x of it for the best acceptance. Any `temp_d` keeps the output exact.
5. **Edge cases:** `p` and `q` with disjoint supports (acceptance 0, residual = `p`); `p` concentrated on one token; `q` putting all mass on one candidate (reduces to today's rule); stop tokens accepted inside a draft (end the cycle, as today); context growth between cycles.

Put the acceptance math in a small pure function (for example `spec_accept(p_ids, p_probs, q_ids, q_probs, draft, rng) -> (accepted, token)`) so it can be tested without models.

## Step 3: validate exactness

- **Host unit tests** (`unittest`, NumPy only, no Core ML): toy `p` and `q` over a small vocabulary, about 1e6 trials, the empirical output distribution of `spec_accept` against direct sampling from `p`, total-variation distance below a fixed bound. Cover top-k / top-p truncation, presence penalty, the edge cases above and multi-position chains (7 steps with prefix-dependent `p` and `q`). Extend [`scripts/qwen38_spec_unit_test.py`](../../scripts/qwen38_spec_unit_test.py) or add `tests/test_spec_sampling.py`.
- **On device, sanity only:** plain sampling against speculative sampling on the same prompt, many seeds, at temperature 0.7 and 1.0; compare first-token and per-position token histograms (KL). A statistical check of the pipeline, not a proof.
- **Regression:** greedy outputs bit-identical to today's server for the fixture at 8K to 64K.

## Step 4: measure the gain

Rerun the Step 1 matrix with the new rule. Report tokens per cycle and tokens/s against the baseline for each temperature, the accepted-draft histogram and the sample-phase time (expected small: 16 candidates per position). Ship when: exactness tests pass, greedy is unchanged, and sampled decode tokens/s improves at temperature 0.7 and 1.0 on both workloads without a regression at any measured context.

## Later options

- **Multi-draft verification:** two sampled drafts per cycle in one 16-row call. Needs per-branch DeltaNet states and new target entries; about +2 ms per extra row at 8K. Consider only after Step 2.
- **Typical acceptance** (Medusa-style, lossy): opt-in fast mode only; it changes the output distribution.
- **Not useful:** shorter verifiers (measured in [../verifier_len.md](../verifier_len.md)).

## Presence penalty notes

`presence_penalty` subtracts a fixed amount from the logit of every token that already appears in the current reply, once per token regardless of count. In this server it is applied before temperature, so 1.5 scales a repeated token's relative probability by `exp(-1.5 / T)`: about 0.12x at T = 0.7 and 0.22x at T = 1.0. The prompt is not counted, and greedy decoding ignores it. The model card recommends 1.5 for non-thinking chat to curb endless repetition and warns of occasional language mixing and slightly lower quality. Code repeats identifiers, keywords, brackets and indentation constantly, so for coding traffic leave it at 0, or test small values with the Step 1 matrix. It also lowers acceptance today, because the drafter proposes the repeated tokens the penalty suppresses; Step 2 applies the same penalty to `q`.

Qwen3.8-27B is developed by the Qwen Team (Alibaba Cloud, Apache 2.0). DFlash2 drafter lineage: [ProCreations/Ternary-Bonsai-2-27B-DFlash2](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2). ANEMLL's work is independent conversion and ANE research.
