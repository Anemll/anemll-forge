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
