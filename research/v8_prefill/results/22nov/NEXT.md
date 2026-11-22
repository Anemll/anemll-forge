# Open ideas, ranked (written 21 Nov 2026, 22:55)

Sources: `results/19nov`, `20nov`, `21nov` READMEs, `21nov/tasks.md`, `22nov_review/lookup_adaptation/RESULTS.md`, and
tonight's served logs. Gains are projections from measured numbers; none is measured end to end yet. "Decode" means
decode tok/s on the long Pi-like agent session (19 turns, 8.6K to 55.5K): the off arms of the 18:59 think-gate A/B,
greedy plus seeds 7100 / 7200 / 7300, both orders. Mean cycle 131 ms.

## Ranked

| # | idea | expected gain | cost | risk | needs a served-default change to test? |
|---|---|---|---|---|---|
| 1 | Finer decode context entries for the tree8 verify, by slicing the last bound KV segment inside the entry | +1.9% decode (+24K, +48K entries), +3.0% (+24K / 40K / 48K / 56K), about +4% (a 4K ladder) | Step A: a one-group scratch probe, about 2 h. Step B: a full temporary build plus an A/B, compile hours and disk. Step C: deploy. | The ANE may still read the whole bound segment (then no gain). More segments instead would break the ≥64K entries at the ~51-binding limit. Compile time and disk. | Steps A and B: no. Step C: yes (Anemll). |
| 2 | Cost-aware chain fallback `TREE_POLICY=dink` vs `branched` on sampled tool-agent requests | 0 to +3.9% on sampled tool-agent requests | Env flag only; about 2 h of forge time while Pi is idle | Low. The deficit it targets may be noise. | No (research flag) |
| 3 | Decode GPU attention Q.K above 64K: K-tile software pipelining; one threadgroup per KV head across all 48 query rows | Ceiling 1.6% (64K) / 2.3% (96K) of decode, only above 64K. The long Pi session never gets there (max 56K). | Kernel work, about half a day | Low | No |
| 4 | fp16 GDN glue intermediates in prefill | Ceiling 1.5% / 2.2% of the prefill wall at 95K / 32K; about 2.7% at 8K if the GPU were the bottleneck (it is co-bound). Prefill is about 20-25% of a Pi turn, so ≤0.6% per turn. | Kernel work, about half a day | Medium. The same table's chunked-recurrence row was 4.7x optimistic (21 Nov item 4). | No |
| 5 | Lookup routed by request type: only explicit full-file rewrites | +18 to +27% on full-file rewrite turns (sampled code cohort); 0 on the long Pi session | A request classifier, held-out cohorts, an A/B | The share of rewrite turns in real Pi traffic is unmeasured | Yes, to serve |
| 6 | 8-row selector refit on sampled traces | Greedy projections −0.50 / −0.08 / −0.18%; sampled untested; expected <1% | Offline | Low | No |
| 7 | Merge the drafter's ctx64 call into the body call | About 0.1% (0.25 ms x 0.5 calls per propose) | A drafter rebuild | Low | Yes |

**Above 2%:** only items 1 and 2, and neither is measured yet. Items 3-7 project below 2% for Pi traffic.

**Status (22 Nov):**
- Item 2 measured 01:09: fails. Long session -5.9% vs branched in both orders; see `README.md`. Closed.
- Item 1 step A measured 01:14: pays. 48K-sliced costs 0.50 of the 32K to 64K step (linear 0.50), bit-exact; projected served verify 117.9 ms vs a 117.7 ms bar. Step B waits for Anemll.
- Item 1 step B measured 08:49: clears the keep rule. Long session pooled +1.92% (order a +1.85%, order b +1.99%), 152/152 replies identical, parity gate KL 0. Matches the +1.9% projection. Build v8_dec8ts kept; deploy is Anemll's call. 4-entry set (projected +3.0%) not timed yet.

### Details

**1. Finer decode entries.**
- **Measured tonight** (served tree8 verify by context entry, requests whose decode stays within one entry): 16K 101.3 ms
  (n=30), 32K 109.8 ms (n=46), 64K 125.5 ms (n=48). That is 0.53 ms per 1K of entry width from 16K to 32K and
  0.49 ms per 1K from 32K to 64K. The ANE's masked attention reads the whole entry width, so a position at 33K pays
  for 64K.
- **Projection** (per-position entry choice over the long session's decode, verify cost linear in width between the
  measured entries, 8K extrapolated):
  - +48K: 1.2%
  - +24K and +48K: 1.9%
  - +24K / 40K / 48K / 56K: 3.0%
  - +12 / 20 / 24 / 28 / 40 / 48 / 56K: 3.6%
  - every 4K: 4.0%
- **The blocker:** a finer KV segment ladder breaks the ≥64K entries at the binding limit (12 Nov: uniform 8K segments
  broke every entry from 64K up; about 51 bound tensors is the most that runs). Intermediate entries therefore have to
  bind the existing segments and slice the last one inside the program.
- **Step A** (no forge touch, Pi idle, banner + 2x HWX): a one-group probe (the `build_tree_probe.py` pattern, layers
  0-3) with entries t8 at 32K, 48K-sliced and 64K. Time each on the ANE. It pays only if 48K-sliced costs about the
  midpoint of 32K and 64K (about 117.7 ms per full verify).
- **Step B** (only if step A pays): a temporary full build and an ABBA on the long session.
- **Step C:** deploying is Anemll's call.

**2. `dink`.**
- **The deficit it targets** (`21nov/tasks.md`): sampled tree8 vs the chain on tool-agent was −1.6% in the first block
  and −3.88% pooled / −3.37% paired in the reversed block, 9 requests each. All hashes changed, so this is within about
  2 standard errors of 0.
- **The rule:** `dink` takes the tree only when the expected extra tokens pay for the tree's extra verify time, using
  running means per entry.
- **Test:** chain / branched / dink ABBA; sampled agent set (2 seeds), sampled code set (the guardrail), and the long
  session (3 seeds). Lookup off.
- **Advance** (the reviewer's rule) if dink brings agent within 2% of the chain and beats branched in both orders,
  while code and the long session stay within 2% of branched.

**3.** From 19 Nov section 16c. At 156 GB/s the Q.K pass would save about 0.13 ms per layer: 2.1 ms per verify step at
64K, 3.2 ms at 96K.

**4.** From 19 Nov section 12's reference rows, not FP8. It would be worth a microbench first, as item 4 of 21 Nov
was.

**5.** From the reviewer's holdout and next experiment. The first step would be measuring how often real Pi turns are
full-file rewrites. That needs Pi request logs, which are Anemll's to share.

## Not ranked: open decisions and upkeep

- **`COLD_SCRUB=kv`:** warm-rewind correctness, 1/9 fresh-equivalent replies before and 9/9 with the scrub. Its latency
  cost is unmeasured. A served change.
- **qwen38_server staleness for a release.**
- **HF V8-96K repo:** V8.1 / V8.2 need only a code and card refresh.
- **192K release repo:** `release.json` and the start script are missing.
- **Deleting old builds:** frees disk; Anemll's call.
- **ANE-only release server rungs** (parked by Anemll 15 Nov): 2K / 4K plus intermediate long rungs, Pi first prompt
  about +4.5%. This is the ANE-only product, not V8.

## Closed, not listed

FP8 / FP8 x FP8 everywhere tried; WL3A8; QKE=tg (deployed); tree widths 9-16 and 16 / 32 rows; dynamic per-cycle
verify size; fork-aware ranking (greedy); prompt-lookup gates (yield, ctx, think) for serving Pi; JetSpec++ (parked);
drafter on the GPU, hybrid split, overlap; 256-row GDN calls and hidden-decode LUT GEMM; short-context attention
placement; chunked fp16 GDN recurrence; GDN FP8; the ANE ↔ GPU handoff probe (stalled the ANE; only with forge down).
