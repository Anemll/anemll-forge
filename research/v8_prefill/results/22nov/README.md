# 22 Nov 2026

Executing the ranked list in `NEXT.md`: items 1 and 2, the only ones projecting 2% or more. Neither needs a
served-default change or a push. Rules: banner + 2x HWX check on every timed ANE package; timed runs only while Pi is
idle; the served config restored and the smoke checksum matched after any forge run; local commits only.

## NEXT.md item 2: `TREE_POLICY=dink` vs `branched` vs the chain (23:00 to 01:09)

`serve_t/dink_ab.sh`, `serve_t/dink_compare.py`. Raw files: `dink/` (ab.log, compare.json, per-arm server and driver
logs), `bench/dk_*` (per-set bench json and logs).

- **Arms** in ABBA order (ch_a, br_a, dk_a, dk_b, br_b, ch_b), all sampled (`--think 1`), lookup off, build v8_dec8tx:
  - ch: `TREE_SAMPLED=0` (the sampled chain)
  - br: the served config (`policy=branched`)
  - dk: `TREE_POLICY=dink`
- **Sets:** agent (u1_code_agent, seeds 4100 and 4200, 384 tokens), code (seed 4100, the guardrail), and the long
  Pi-like session (19 turns, 8.6K to 55.5K, seeds 7100 / 7200 / 7300, 512 tokens).
- **Checks:** every banner shows the intended policy; Pi requests 0 and errors 0 in all six arms. Restored at 01:08
  (forge pid 15402, banner `policy=branched sampled=on lookup off`). Smoke checksum `29aef2ec9db18a3f` matches.

Client decode tok/s, pooled over both orders, with the change in order a / order b:

| stratum | ch | br | dk | br/ch | dk/ch | dk/br (a / b) | dk_a = dk_b | dk = br |
|---|---|---|---|---|---|---|---|---|
| agent | 46.31 | 45.23 | 47.38 | -2.3% | +2.3% | **+4.8%** (+5.5 / +4.0) | 3/18 | 1/18 |
| code | 32.83 | 36.71 | 37.10 | +11.8% | +13.0% | +1.1% (+1.3 / +0.8) | 8/9 | 3/9 |
| long 8-16K | 34.58 | 36.36 | 35.75 | +5.2% | +3.4% | -1.7% (-2.5 / -0.9) | 9/12 | 3/12 |
| long 16-32K | 33.51 | 34.62 | 33.33 | +3.3% | -0.5% | -3.7% (-3.9 / -3.6) | 15/18 | 1/18 |
| long 32-57K | 27.40 | 32.38 | 28.87 | +18.2% | +5.4% | **-10.8%** (-12.2 / -9.5) | 19/27 | 11/27 |
| long all | 31.22 | 34.35 | 32.33 | +10.0% | +3.6% | **-5.9%** (-6.5 / -5.3) | 43/57 | 15/57 |

- **Per seed, dk/br:**
  - agent: 4100 -0.9%, 4200 +10.2%. The agent gain comes from one seed.
  - code: 4100 +1.1%.
  - long: 7100 -12.5%, 7200 -4.0%, 7300 -1.9%.
- **Determinism:** ch and br are identical across orders (18/18, 9/9, 57/57). dink is not (3/18 on agent): its
  choice uses running means of measured verify ms, so the sampled stream depends on timing.
- **Why dink loses on the long session** (server tree lines, cycles by prompt band):

  | band | br: tree cycles (tok/verify, ms) | br: chain cycles | dk: tree cycles | dk: chain cycles |
  |---|---|---|---|---|
  | <8K | 1216 (3.27, 99.0) | 660 (6.76, 96.4) | 884-894 (3.2, 99.5) | 997-1030 (5.8-5.9, 95.9) |
  | 16-32K | 871 (3.25, 112.5) | 341 (6.64, 110.9) | 419-465 (3.1, 113.3) | 688-707 (4.5, 109.1) |
  | 32-57K | 652 (3.22, 128.0) | 288 (6.55, 126.5) | 495-537 (2.9, 128.5) | 527-530 (4.4-4.8, 124.4) |

  - Above 16K, dink sends about twice as many cycles to the chain, and those extra chain cycles accept far less than
    branched's (4.4-4.8 vs 6.6 tokens per verify).
  - The verify cost it saves is only about 4 ms of 125 (the chain and tree verify both run 8 rows).
  - Its running mean of chain acceptance comes mostly from the confident cycles that took the chain. That inflates
    E_chain, so it gives up the tree where the tree pays.
  - Below 8K, dink's cycle tok/s is +3.9% / +2.8% over branched (47.7 / 47.2 vs 45.9). The long Pi session has no
    decode below 8K.
- **Decision rule** (NEXT.md: dink brings agent within 2% of the chain and beats branched in both orders, while code
  and the long session stay within 2% of branched):
  - agent: passes (+2.3% vs the chain, +5.5 / +4.0 vs branched).
  - code: passes (+1.1%).
  - long session: **fails** (-5.9%, both orders; -10.8% at 32-57K).

STATUS 01:12 item 2 (`TREE_POLICY=dink`): FAILS the guardrail, not advanced. The long session is -5.9% vs branched in
both orders (-10.8% at 32-57K). The cause is the running-mean chain estimate, which is inflated by its confident cycles.
Agent +4.8% rests on one seed (+10.2 / -0.9). Branched stays. A below-8K-only dink projects 0 on the long session, so
it is not run.

## NEXT.md item 1, step A: the sliced-entry rung probe

Queued at 23:02 behind the A/B. That queue gave up at 01:03 after its 2 h idle wait (the A/B ran past 2 h). Requeued
at 01:04.
