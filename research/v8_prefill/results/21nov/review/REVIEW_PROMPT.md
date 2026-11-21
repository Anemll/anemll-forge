# Review brief: speculative decoding on the M6 (tree verify, served tree8, JetSpec++), 19-21 Nov 2026

You are reviewing two days of decode-speed research on Anemll's M6 Mac (Apple silicon: ANE + GPU). It serves
Qwen3.8-27B (dense, 3-bit VQ weights, Gated DeltaNet + attention hybrid) through "forge", an OpenAI-compatible
server. **Goal:** find where decode can get faster on this machine, check that what was built is correct, and propose
the next tests, ranked by expected gain per hour of work. Treat every conclusion below as a claim to check against
the raw data, not as fact; disagreeing with it is useful.

## Access
- `ssh m6.local` (user anemll). Python: `/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python`
  (torch, transformers, numpy are there; do not install packages).
- forge serves on 127.0.0.1:8765 and is used live by Anemll's Pi coding agent.

## Hard rules (read before running anything)
1. Read-only by default. Offline analysis of logs, JSON results and the replay simulators is fine at any time.
2. Do not stop, restart or send requests to forge on 8765, and do not run any ANE or GPU measurement, without
   Anemll's explicit OK for that window. Bench traffic interleaves with Pi, evicts Pi's prompt cache and skews both.
   In an approved window: launch only through the serving launcher, print the banner
   (`grep -E "drafter ring snapshot|v8: variant|dec8: BANNER|tree: BANNER" <server log>`), and restore the served
   default at the end.
3. Never reboot. Never touch other ports or services (8766, 8771, 8799, anything that is not forge).
4. One model at a time: forge holds about 19 GB wired. A replay, a second server or a build needs forge down first
   (a JetSpec++ replay beside forge got forge killed on 20 Nov). Arm the swap watchdog on long jobs:
   `/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh <pid> 1024 <log>`.
5. Builds: one ANE compile at a time, through the existing build scripts. Journal every Core AI load
   (`/Users/anemll/Documents/Codex/ios-prefill-20261030/deployment_journal`). Delete Core AI caches only with
   `anemll-forge/scripts/coreai_cache_delete.py`. Ask before deleting anything (the superseded builds v8_dec8t and
   v8_dec8tc and `jetspec_pp/calib_inputs` stay).
6. Do not edit served files in place (pipe/*.py, launchers, builds). Work on copies. Put new code and every raw result
   under `RR/results/22nov_review/` with an index README.
7. Do not modify `anemll-forge/scripts/qwen38_server.py` (Anemll's uncommitted edits). No git push; local commits only,
   on anemll-forge-v8 branch v8-gpu-ane-prefill. Never echo or store tokens.
8. Never invent timings: every number you state comes from a file you name or a run you log.

## Where things are
C = `/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work` (working sources).
RR = `/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill` (git copies + results).
- **Results** (number-first READMEs):
  - `RR/results/19nov/README.md`: sections 6, 13, 13b, 15 (tree drafting, single-call tree verify, dynamic verify size);
    7 (JetSpec++ port)
  - `RR/results/20nov/README.md`: Summary; A1-A8 (tree verify 8 / 16 / 32 rows); B, B7, B8, C1-C5, D1-D4, E0-E3
    (JetSpec++); T and U (why Splash's tree pays, prompt lookup, 16-row cost)
  - `RR/results/21nov/README.md` (tree8 served, fresh vs used server, ring fix, sysk crash), `RR/results/21nov/pi_decode/`
    (real Pi sessions), `RR/results/21nov/tree_sampled/` (tree8 for sampled requests)
- **Server:** `C/pipe/v8_server.py` (V8 engine: GPU + ANE prefill, decode on the ANE with GPU attention, prompt cache);
  `C/pipe/tree_verify.py` (tree8: build_tree, masks, call, compact, walk_sampled); the base
  `/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02/scripts/qwen38_server.py` (Engine.draft_cycle, dist,
  sampling) and `dflash2_coreai_drafter.py` in the same folder.
- **Launchers:** `C/pipe/v8_server_dec8_qk.sh` (served; rollbacks `*.pre_*.sh`) -> `v8_server_tree.sh` -> v8_server.py.
- **Builds:** `/Volumes/SSD4TB/anemll-forge-research/v8_dec8tx` (served: chain + tree8 entries, corrx0 conv), `v8_dec8`
  (chain). Build scripts `C/v8/build_dec8t_chunk.py`, `build_tree16_probe.py`, `build_u2_decomp.py`.
- **Offline simulators:** `C/pipe/unit/power/tree_dyn2.py` (tree arms on fixed texts), `t3_lookup_sim.py`,
  `t4_replay_sim.py`. Use the trajectory replay: per-cycle upper bounds overcounted copied spans by 2-3x.
- **Benches:** `C/bench/tree/` (tree_bench.py with banner checks and `--temp/--think/--seed`, run_arm.sh,
  realistic_v1.json, u1_code_agent.json, make_prompts*.py).
- **JetSpec++:** `C/jetspec/` (jsp_replay.py, quant_adapter_gptq.py, gpu_lut4_pass.py); release
  `/Volumes/SSD4TB/anemll-forge-research/jetspec_pp/`.
- **Live log:** `/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log`, rotated per restart as
  `v8_server.<mtime>.log`. Each request ends with "prompt N (cached ..) | gen G in T s (X tok/s, Y tok/verify, accept
  ..%, ms/cycle M (draft d verify v sample s ctx c host h))". Sampled requests (temperature > 0) are Pi's.

## State and the numbers that matter
- **Cycle budget:** DFlash2 drafter (7 drafts, ANE) about 13 ms + one 8-row target verify (ANE FFN and DeltaNet, GPU
  attention with a clock keep-alive): 94-97 ms at 2-8K, about 106 at 16-32K, about 121 at 32-64K, about 136 at 57K.
  Host work under 1 ms. Verify is about 85% of a cycle; tok/s = 1000 x tok/verify / ms per cycle.
- **Served config (21 Nov):** tree8 (a dynamic top-7 tree in the 8 verify rows, used only when it differs from the
  chain), prompt lookup off, DRAFT_RING_SNAP=fix, TREE_SAMPLED=1 (tree8 for sampled requests, from 21 Nov 10:32).
- **Greedy tree8 vs chain, end to end** (20 Nov U1): +4.6 / +6.5 / +4.9 / +1.9% tok/s at 2K / 8K / 32K / 64K. On
  identical texts the gain is +0.5-3% (chat +8%). Prompt lookup (NGRAM_DRAFT) gives +101% on full-file code edits and
  is neutral elsewhere.
- **16-row verify:** an ANE tree16 costs +12.6% / +20% per group at 8K / 32K (DeltaNet UT solve, tree conv taps,
  attention score passes); a 16-row chain costs x1.085 at 8K. Splash 1.7.0's tree gains +17-22% tokens per cycle from
  16-row width at about 0 GPU cost.
- **JetSpec++** (exact replay of our model's streams): x1.4-1.8 tokens per round vs DFlash2, but as a serial ANE
  drafter it loses 8-45%. Its adapter pass is 22 ms at LUT4, memory-bound at 110 GB/s, with several passes per round.
  Parked until there are two execution units.
- **Real Pi use** (sampled at temperature 1.0, thinking on, 6K-57K context): the 21 Nov session ran at 33.5 tok/s and
  4.10 tok/verify. Until 21 Nov tree8 skipped every sampled cycle (tree_verify.py: sampled -> chain).
- **Sampled tree8 A/B** (same prompts and seeds, fresh launches): +8.5% tok/s overall (27 of 36 paired requests faster): 8K +9.0%, 32K +11.6%, code +10.9%,
  agent -1.6% (noise level; it already accepts 5.2 per verify). tok/verify 3.91 -> 4.35, ms per cycle +2.4%.
  The tree runs in 52-82% of sampled cycles (`RR/results/21nov/tree_sampled/`).
- **Power:** about 22-25 W total during decode (CPU + GPU about 2 W): 0.48-0.62 J per generated token greedy, 0.73-0.88
  Pi-like.
- **Known open items:**
  - (a) Stale KV rows past the position change results (used server 1-2/9 identical to fresh; 9/9 when a cold start
    resets them, `COLD_SCRUB=kv`; results/21nov/leftover/). The masking is not exact somewhere; the kernel is not
    localized, and warm restores to a shorter snapshot are not covered.
  - (b) The release server (anemll-forge-v02 / anemll-forge `qwen38_server.py`) can restore a 'system' snapshot whose
    target KV rows another prompt overwrote. It has no `_prune` / 'sysk'.
  - (c) Sampled tree8 reuses the greedy-calibrated tree (tau 1.5, top-4 children per node).

## What to deliver
`RR/results/22nov_review/REVIEW.md`, number-first, in plain language:
1. **Correctness review** of tree_verify.py (the greedy walk and the new walk_sampled) and of the ring-snapshot and
   'sysk' fixes in v8_server.py: concrete bugs or risks, each with file:line and a failing scenario.
2. **Ranked speedup opportunities** on this M6, decode first, prefill second. For each: the mechanism, the expected
   tok/s gain derived from measured numbers (show the arithmetic and cite the file), the cost to try, and what result
   would rule it out. Include ideas we have not tried.
3. **A test plan:** exact commands; which tests need forge down or a bench window, and for how long; a go / no-go gate
   for each.
4. **What not to pursue,** with the measured reason.
5. **Where our methods or conclusions look wrong:** about 4% run-to-run tok/s noise, non-identical texts between arms,
   per-cycle vs trajectory estimates, sampled vs greedy.

Run offline checks as needed. Ask Anemll before anything under rules 2 or 4.
