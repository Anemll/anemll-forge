# 21 Nov 2026: tree8 is the served default

Decision (Anemll, 21 Nov 02:30): serve tree8 (v8_dec8tx, corrx0, TREE=1, policy branched). Prompt lookup stays off
(NGRAM_DRAFT=0). No new package builds (no 16-row entry, no GPU attention split).

## Result

- Forge on 127.0.0.1:8765 serves tree8 from the standard serving launcher `pipe/v8_server_dec8_qk.sh`, using the
  launcher defaults only (no tree or lookup env). pid 55758 since 02:43:29.
- Decode tok/s, realistic 2K rung: 45.80 vs chain 42.74. **Correction (03:30):** that run was on a used server, and
  its replies differ from a fresh start (see "Fresh vs used server" below). With the ring fix, a used server gives
  the fresh replies at **44.91** (U1 tree8, fresh: 44.70). Like for like against the fresh chain that is **+4.6 to
  +5.1%**, not +7.2%.
- Code set: **40.66** vs chain 39.46 (+3.0%). Agent set: **54.04** vs chain 54.48 (-0.8%).
- Smoke (realistic 2K, positions 1 and 8): a fresh launch reproduces itself and the U1 tree8 arm exactly (checksum
  29aef2ec9db18a3f). The second run on the same server also reproduces exactly (f7f2e615cb9623bf). The chain
  behaves the same way: a second run on one server changes the chat reply. Tree8 did not introduce this.

## Launcher

`pipe/v8_server_dec8_qk.sh` (backup of the 19 Nov chain version: `v8_server_dec8_qk.pre_tree8.sh`):

- Sets defaults `PORT=8765`, `V8_SRV_DYLIB=libgpu_v8_f8q.dylib`, `QKE=tg` and `NGRAM_DRAFT=0`.
- Then sets `BUILD=v8_dec8tx`, `GM=w_dec8tx/manifest_best.json` and `TREE=1`, and execs `v8_server_tree.sh`.
  That script loads libdec_gpu_tree.dylib and dec_kernels_tree.metal, with F8=x, the INT8 KV flags and the
  keep-alive as in v8_server_dec8.sh.
- Rollback to the 19-20 Nov served chain: `SERVED_TREE=0 v8_server_dec8_qk.sh`. This execs v8_server_dec8.sh on
  v8_dec8 with libdec_gpu_clk.dylib, and was used for det_chain_fresh1 below.
- `TREE=0` keeps this build and its binaries with the chain only.
- `bench_tree/run_arm.sh` now takes `RDIR`, the results directory (default results/20nov).

## Banner (server log of the running instance, pid 55758)

```
[qwen38-27b-ane] started 2026-11-21 02:43:29  pid 55758  listen 127.0.0.1:8765
[qwen38-27b-ane] model dir  /Volumes/SSD4TB/anemll-forge-research/v8_dec8tx  (max ctx 98304; verify-8 contexts [8192, 16384, 32768, 65536, 98304], prefill-64 contexts [same], KV kv8, one weight copy)
[qwen38-27b-ane] speculative   ON  drafter .../drafter/all3/dflash2_lut4_gptq_nohead.aimodel   (no prompt-lookup line: lookup off)
v8: variant F8=x F8VS=1.0 FOLD=off KV8=true K8V16=true QK8=true TK=2048 Lmax=98304 T=128 | prep=attn_prep8qv_f8v qk=attn_qk8isk_tg softmax=attn_softmax8n_f8x pv=attn_pv_f8v fill=kv8_v_to_f8v | V copy=e4m3 | dylib engine/libgpu_v8_f8q.dylib sha256 ea591258bceafe6d
dec8: BANNER decode GPU dylib engine/libdec_gpu_tree.dylib sha256 dfe059c68dce376a362ec502230dd0c2f597f72ce3a0f78b7800d54621629598 | kernels dec_kernels_tree.metal sha256 674918fb1e682a2c... | attn=dec_attn_r4 | V8_DEC8=auto
dec8: keep-alive hold 60 ms (DEC_KA_HOLD_MS), switch /Volumes/SSD4TB/anemll-forge-research/contention/ctl/ka
tree: BANNER mode=tree8 policy=branched topc=4 tau=1.5 t8 contexts [8, 16, 32, 64]K | split tree on (tdva/tdvba + dec_set_tree) | build /Volumes/SSD4TB/anemll-forge-research/v8_dec8tx | lookup off
serving OpenAI API on http://127.0.0.1:8765/v1 (V8 prefill, speculative ON, ctx 98304)
```

Every bench below checked these banner parts before timing (tree_bench `--expect`): qk=attn_qk8isk_tg, F8=x,
KV8=true, libgpu_v8_f8q.dylib, libdec_gpu_tree.dylib, mode=tree8, policy=branched, v8_dec8tx, lookup off. The chain
arm checked libdec_gpu_clk.dylib and mode=chain instead. The keep-alive hold of 60 ms was checked by grep.

## Smoke determinism (realistic_v1 2K, positions 1 = code:1 and 8 = chat:2, greedy, gen 320)

| run | server state | code:1 | chat:2 | checksum |
|---|---|---|---|---|
| U1 tree8 (20 Nov, 2K rung of 9) | fresh launch, 9-prompt rung | 1681b92d273f64ad | a773ceabaa6312d7 | |
| served_tree8_smoke1 | fresh launch | 1681b92d273f64ad | a773ceabaa6312d7 | 29aef2ec9db18a3f |
| served_tree8_smoke2 | 2nd run, same server | 1681b92d273f64ad | 13dd26d24dd1cf65 | f7f2e615cb9623bf |
| served_tree8_smoke2b | 3rd run (after 2K + code/agent sets) | 1681b92d273f64ad | 6f98638c8ba91bc7 | acbaaf3f8b265563 |
| served_tree8_smoke3 | fresh launch | 1681b92d273f64ad | a773ceabaa6312d7 | 29aef2ec9db18a3f |
| served_tree8_smoke3b | 2nd run, same server | 1681b92d273f64ad | 13dd26d24dd1cf65 | f7f2e615cb9623bf |
| chain final_served3-7 (20 Nov) | fresh launch | 1681b92d273f64ad | 3b060108f4af10b0 | |
| det_chain_fresh1 | fresh launch (SERVED_TREE=0) | 1681b92d273f64ad | 3b060108f4af10b0 | c4c4be331d40e7f1 |
| det_chain_same2 | 2nd run, same server | 1681b92d273f64ad | 764b46dd5fc2b4ae | fce3854ba8d742dd |

- A given request sequence on a given server gives the same text. This holds across launches (smoke1 = smoke3 =
  U1) and for the second run on a server (smoke2 = smoke3b).
- What a server ran before changes the low-acceptance chat reply (2.2-2.9 tok/verify). The chain has the same
  property (det_chain_same2), so tree8 did not add it.
- The replies split at near-tie greedy tokens. Chat:2 is identical for its first 35 tokens, then continues "been
  the anchor of" in smoke1 and "watched the seasons turn" in smoke2.
- The source of the history dependence is not found. The tree code keeps no cross-request numeric state: plans are
  keyed by (entry, parity), and anc / ctap are rewritten on every call. The prefill split alone does not explain it:
  U1's chat:2 restored 11 cached tokens and still matched smoke1, which had 0 cached. Open item.
- The same history effect is visible at the rung level. served_tree8_2k ran after two smoke runs and matches U1
  tree8 (a fresh launch) on 4 of 9 replies. The code set matches 9/9 and agent 8/9.
- serve_tree8.log's "vs u1_tree8" column keyed rows by index alone, which repeats across categories, so it reported
  code:1 as DIFFER. Keyed by (cat, index), smoke1 = U1 on both. The driver is fixed (ops.log 02:45).

## Decode check (tok/s, token-weighted client times; served_tree8_2k, served_tree8_ca)

| set | chain (U1, 20 Nov) | tree8 (U1, 20 Nov) | tree8 served (21 Nov) | vs chain | tok/verify served |
|---|---|---|---|---|---|
| realistic 2K | 42.74 | 44.70 | 45.80 (used server); **44.91** with the ring fix (= the fresh replies) | +7.2%; **+5.1%** like for like | 5.007; 4.862 |
| code (2.1-4.6K prompts) | 39.46 | 41.01 | **40.66** | +3.0% | 4.491 |
| agent (3.2-4.5K prompts) | 54.48 | 54.71 | **54.04** | -0.8% | 5.562 |

The 54.5 / 39.5 chain baselines in the request are read as U1's agent (54.48) and code (39.46) sets. The realistic
2K chain is 42.74.

## Files

- Drivers: `serve_t/serve_tree8.sh` (restart, banner, smoke x2, 2K + code / agent) and `serve_t/det_tree8.sh`
  (third same-server run, chain fresh + same, tree8 fresh + same; leaves tree8 serving).
- Raw output: `serve_tree8.log`, `det_tree8.log`, `bench/*.json|log`, `ops.log`, `*.driver.log`. Server logs are
  in coreai_seg_logs/v8_server.<timestamp>.log; the live one is v8_server.log.

## Fresh vs used server: root cause (Anemll, 03:17; ring/, bench/ring_*)

**Cause.** The drafter's context ring is restored by slot position only.
- The 'system' snapshot, and 'sysk', its copy with KV rows, store the drafter as `slot_pos` plus the pending
  queue. They do not store the ring's K / V.
- `_restore` keeps every slot whose position still equals the snapshot's (qwen38_server `Engine._restore`:
  `np.where(d.slot_pos == slot_pos, slot_pos, -1)`).
- Another prompt can rewrite those slots at the same positions. On a used server that is tree_bench's "Say OK."
  warm-up, a cold request with no system prompt: `drafter.reset()`, then its rows 0-6 go into slots 0-6.
- A later restore of 'system' or 'sysk' then keeps those slots as valid. The drafter drafts the next request with
  the warm-up's rows at positions 1-2 instead of the system prompt's.
- Different drafts give a different verify split. The target's greedy text depends on the split at near ties
  (D4), so the reply changes.
- The target model state itself is restored correctly: the same prefill calls and the same DeltaNet snapshot.
  Seed, sampling, presence penalty and keep-alive play no part: greedy requests skip the sampler (`dist` returns
  None at temperature 0).

**Is it a bug?** Yes, in the prompt cache's drafter restore. It is not in the model math and not in tree8.
- The chain and tree8 share it.
- On a fresh server, or when the previous prompts had the same system prompt (U1's rung order), the kept slots hold
  the same tokens, so it is invisible.
- Real-world trigger: any request that does not start with the 'system' snapshot's tokens, followed by one that
  restores 'system' / 'sysk' (Pi's summary request, another conversation, a title request).
- With a long system prompt, up to the last 2048 rows before the cut (the drafter window) would be another prompt's.
  The effect on acceptance in that case was not measured.

**Evidence** (each arm is a fresh launch of the served launcher with tree8 unless noted; smoke = realistic 2K
positions 1 and 8; the usual banner was checked, and the server log shows `drafter ring snapshot: <mode>`):

| arm | DRAFT_RING_SNAP | run 1 (fresh) | run 2 (same server) | 2K rung on the used server vs U1 tree8 (fresh) | run 3 |
|---|---|---|---|---|---|
| this morning (served, before the switch existed) | | 29aef2ec9db18a3f | f7f2e615cb9623bf | 4/9 identical (068b970f1430c571) | |
| ring_log (control) | log | 29aef2ec9db18a3f | f7f2e615cb9623bf | 4/9 (068b970f1430c571, = this morning) | |
| ring_fix | fix | 29aef2ec9db18a3f | **29aef2ec9db18a3f** | **9/9** (576e79b18042345a; tok/verify per request = U1's) | **29aef2ec9db18a3f** |
| chain, det (no switch) | | c4c4be331d40e7f1 | fce3854ba8d742dd | | |
| chain, ring_fixc | fix | c4c4be331d40e7f1 | **c4c4be331d40e7f1** | | |
| ring_served_final | off (served) | 29aef2ec9db18a3f | | | |

- The log arm's restore lines, on a used server after the warm-up, read `snapshot slots 3, kept by position 3 (2 hold
  another prompt's rows), hidden 0`. When the previous request shared the system prompt they read `0 hold another
  prompt's rows`.
- The 11-token system prompt flushes 3 rows to the ring before the cut; the other 8 stay in the pending queue, which
  the snapshot does keep. Slot 0 (`<|im_start|>`) is the same token in both prompts, so only positions 1-2 differ.
- Off and log produce identical output, so the control reproduces this morning's sequence exactly. The history
  effect is deterministic.

**Residual, not the ring.** A long prompt on a used server restores 11 cached tokens. Its V8 prefill then starts at
an aligned point instead of position 0, a different target prefill slicing.
- The chain's code:1 (279 tokens) shows it: tok/verify 7.62 fresh and 7.80 used, with or without the fix. The text
  is identical in every run.
- This comes with caching by design: the cached prefix was computed in one slicing.

**Smallest fix** (prototype in `pipe/v8_server.py` behind `DRAFT_RING_SNAP`, default **off** = the served behavior,
checked: ring_served_final = fresh):
- `fix` copies the ring K / V of the slots the 'system' snapshot can see into the snapshot (`draft_ring`), and
  writes them back on restore. 'sysk' carries the same arrays.
- The code is about 30 lines (`_ring_save`, `_ring_restore`).
- Memory: 20 KB per visible slot (5 layers x K, V x 8 x 128 fp16), at most 40 MB at 2048 slots. Here: 3 slots, 60 KB.
- Only 'system' needs it: `_prune` drops every other snapshot once another prompt rewrites its KV rows.
- It changes served replies on a used server, back to the fresh-server ones, so it is not applied. To serve it:
  `DRAFT_RING_SNAP=fix` in the serving launcher (Anemll's call).
- The release server (anemll-forge `scripts/qwen38_server.py`, `Engine._snapshot` / `_restore`) has the same
  position-only restore. It is not changed here: that file has uncommitted edits.

Files: `serve_t/ring_snap.sh` (the driver), `ring/ring.log` (shas, banners, every restore line), `ring/*.driver.log`,
`bench/ring_*`, `pipe/v8_server.pre_ringsnap.py` (before the switch).

## 07:27 to 07:42: ring fix served, release port, a 'sysk' crash fixed

**Served** (Anemll 07:27): `pipe/v8_server_dec8_qk.sh` exports `DRAFT_RING_SNAP=${DRAFT_RING_SNAP:-fix}` (backup
`v8_server_dec8_qk.pre_ringfix.sh`). Banner: `drafter ring snapshot: fix`, tree8 branched, lookup off, libgpu_v8_f8q,
libdec_gpu_tree dfe059c68dce376a, keep-alive 60 ms, ctx 98304. Forge pid 85554 since 07:40:54.

**Checks, incomplete.** The user's Pi session used the server from 07:31:24, so nothing after the restart was fresh.
- smoke1 (07:32-07:33): code:1 1681b92d = fresh; chat:2 e4b2451b, fresh a773ceab, although both were cold starts
  (tok/verify 2.69 vs 2.67). Checksum 43f66f649f26723e, not 29aef2ec9db18a3f. Something besides the ring carries
  over after a long sampled Pi request (5953 prompt tokens, about 7000 KV rows). Not explained yet.
- The 9-prompt used-server rung was stopped after 3 requests (it triggered the crash below in the user's session).

**Crash** (StopIteration at `v8_server.py:540`, 3 times: 07:33:07 and 07:35:32 Pi, 07:36:16 bench). Two
conversations with different system prompts alternating: the request picks `sysk` (its own system prompt with KV),
then `_save_sysk` replaces `sysk` with the other conversation's `system` before the key lookup, so the lookup finds no
key. Code from 9c18f06 (16 Nov), not the ring switch. Pi retried each time with a cold prefill (9 s / 18 s).
- Fix: the key is looked up before `_save_sysk`, which is skipped when `sysk` itself is restored (restoring it
  rewrites those rows and makes it `system`; the other conversation goes cold next time, as with one `sysk`).
  Backup `pipe/v8_server.pre_sysk_fix.py`. Replies unchanged where nothing crashed.
- Repro after the restart (`sysk_fix_repro.log`): X (system A), Y (system B), X again: the third request restored
  `sysk`, 11 tokens reused, no exception.

**Release port** (anemll-forge `scripts/qwen38_server.py`, not committed, not staged): +16/-1 lines on top of
Anemll's uncommitted edits, `DRAFT_RING_SNAP` default `fix` there (`off`: before). Backup
`scripts/qwen38_server.py.bak-20261121`; his edits `scripts/qwen38_server.py.anemll-edits-20261121.patch`; the ring
change `scripts/qwen38_server.py.ringsnap-20261121.patch` (copies in `release_port/`). Unit check
`serve_t/release_ring_unit.py` (`release_port/unit.log`): fix restores the system prompt's rows after a cold
request and after a ring wrap; off keeps the other prompt's rows. The release server has no `_prune` / `sysk`: a
restored `system` snapshot can still pair with another prompt's target KV rows there (not changed).

Files: `serve_t/ringfix_serve.sh`, `ringfix_serve.log`, `ringfix_server_0731.log` (server log 07:31-07:40, prompt
excerpts removed), `bench/served_ringfix_*`, `sysk_fix_repro.log`.

## 10:22 to 10:42: tree8 for sampled requests (Pi), served (`tree_sampled/`)

Pi samples at temperature 1.0, and until now `tree_verify.py` sent every sampled cycle to the chain, so Pi never
used tree8 (`pi_decode/`: per-cycle cost equal to the 19 Nov chain, all differences were content).

**Change** (`pipe/tree_verify.py`, backup `tree_verify.pre_sampled.py`; switch `TREE_SAMPLED`, launcher default 1,
backup `v8_server_dec8_qk.pre_treesampled.sh`):
- `walk_sampled`: per node the served chain's rule child by child. Accept child c with probability r(c), where r is
  the row's sampling distribution with the rejected siblings removed and renormalized; all rejected: the next token
  from r. Every token keeps its plain-sampling probability. A one-child node makes exactly the chain's draws.
- `_chain` (chain-shaped cycles) runs the release chain's sampled accept line for line.
- Unit checks (`serve_t/tree_sampled_unit.py`, `tree_sampled/unit.log`), all pass: (A) lossless on a synthetic
  tree with a child outside the support and a stop-token child (66 sequences, chi2 z +1.11, 400k walks); (B) a
  chain-shaped walk = the release loop with the release `dist()` (2000/2000 identical incl. RNG state, 396 cases
  with all 7 accepted); (C) `Tree._chain` sampled = the release `draft_cycle` (2000/2000).

**A/B** (`serve_t/tree_sampled_ab.sh`; two fresh launches, the same prompts and seeds; thinking on, temperature 1.0,
the server's thinking defaults, 384 tokens; Pi idle by agreement, checked in the server logs):

| set | chain_s tok/s, tok/verify, ms/cycle | tree_s tok/s, tok/verify, ms/cycle | tree cycles | tok/s | paired (geo mean) | faster |
|---|---|---|---|---|---|---|
| 8K | 33.87, 3.75, 111.7 | 36.91, 4.18, 114.4 | 70% | +9.0% | +7.5% | 7/9 |
| 32K | 29.06, 3.60, 125.1 | 32.44, 4.06, 126.6 | 74% | +11.6% | +8.6% | 8/9 |
| code | 32.73, 3.60, 110.6 | 36.32, 4.13, 114.9 | 82% | +10.9% | +10.9% | 8/9 |
| agent | 48.09, 5.24, 112.1 | 47.34, 5.27, 114.2 | 52% | -1.6% | -2.4% | 4/9 |
| all | 34.52, 3.91, 114.8 | 37.47, 4.35, 117.6 | 72% | **+8.5%** | +6.0% | 27/36 |

- Sampled gains are larger than greedy ones (+1.9-6.5%): at temperature 1.0 the target's mass is spread, and
  siblings catch more of it. Agent edits already accept 5.2 per verify; the tree adds nothing there.
- Greedy is unchanged: both arms' fresh smoke = 29aef2ec9db18a3f (the clean fresh check, ring fix on).
- The comparison first keyed requests by (rung, cat, index), which repeats inside a rung (humaneval:0 and mbpp:0);
  it now keys by set too.
- Served: forge pid 61702 since 10:32 (banner `sampled=on`, ring fix, tree8, lookup off).
- Not done yet: the 9-prompt used-vs-fresh greedy check of 07:27 (stopped twice by live Pi traffic).

## 10:48 to 10:56: used vs fresh (9 prompts) and the chat:2 leftover: stale KV rows (`leftover/`)

Anemll 10:45, Pi idle. Driver `serve_t/leftover.sh`, log `leftover/leftover.log`. Realistic 2K rung, all 9 prompts,
greedy, served config (tree8, TREE_SAMPLED=1, lookup off, DRAFT_RING_SNAP=fix).

| run | state before the rung | identical to the fresh rung |
|---|---|---|
| lo_fresh_2k | fresh launch | (reference; = u1_tree8 of 20 Nov: 9/9) |
| lo_used_ab | the server after the sampled A/B (8K / 32K / code / agent) | **2/9** |
| lo_used_ctl | fresh launch + its rung + one long sampled request (8k task 0, seed 777, 1500 tokens, past 8K) | **1/9** |
| lo_scrub_fresh_2k | fresh launch with COLD_SCRUB=kv | 9/9 |
| lo_scrub_used_2k | the same long request (identical output, a6a5c4bb) on the COLD_SCRUB=kv server | **9/9** |

**Cause:** KV cache rows past the current position. A cold start resets the DeltaNet states and the drafter, not the
KV rows; they are "masked by position". A fresh process has them at 0 with the unused scale (1.0); after a longer
request they hold that request's rows, and they change a later request's numbers enough to flip near-ties. Putting
every KV row back in its fresh state on a cold start (`COLD_SCRUB=kv`, `pipe/v8_server.py` `_cold_scrub`: 258 MB at
the 8K entry, 2-8 ms) makes a used server give the fresh replies 9/9. So the masking of rows past the position is not
exact somewhere; which kernel leaks (bridge prefill, chain verify, tree entry, GPU split) is not localized yet.
- The ring fix (07:40) holds, but it only covered histories of short prompts; after long requests the KV leftover
  dominates (2/9).
- The same leak can touch warm requests: after a restore to a shorter snapshot, the rows past it hold the abandoned
  continuation. The cold-start scrub does not cover that.
- Served stays as it was (no COLD_SCRUB): the fix changes served replies on used servers (Anemll's call). Smallest
  fix: `COLD_SCRUB=kv` in the launcher (cold starts). Fuller fix: also reset rows [n, high-water mark) after a
  restore. Better still: find the kernel whose mask leaks (a one-layer probe, zero vs stale rows past the position).
- Served restart 10:55:50, forge pid 73029; smoke (positions 1, 8) 29aef2ec9db18a3f.

## 12:55 to 13:35: M6 ANE bot batch: fork-aware tree ranking, sampled lookup on chat / math (`forkrank/`, `lookup_cm/`)

**Fork-aware ranking at 8 rows: no gain, greedy or sampled.** `serve_t/fork_rank_sim.py`. The served tree's top-4
children were re-ranked by a calibrated edge value:
- **Features:** rank among the top 4, softmax(s) bucket, depth, and the fork distance of the parent's path (spine / +1 / +2..).
- **Smoothing:** 20 pseudo-counts toward the served edge probability.
- **Fit / score:** fit on one half of the requests, scored on the other.
- **Tree:** best first on the product of edge values, 7 nodes.
- **Replica check:** the served replica equals `tree_verify.build_tree` (0 mismatches).

| | 2K | 8K | 32K |
|---|---|---|---|
| greedy (chainlog_v1, 1,174 cycles): tok/verify served -> fork-aware | 4.866 -> 4.838 | 5.299 -> 5.291 | 5.407 -> 5.395 |
| greedy projected tok/s | −0.50% | −0.08% | −0.18% |
| sampled (1,887 logged cycles): expected tok/verify served -> fork-aware | 4.110 -> 4.106 | 4.190 -> 4.186 | 3.951 -> 3.947 |
| sampled projected tok/s | −0.09% | −0.11% | −0.13% |

- **Sampled data:** the served config plus `TREE_LOG` + `TREE_LOG_DIST=1`. This new research switch in
  `pipe/tree_verify.py` (off by default) records per cycle the drafter's cand / unary / hp and the target's sampling
  distribution of every verified row. It runs before the accept walk and uses no RNG (`serve_t/tree_logdist_unit.py`).
  The workload was realistic 2K / 8K / 32K, all 9 tasks, thinking on, temperature 1.0, seed 6100.
- **Sampled metric:** expected tok/verify = 1 + sum over nodes of the product of target probabilities along the
  path (the walk's acceptance law). It is exact for 99.8% of the fork-aware tree's expectation; 0.2% comes from nodes
  without a verified parent.
- **Calibration check:** expected 4.110 / 4.190 / 3.951 against the logged realized 3.980 / 4.260 / 3.944 tokens
  per cycle, within the sampling error of about ±0.1. The server's gen / cycles is about 0.16 higher (tokens outside
  draft cycles).
- **Sampled tau sweep:** best-first order, expected tok/verify (unmeasured nodes 0):

  | tau | 2K | 8K | 32K |
  |---|---|---|---|
  | 1.0 | 4.068 | 4.144 | 3.892 |
  | 1.25 | 4.110 | 4.183 | 3.939 |
  | 1.5 | 4.110 | 4.190 | 3.951 |
  | 2.0 | 4.069 | 4.161 | 3.911 |
  | 3.0 | 3.800 | 3.961 | 3.704 |

  The served 1.5 is best under sampling as well.
- **Why:** the served path probability is already about as calibrated as these bins can make it. Dropping
  post-fork continuations (inventory: kept 14-24% one step after a fork) is the correct trade for the 7-node budget,
  because after a fork the true continuation is less likely. Re-weighting cannot recover it. Better post-fork
  candidates could, for example context n-gram continuations as children (the review's merged lookup + neural trie).

**Sampled prompt lookup on math and chat: neutral overall.** `serve_t/lookup_chatmath2.sh`, `lookup_cm/run.log`.
- **Setup:** fresh launches off1 / on1 / on2 / off2, NGRAM_DRAFT=6 when on; tree8 sampled, banner checked first.
  realistic_v1 math + chat tasks (3-8) at 2K / 8K / 32K, Pi settings, seed 5100, 384 tokens.
- **Determinism:** the two orders gave byte-identical outputs per config (a fixed seed is deterministic on a fresh
  process).

| stratum | off tok/s | on tok/s | change (both orders) | tok/verify | lookup cycles | faster |
|---|---|---|---|---|---|---|
| 2K math | 50.76 | 48.04 | **−5.4%** | 5.72 -> 5.39 | 6% | 0/6 |
| 2K chat | 24.17 | 25.79 | +6.7% | 2.69 -> 2.85 | 4% | 5/6 |
| 8K math | 52.56 | 50.93 | −3.1% | 5.93 -> 5.72 | 4% | 1/6 |
| 8K chat | 25.35 | 26.49 | +4.5% | 2.82 -> 2.96 | 3% | 4/6 |
| 32K math | 42.81 | 40.82 | **−4.7%** | 5.37 -> 5.10 | 6% | 0/6 |
| 32K chat | 23.38 | 23.62 | +1.0% | 2.90 -> 2.90 | 3% | 2/6 |
| all | 33.56 | 33.58 | +0.1% | 3.90 -> 3.88 | 4% | 12/36 |

- **Math loses:** a lookup cycle there replaces a DFlash2 / tree cycle that accepts more.
- **With the code / agent result** (sampled +26.0% / +6.6%, `yolo/`): a global default would help code-heavy Pi
  traffic and cost math 3-5%. The candidate is a per-cycle gate (take a lookup proposal only when its match is long or
  the request's lookup cycles have been accepting), not a global switch. Defaults not changed.
- **Run notes:** the first run of these arms failed at argument parsing (zsh does not split `${3:+--tasks $3}`;
  `lookup_cm/run.log` 13:01-13:04); rerun 13:12-13:30 as `lkc2_*`. The sampled tree-log launch of the first run was
  unaffected.
- **No Pi traffic** in any arm. Served config restored 13:29:58: forge pid 45775, smoke 29aef2ec9db18a3f, banner
  tree8 sampled=on, lookup off, ring fix.

## 13:40 to 15:16: lookup gate (LOOKUP_GATE) and a combined lookup + drafter tree (`lookup_gate/`)

**Result:** `NGRAM_DRAFT=6 LOOKUP_GATE=yield:1.25` keeps 97% / 106% of ungated lookup's code gain (greedy / sampled),
removes the math loss, and turns agent positive. Fresh-process A/B, defaults unchanged. A combined lookup + drafter
tree does not pay (offline).

**Data.** `serve_t/lookup_observe.sh`, 13:53-14:10: one launch of the served config with NGRAM_DRAFT=6
LOOKUP_OBSERVE=1 TREE_LOG TREE_LOG_DIST=1.
- **New research switch:** LOOKUP_OBSERVE in `pipe/tree_verify.py`, off by default. It never takes a lookup cycle
  and records each cycle's lookup proposal (drafts, match length) next to the tree.
- **Workloads:** code / agent and math / chat, greedy and sampled, the seeds of the earlier lookup A/Bs.
- **Size:** 4,397 cycles, 72 requests, 0 lookup cycles taken, 0 Pi, 0 errors.
- **Estimate of a lookup cycle's tokens:** 1 + the draft's match length against the realized continuation. Under
  lossless speculative sampling, P(accept m drafts) = P(the target's sample starts with them), so this is unbiased for
  sampled and exact for greedy.
- **Expected tokens per lookup cycle by match length (greedy / sampled):** 6: 4.0 / 4.8; 8: 5.4 / 6.0; 12: 6.8 / 6.7;
  16: 7.2 / 7.0; 48+: 8.0 / 7.9 (`e_table.json`).

**Offline gate screen.** `serve_t/lookup_gate_sim.py`, `sim.log`. Fixed anchors; per-cycle costs from the clean
lookup-off runs; a lookup cycle = chain cycle minus the drafter.
- **Calibration vs the measured ungated A/B:** greedy code +98.7% vs +94.9%, but sampled code +43.8% vs +26.0% and
  sampled math −1.3% vs −4.2%. Good for ranking gates, not for absolute gains.
- **Ranking:**
  - min match 16-24 removes the math loss and keeps 90-94% of the code gain, but drops agent;
  - the yield gate ×1.25 is best overall.

**Combined tree** (offline, not built; one lookup arm of m nodes + the drafter's best-first 7 - m nodes, verified
together; sampled values exact for drafter nodes, the unbiased indicator for lookup-only nodes):
- m = 1-4: −2..−15% on agent / math, −2..−6% overall.
- m = 7 (all lookup): code +71% greedy / +36% sampled, math −9% / −4%.
- A match-length rule for m: +19.5% / +7.1% overall, agent −8% / −5%.
- **Why it loses:** it still pays the drafter call and the tree verify, which lookup-only cycles skip. So a separate
  lookup cycle with a good gate is better than a shared tree.

**The gate.** `LOOKUP_GATE` in `pipe/tree_verify.py` (research, default off; with NGRAM_DRAFT > 0):
- `len:L` takes a lookup draft only when its match is >= L tokens.
- `yield:M` takes it only when E(lookup tokens | match length) / (cycle - drafter ms) > M x the request's running
  tree / chain tokens per ms. Without tree history yet: match >= 16.
- The per-request tree line counts gated proposals; the banner shows `lookup_gate=`.

**A/B.** `serve_t/lookup_gate_ab.sh`, 14:14-15:16. Fresh launches off1 / len:16 / yield:1.25 / off2, banner first;
the same prompts and seeds; the ungated column is from the earlier lk_* / lkc2_* A/Bs.
- **Noise:** off1 vs off2 within ±0.1%, identical texts 9/9 per stratum.
- **No Pi** in any arm; 0 errors. Gated proposals: 318 (len:16), 196 (yield).

| stratum | off tok/s | len:16 | **yield:1.25** | ungated (earlier) | yield keeps |
|---|---|---|---|---|---|
| greedy code | 40.84 | +85.2% | **+92.1%** | +94.9% | 97% |
| greedy agent | 54.03 | −1.7% | **+10.9%** | +0.3% | |
| greedy math | 58.02 | −0.2% | **−0.0%** | | |
| greedy chat | 25.42 | −2.9% | −2.7% * | | |
| sampled code | 37.59 | +20.1% | **+27.4%** | +26.0% | 106% |
| sampled agent | 45.98 | −1.3% | **+4.6%** | +6.6% | 70% |
| sampled math | 45.97 | +3.4% | **+0.5%** | −4.2% | |
| sampled chat | 24.83 | +3.3% | −1.0% | +4.0% | |

\* No overhead from the gate. The yield arm took 0 lookup cycles in greedy chat and math, and on the requests with
identical texts it runs at the off arm's speed (chat −0.0%, math −0.1%, same ms / cycle). The changed texts (2 of 9
chat) come from server history: the arm's earlier code / agent requests took lookup cycles, which changes the stale KV
rows a later cold request sees (the 10:56 finding; COLD_SCRUB off).

**Candidate default** (Anemll's call; not applied): `NGRAM_DRAFT=6 LOOKUP_GATE=yield:1.25` in the serving launcher.
Before that, a repeat in the reverse order and a check on long Pi-like agent sessions (6-57K context) would be
prudent.

Served config restored 15:15:16: forge pid 93698, smoke 29aef2ec9db18a3f, tree8 sampled=on, lookup off, ring fix.

## 15:26 to 16:27: lookup gate, reverse order + long Pi-like agent session (`lookup_gate/ab_rev_long.log`)

`serve_t/gate_reverse_long.sh`; fresh launches, banner first; no Pi traffic, 0 errors in all 6 arms. Served config
restored 16:26 (forge pid 26797, smoke 29aef2ec9db18a3f).

**Reverse order (off, yield:1.25, len:16)** reproduces the first order within ±0.2 points. The outputs are
deterministic per config, so only the timing could move:

| stratum | len:16 (1st / reverse) | yield:1.25 (1st / reverse) | ungated (earlier) |
|---|---|---|---|
| greedy code | +85.2 / +85.4% | +92.1 / **+92.3%** | +94.9% |
| greedy agent | −1.7 / −1.8% | +10.9 / **+11.0%** | +0.3% |
| greedy math | −0.2 / −0.2% | −0.0 / **−0.2%** | |
| greedy chat | −2.9 / −2.7% | −2.7 / −2.8% | |
| sampled code | +20.1 / +20.0% | +27.4 / **+27.4%** | +26.0% |
| sampled agent | −1.3 / −1.3% | +4.6 / **+4.8%** | +6.6% |
| sampled math | +3.4 / +3.4% | +0.5 / **+0.6%** | −4.2% |
| sampled chat | +3.3 / +3.5% | −1.0 / −0.9% | +4.0% |

**Long Pi-like agent session.**
- **Set:** `bench_tree/long_agent_v1.json`, `make_prompts_long.py`. Synthetic, from repo files, no user data.
  - A 5,474-token system prompt with the coding-agent tools.
  - 19 turns of user request, read call and file result, growing from 8.6K to 55.5K of context.
  - Earlier turns carry fixed replies, so every arm sees the same prompts and the server continues from its cache.
- **Run:** sampled with Pi settings, seed 7100, 512 tokens. Arms off1 / yield:1.25 / off2. off1 vs off2: 19/19
  identical texts, ±0.8% by band.

| context band | turns | off tok/s (tok/verify) | yield:1.25 tok/s (tok/verify, lookup cycles) | change | faster turns |
|---|---|---|---|---|---|
| 8-16K | 4 | 37.71 (4.42) | 36.69 (4.26, 4%) | −2.7% | 2/4 |
| 16-32K | 6 | 35.16 (4.30) | 33.70 (4.14, 4%) | −4.2% | 1/6 |
| 32-57K | 9 | 29.54 (3.99) | 31.91 (4.40, 10%) | **+8.0%** | 5/9 |
| all | 19 | 33.73 (4.22) | 33.92 (4.28, 6%) | +0.6% | 8/19 |

**Verdict.**
- **Short sets:** `yield:1.25` holds in both orders (code +92% / +27%, agent +11% / +5%, math −0.2% / +0.6%).
- **The long agent session:** neutral overall, +0.6%. It loses 3-4% at 8-32K (lookup cycles there accept less than
  the tree they replace: tok/verify 4.42 -> 4.26, 4.30 -> 4.14) and wins 8% at 32-57K, where the tree's yield falls
  and the copies are longer.
- **Not yet a default.** A context-aware margin (stricter below 32K) or a calibration of E on long agent traces is the
  next lookup step if wanted. Defaults unchanged.

## 16:51 to 18:11: context-aware lookup gate (`lookup_gate/ctx_gate.log`, `lookup_gate/ctx_vs_yield.log`)

**Gate.** `LOOKUP_GATE=ctx:C:L:M` in `pipe/tree_verify.py`, default off, served defaults unchanged.
- Below C context positions, a lookup proposal is taken only when its match is at least L tokens and it passes
  `yield:M`. With `L = off`, there is no lookup below C.
- At or above C, the gate is `yield:M`.
- Tested: `ctx:32768:48:1.25`. 48 keeps the long full-file copies, where most of the code gain is.

**Run.** `serve_t/ctx_gate_ab.sh`: four fresh launches in ABBA order (off_a, ctx_a, ctx_b, off_b), banner first.
- **Sets per arm:**
  - u1 code + agent.
  - Realistic math + chat (tasks 3-8, 2k / 8k / 32k).
  - Both greedy and sampled. Sampled uses Pi settings: thinking on, temperature 1.0, seeds 4100 / 5100.
  - The long Pi-like agent session: sampled, seed 7100, 512 tokens.
- **Conditions:** no Pi traffic and 0 errors in all 4 arms. The gate turned down 811 proposals per ctx arm.
- **Restore:** served config back at 18:10 (forge pid 74081). Banner: tree8 sampled=on, lookup off, ring fix. Smoke
  29aef2ec9db18a3f.

**Determinism.** off_a = off_b and ctx_a = ctx_b on every request (91/91 texts each). All 24 off arms of the three
gate runs (lga, lgr, lgl, lcg) give the same texts. So each gate is compared with one off text set, and the yield:1.25
column below comes from the earlier runs with the same prompts and seeds.

**Columns.** "Same text" is the change over only the requests whose text equals off's. It shows what the gate itself
gains or costs, free of text noise.

| stratum | off tok/s | ctx:32768:48:1.25 (order a / b) | ctx, same text | yield:1.25 | yield, same text |
|---|---|---|---|---|---|
| greedy code | 40.77 | **+60.6%** (+60.6 / +60.5) | +68.2% (8/9) | +92.2% | +98.7% (8/9) |
| greedy agent | 54.07 | +0.1% (+0.0 / +0.2) | +0.1% (9/9) | +11.0% | +10.9% (5/9) |
| greedy math | 57.98 | +0.5% (+0.5 / +0.5) | −0.0% (7/9) | −0.1% | −0.2% (6/9) |
| greedy chat | 25.41 | −3.3% (−3.4 / −3.3) | −0.1% (7/9) | −2.8% | −0.0% (7/9) |
| sampled code | 37.59 | **+18.0%** (+17.9 / +18.1) | +16.3% (7/9) | +27.4% | n/a (0/9) |
| sampled agent | 46.02 | +1.1% (+1.2 / +1.1) | −0.1% (8/9) | +4.7% | −0.3% (2/9) |
| sampled math | 46.01 | −0.1% (−0.0 / −0.2) | −0.1% (9/9) | +0.5% | +0.0% (3/9) |
| sampled chat | 24.84 | +0.8% (+0.9 / +0.6) | −0.1% (8/9) | −1.0% | +0.0% (1/9) |
| long 8-16K | 37.94 | −0.3% (−0.0 / −0.5) | −0.3% (4/4) | −2.7% | −2.9% (1/4) |
| long 16-32K | 35.16 | −0.0% (+0.1 / −0.2) | −0.0% (6/6) | −4.2% | −1.8% (3/6) |
| long 32-57K | 29.47 | **+8.3%** (+8.3 / +8.3) | −1.5% (6/9) | +8.0% | −1.8% (6/9) |
| long all | 33.77 | **+2.8%** (+2.9 / +2.6) | −0.4% (16/19) | +0.6% | −1.9% (10/19) |

Each yield:1.25 change is measured against the off arms of its own run (lga + lgr, lgl).

**Text noise.** A request whose text differs from off moves on the text alone.
- **Per-request spread:** taken from the requests that took no lookup cycle but still diverged (the process history
  differs: stale KV rows, see `leftover/`). It is 4.9% greedy (n=9) and 10.2% sampled (n=4, few).
- **Per-stratum standard error:** each request is weighted by its decode time.
  - greedy chat: 1.7%
  - long 8-16K / 16-32K under yield:1.25: 5.7% / 4.4%
  - long 32-57K: 4.6% for either gate
  - long all: 1.9% ctx / 2.9% yield

**Findings.**
- **Both orders agree** within 0.5 points in every stratum.
- **The ctx gate costs nothing measurable.**
  - On same-text requests, every stratum is within ±0.3%. The one exception is long 32-57K (−1.5%), where both gates
    take lookup.
  - The greedy chat −3.3% is text divergence on the 2K chat requests 1 and 2, which took no lookup cycle. Same-text
    change there is −0.1%, and yield:1.25 shows the same −2.8%.
- **Below 32K it gives up gains that are real.** The 48-token rule blocks the 16-47-token copies, which carry a third of
  the code gain and all of the short agent gain:
  - greedy code +60.6% vs +92.2% under yield:1.25 (same text: +68.2% vs +98.7%)
  - sampled code +18.0% vs +27.4%
  - greedy agent +0.1% vs +10.9% same text (5/9 requests)
- **The 8-32K dip under yield:1.25 on the long session is small and mostly noise.**
  - The band standard errors (5.7%, 4.4%) exceed the dips (−2.7%, −4.2%).
  - The same-text turns lose 1.8-2.9%. They come from 1-2 lookup cycles per turn that accepted 1-3 tokens, in
    sampled thinking text.
  - The ctx gate removes the dip because it takes no lookup cycle below 32K, giving +2.8% overall vs +0.6%.
- **Above 32K both gates take the same lookup cycles on every turn** (+8.3% vs +8.0%). That band's gain has a 4.6%
  standard error; 3 diverged turns carry most of the weight.
- **The text, not the context length, separates the dip from the gain.** The dip is sampled thinking on the long
  session. The gain is tool-result copies on the short agent set, whose prompts (3-5K) sit below the long session's
  8-32K.

**Verdict.**
- **ctx:32768:48:1.25 is the conservative gate:** no measured loss anywhere; +60.6% / +18.0% on code, +2.8% on the
  long session.
- **yield:1.25 has the larger total:** +92% / +27% on code, +11% / +5% on agent. It costs about 2-3% on long sampled
  thinking turns at 8-32K, which is within noise here.
- **If lookup is served, yield:1.25.**
- **Next lookup step, if wanted:**
  - A text-aware rule: stricter inside the reasoning text than in tool output or code.
  - More seeds on the long session, to settle the 8-32K dip.
- **Served defaults are unchanged.**

**Files.**
- Driver and analysis: `serve_t/ctx_gate_ab.sh`, `serve_t/ctx_gate_compare.py` (this A/B), and
  `serve_t/ctx_gate_vs_yield.py` (next to yield:1.25, with the same-text and noise columns).
- Results in `lookup_gate/`:
  - `ctx_gate.log` and `ctx_gate_compare.json`
  - `ctx_vs_yield.log` and `.json`
  - `lcg_*.driver.log` and `lcg_*.server.log`
  - `restore_ctx.driver.log`
- Raw bench rows: `bench/lcg_*`, `bench/served_after_lcg.*`.

## 18:38 to 20:55: lookup gate inside the reasoning span (`think_gate/`)

The offline sim finished at 18:38. The live run took 18:59 to 20:55. It started once the reviewer's window ended
(18:58) and Pi was idle.

**Gate.** `LOOKUP_GATE=think:T:BASE` in `pipe/tree_verify.py`, default off, served defaults unchanged.
- **Inside the reasoning span:** a lookup proposal is taken only when its match is at least T tokens (`T = off`: none)
  and it also passes BASE.
- **Outside the span:** BASE alone.
- **Span test:** the last `<think>` (248068) comes after the last `</think>` (248069) in history plus anchor. The scan
  is incremental per request.
- **Tree line:** a think gate adds `in reasoning: lookup N, gated M`.
- **Checks:**
  - `serve_t/think_gate_unit.py`: 17 checks, ALL OK.
  - The refactored `_gate` decides as before for yield, ctx and len rules (16,000 random cases, 0 differ).
  - The tree unit tests pass.

**Offline sim.** `serve_t/think_gate_sim.py`, output in `think_gate/sim.log` and `.json`.
- **A. Short sets, exact cycles** from the lgo21 tree traces.
  - Reasoning is 44-55% of the sampled tokens.
  - The short greedy benches run thinking off, so the rule never acts there.

| sampled stratum | yield:1.25 | ctx:32768:48:1.25 | think:off:yield:1.25 | think:24:yield:1.25 | measured yield / ctx |
|---|---|---|---|---|---|
| code | +43.6% | +30.2% | +41.8% | +44.2% | +27.4% / +18.0% |
| agent | +5.7% | +1.7% | +2.4% | +4.8% | +4.7% / +1.1% |
| math | −0.2% | +0.0% | +0.0% | −0.0% | +0.5% / −0.1% |
| chat | −0.1% | −0.2% | +0.0% | −0.1% | −1.0% / +0.8% |
| all | +13.9% | +9.2% | +12.7% | +14.0% | |

  - **The sim runs high** (code +43.6% vs +27.4% measured) but keeps the measured order.
  - **Why it runs high:** the tree yields more on cycles where lookup has a match. Sampled tree tokens per cycle are
    3.65 overall, 5.58 at a 6+ match, and 6.32 at 16+. A replay with one average tree yield therefore overstates what
    lookup adds.
  - **On the short sets, reasoning lookups pay.** Turning them off (think:off) loses half the agent gain. A 24-token
    minimum there costs almost nothing.
- **B. Long session replay.**
  - Each reply is rebuilt from the bench text: reasoning, `</think>`, answer, and the tool-call XML from the template.
    It lands within 1 token of the server's counts. Lookup is then run at every position.
  - **Validation is weak.** On the 10,446-token turn the sim finds 4 ctx lookups where the live run took 0 on the same
    text. At 32-57K it gives −1.1% to −2.9% where the live run measured +8%.
  - **What it does show:** 17 of the 40-44 yield lookups below 32K fall in reasoning. The think rules move 8-32K by
    about +1 point. The 16-32K dip comes from lookups in the answer, which the think rule leaves in place.
  - **Totals** (tree yield scaled by match length): yield −1.6%, think:24:yield −0.6%, ctx −1.1%.
- **Picked for the live run: `think:24:yield:1.25`.** It keeps yield's short-set gain and is tied or best in the long
  replay.

**Live run.** `serve_t/think_gate_ab.sh`, results in `think_gate/ab.log` and `compare.json`.
- **Arms:** six fresh launches in ABBA order (off_a, ctx_a, thk_a, thk_b, ctx_b, off_b), banner first.
  - ctx = `ctx:32768:48:1.25`, thk = `think:24:yield:1.25`.
- **Per arm:** the long Pi-like agent session (19 turns, 8.6K to 55.5K, 512 tokens, thinking on, budget 256).
  - Greedy once.
  - Then sampled (Pi settings, temperature 1.0) with seeds 7100, 7200 and 7300.
  - Unlike the short greedy benches, greedy here has thinking on, so the rule acts in greedy too.
- **Conditions:**
  - Every banner shows the right gate. Pi requests 0 and errors 0 in all six arms.
  - ctx turned down 447 proposals per arm.
  - thk turned down 358 / 367. Of those, 207 were in reasoning, and 29 reasoning lookups passed.
- **Restore:** served config back at 20:55 (forge pid 56175). Banner: tree8 sampled=on, lookup off, ring fix. Smoke
  29aef2ec9db18a3f.

**Determinism.** Repeats of one config give the same texts:
- **off:** 19/19 greedy, 57/57 sampled.
- **ctx:** 19/19 and 57/57.
- **thk:** 19/19 and 56/57.
  - The one thk difference is seed 7100 on the 24,208-token turn: 3 lookup cycles in thk_a, 6 in thk_b.
  - The yield rule reads measured cycle times, so a borderline decision can flip from run to run. In sampled mode a
    different draft changes the sampled text.
- **Timing drift:** off_a and off_b agree within 0.3% (38.97 / 39.02 tok/s greedy), so drift is negligible.

| stratum | off tok/s | ctx (order a / b) | ctx, same text | think:24:yield (order a / b) | think, same text | think vs ctx |
|---|---|---|---|---|---|---|
| greedy 8-16K | 43.39 | −0.3% (−0.1 / −0.5) | −0.3% (4/4) | +0.4% (+0.4 / +0.5) | +0.1% (3/4) | +0.7% |
| greedy 16-32K | 40.41 | −0.1% (−0.0 / −0.2) | −0.1% (6/6) | −0.8% (−0.8 / −0.9) | −0.8% (6/6) | −0.7% |
| greedy 32-57K | 34.56 | −1.2% (−1.0 / −1.4) | −1.4% (6/9) | −3.7% (−3.6 / −3.7) | −3.1% (5/9) | −2.5% |
| greedy all | 38.99 | −0.6% (−0.5 / −0.8) | −0.5% (16/19) | −1.6% (−1.6 / −1.6) | −1.0% (14/19) | −1.0% |
| sampled 8-16K | 37.80 | +0.0% (+0.0 / −0.0) | −0.1% (11/12) | +0.1% (+0.2 / −0.1) | −0.1% (6/12) | +0.0% |
| sampled 16-32K | 35.14 | −4.6% (−4.6 / −4.6) | −0.0% (13/18) | −1.2% (−1.6 / −0.8) | +0.9% (7/18) | +3.5% |
| sampled 32-57K | 30.05 | −1.7% (−1.7 / −1.8) | −4.4% (10/27) | +1.4% (+1.5 / +1.4) | −2.6% (11/27) | +3.2% |
| sampled all | 33.56 | −1.5% (−1.5 / −1.5) | −0.4% (34/57) | +1.6% (+1.5 / +1.7) | −0.2% (24/57) | +3.1% |

**Per seed, sampled whole session:**

| seed | ctx | think:24:yield |
|---|---|---|
| 7100 | −3.1% | −1.1% |
| 7200 | −4.5% | −1.0% |
| 7300 | +4.1% | +7.3% |

From the seed spread, the standard error is about 2.7% for either gate.

**Text-only noise** came from very few requests this time (greedy n=1, sampled n=6 at 19.2%). So the per-stratum
noise columns in `compare.json` are rough. The seed spread gives a similar size.

**Findings.**
- **Neither gate is a measured gain on the long session.**
  - Sampled pooled: ctx −1.5%, think +1.6%. Both are within one seed standard error.
  - The think gate's +1.6% comes from seed 7300 (+7.3%). Seeds 7100 and 7200 lose 1.0-1.1%.
- **On same-text sampled requests, the gate effect is about zero:** ctx −0.4%, think −0.2%. The whole-session changes
  are text divergence. The ctx 16-32K −4.6% is the clearest case: same text −0.0% (13/18), with 1% of cycles taking
  lookup.
- **Greedy shows a small real cost.** Its texts repeat exactly, so these numbers are clean.
  - ctx −0.6%, think −1.6% (same text −0.5% / −1.0%), mostly above 32K (−1.2% / −3.7%).
  - Above 32K, think differs from ctx only by the 24-token minimum inside reasoning. On the 33,336-token turn (same
    text) ctx took 4 lookup cycles at 34.30 tok/s and think took 2 at 32.55 (off 32.94). So long-context reasoning
    lookups can pay, and blocking them is not free.
- **The earlier single-seed +2.8% for ctx on this session (16:51 section) did not hold.** At seed 7100 it now measures
  −3.1%.
  - The off texts also differ from that run (checksum 6dc42bac vs 22bc6790).
  - The reason: here the greedy pass ran first in each process, so the first sampled turn reused 5,376 cached prompt
    tokens. That is a different process history (see `leftover/`). Inside this run every arm has the same history.
  - One seed moves this session by about ±4%, which covers the earlier +2.8%.
- **The offline sim's long-session answer held** (no gain within about ±2%), and so did its caveat: its sizes do not
  predict per-band results.
- **It agrees with the reviewer's fresh holdout** (`../22nov_review/lookup_adaptation/RESULTS.md`, commit 38a964f).
  On 19 new coding turns: yield +1.5%, ctx +0.6%, with ctx going from −4.5% on one seed to +6.1% on the other above
  32K.

**Verdict.**
- **Keep lookup off for serving Pi.** On the long session no gate gives a gain that survives three seeds.
- **think:24:yield:1.25 adds nothing over yield:1.25.**
  - Offline on the short sets it ties yield (+14.0% vs +13.9% sampled).
  - On the long session it is within seed noise of ctx and off. Greedy above 32K it costs more than ctx.
  - The flag stays in, default off.
- **If lookup is served for code edits:** yield:1.25, or the reviewer's next test (route by request type: full-file
  rewrites vs ordinary tool turns).
- **Served defaults are unchanged.**

**Files.**
- Gate: `pipe/tree_verify.py` (`think:T:BASE`, `_in_think`, reasoning counts on the tree line).
- Scripts in `serve_t/`: `think_gate_unit.py`, `think_gate_sim.py`, `think_gate_ab.sh`, `think_gate_compare.py`.
- Results in `think_gate/`:
  - `sim.log` and `sim.json`
  - `ab.log` and `compare.json`
  - `driver.log`, `tgl_*.driver.log`, `tgl_*.server.log`, `restore.driver.log`
- Raw bench rows: `bench/tgl_*`, `bench/served_after_tgl.*`.

## 21:20 onward: evening batch (M6 ANE bot): GDN FP8, WL3A8, drafter split, JetSpec++ vs tree8

All four items come from existing records plus tonight's server logs. No new build or timed run was needed, so forge
stayed on the served config (pid 56175) throughout.

### 1. FP8 on the Gated DeltaNet layers, prefill estimate

STATUS 21:22: closed, no build. Best case about 1.6% (8K), under the 3% gate. This is 19 Nov item 12 (`../19nov/README.md`
section 12), here scaled to Pi's shorter spans.

**The 19 Nov estimate.** It came from the section 3 profile at 95K / 32K, the most each candidate can save of the wall:
- FP8 activations and FP8-decoded weights in the LUT GEMMs: 0. INT8 MMA (39.5 TOPS) is already at least as fast as
  FP8 (36.9), and the activations are already 1 byte.
- FP8 in today's recurrence: 0. It is an fp32 FMA loop, and there is no FP8 ALU path.
- FP8 operands in a chunked recurrence, the extra over an fp16 chunked form: 0.6% / 0.8%.
- FP8 conv and gates: 0.4% / 0.5%.
- FP8 low-rank correction: about 0.
- Sum of the FP8 candidates: 1.0% / 1.3%.

**Scaled to shorter contexts.** The saving scales with DeltaNet's share of the wall, and only counts while the GPU is
the critical path:

| prefill | DeltaNet s / wall s (share) | GPU / ANE busy | FP8 ceiling (share x 1.3% / 52%) | what bounds the wall |
|---|---|---|---|---|
| 2K | 1.9 / 3.1 (61%) | 83% / 96% | 1.5% on paper, about 0 | the ANE (the GPU waits on it) |
| 8K | 7.6 / 11.8 (64%) | 95% / 97% | 1.6% | both, so at most this |
| 32K | 30.0 / 57.8 (52%) | 99% / 85% | 1.3% | the GPU |
| 95K | 88.1 / 243 (36%) | 99% / 58% | 1.0% | the GPU |

- The FP8 total stays at or below 1.6% everywhere.
- **The bigger candidate is not FP8.** The chunked recurrence in fp16 on the matrix units was worth 2.4% / 3.5% at
  95K / 32K. Scaled the same way it is about 4% at 8K, which is above the 3% gate. It is a non-FP8 code change, and
  the FP8 extra above only exists once it is built. It is left for Anemll to decide; nothing was started.

### 2. WL3A8 temporary build A/B

STATUS 21:22: skipped, closed. This is 19 Nov item 4 (`../19nov/README.md` section 4), and the batch's own skip list
names it.
- **19 Nov result:** WL3A8 was 0.3-3.1% slower in prefill at all six contexts, 2K to 95K. Its teacher-forced KL was
  0.0012, as much again as the whole FP8 x change. The build was deleted.
- **Decode** (the greedy + sampled suite) reads the FFN from the same 3-bit index constants. 19 Nov found the FFN
  bound by that weight stream, so INT8 activations cannot shorten it in decode either.
- **Cost of a rerun:** 6.3 GB of packages plus about 6.8 GB of Core AI cache and the compile, for an expected gain of
  0 or less. Not low risk, so it was not rebuilt. A decode-only confirmation would be Anemll's call.

### 3. Hybrid drafter split: ANE vs GPU placement (offline, proposal only)

STATUS 21:30: analysis done, no served change proposed. A single placement change reaches at most 2.0% per cycle even
with free handoffs. All of them together reach at most 4.8% on paper, and that needs handoff costs nobody has
measured.

**The drafter's share of a tree8 cycle, measured tonight.** From the `ms/cycle` split of the 106 requests in the
off_a arm of the think-gate A/B (`think_gate/tgl_off_a.server.log`, served config, 8.6K to 55.5K):

| part | median ms | range |
|---|---|---|
| cycle | 129 | 109-168 |
| draft | 13.2 | 13.0-15.6 |
| verify | 110.7 | 94.6-148.6 |
| sample | 0.8 | 0.1-1.6 |
| ctx | 1.0 | 0.5-3.1 |
| host | 0.2 | 0.1-7.1 |

- **Draft share:** 11% of a cycle at 8-16K (116-120 ms) and 9% at 33-55K (141-148 ms).
- **The cycle is fully attributed:** the parts sum to the cycle within 0.6 ms.
- **The host outliers (up to 7.1 ms) are not the drafter.** They come only on requests where the thinking budget forced
  the close (`THINK closed at N (budget)`).

**Inside the draft phase** (19 Nov section 5, `DRAFT_PROF=1`). That was the chain era, but the total is unchanged
under tree8 (13.1-13.2 ms):
- **Body:** 6.5 ms, the ANE `draft` entry. 674 MB, about 104 GB/s.
- **Target head on the 7 draft rows:** 4.3 ms on the ANE. 636 MB, about 148 GB/s.
- **Host:** 1.8-2.0 ms (top-k, selector, rope / mask fill, ctx64 calls).
- **Gap:** 1 ms.

**Placement options, ceilings at a 118 / 145 ms cycle (8K / 55K).** The DRAM ceilings use the measured 17 Nov peaks:
GPU 156, ANE 121, both together 164 GB/s.

| option | most it can save | of the cycle | what stands in the way |
|---|---|---|---|
| A. Whole drafter on the GPU | 1.0 ms (best kernel 5.5 vs 6.5 ms body); today's kernel loses 2.4 ms | ≤0.8% / 0.7% | Closed 19 Nov section 5. Below 64K the GPU sits at its low clock between calls, and the gapped twin ran 2x slower. |
| B. Drafter overlapping the verify | 0 | 0 | Closed 19 Nov section 5. The draft needs the verify's accepted rows and the bonus token. |
| C. Body split across both units (gate on one unit, up on the other, and the like) | 2.4 ms (674 MB at 164 GB/s = 4.1 ms) | ≤2.0% / 1.7% | One ANE call becomes about 20 ANE↔GPU joins (2 split groups per layer, 5 layers). It only breaks even if a join costs under 0.12 ms, and join cost was never measured. The GPU streams the narrow k / v / conv matrices at only 36-41 GB/s. |
| D. Head split by vocab across both units | 0.4 ms (already 148 of the 164 GB/s peak) | ≤0.3% | Near the DRAM peak already. |
| E. Host draft work moved onto a device (top-k from the zero-copy logits, or inside the head program) | up to 2.9 ms (1.8-2.0 host + 1 gap), not all removable | ≤2.5% / 2.0% | This is not an ANE-vs-GPU choice. A small GPU kernel once per cycle runs at the low clock below 64K. How the 1.8-2.0 ms splits under tree8 is not measured. |
| C + D + E, all ideal | 5.7 ms | ≤4.8% / 3.9% | Free handoffs and fully removable host work. |

**Proposal (nothing changed, nothing run).**
- Keep the drafter as served, all on the ANE.
- If item 3 is to go further, two cheap measurements first, in this order:
  1. **The tree8 draft-phase split.** One fresh launch with the existing timing-only flag `DRAFT_PROF=1`, otherwise the
     served config, then restore and smoke. It shows how much of the host 1.8-2.0 ms is top-k and so how much of
     option E is real. About 25 minutes with Pi idle.
  2. **An ANE↔GPU join probe.** A tiny ANE program ping-ponging with a tiny Metal kernel through a shared event. It
     prices option C, and it needs a tiny compiled package (built via forge.py) and Pi idle.
- Write code only if those two show at least 3% per cycle together.

### 4. JetSpec++ port feasibility vs tree8 (note, no code)

STATUS 21:38: note done. JetSpec++ gets more tokens per verify than tree8: about 1.6-1.75x on code and math at the
reference schedule. But it would run about 13-48% slower than tree8 on this M6. A port is roughly 6-8 working days, so
it stays parked as Anemll decided on 20 Nov. This builds on closed 19 Nov item 7 and the 20 Nov replay work
(`../20nov/README.md` B, A7, B7, B8, C5, D, E). Nothing was rerun.

**Tokens per verify vs tree8.** JetSpec++ was replayed losslessly on our served model's own greedy streams, and
compared there with the DFlash2 chain:

| set | DFlash2 chain | JetSpec++ J3-R3 (reference) | J0-R0 (cheapest schedule) |
|---|---|---|---|
| gsm8k 0-31 | 5.75 | 9.50 (x1.65) | 6.12 (x1.06) |
| humaneval | 7.30 | 13.20 (x1.81) | 8.59 (x1.18) |
| mbpp | 5.89 | 10.55 (x1.79) | 6.64 (x1.13) |
| mt-bench | 3.48 | 4.94 (x1.42) | |

- **What tree8 adds over the chain:** about +3% tokens per verify (5.12 vs 4.95 on the realistic set, 20 Nov A7).
  Like for like in tok/s that is +0.3 / +0.6% on code / math and +8% on chat.
- **So against tree8:** about x1.6-1.75 on code and math and about x1.3-1.4 on chat at J3-R3. At J0-R0 it is about
  x1.03-1.15. These come from dividing by tree8's gain, because tree8 was not replayed on the JetSpec sets.

**Throughput vs tree8 (projected from measured pieces, 8K entry, serial ANE).**
- **Measured against the chain** (20 Nov D2, LUT4 GPTQ adapter): Jet Flash runs −12 to −24% at J0-R0 and −33 to −46%
  at J3-R3.
- **Converted to tree8:** dividing by tree8's like-for-like +0.5 to +3% gives about −13 to −26% (J0-R0) and −34 to
  −48% (J3-R3).
- **Why it loses:** drafting takes 37-43 ms per round even at J0-R0, against DFlash2's 13 ms. A 16-row verify costs
  x1.085 (8K) to x1.27 (64K) of the 8-row verify. The verify is already 72-80% of the cheap schedules' round.
- **What would make it pay** (20 Nov): an adapter 2-3x cheaper per pass (it is already bound by reading its weights,
  and INT4 does not help), or drafting overlapped with the verify. Overlap needs a second execution unit; on this M6
  the GPU overlap loses to DRAM contention (C2).

**Effort if it is ever unparked.** These are estimates, not measurements. The starting points are `jsp_replay.py`,
the LUT4 / LUT6 adapter tensors and the captures on the SSD.

| step | work | estimate |
|---|---|---|
| 1 | ANE packages for the adapter (LUT4 GPTQ; fix the per-matrix Hadamard seed bug first, 20 Nov E), the B16 head (LUT6) and the anchor decoder, all via forge.py; HWX checks | 1.5-2 days plus compiles |
| 2 | A 16-row chain verify entry for the served target (tree8 has 8 rows): a target rebuild via forge.py, with a second build set on disk during the A/B | 1 day plus compile |
| 3 | Server integration: the multi-pass round schedule, acceptance up to 16, DeltaNet state and KV rollback at 16 rows, the sampled path, the drafter ring | 2-3 days |
| 4 | Identity, ABBA on the standard suites, the long Pi session, memory under Pi load (the LUT4 adapter alone is 2.55 GB wired, next to forge's about 19 GB) | 1 day |
| | total | about 6-8 working days |

**Verdict:** not worth porting as a serial ANE drafter. Keep it parked until there is a second execution unit, or an
adapter that is 2-3x cheaper per pass.

## 21:45 onward: second evening batch (M6 ANE bot): handoff probe, DRAFT_PROF, drafter-split decision, chunked GDN recurrence

### 1. ANE ↔ GPU handoff probe, standalone (`handoff/`)

STATUS 22:06: stopped, so the probe produced no handoff medians. It stalled the ANE for 100 s and then got "Request
cancelled", in 2 of 3 runs. The ANE recovered each time, no Pi traffic arrived during either stall, and forge was not
touched. Item 3 uses the 16-17 Nov handoff measurement instead.

**Setup** (sources in `handoff/src/`):
- **Package:** `v8/build_hop_probe.py`, a scratch build of 49.2 MB, deleted after with `coreai_cache_delete.py` and
  then the files. Its input x is (1, 5120, 1, 16) fp16, the drafter's width and rows. Three functions:
  - `cp`: y = x + z
  - `mm`: a 5120 x 5120 matrix as a 6-bit vector LUT over output-channel pairs (the drafter's format, 9.8 MB)
  - `mm4`: four such matrices in one call
- **Compile and loads:** compiled via forge.py (`journal_compile.py`); every load is journaled.
- **HWX check** (`hwx_mode.py`), before timing in every run: 47.1 MB, zin_ane_compiler v10.302.2, bonded only over ANE
  instances [0, 1], 6 streams, 10 fp16-lut6bit NE tasks (ActiveNE 4). The second check, after timing, was never reached.
- **GPU side:** `hop_gpu.swift` wraps the ANE's IOSurface buffers zero-copy (`makeBuffer(bytesNoCopy:)`, the served
  `dec_wrap`'s method).
- **Driver:** `hop_probe.py`. Banner first. It times the serial (served) pattern and the fork-join (body-split)
  pattern, and redoes a block if Pi traffic overlaps it.

| run | time | where it stopped |
|---|---|---|
| 1 | 21:57 | At the cross-device value check. The data did cross both ways (each ANE output equalled the GPU's previous output exactly), but my tolerance was wrong: Metal's half add moved the value by one fp16 step (0.00049), not 0.001. Fixed. |
| 2 | 21:58:52 to 22:00:32 | The first `cp` call hung 100 s, then `ANEProgramProcessRequestDirect() Failed with status=0xf : statusType=0x9: Request cancelled`. |
| 3 | 22:02:45 to 22:04:26 | Warm-up of all three functions passed, run before Metal loaded. The value chain passed both ways. Calibration passed: ANE `mm` 0.297 ms back to back, GPU busy kernel 0.288 ms. Then the first back-to-back `cp` block hit the same 100 s stall and cancel. |

**After each stall:**
- A fresh process without Metal ran cp / mm / mm4 / cp normally (0.48-0.98 ms single calls), so the ANE itself
  recovered.
- Forge (pid 56175) logged no request in that time.

**Not root-caused.**
- The same zero-copy wrap runs in the served split verify without stalls. Run 1 also made 6 `cp` calls with Metal
  loaded.
- The stall appeared only in processes with the Metal library loaded and the buffers wrapped.
- Each attempt can block the ANE for 100 s, and a Pi request arriving then would hit forge. So I made no further
  attempt. A rerun belongs in a window with forge stopped, which is Anemll's call.

**Handoff numbers used instead** (16-17 Nov, `pipe/unit/DECODE_GPU_ATTENTION.md`). These were measured on the served
split verify (ANE part A, GPU attention, ANE part B). Its IOSurface tensors are about 100 KB each, against the
drafter's 160 KB. They are typical values; no p90 was recorded.
- **Each GPU round trip:** about 0.65 ms on top of the kernel.
  - commit to GPU start: 0.28-0.34 ms (the GPU idles during the ANE call)
  - GPU end to CPU wake: about 0.1 ms
  - encoding: 6 us
- **One extra ANE call:** about 0.25 ms. Fusing part B of chunk k with part A of chunk k+1 saved about 0.25 ms per
  chunk.
- **Already tried then, and worse:**
  - spinning on `cb.status` or a shared event
  - a GPU-side flag spin (the GPU never sees the store)
  - a warm-up kick (the ANE call gets about 1.3 ms longer)

### 2. DRAFT_PROF=1 launch, tree8 draft phase (`draft_prof/`, `serve_t/draft_prof.sh`)

STATUS 22:17: done. The tree8 draft phase is 13.3 ms per cycle, and 12.3 ms of it is ANE work. Only about 1.0 ms is host
work. Served config restored (pid 95936, banner tree8 sampled=on, lookup off, ring fix), smoke 29aef2ec9db18a3f.

**Run:**
- **Precondition:** Pi had been idle for 4,294 s at 22:06:20, and no reviewer window was running.
- **Launch:** one fresh launch of the standard launcher with `DRAFT_PROF=1` and otherwise the served config, banner
  first. The banner shows "DRAFT_PROF timing armed".
- **Bench:** the long Pi-like agent session, greedy and then sampled seed 7100 (thinking on, 512 tokens).
- **Output identity:** both runs gave exactly the served off texts (greedy 119a784ffdd31d68, seed 7100
  6dc42bac15e9127d).
- **Conditions:** Pi requests 0, errors 0, 42 `drafter prof:` lines. The abort guard (on Pi traffic or a reviewer
  window) never fired.
- **Speed:** 38.32 / 32.73 tok/s, against 38.97-39.02 / 33.27-33.28 for the off arms of the 18:59 run on the same texts
  (−1.7%). The cause was not checked. This flag is never served.

**The draft phase per propose.** Medians over the 29 requests with at least 30 proposes, 8.6K to 55K. The ANE parts do
not change with context: body 6.75-6.94 ms, head 4.37-4.44 ms.

| part | where | ms per propose |
|---|---|---|
| drafter body (`draft` entry) | ANE | 6.86 |
| target head on the 7 draft rows | ANE | 4.41 |
| ctx64 context calls (1.97 ms per call, 0.50 calls per propose) | ANE | 0.99 |
| host work inside propose (top-k, selector, rope / mask fill) | CPU | 0.78 |
| outside propose (tree assembly), draft minus propose | CPU | about 0.22 (0.12-0.35) |
| total (the cycle split's draft) | | 13.3, of a 129 ms cycle (10%) |

The 19 Nov chain-era split had "host 1.8-2.0 ms" plus a 1 ms gap. That host figure included the ctx calls, which
today's `propose host` (1.76 ms) also does. Without the ctx calls the CPU part is 0.78 ms, and the gap is gone.

### 3. Drafter-split proposal

STATUS 22:20: not written, because items 1 and 2 together come nowhere near 3%. The best case of any drafter placement
change is about 0.8% per cycle, and the split options lose once handoffs are paid. Nothing changed.

| option (item 3 of the first batch) | ceiling before handoffs | handoffs it needs | net |
|---|---|---|---|
| C. body split across ANE + GPU | ≤2.4 ms | about 20 joins. A fork-join costs at least an extra ANE call (about 0.25 ms) plus the CPU wake (about 0.1 ms), and a serial round trip about 0.65 ms (16-17 Nov). | −4.6 to −10.6 ms per cycle: loses |
| D. head split by vocab | ≤0.4 ms | at least 1 join (0.35-0.65 ms) | about 0 or a loss |
| E. host draft work onto a device | ≤1.0 ms (0.78 inside propose + 0.22 outside, measured in item 2) | at least 1 dispatch; below 64K the GPU is at its low clock | ≤0.8% of a 129 ms cycle |
| merge the ctx64 call into the body call | the per-call overhead only: about 0.25 ms x 0.5 calls = 0.13 ms | none | about 0.1% |

The drafter's 13.3 ms is 12.3 ms of ANE work that already streams near its DRAM ceiling: the body at about 104 GB/s
and the head at about 148 GB/s against the 164 GB/s that both units reach together. Moving it does not shorten it.
The only remaining lever is less drafter work per cycle, which is not a placement question.

### 4. Chunked fp16 GDN recurrence on the matrix units, standalone prototype (`gdn_chunk/`)

STATUS 22:32: done. The ~4% at 8K does not hold. The chunked form matches the reference, but even its best version
(chunk 32) is 1.6x slower than the served `gdn_rec` alone (0.471 vs 0.288 ms per 128-token call) and 1.9x slower
under DRAM load (0.631 vs 0.337). Nothing was integrated. Forge was not touched, and every timed run waited for my
timing lock and for Pi to be idle.

**Setup.** `pipe/unit/power/gdn_chunk_proto.swift` and `gdn_chunk.metal`, copied to `gdn_chunk/src/`.
- **Served kernel:** run exactly as served (kernels.metal sha 421b61512afa8ead, T=128).
- **Shapes:** q, k [16][128][128]; v, o [48][128][128]; g, beta [48][128]; state S [48][128][128] fp32.
- **Chunked form:** fp16 operands on simdgroup matrices, fp32 accumulation. The S0 / P / X copies stay fp32.
- **Inputs:** synthetic but realistic: unit-norm q / k with key correlation rho 0.3 (0.7 as a stress case), decay near
  1, beta in (0, 1).
- **Banner** first in every timing log.

**Parity.** Against a CPU double-precision reference, starting from S = 0 and carrying the state from call to call:

| chain | heads checked | output RMS rel | output max rel | state max rel |
|---|---|---|---|---|
| 2K (16 calls) | 48 | 5.0e-4 | 6.8e-4 | 5.9e-4 |
| 2K, rho 0.7 | 48 | 5.8e-4 | 7.4e-4 | 8.9e-4 |
| 8K (64 calls) | 6 | 5.2e-4 | 6.4e-4 | 6.1e-4 |
| 32K (256 calls) | 6 | 5.2e-4 | 6.8e-4 | 6.5e-4 |

- There is no drift over 256 calls; the error stays at fp16 operand level.
- For comparison, the served fp32 `gdn_rec` is at 4.4e-7 RMS against the same reference.

**Timing.** Milliseconds per 128-token call for one layer, median of 15, back to back. The same at the 2K, 8K and 32K
chain lengths; p90 within 2% of the median.

| kernel | alone | under `dram_hog 2` |
|---|---|---|
| `gdn_rec` (served) | 0.288 | 0.337 |
| chunked, chunk 64 | 0.707 | 1.032 |
| chunked, chunk 32 (best) | 0.471 | 0.631 |
| chunked, chunk 16 | 0.521 | not run |

**Ablations** (8K, alone). None closes the gap:
- fp16 copies of S0 / X / P: slower (0.55-0.84).
- Each head's value rows split over 2 to 4 threadgroups: slower (0.58-1.35). The redundant per-head work costs about
  0.1 ms per extra copy, so the kernel is limited by total work, not by latency.
- Operands stacked so it needs 5 matrix ops instead of 9: slower (0.54-0.92).
- The triangular solve is not the main cost: 0.06 ms at chunk 32, 0.14 ms at chunk 64.
- On these small per-head tiles the matrix units reach about 1.8 TFLOP/s.

**Projected prefill change** (the measured per-call change x calls, over the 19 Nov section 3 wall; positive means
slower):

| prefill | calls | alone | under load | what bounds the wall |
|---|---|---|---|---|
| 2K | 768 | +0.14 s | +0.23 s | the ANE, so about 0 |
| 8K | 3,072 | +0.56 s (+4.7%) | +0.90 s (+7.7%) | both |
| 32K | 12,288 | +2.2 s (+3.9%) | +3.6 s (+6.3%) | the GPU |
| 95K | 35,712 | +6.5 s (+2.7%) | +10.5 s (+4.3%) | the GPU |

**Why the estimate was wrong.** The 19 Nov estimate (2.4% / 3.5% at 95K / 32K) assumed the chunked form would cut the
recurrence to about 0.1 ms per call at large-matrix tensor-op efficiency. To reach that, this form would have to run
about 4.7x faster than the best version here. The served per-token fp32 loop is already the faster form on this GPU.

**Not recorded:** the DRAM load's achieved bandwidth. `dram_hog` was stopped before it printed; 19 Nov measured this
setting at about 140 GB/s.

**Verdict:** drop the chunked recurrence. The served DeltaNet recurrence stays as it is.
