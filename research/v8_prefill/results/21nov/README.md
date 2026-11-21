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
