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
