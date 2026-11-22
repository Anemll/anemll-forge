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

## NEXT.md item 1, step A: the sliced-entry rung probe (01:10 to 01:14)

Sources in `rung_probe/src/` (masters: `C/v8/build_rung_probe.py`, `C/pipe/unit/power/rung_probe_time.py`,
`rung_probe.sh`). Raw files: `rung_probe/` (run.log, build.log, compile.log, rung_probe.log, rung_probe.json, watchdog
logs).

- **Queue:** queued at 23:02 behind the A/B. That run gave up at 01:03 after its 2 h idle wait, because the A/B ran past
  2 h. Requeued at 01:04. Built 01:10:50 to 01:12:38 and compiled via forge.py (`journal_compile.py`, 26 s). Timed at
  01:14:01 with Pi idle and no other timing lock. Forge was listening and not touched.
- **Package:** one scratch group (layers 0-3) in the served tree8 build's environment, KV_SEG 8K, 8K, 16K, 32K, 32K;
  main.hash sha 45af98b41a2c99af. It is deleted now (`rung_probe.sh clean`: Core AI entries via coreai_cache_delete.py,
  then the files).

  | entry | keys attended | segments bound | bound tensors |
  |---|---|---|---|
  | tree8_16k | 16384 | 2 | 38 |
  | tree8_24ks | 24576 | 3 | 42 |
  | tree8_32k | 32768 | 3 | 42 |
  | tree8_48ks | 49152 | 4 | 46 |
  | tree8_64k | 65536 | 4 | 46 |

  A sliced entry binds the bigger entry's segments, so it adds no bound tensors.
- **HWX, both checks:** before timing, 573.5 MB, zin_ane_compiler v10.302.2, bonded only (disable-bonded=false), ANE
  instances [0, 1], 10 streams. After timing: identical (`hwx_same_after: true`).
- **Banner:** `BANNER rung_probe_time | ...rung_p8_L00-03.aimodel main.hash sha 45af98b41a2c99af | ... |
  MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 | forge 8765 LISTENING (not touched) | HWX before: ... ANE instances [0, 1]`.
- **Swap:** 686 MB before and after both the build and the timing (watchdog peak 686 MB).
- **Slice check:** each sliced entry is bit-exact against the bigger entry with the keys past its width masked off.
  - 24ks vs 32k masked: y rel error 0.00.
  - 48ks vs 64k masked: y rel error 0.00.
  - Against the unmasked bigger entry, the errors are 5.9e-3 and 4.6e-3, so the check would catch a slice that read the
    tail.
- **Timing:** ANE ms per group call, median / p90 over 50 calls after 5 warm-up. Pass a runs 16k to 64k, pass b the
  reverse. Both passes were clean (no Pi traffic).

  | entry | pass a | pass b | mean of medians |
  |---|---|---|---|
  | tree8_16k | 5.674 / 5.920 | 5.623 / 5.651 | 5.648 |
  | tree8_24ks | 5.913 / 6.003 | 5.903 / 5.990 | 5.908 |
  | tree8_32k | 6.165 / 6.209 | 6.125 / 6.185 | 6.145 |
  | tree8_48ks | 6.627 / 6.720 | 6.616 / 6.690 | 6.622 |
  | tree8_64k | 7.084 / 7.126 | 7.106 / 7.179 | 7.095 |

- **Position in the step** (fraction of the way from the smaller to the bigger entry; linear in width would be 0.50):
  - 48ks: 0.50 of 32k to 64k. Saves 0.474 ms per group, so 7.6 ms per 16-group verify against 64k.
  - 24ks: 0.52 of 16k to 32k. Saves 0.238 ms per group, so 3.8 ms per verify against 32k.
- **The ANE reads only the slice.** The cost follows the attended width, not the bound segments.
- **Against the served verify** (per entry tonight: 16K 101.3, 32K 109.8, 64K 125.5 ms):
  - The probe's steps times 16 are 7.96 ms (16K to 32K) and 15.2 ms (32K to 64K). Served: 8.5 and 15.7 ms. The ANE
    width cost is about 94-97% of the served step.
  - Projected served verify at 48ks: 125.5 - 7.6 = **117.9 ms**. The bar was the midpoint, 117.7 ms.
  - Projected served verify at 24ks: 109.8 - 3.8 = 106.0 ms. Midpoint 105.6 ms.
- **What this projects** (unchanged from NEXT.md, now with a measured slice cost): +1.9% decode on the long Pi session
  (adding 24K and 48K), +3.0% (24K / 40K / 48K / 56K), about +4% (a 4K ladder).
  - Sliced entries are bit-exact against the masked bigger entry, so replies would not change.
- **Step B is not started.** It needs a temporary full build: 8 chunks, about 10 GB on the SSD plus Core AI and aned
  caches on the internal disk, which has 29 GiB free. It also needs entry selection for the sliced entries in the tree
  verify path, behind a research flag.

STATUS 01:14 item 1 step A (sliced decode entries): PAYS. 48K-sliced = 0.50 of the 32K to 64K step (linear 0.50),
bit-exact against masked 64K. Projects 117.9 ms served verify against a 117.7 ms bar. HWX bonded [0, 1], unchanged
after. Scratch build deleted. Step B (temporary full build + long-session ABBA) waits for Anemll.

## NEXT.md item 1, step B: temporary build with sliced entries + long-session A/B (from 01:50)

Anemll's go-ahead 01:50, with guardrails:
- the temporary build on the SSD
- internal free space checked before compiling and hourly; stop below 15 GiB, and delete no cache I did not create
- sliced selection only behind a research flag
- the usual timing and restore rules
- ABBA plus KL / greedy parity, 24K+48K first
- the build deleted unless it clears +1.5% on the long session in both orders

- **Code** (commit 361b6fd):
  - `build_dec8t_chunk.py` with `TREE_SLICED` adds `g<i>_v8s_<w>k` (chain) and `g<i>_t8s_<w>k` (tree) per group at
    24K / 40K / 48K / 56K. Each binds the next ladder entry's KV segments (40 to 46 tensors) and attends over w keys.
  - `v8/build_dec8ts_all.sh` is the build driver with the disk guard.
  - `pipe/rung_slice.py`: research only, installed by `tree_verify.install` when `RUNG_SLICE` is set (a 3-line hook;
    unset = the served path).
  - `serve_t/slice_ab.sh` / `slice_compare.py`: the parity gate, then the ABBA.
  - `tree_bench.py` banner: a `slice` key.
  - One build carries all four widths; `RUNG_SLICE=24,48` or `24,40,48,56` picks the set at launch. Chain cycles (about a
    third of the long session's verifies) use the sliced entries too.
- **Internal disk:** per compiled chunk, about 1.2-1.4 GB of Core AI entry plus about 1.4 GiB of aned HWX. These are the
  served build's own numbers: 10.4 GB and 11.2 GiB for 8 chunks.
  - 29.0 GiB free at the start, so a full build would end near 8 GiB.
  - The driver starts a compile only with ≥ 18 GiB free (guard + one chunk) and stops a running one below 15 GiB.
  - Expected: a pause after about 5 of 8 chunks unless space is freed. That is Anemll's call. Candidates are the
    superseded 20 Nov tree builds v8_dec8tc and v8_dec8t, about 21 GiB each with aned.
- **First export, 02:12: failed.** The palettizer reported `NUM_PALETTES 256` (and 64) for 4-bit indices on 8 weights:
  FFN chunk (9216 x 5120, 4-bit LUT) and DeltaNet out_proj (5120 x 6144, 2x64 LUT).
  - This happens when a weight no longer matches its registered LUT and the UNIQUE fallback runs.
  - That run had TMPDIR on the exFAT SSD; the served builds used the default.
  - Retried 02:16 with the default TMPDIR (internal, transient).
- **Retry, 02:24: chunk 0 exported** (81 entries, 1193 MB; the served chunk is 65 entries, 1189 MB). The exFAT TMPDIR
  was the cause, and TMPDIR stays the default from here.
- **Chunk 0 compiled** at 02:27 (3 min 4 s). Persistent cost: a 1.1 GB Core AI entry plus a 1.3 GB aned HWX, about
  2.4 GB, as projected. Chunk 1 exported at 02:36 (713 s).
- **Stopped at the disk guard, 02:38:50.** The compile of package 1 started at 02:36:02 with 26.6 GiB free, while
  chunk 2 exported alongside it. Free fell to 14 GiB at 02:38:00 and 11.2 GiB at 02:39.
  - The driver polled the guard only while waiting for a compile, so the overlapping compile ran unwatched. The build
    driver, the compile (coreai_compile.py) and the export were killed. Nothing was deleted, and forge kept serving
    (pid 15402).
  - Mine in that window: 1.1 GB of compile temp (`$TMPDIR/com.apple.MetalPerformanceShadersGraph/mpsgraph-60774-*`)
    and a 1.1 GB aned HWX written at 02:37:15 with no Core AI entry (orphaned by the kill). No Core AI entry was
    written for chunk 1.
  - The other ~13 GiB of the drop is in no file this user can read. Candidates: root-only compiler temp of the killed
    compile, or the macOS update staging (`com.apple.os.update-MSUPrepareUpdate` is the newest snapshot). The update
    services were idle when checked. Free held at 11.2 GiB after the stop.
- **Driver fixed for a resume.** A watcher now polls free space every 15 s for the whole run and stops this build's
  compile / export below 15 GiB, and export and compile no longer overlap.
- **What a resume needs.** 6 chunks to compile at about 2.4 GB each (about 15 GB persistent). Each compile's transient
  peak is unknown but was several GiB here. With the 15 GiB guard, that means about 25 to 30 GiB freed first.

STATUS 02:41 item 1 step B: STOPPED at the 15 GiB internal-disk guard (11.2 GiB free). 2 of 8 packages exported, 1
compiled; build killed; nothing deleted; forge still serving. Waiting for Anemll's call on disk space.

STATUS 08:49 item 1 step B (sliced decode entries, temporary build v8_dec8ts, RUNG_SLICE=24,48): CLEARS the keep rule.
Long session pooled +1.92% (order a +1.85%, order b +1.99%; rule: +1.5% in both orders), 152/152 replies identical to
served (greedy and seeds 7100 / 7200 / 7300). Build kept; deploy waits for Anemll. Served config restored, forge pid
14551, smoke 29aef2ec9db18a3f MATCH.

### Step B result (22 Nov, 07:27 to 08:49, macOS 27.3 26C5047e)

Parity gate (kc, RUNG_SLICE=24,40,48,56 RUNG_SLICE_CHECK=1, long greedy): 471 sliced calls over 22 requests, max |logit
diff| 0, max KL 0, argmax 3762/3762 rows. PASS.

Block 1, ABBA (sv_a, s2_a, s2_b, sv_b), pooled client decode tok/s, long session (19 turns, 8.6K to 55.5K), 4 sets:

| band   | served | 24K+48K | pooled | order a | order b | identical replies |
|--------|-------:|--------:|-------:|--------:|--------:|------------------:|
| 8-16K  | 40.06  | 40.17   | +0.29% | +0.19%  | +0.40%  | 32/32   |
| 16-32K | 36.49  | 37.17   | +1.86% | +1.83%  | +1.89%  | 48/48   |
| 32-57K | 29.76  | 30.70   | +3.15% | +3.07%  | +3.23%  | 72/72   |
| all    | 34.92  | 35.59   | +1.92% | +1.85%  | +1.99%  | 152/152 |

Per set (order a / order b): greedy +1.61% / +2.40%, s7100 +2.00% / +2.01%, s7200 +1.97% / +2.07%, s7300 +1.74% /
+1.50%. Verify calls: 1789 of 4212 sliced (t8s_24k 687, t8s_48k 626, v8s_24k 277, v8s_48k 199); tree cycle 111.7 ->
109.2 ms, chain 107.2 -> 105.1 ms. Block 2 (4-entry set) skipped: it would start after 07:00. HWX: 8 referenced, 0
missing, for both w_dec8tx and v8_dec8ts_w after the run. Pi requests 0 and errors 0 in every arm.

Build notes (05:43 to 07:26): resumed after the 27.3 update (which freed the 26C5035r caches: 272 GiB free at the end,
guard never near). Sliced chunks compile in about 3m12s vs 2m20s served (81 vs 65 functions). GPU tables: the 4-part
parallel prep (prep_engine_par.py, = serial: sliced/prep_par_check.txt) hit its swap watchdog at 07:01 with forge
serving; a 2-part rerun then pushed system swap until forge's own watchdog killed forge at 07:04:23 (down to about
07:06:30, restored, smoke MATCH). Fix: prep_engine.py keeps one chunk's HWX weights in memory (80/80 files identical:
sliced/prep_lean_check.txt), tables rerun as 1 part under a +512 MB watchdog, done 07:23 with no swap growth.
Sources in sliced/src/.

OS update recovery (04:51 to 05:32): 27.3 recreated aned's cache without the read ACL (Anemll re-granted it); served
v8_dec8tx recompiled (17 packages, 19m36s); w_dec8tx GPU tables regenerated (1346 files byte-identical, manifests differ
only in the HWX path build dir; old manifests kept as manifest*.26C5035r.json).
