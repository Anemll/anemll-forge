# 24 Nov 2026: harness-hub super-adapter on the forge Core AI model (yolo run, Anemll)

Adapter: `harness-hub/Qwen3.8-27B-super-adapter` (blog "harness adapters at scale", 20 Nov 2026): PEFT LoRA, r 96,
alpha 192 (scale 2.0), the pi + hermes + opencode RL adapters concatenated at weight 1/2; targets q / k / v / o of the
16 full-attention layers and gate / up / down of all 64 MLPs; DeltaNet, embeddings and head untouched; 478M params
(F32 1.9 GB). Its chat template differs from ours only by `{% generation %}` training markers (same rendered text).
Local copy: `/Volumes/SSD4TB/anemll-forge-research/adapters/qwen38-27b-super-adapter`.

## Method: an exact fp16 side path, the 3-bit weights untouched

`adapter/adapter_export.py` writes a new export: every adapted matrix gets `lr_a = 2 B`, `lr_b = A` (original basis,
the export's own convention; the anemll-forge-v02 builder moves b into the rotated MLP / mixer basis). Attention already
carries the release's rank-64 corrections: the adapter is appended (rank 160). Everything else is symlinked.

- `exports/release_vq3pA_sa`: 256 matrices, 0.89 GiB fp16 factors (8K test build)
- `exports/release_vq3pA_sa_p64`: the same with the attention factors zero-padded to rank 192 (exact; the V8 GPU prefill
  computes them in 64-row tiles) for the 96K tree build

Checks (layer 3, against the BF16 checkpoint and the adapter):

| matrix | adapter delta / W (Frobenius) | release quantization error | adapted - base export vs delta | builder path (rotated basis) vs W_sa x | rank |
|---|---|---|---|---|---|
| mlp gate | 0.0018 | 0.214 | 0.00029 | 0.00000 | 96 |
| mlp up | 0.0019 | 0.214 | 0.00029 | 0.00000 | 96 |
| mlp down | 0.0010 | 0.240 | 0.00028 | 0.00000 | 96 |
| q_proj | 0.0012 | 0.359 | 0.00029 | 0.00000 | 160 |
| k_proj | 0.0012 | 0.010 | 0.00029 | 0.00000 | 160 |
| o_proj | 0.0012 | 0.418 | 0.00030 | 0.00000 | 160 |

The adapter moves each weight by 0.1-0.2 % while the 3-bit quantization error is 21-42 % (k / v: INT8, 1 %): merging and
re-quantizing would bury it; the side path keeps it exact.

Tree-build MLP (`v8_work/v8/chunkmlp.py`, backup `chunkmlp.pre_lr.py`): optional low-rank factors (gate / up share
one b x, down sums b over the chunks). Layer 3, fp32: chunked + adapter vs `LayerW.mlp` (QConv + adapter) 1.9e-6 (the
chunking's own summation order: 1.8e-6 without factors); adapter effect on the MLP output 0.24 %.

GPU prefill (`pipe/engine/gpu_v8_f8q.swift`, backup `gpu_v8_f8q.pre_lrrank.swift`): `lowrank()` takes the rank from the
B buffer (64 unchanged, up to LRMAX 192), TR / TR16 sized for 192. Built as `libgpu_v8_f8q_uwlr.dylib` (served `_uw`
dylibs untouched).

Build scripts: `v8/build_dec8t_one.sh` and `v8/build_dec8ts_all.sh` take EXPORT_DIR (and LT, RDIR) from the
environment; defaults = the 22 Nov build (backups `*.pre_env.sh`).

## Runs

1. 8K Core AI test build (`adapter/build_8k_sa.sh`): anemll-forge-v02 `qwen38_coreai_build.py all --ctx 8192
   --pctx 8192 --kv-cache-dtype v8`, the numerics of the base 8K build `v3_tierA_s600_dual/..._kvv8`, 8K entries only;
   `forge.py compile --follow` (journaled). Output `/Volumes/SSD4TB/anemll-forge-research/sa_test/release_vq3pA_sa_kvv8`.
2. `adapter/after_8k.sh`: KL-512 (same BF16 trace as `gate_kl512_tierA_s600`), KL(adapter || base build), gate,
   cleanup of the 8K test's Core AI entries, then the 96K tree build `v8_dec8ts_sa` + tables `v8_dec8ts_sa_w`.

Results: below as they come in (raw: `run.log`, `kl512_*.json`, `kl_adapter_vs_base.json`, `build96/`).

## 8K result (03:13-04:08): gate PASS

Build 16 chunks + head (v8_8k / p64_8k), export 03:13-03:58, `forge.py compile --follow` 17 packages in 41m36s.
KL-512 on the BF16 trace of `gate_kl512_tierA_s600` (64 sequences, 40,023 positions, the v8_8k entry, teacher forcing):

| build | mean KL vs BF16 | median | top-1 agree | ppl (BF16 ref 2.136) |
|---|---|---|---|---|
| base 8K (release, 9 Oct run) | 0.0518 | | 0.9189 | 2.131 |
| + super-adapter (side path) | 0.0533 | 0.0088 | 0.9167 | 2.131 |

KL(adapter build || base build) on the same partition: mean 0.0016, median 0.00021, p99 0.0172, max 0.926; top-1 the same
at 98.6 % of positions. The adapter shifts 1.4 % of next-token choices on this generic (non-agent) trace and leaves the
distance to BF16 and the perplexity where they were: not broken. Whether the shift is the adapter's intended one needs
an agent / Pi test (no BF16 + adapter reference here). Raw: `kl512_sa8k_summary.json`, `kl512_base8k_summary.json`,
`kl_adapter_vs_base.json`, `run.log`.

Cleanup (04:08): `coreai_cache_delete.py --delete` on the 17 test packages (`cleanup_8k_plan.txt`, `cleanup_8k.log`);
none shared a main.hash with a kept build (the served head 34358c7c9abe stays cached; the 8K head was fe3ea3737776).
The 8K build files stay on the SSD (`sa_test/`).

## 96K tree build `v8_dec8ts_sa` (04:08-06:10)

`v8/build_dec8ts_all.sh` with EXPORT_DIR=`exports/release_vq3pA_sa_p64`, OUT `v8_dec8ts_sa`, GW `v8_dec8ts_sa_w`: 8 packages x
81 entries (1318-1538 MB each, served 1297-1583 MB), export + compile 04:08-05:46, GPU tables 05:46-06:10 (1.2 GB;
attention lra / lrb rank 192, DeltaNet unchanged at 64); 8 HWX referenced, none missing. Logs: `build96/`.

## Repair of the served build (06:23-06:44): HWX purged on 23 Nov

The first A/B start (06:11) failed: `BridgeError: nilError` loading `v8_dec8ts`. aned's `model.hwx` of 4 served packages
(L16-23, L32-39, L40-47, L48-55), of the served head (`coreai_seg2/..._kv8/head_T8.aimodel`, aned 2CBF43E551A6) and of
the all3 drafter (`dflash2_lut4_gptq_nohead.aimodel`, C0A30F694542) were gone, `model.src` stubs left, directory mtimes
23 Nov 17:13-17:15 (internal disk low that day; not this run). Their Core AI entries stayed, so a load does not
recompile: it fails. The served forge could not have started since then.
Fix (`adapter/repair_dec8ts.sh`, `adapter/repair_head_drafter.sh`): `coreai_cache_delete.py --delete` of those 6 entries,
journaled `forge.py compile` (13m15s + 25s); `gpu_tables_kit.py install` of the HF kit: 8 x weight section MATCH (same
programs at the same paths), `verify` against the served `w_dec8ts`: 1344 table files identical; all 20 ANE programs of
both arms present. Base smoke afterwards: `29aef2ec9db18a3f` (= served; also proves `libgpu_v8_f8q_uwlr.dylib` is the
served math at rank 64). Logs: `repair/`.

## 96K A/B (06:44-07:11), port 8767, served launcher, libgpu_v8_f8q_uwlr + libdec_gpu_tree_uw in both arms

`adapter/ab_96k.sh`, raw `ab96/` (failed starts: `ab96_try*`). Greedy, gen 320, tree8 + slices, one fresh server per arm,
base first. Decode tok/s token-weighted (tree_bench SUMMARY), tok/verify from the server log:

| set | base tok/s | adapter tok/s | change | tok/verify base / adapter | ms per verify base / adapter | mean TTFT change |
|---|---|---|---|---|---|---|
| smoke 2K (2) | 36.41 | 30.07 | -17.4 % | 3.98 / 3.64 | 109 / 121 | +34 % |
| realistic 2K (9) | 45.48 | 40.40 | -11.2 % | 4.86 / 4.75 | 107 / 117 | +13 % (2K + 32K) |
| realistic 32K (9) | 47.81 | 42.32 | -11.5 % | 5.59 / 5.38 | 117 / 127 | |
| realistic 96K (3) | 52.30 | 48.47 | -7.3 % | 6.74 / 6.73 | 129 / 139 | +7.5 % |
| U1 agent, tools (9) | 56.41 | 49.60 | -12.1 % | 5.66 / 5.53 | 100 / 111 | +17 % |
| long agent session 8K-55K (19) | 50.83 | 48.22 | -5.1 % | 4.57 / 4.88 | 90 / 101 | +9 % |

Cost: every verify cycle +8-11 % (the 0.9 GB fp16 side path read per step; prefill +7-17 % TTFT), drafter acceptance
-1..-9 % on short prompts, equal at 96K, +7 % on the long session. Footprint peak 11264 MB in both arms (watchdog limit
12288); swap rose 664 -> 2242 MB during the adapter's 96K set (logged only).

Behaviour (single-turn replays, not task completion): replies identical in 21 / 51 requests (realistic 1 / 18, 96K 2 / 3,
agent 4 / 9, long 14 / 19); tool calls agent + long 13 -> 12; completion tokens agent 1488 -> 1377, long 558 -> 639.
The divergent agent turns go the way the adapter was trained (fewer exploratory calls): agent[7] base makes a no-op
edit (adds a `# noqa` comment), the adapter answers "no changes needed"; long[0] base greps first, the adapter states
the answer and edits directly; long[4] base greps to verify, the adapter reports done without checking. Whether that
raises task success needs multi-turn Pi tasks (these sets replay fixed turns).

State after the run: no server on 8765 / 8767 (forge was not serving before the run either). The adapter build and its
Core AI entries are kept for a Pi trial (`BUILD=.../v8_dec8ts_sa GM=.../v8_dec8ts_sa_w/manifest_best.json
V8_SRV_DYLIB=libgpu_v8_f8q_uwlr.dylib DEC_DYLIB=libdec_gpu_tree_uw.dylib v8_server_dec8_qk.sh`); the 8K test's entries
were deleted.

## Pi on the adapter build (24 Nov 07:20-07:40, 8765, thinking medium, sampled T 1.0)

Voxel pagoda prompt (same as the 22 Nov runs), compared with the base tree build's 22 Nov 12:51 run (thinking medium):

| | base 22 Nov | adapter 24 Nov |
|---|---|---|
| wall / model time | 82.3 min / about 33 min (45 min waiting for "continue") | 10.5 min / about 9.4 min |
| model turns, output tokens | 33, 55.0K | 4, 16.9K |
| tool calls | bash 23, edit 4, write 1, read 2 (node stub tests, Chrome screenshots, pixel stats) | write 1, bash 1 (node --check only) |
| first turn | 14,026 tok, 27.4 tok/s, 3.66 tok/verify, budget 6144 reached, then a tool call | 11,329 tok, 27.9 tok/s, 3.52 tok/verify, budget 6144 reached, code printed in chat, turn ended (no write) |
| user nudges | 2 x "continue" (+ "120s is too high?") | "did you save it?" |
| result (headless Chrome, `pi/`) | pagoda broken: roof tiers missing, floating columns | coherent 5-tier pagoda, torii, bridge, pond, trees; no console errors |

Decode speed was the same; the adapter was faster because it did far less verification. "Stopped midway" = its first
reply printed the HTML and ended without a `write` call (Pi ends the loop on a reply without tool calls); one nudge fixed it.
Tetris (07:20-07:29): wrote the file at once, mocked-DOM tests, "done" at 2.5 min, but missed a hidden start overlay
(its own test printed empty score readouts and it explained them away); fixed in 1.1 min after the report.
One sample each at T 1.0: suggestive, not a measurement.
