# Hybrid Neural Engine + GPU (research name V8, M6, 13 to 17 Nov 2026)

Qwen3.8-27B (release 0.2, vq3pA) prefill on the M6 with every FFN on the Neural Engine and every mixer (DeltaNet and
attention) on the GPU, plus an OpenAI-compatible server built on it.

**Status: research and experimental.** Not part of the release and not supported. It needs System Integrity
Protection disabled and root once to open a system cache to your user, so use it only on a research Mac: see
[Requirements: root and SIP](#requirements-root-and-sip). User-facing docs call this path the **hybrid** (the Neural
Engine and the GPU running one model at the same time); V8 is its research name in these notes. It is unrelated to
the "V8" INT8 V cache format of the main README, and it is not the earlier V1 hybrid prototype in `prototypes/`.

The server (96K context, DFlash2 all3 drafter, greedy, thinking off, the GPU otherwise idle; `pipe/unit/cmp_bench.py`,
times to the first streamed token at the client). Prefill and follow-ups are the 16 Nov evening run; decode is shown
before and with SPLIT+GPU (the 17 Nov rerun, `results/cmp_v8dec8_96k.jsonl`, prefill unchanged within run noise):

| prompt | cold prefill | follow-up (about 320 new tokens) | decode cycle: 16 Nov | with SPLIT+GPU | decode, cold turn: 16 Nov | with SPLIT+GPU |
|---|---|---|---|---|---|---|
| 2K | 586 tok/s (3.6 s) | 1.04 s | 108 ms | 107 ms | 34.4 tok/s | 36.0 tok/s |
| 8K | 658 tok/s (12.5 s) | 1.15 s | 112 ms | 111 ms | 24.6 tok/s | 23.2 tok/s |
| 16K | 620 tok/s (26.5 s) | 1.90 s | 120 ms | 119 ms | 20.4 tok/s | 20.9 tok/s |
| 32K | 541 tok/s (61 s) | 1.75 s | 134 ms | 133 ms | 19.3 tok/s | 20.0 tok/s |
| 64K | 420 tok/s (156 s) | 2.36 s | 153 ms | **141 ms** | 18.5 tok/s | 20.6 tok/s |
| 95K | 347 tok/s (274 s) | 2.92 s | 155 ms | **149 ms** | 18.9 tok/s | 20.4 tok/s |

The ANE-only target build of the same model prefills about 300 tok/s at 2K to 8K and 180 at 96K. Decode speed is the
cycle time over the accepted draft tokens: 2.4 to 3.7 per cycle on these cold turns, up to 7.8 when the answer
repeats earlier text (follow-ups: 20 to 67 tok/s).

**SPLIT+GPU decode (17 Nov).** The 8-layer build `v8_dec8` (`v8/build_dec8_all.sh`, server manifest
`v8/dec8_srv_manifest.py`) adds three Neural Engine entries per package, `dva` / `dvba` / `dvb`, that run each
verification step's layers around a GPU attention over the KV history (`pipe/dec_split.py`,
`pipe/engine/dec_gpu.swift` and `dec_kernels.metal`). They bind no KV and no mask, so one program serves every context
length. Served by `pipe/v8_server_dec8.sh` (the split runs in the 96K entry only): decode cycle at 64K 153 → 141 ms and
at 95K 155 → 149 ms (splash 1.3.2: 141 / 148), unchanged at 2K to 32K. Details and the rejected variants:
[pipe/unit/DECODE_GPU_ATTENTION.md](pipe/unit/DECODE_GPU_ATTENTION.md).

**Served since 22 Nov: tree8 + sliced entries (`v8_dec8ts`).** The 8-layer build with tree verify entries (tree8, 21
Nov) and sliced verify entries at 24K / 40K / 48K / 56K (22 Nov), the KV-wrap memory fix (22 Nov) and the GPU-table
kit. Write-up: [TREE_VERIFY.md](TREE_VERIFY.md). Build, deploy and test on another Mac: [hf/DEPLOY.md](hf/DEPLOY.md)
(the portable launcher `hf/serve_v8_96k.sh`, `pipe/engine/build_engines.sh`, `pipe/gpu_tables_kit.py`,
`pipe/build_gpu_tables.sh`). On the development M6: `pipe/v8_server_dec8_qk.sh` (defaults in its header; the leak fix
with `V8_SRV_DYLIB=libgpu_v8_f8q_uw.dylib DEC_DYLIB=libdec_gpu_tree_uw.dylib`).

- [PREFILL_RESEARCH.md](PREFILL_RESEARCH.md): what was tried, what was kept or rejected and why, numbers, lessons.
- [pipe/V8_SERVER.md](pipe/V8_SERVER.md): the server (run, options, request flow, results, KV8 findings).

## Tree verification (tree8)

Decoding is speculative: the DFlash2 drafter proposes a block of 7 tokens per cycle on the Neural Engine (16
candidates per position), and one 8-row target call verifies them. Tree verification (tree8, served since 21 Nov)
keeps those 8 rows but fills them with a tree instead of one chain:

- **The tree.** Row 0 is the anchor; the other 7 rows are the drafter's most probable continuations, chosen best first
  on path probability (up to 4 children per node). Each row attends to the KV history and to its own ancestors only.
  A draft with a single path runs on the served chain entry; only a fork pays for the tree entry (ancestor masks and
  the `corrx0` DeltaNet short-convolution correction, which keeps chain replies bit-identical: 45/45).
- **The accepted path** is committed through the existing chain routine: the selected DeltaNet pending state and
  convolution rows are packed, the new KV rows compacted and the drafter's features gathered in path order. Sampled
  requests (Pi, temperature 1.0) accept child by child with the chain's rejection rule.
- **Across both engines.** At the 96K entry the verification's attention runs on the GPU between Neural Engine calls,
  and the Metal kernel applies the same ancestor mask.
- **Sliced KV widths (22 Nov).** Verify entries at 24K / 40K / 48K / 56K bind the KV segments of the next larger
  entry and attend only over the prefix in use (logits identical on 471 checked calls).

| measurement (M6, client tok/s) | result |
|---|---|
| greedy, 2K to 64K rungs | +1.9% to +6.5% over the chain (2K: 45.80 against 42.74) |
| sampled (Pi-like), two 36-request blocks | +8.5% and +10.2% pooled; tool-agent subset -1.6% / -3.9% |
| sliced widths, 19-turn Pi-like session (8.6K to 55.5K) | +3.4% pooled (+2.5% without one slow baseline seed; +5.7% at 32K to 57K), 152 of 152 replies identical |

Wider trees do not pay on the Neural Engine: 16 or 32 verify rows cost +26-39% or +87-116% per call, and timing probes
with replay put the best wider tree only 1.6% ahead at 8K. Also measured and not kept: per-cycle tree / chain choice, prompt
lookup for Pi, JetSpec++ (projected 12-46% slower on the serial schedule), drafting ahead during the verify, and
rescoring the tree. Switches: `TREE=0` serves the chain, `RUNG_SLICE=` (empty) the build without slicing. Write-up,
prior art and every result: [TREE_VERIFY.md](TREE_VERIFY.md); deployment: [hf/DEPLOY.md](hf/DEPLOY.md).

## Requirements: root and SIP

Core AI is not designed to run one model on the Neural Engine and the GPU at the same time: it gives the GPU no access
to the Neural Engine's compiled weights, and its own GPU path is far too slow for the GPU half (`ffn2_L0` 89.5 ms
against 4.6 ms in our Metal engine). Apple may support this in the future; this research setup works around it.

The GPU engine does not load its own copy of the weights. It maps, read-only and zero-copy, the weight segments
(`__KERN_0`) of the Neural Engine programs that macOS compiled on this Mac, and decodes their palettized layout in its
kernels (`pipe/engine/gpu_v8.swift`; the `hwx` list of the table manifest). That keeps the weights in memory once: the
eight `v8_dec8` packages map 10.8 GiB, and a GPU-only copy would add as much again on a 32 GB Mac.

Those programs exist only in aned's cache. A Core AI package carries no compiled Neural Engine program: the system's
ANE compiler service, `aned`, compiles it on first load into
`/Library/Caches/com.apple.aned/<macOS build>/ModelAssetsCache/-_unsigned/<hash>/<hash>/model.hwx`. Two protections
stand in the way:

1. **Owner-only directory (sudo, once).** The cache directory is `drwx------ root admin`. The `model.hwx` files
   themselves are world-readable (`-rw-r--r--`), but a normal user cannot enter the directory. An inherited ACL grants
   one user list and read access:
   ```
   sudo chmod -R +a "user:$(id -un) allow list,search,read,readattr,readextattr,readsecurity,file_inherit,directory_inherit" /Library/Caches/com.apple.aned
   ls -le /Library/Caches | grep -A1 com.apple.aned     # shows the grant
   sudo chmod -R -N /Library/Caches/com.apple.aned       # undo: removes every ACL there (on this M6 the grant is the only one)
   ```
   Nothing is written to the cache. aned recreated its cache directory once (27 Oct 2026), which dropped the grant:
   grant it again when the GPU engine cannot open the files.
2. **Data vault (SIP must be off).** The cache and its files carry the `datavault` flag (`ls -lO`). macOS enforces
   data vaults through System Integrity Protection, even against root, so with SIP on neither sudo nor the ACL can
   read them. Disable SIP from macOS Recovery (Terminal: `csrutil disable`, then restart), check with
   `csrutil status`, and re-enable it the same way with `csrutil enable` when done. SIP off lowers the security of
   the whole Mac: keep this to a dedicated research machine.

The GPU tables point into those compiled files, so they are rebuilt after every recompile (a macOS update or a cache
purge; Build chain, step 3). Building them needs the public release 0.2 export ([anemll/anemll-quantized-qwen3.8-27b-for-CoreAI](https://huggingface.co/anemll/anemll-quantized-qwen3.8-27b-for-CoreAI): `export/release_vq3pA_mixh_s600_k1_mat/` and `model/` for the
small tensors, byte-identical to ours; no HWX parser: the package -> `model.hwx` lookup reads the Core AI cache
manifest), and they match only this compiler's output (M6, macOS 27.3; the kit of `hf/DEPLOY.md`: build 26C5047e):
other chips and macOS builds are untested.

The research scripts are committed as they ran: they use absolute paths on the M6 (below). The release-code
commits of this branch hold the builder and runtime changes they need (`coreai/`, `scripts/` of this checkout = the
`anemll-forge-v02` working tree they import from; `V8_V02=<tree>` points the server at another one):
`Segmented KV8 history` (KV_SEG), the host-side speedups taken over from Greg Martin's anemll-forge-speedup fork
(credited in that commit), and the `kv_seg` class default.

## Layout

| path | what | original location |
|---|---|---|
| `pipe/v8_server.py`, `v8_engine.py`, `v8_server*.sh`, `srv_stop.sh` | the V8 server (wraps `scripts/qwen38_server.py`, replaces `Engine.prefill`); `v8_server_dec8.sh` is the served script since 17 Nov | `C/v8_work/pipe` |
| `pipe/dec_split.py` | SPLIT+GPU in the server: the split verify (dva -> GPU -> dvba -> GPU -> dvb per package), 16 logical chunks over 8 packages loaded once each | `C/v8_work/pipe` |
| `pipe/engine/dec_gpu.swift`, `dec_kernels.metal`, `dec_kernels_r.metal` | decode attention (`libdec_gpu.dylib`, kernel `dec_attn_r4`; `DEC_ATTN=dec_attn` is the previous one), research kernels | `C/v8_work/pipe/engine` |
| `pipe/v8_server_split.py` | earlier server (snapshots split the V8 run) | `C/v8_work/pipe` |
| `pipe/engine/gpu_v8.swift`, `kernels.metal` | GPU engine: mixers, attention, KV8 / K8V16, A8 expanders, state handover | `C/v8_work/pipe/engine` |
| `pipe/engine/gpu_layers.swift` | V1 engine (contiguous GPU layers, the hybrid prototype) | same |
| `pipe/engine/patch_*.py` | one script per engine change (already applied to these sources) | same |
| `pipe/engine/a8_cpu.c` | A8 CPU (NEON) weight expander | same |
| `pipe/prep_engine.py`, `prep_kv8.py`, `prep_ffn.py`, `merge_manifests.py`, `ref_layers.py` | GPU tables (zero-copy into the compiled HWX) and the float32 reference | `C/v8_work/pipe` |
| `pipe/v8_a8.py`, `prep_a8.py`, `prep_a8att.py`, `prep_a8out.py` | A8 driver (runtime-INT8 FFN on the ANE) and its preps | `C/v8_work/pipe` |
| `pipe/unit/` | unit checks, profilers, A/B and sweep scripts, server bench (`srv_bench.py`, `srv_matrix.sh`), server-log prefill paths (`srv_log_prefill.py`), the offline prefill / snapshot check (`shortfix_check.py`) | `C/v8_work/pipe/unit` |
| `v8/build_seg2_96.sh` | base KV8 segmented target build (coreai_seg2) | `SSD/coreai_seg_logs` |
| `v8/build_shared2_all.sh`, `build_shared2_one.sh`, `build_shared_chunk.py`, `chunkmlp.py`, `build_v8_ffn.py`, `build_v8_mix.py` | the shared-weight V8 build (v8_shared2) | `C/v8_work/v8` |
| `v8/build_shared_all.sh`, `build_shared_probe.sh`, `build_ffn.py`, `build_ffn2_probe.py`, `v8_handoff_probe.py` | earlier builds and probes (v8_shared, FFN-only, handoff) | same |
| `v8/share/`, `v8/trace/` | probes: shared weights, attproj, ANE context cost, ANE load, timelines | same |
| `v8/build_dec8_chunk.py`, `build_dec8_one.sh`, `build_dec8_all.sh`, `dec8_manifest.py`, `dec8_srv_manifest.py`, `journal_compile.py` | the 8-layer SPLIT+GPU build (v8_dec8), its manifests, journaled compiles | `C/v8_work/v8` |
| `v8/build_dec_att*`, `build_dec_split_probe*`, `build_sync_bridge*` | decode probes: ANE partial attention by key range (rejected), split probes | same |
| `pipe/unit/dec*`, `decode_*.swift`, `gpu_wake_lat.swift`, `bw/`, `DECODE_GPU_ATTENTION.md` | decode checks, kernel tests, bandwidth tests and the decode write-up | `C/v8_work/pipe/unit` |
| `bench/cron/`, `bench/verbosity/` | Pi coding comparison: the cron prompt and hidden grader, the single-turn verbosity probe | `C/v8_work/bench` |
| `ARTICLE_ANE_GPU_M6.md`, `.html`, `img/`, `make_article_html.py` | the human-readable article on this work, its HTML version (rendered in the browser; rerun `make_article_html.py` after edits) and its 12 slides (1920 px PNG) | |
| `w8a8/` | ANE INT8 program builders / probes: V7 W8A8, A8 FFN shape and real probes, A8 attention, LUT split | `C/v8_work/w8a8` |
| `prototypes/v8_prefill.py` | the standalone V8 driver (all research runs: ref, v8, handover, kv8, attproj, sweeps) | `C/claude_full96_prep/freeze_kv_slice_v1` |
| `prototypes/hybrid_*.py`, `gpu_ane_*.py`, `w8a8_mlp_probe.py`, `capture_hidden.py` | V1 hybrid and earlier probes | same |
| `prototypes/kv_slice_probe*.py`, `view_resize_check.py`, `e2e_seg_check.py`, `e2e_server_check.py`, `hwx_share_check.py` | KV segment / view probes and checks behind commit 1 | same |
| `a8probe/` | A8 and long-context run scripts, A8 probe results (json) | `SSD/a8probe` |
| `splash132/` | splash 1.3.2 serve and bench scripts, `cmp_96k.sh` (the same requests on V8 and splash 1.3.2), `cmp_96k_gpu.sh` (splash 1.3.2 GPU only), `serve_pi.sh` | `C/v8_work/splash132` |
| `tools/swap_watchdog.sh` | swap guard used by every long job | `SSD/tools` |
| `tools/forge_watchdog.sh` | the served forge's own memory guard (footprint) | `SSD/tools` |
| `tools/deploy_journal.py` | portable copy of the deployment journal (root `DEPLOY_JOURNAL_ROOT`) for `hf/serve_v8_96k.sh` | `C/claude_full96_prep/freeze_eight_row_v4` |
| `hf/` | the Hugging Face card, `DEPLOY.md`, `serve_v8_96k.sh` | |
| `pipe/tree_verify.py`, `rung_slice.py`, `stream_pace.py`, `gpu_tables_kit.py`, `build_gpu_tables.sh`, `prep_engine_par.py`, `engine/build_engines.sh` | tree8 verify, sliced entries, stream pacing (research, off), the GPU-table kit and full rebuild, parallel table prep, engine build | `C/v8_work/pipe` |
| `bench_tree/` | tree benches (`tree_bench.py`, prompt sets incl. `smoke_2k_v1.json`, `run_arm.sh`) | `C/v8_work/bench/tree` |
| `results/` | server matrix, K8V16 64K run, splash 1.3.2 runs, `cmp_{v8,splash}_96k.jsonl`, `cmp_splash_gpu_96k.jsonl`, `cmp_v8dec8_96k.jsonl`; `results/17nov/` (indexed) | `SSD/coreai_seg_logs` |

Absolute paths in the scripts:

| name | path |
|---|---|
| C | `/Users/anemll/Documents/Codex/ios-prefill-20261030` |
| V02 | `/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02` (fusedqkv + the KV_SEG changes of commit 1) |
| SSD | `/Volumes/SSD4TB/anemll-forge-research` (builds `coreai_seg2/`, `v8_shared2/`, `v8_dec8/` (served), logs `coreai_seg_logs/`, `a8probe/`) |
| model | `/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K` (HF config, tokenizer, embedding, drafter `all3`) |
| export | `/Users/anemll/Models/vq27b/export/release_vq3pA_mixh_s600_k1_mat` (release VQ export; public, byte-identical: `export/release_vq3pA_mixh_s600_k1_mat/` of anemll/anemll-quantized-qwen3.8-27b-for-CoreAI), checkpoint `/Volumes/SSD4TB/Qwen3.8-27B` (the small tensors the builder reads are in that repo's `model/`) |
| runtime python | `anemll-forge/.venv` (server, drivers, `forge.py compile`) |
| builder python | `ane-vector-lut/coreai/.venv` (torch + Core AI tools: builds, preps) |
| journal | `deploy_journal.py` from `C/claude_full96_prep/freeze_eight_row_v4` (every Core AI load is journaled) |

Downloaded, or generated on your Mac, so they do not need to be in this repository:

- **Model weights (downloaded):** the quantized release 0.2 export,
  [anemll/anemll-quantized-qwen3.8-27b-for-CoreAI](https://huggingface.co/anemll/anemll-quantized-qwen3.8-27b-for-CoreAI)
  (`export/release_vq3pA_mixh_s600_k1_mat/`, and `model/` for the small tensors).
- **The SPLIT+GPU build (downloaded, compiled on your Mac):**
  [anemll/anemll-forge-qwen3.8-27B-V8-96K](https://huggingface.co/anemll/anemll-forge-qwen3.8-27B-V8-96K)
  (`coreai/m6-96k-dec8/`, with the drafter and model files); `forge.py compile` compiles it for the Neural Engine on
  first use (Build chain, step 5).
- **GPU tables (generated on your Mac):** `pipe/w_*`, 0.3 to 0.9 GB each, built from your own compiled programs and the
  downloaded export (Build chain, steps 3 and 6); they point into your Mac's compiled HWX files, so each Mac makes its
  own.
- **Dylibs (built on your Mac):** from the sources in this repository (Build chain, steps 4 and 6).
- **Reference logits, probe packages and hidden-state dumps (generated):** `v8/ref_*.npz` from
  `v8_prefill.py --mode ref`, and the A8 probes' outputs, written by the scripts in this repository.
- **The `.pre_*` source backups:** not needed; every later variant is an env switch in the final `gpu_v8.swift` /
  `kernels.metal`.

## Build chain

1. Base target build, KV8 with segmented history (coreai_seg2, contexts 8K to 96K):
   `v8/build_seg2_96.sh` (runs `coreai/qwen38_coreai_build.py all ... --kv-cache-dtype kv8` from V02 with
   `KV_SEG=8192,8192,16384,32768,32768`, `ATT_INT8MM=s8,s8b,sm8,pvf8` and the rest of its env).
2. Shared-weight V8 build (v8_shared2, 16 chunks, same size as the plain target): `v8/build_shared2_all.sh`. Per chunk
   `build_shared2_one.sh` -> `build_shared_chunk.py` with the seg2 env plus `V8_SPLIT=8192,9216 V8_ATTPROJ=1`: target
   entries with the chunked FFN, plus `ffn_L` / `ffn2_L` (FFN), `mix_L` / `mix2_L` (DeltaNet), `ffnq2_L` / `offn2_L`
   (FFN fused with the attention projections) on the same weight constants. Then the manifest and
   `forge.py compile --build` (about 40 min build, 20 min compile). After an aned cache purge:
   `FORCE=1 pipe/unit/rebuild_shared2.sh` (recompile + GPU tables).
3. GPU tables (zero-copy into the compiled HWX, so rerun after every compile):
   `PREP_NO_FFN=1 EXPORT_DIR=<export> MODEL=<checkpoint> <builder python> pipe/prep_engine.py SSD/v8_shared2 0 64 pipe/w_shared2r 2048`,
   then `EXPORT_DIR=<export> <builder python> pipe/prep_kv8.py pipe/w_shared2r/manifest.json pipe/w_shared2r/manifest_best.json`
   (int8 k / v weights for I8KV; the server uses `manifest_best.json`).
4. GPU engine: `cd pipe/engine && swiftc -O -emit-library -module-name gpuv8 -o libgpu_v8_srv3.dylib gpu_v8.swift -framework Metal -framework Accelerate`
   (the server's default; srv2 is the same engine without `patch_fillfrom.py`).
   `kernels.metal` is compiled at load (Metal 4.1). A8 CPU expander: `clang -O3 -shared -o liba8cpu.dylib a8_cpu.c`.
5. Drafter (once): `forge.py compile --build SSD/v8_shared2 --draft <model>/drafter/all3/dflash2_lut4_gptq_nohead.aimodel`.
6. SPLIT+GPU build (v8_dec8, served since 17 Nov), replacing v8_shared2: `v8/build_dec8_all.sh`. Each of the 8 packages
   (`build_dec8_one.sh` -> `build_dec8_chunk.py OUT_DIR FIRST_LAYER TAPS`, the v8_shared2 env) carries the target and
   V8 entries of its two 4-layer groups (`g0_` / `g1_`) plus `dva` / `dvba` / `dvb`, and is compiled
   (`journal_compile.py` -> `forge.py compile`, one at a time) while the next one exports. Then `prep_engine.py` into
   `pipe/w_dec8` with a check that its 8 HWX files exist, `prep_kv8.py` as in step 3 (`manifest_best.json`), and
   `v8/dec8_srv_manifest.py` (16 logical chunks with aliases; the file-level manifest stays as `manifest_files.json`).
   Decode attention: `cd pipe/engine && swiftc -O -emit-library -o libdec_gpu.dylib dec_gpu.swift -framework Metal`
   (`dec_kernels.metal` is compiled at load).
7. Tree8 + sliced build (v8_dec8ts, served since 22 Nov), replacing v8_dec8: `v8/build_dec8ts_all.sh` (each package by
   `v8/build_dec8t_one.sh` -> `v8/build_dec8t_chunk.py` with `TREE_FORM=corrx0 TREE_SLICED=24576,40960,49152,57344`,
   compiled while the next one exports; guards for internal disk space), then `v8/build_dec8ts_tables.sh` (tables with
   `pipe/prep_engine_par.py` + `prep_kv8.py`, the server manifest, the 8-HWX check). GPU engines:
   `zsh pipe/engine/build_engines.sh uw` (`libgpu_v8_f8q_uw.dylib`, `libdec_gpu_tree_uw.dylib`). Portable table kit:
   `pipe/gpu_tables_kit.py pack | install | verify`. All of it for another Mac: [hf/DEPLOY.md](hf/DEPLOY.md).

## Run

```
PORT=8765 pipe/v8_server_dec8_qk.sh     # served since 22 Nov: tree8 + sliced (v8_dec8ts, w_dec8ts); SERVED_SLICE=0 / SERVED_TREE=0 roll back
REPO=<HF download> hf/serve_v8_96k.sh    # the same configuration on another Mac (hf/DEPLOY.md)
PORT=8765 pipe/v8_server_dec8.sh        # 17 to 20 Nov: SPLIT+GPU chain only (v8_dec8, w_dec8), the port Pi's ane-qwen38 provider uses
pipe/v8_server.sh                       # the 16-chunk v8_shared2 server without the split: port 8767 (PORT=...), ctx 98304 (CTX=...)
splash132/serve_pi.sh                   # splash 1.3.2 for Pi (provider splash-local, port 8000); one server at a time
pipe/srv_stop.sh                        # stops whatever listens on $PORT
pipe/unit/srv_matrix.sh OUT "tag|opts|2000 6100 16200"    # cold + follow-up benchmark per config
```

Standalone driver (prefill-only or real end to end, KL against an ANE-only reference), e.g. 6K with the final
engine: `export BUILD=SSD/v8_shared2 MAN=pipe/w_shared2r/manifest_best.json`, then
`FUSE=all I8=1 I8T=1 I8TV=_k256 LUTC=1 I8A=1 I8KV=1 LRM=1 QK8=1 TKV=2048 pipe/unit/run_wd.sh 6144 ref_6k.npz TAG --shared --streams 8 --dylib libgpu_v8_srv2.dylib --handover --ho-overlap`
(always export BUILD and MAN: without them `run_wd.sh` falls back to another build and the bridge compiles it).
A8: `a8probe/e2e_len.sh TOKENS REF REPS` (baseline vs A8 back to back).
