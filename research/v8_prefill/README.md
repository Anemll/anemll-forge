# V8 GPU + ANE prefill (research, M6, 13 to 16 Nov 2026)

Qwen3.8-27B (release 0.2, vq3pA) prefill on the M6 with every FFN on the Neural Engine and every mixer (DeltaNet and
attention) on the GPU, plus an OpenAI-compatible server built on it. Cold prefill 634 tok/s at 2K, 663 to 700 at 6K,
350 at 95K (ANE-only: about 300 / 180; splash 1.3.2 on the same machine: 500 / 288).

- [PREFILL_RESEARCH.md](PREFILL_RESEARCH.md): what was tried, what was kept or rejected and why, numbers, lessons.
- [pipe/V8_SERVER.md](pipe/V8_SERVER.md): the server (run, options, request flow, results, KV8 findings).

The research scripts are committed as they ran: they use absolute paths on the M6 (below). Commit 1 of this branch
(`Segmented KV8 history`) holds the builder and runtime changes they need (`coreai/`, `scripts/` of this checkout =
the `anemll-forge-v02` working tree they import from).

## Layout

| path | what | original location |
|---|---|---|
| `pipe/v8_server.py`, `v8_engine.py`, `v8_server*.sh`, `srv_stop.sh` | the V8 server (wraps `scripts/qwen38_server.py`, replaces `Engine.prefill`) | `C/v8_work/pipe` |
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
| `w8a8/` | ANE INT8 program builders / probes: V7 W8A8, A8 FFN shape and real probes, A8 attention, LUT split | `C/v8_work/w8a8` |
| `prototypes/v8_prefill.py` | the standalone V8 driver (all research runs: ref, v8, handover, kv8, attproj, sweeps) | `C/claude_full96_prep/freeze_kv_slice_v1` |
| `prototypes/hybrid_*.py`, `gpu_ane_*.py`, `w8a8_mlp_probe.py`, `capture_hidden.py` | V1 hybrid and earlier probes | same |
| `prototypes/kv_slice_probe*.py`, `view_resize_check.py`, `e2e_seg_check.py`, `e2e_server_check.py`, `hwx_share_check.py` | KV segment / view probes and checks behind commit 1 | same |
| `a8probe/` | A8 and long-context run scripts, A8 probe results (json) | `SSD/a8probe` |
| `splash132/` | splash 1.3.2 serve and bench scripts (comparison) | `C/v8_work/splash132` |
| `tools/swap_watchdog.sh` | swap guard used by every long job | `SSD/tools` |
| `results/` | server matrix, K8V16 64K run, splash runs | `SSD/coreai_seg_logs` |

Absolute paths in the scripts:

| name | path |
|---|---|
| C | `/Users/anemll/Documents/Codex/ios-prefill-20261030` |
| V02 | `/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02` (fusedqkv + the KV_SEG changes of commit 1) |
| SSD | `/Volumes/SSD4TB/anemll-forge-research` (builds `coreai_seg2/`, `v8_shared2/`, logs `coreai_seg_logs/`, `a8probe/`) |
| model | `/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K` (HF config, tokenizer, embedding, drafter `all3`) |
| export | `/Users/anemll/Models/vq27b/export/release_vq3pA_mixh_s600_k1_mat` (release VQ export), checkpoint `/Volumes/SSD4TB/Qwen3.8-27B` |
| runtime python | `anemll-forge/.venv` (server, drivers, `forge.py compile`) |
| builder python | `ane-vector-lut/coreai/.venv` (torch + Core AI tools: builds, preps) |
| journal | `deploy_journal.py` from `C/claude_full96_prep/freeze_eight_row_v4` (every Core AI load is journaled) |

Not committed: weights and GPU tables (`pipe/w_*`, 0.3 to 0.9 GB each), builds and compiled packages, dylibs, reference
logits (`v8/ref_*.npz`, from `v8_prefill.py --mode ref`), A8 probe packages and hidden-state dumps, and the `.pre_*`
source backups (every later variant is an env switch in the final `gpu_v8.swift` / `kernels.metal`).

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

## Run

```
pipe/v8_server.sh                       # OpenAI API, port 8767 (PORT=...), ctx 98304 (CTX=...), drafter on
PORT=8765 pipe/v8_server.sh             # the port Pi's ane-qwen38 provider uses
pipe/srv_stop.sh                        # stops whatever listens on $PORT
pipe/unit/srv_matrix.sh OUT "tag|opts|2000 6100 16200"    # cold + follow-up benchmark per config
```

Standalone driver (prefill-only or real end to end, KL against an ANE-only reference), e.g. 6K with the final
engine: `export BUILD=SSD/v8_shared2 MAN=pipe/w_shared2r/manifest_best.json`, then
`FUSE=all I8=1 I8T=1 I8TV=_k256 LUTC=1 I8A=1 I8KV=1 LRM=1 QK8=1 TKV=2048 pipe/unit/run_wd.sh 6144 ref_6k.npz TAG --shared --streams 8 --dylib libgpu_v8_srv2.dylib --handover --ho-overlap`
(always export BUILD and MAN: without them `run_wd.sh` falls back to another build and the bridge compiles it).
A8: `a8probe/e2e_len.sh TOKENS REF REPS` (baseline vs A8 back to back).
