# Build, deploy and test: V8-96K tree8 + sliced build (`coreai/m6-96k-dec8ts`)

Research preview, 22 Nov 2026. This is the configuration served on the development M6 since 22 Nov: the hybrid
prefill (FFN on the Neural Engine, token mixers on the GPU), tree verification (tree8) for greedy and sampled
requests, sliced decode entries at 24K / 40K / 48K / 56K, the decode attention split onto the GPU at the 96K entry,
and the KV-wrap memory fix of 22 Nov. Everything below runs on the Mac that serves; nothing is uploaded anywhere.

Coding agents: [AGENTS.md](AGENTS.md) and `hf/agent_deploy.sh` run these steps as stages with exit codes.

Paths: `REPO` is the downloaded Hugging Face repository, `CODE=$REPO/code/anemll-forge` the source snapshot inside
it, `V=$CODE/research/v8_prefill`.

## 1. What is in the repository

| path | what |
|---|---|
| `coreai/m6-96k-dec8ts/` | the build: 8 Core AI packages of 8 layers, 81 entry points each on one copy of the weights (target verify and prefill per context, hybrid prefill parts, decode split `dva` / `dvba` / `dvb`, tree verify `t8_*`, sliced verify `v8s_*` / `t8s_*`), the output head `head_T8.aimodel`, the server manifest `manifest.json` (16 logical 4-layer chunks) and the file-level `manifest_files.json` |
| `gpu/w_dec8ts/` | the GPU-table kit: the tables of the served M6 (1344 files, 0.9 GB), `kit.json` (sha256 of every file, and per package the compiled program's weight section: offset, size, sha256, macOS build), manifests with the machine paths removed |
| `drafter/all3/` | the DFlash2 drafter (unchanged from release 0.2.1) |
| `model/` | tokenizer, configuration, FP16 host embedding table (release 0.2) |
| `code/anemll-forge/` | the source snapshot: runtime and builder (`scripts/`, `coreai/`, `forge.py`), the server and GPU engines (`research/v8_prefill/pipe/`), build scripts (`research/v8_prefill/v8/`), this file |
| `coreai/m6-96k-dec8/`, `coreai/m6-96k-v8/` | the earlier builds (17 Nov chain-only SPLIT+GPU, 16 Nov hybrid): kept for comparison, not needed for this configuration |

## 2. Requirements

- **M6 Mac, 32 GB, macOS 27.3.** The kit's tables were made on build 26C5047e. Other chips and macOS builds are
  untested; on another macOS build the compiled programs differ and the tables are rebuilt from the export (step 5b).
- **SIP off and one ACL grant** so the GPU can map the Neural Engine's compiled weights: see
  [README: Requirements, root and SIP](../README.md#requirements-root-and-sip). Use a dedicated research Mac.
- **Disk:** about 15 GB for this build plus head, drafter, model and tables (download only what step 3 lists), plus about
  25 GB on the internal disk for the compile caches (Core AI and aned, about 2.7 GB per package).
- **Xcode command line tools** (`swiftc`, Metal) for the engine dylibs and the Core AI Swift bridge.
- **Runtime Python** (the server and `forge.py`): a venv with `requirements-inference.txt` of `code/anemll-forge`.
  The compile cache is keyed by the executable name: compile and serve with the same `python`.
- **Only for a full table rebuild (step 5b):** the builder Python (`requirements-conversion.txt`: torch and the Core
  AI Python tools) and the public release 0.2 export,
  [anemll/anemll-quantized-qwen3.8-27b-for-CoreAI](https://huggingface.co/anemll/anemll-quantized-qwen3.8-27b-for-CoreAI)
  (`export/release_vq3pA_mixh_s600_k1_mat/` and `model/`).
- **Only for the bench in step 7:** `mactop` (power sampling; `brew install mactop`).

## 3. Download

```
hf download anemll/anemll-forge-qwen3.8-27B-V8-96K --local-dir ~/Models/anemll-forge-qwen3.8-27B-V8-96K \
  --include "coreai/m6-96k-dec8ts/*" "gpu/w_dec8ts/*" "drafter/*" "model/*" "code/*" "*.md" "*.json" LICENSE NOTICE
export REPO=~/Models/anemll-forge-qwen3.8-27B-V8-96K CODE=$REPO/code/anemll-forge V=$REPO/code/anemll-forge/research/v8_prefill
```

## 4. Build on the Mac

a. **Core AI Swift bridge** (once, as for release 0.2): `cd $CODE && bash coreai/swift_bridge/build.sh`
   (`code/anemll-forge/README.md`: macOS 27 with a matching Xcode 27 / Core AI SDK).

b. **Compile for the Neural Engine** (once per macOS build; about 30 min, 8 packages + head, about 3 min each, then
   seconds per load). The drafter is compiled with it:
   ```
   cd $CODE && MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 python forge.py compile --build $REPO/coreai/m6-96k-dec8ts \
     --draft $REPO/drafter/all3/dflash2_lut4_gptq_nohead.aimodel
   ```
   One compile at a time. Keep at least 25 GB free on the internal disk.

c. **GPU engine dylibs** (Swift; their Metal kernels compile at load):
   ```
   zsh $V/pipe/engine/build_engines.sh uw
   ```
   It builds `libgpu_v8_f8q_uw.dylib` (prefill mixers and attention) and `libdec_gpu_tree_uw.dylib` (decode attention
   with tree masks) in `pipe/engine/`, checks their exported C API and prints each one's sha256 (the server banner
   shows the first 16 hex digits). The builds are not byte-reproducible (each gets a new LC_UUID). `OUT=dir` builds
   elsewhere: never rebuild a dylib in place while a server has it loaded.

## 5. GPU tables

The GPU engine reads the palettized mixer weights straight from the compiled programs in aned's cache, so its
tables hold offsets into this Mac's `model.hwx` files.

a. **Install the kit** (seconds; needs the compile of step 4b and the ACL grant):
   ```
   python $V/pipe/gpu_tables_kit.py install $REPO/gpu/w_dec8ts $REPO/coreai/m6-96k-dec8ts $REPO/gpu/w_dec8ts_local
   ```
   For each package it finds the local `model.hwx` through the Core AI cache, checks that its weight section has the
   kit's offset, size and sha256 (the same compiled program, so the shipped offsets are valid), copies the files with
   a sha256 check and writes `manifest.json` / `manifest_best.json` with the local paths. Every line should say
   `MATCH`.

b. **Exit 2 (`DIFFERS`): rebuild from the export** (25 to 40 min; another macOS build or compiler):
   ```
   BUILD=$REPO/coreai/m6-96k-dec8ts OUT=$REPO/gpu/w_dec8ts_local EXPORT_DIR=<export>/release_vq3pA_mixh_s600_k1_mat \
   MODEL=<export repo>/model BPY=<builder python> PY=python V8_V02=$CODE zsh $V/pipe/build_gpu_tables.sh
   python $V/pipe/gpu_tables_kit.py verify $REPO/gpu/w_dec8ts_local $REPO/gpu/w_dec8ts
   ```
   The verify step names the files that differ from the kit's; only `.tab` files (offsets) and the manifests' HWX
   paths are expected to change.

The tables are tied to the compiled programs: after a macOS update or an aned cache purge, compile again (4b) and
repeat step 5.

## 6. Deploy

```
REPO=$REPO PORT=8765 zsh $V/hf/serve_v8_96k.sh
```

The launcher runs the served configuration of the development Mac with every path a variable (`TABLES`, `PORT`,
`LOGDIR` (default `~/Library/Logs/anemll-forge`), `PY`, `CTX`, `RUNG_SLICE`). It starts forge's memory watchdog
(`tools/forge_watchdog.sh`: stops the server above `FORGE_FP_MAX_MB`, default 12288 MB of footprint) and journals every
Core AI load (`tools/deploy_journal.py`, under `DEPLOY_JOURNAL_ROOT`, default `~/.anemll-forge`). It serves an
OpenAI-compatible API on `127.0.0.1:$PORT` (`/v1/chat/completions`, streaming, tools; `/health`). Once compiled,
loading takes seconds (5 s on the development M6 with the files in memory).

Switches (environment, one launch): `RUNG_SLICE=` (empty) serves the build without the sliced entries; extra options
go to `v8_server.py` (`python $V/pipe/v8_server.py --help`). Stop: `PORT=8765 zsh $V/pipe/srv_stop.sh`.

## 7. Test

a. **Banners.** The log (`$LOGDIR/v8_server.log`) should show, before `serving OpenAI API`: `v8: variant ... qk=attn_qk8isk_tg`
   with `libgpu_v8_f8q_uw.dylib`, `dec8: BANNER decode GPU dylib .../libdec_gpu_tree_uw.dylib`, `tree: mode=tree8
   policy=branched ... sampled=on`, `slice: sliced decode entries [24, 40, 48, 56]K` and `lookup off`.

b. **Smoke** (2 greedy requests, 1 min):
   ```
   python $V/bench_tree/tree_bench.py --prompts $V/bench_tree/smoke_2k_v1.json --rungs 2k --tasks 1,8 \
     --label smoke --out /tmp/smoke.json --log $LOGDIR/v8_server.log --idle-s 5
   ```
   On the development M6 (macOS 26C5047e) this prints `2k SUMMARY tok/s 36 to 37 ..., tok/verify 3.978, output
   checksum 29aef2ec9db18a3f` every time. The same checksum on your Mac means the same replies token for token. A
   different one on the same macOS build points at a setup problem (tables, dylibs, build); on another macOS build the
   compiler may change low-order numerics.

c. **Memory over context switches** (about 10 min; the 22 Nov leak test):
   ```
   python $V/serve_t/memleak_alt.py /tmp/memleak.json --cycles 3 --big 56000
   ```
   It alternates cold 2K and 56K requests and records forge's footprint after each. With the `_uw` dylibs the
   footprint returns to about 4.4 GB after every 2K request (results: `results/22nov/memleak/README.md`). It expects
   the server on port 8765 and reads `model/tokenizer.json` from the development path: edit `TOK` / `SRC` at the top
   for your Mac.

d. **Longer benches.** `bench_tree/tree_bench.py` with `long_agent_v1.json` (a 19-turn Pi-like session, 8.6K to
   55.5K), `u1_code_agent.json` (agent set, tools) and the realistic set (`make_prompts.py`) are the sets behind the
   numbers in `TREE_VERIFY.md`; `--temp 1.0 --think 1 --seed N` gives Pi-like sampled requests.

## 8. Rebuilding the build itself

`coreai/m6-96k-dec8ts` was made from the release 0.2 export by `v8/build_dec8ts_all.sh` (export of each 8-layer
package by `v8/build_dec8t_one.sh` -> `v8/build_dec8t_chunk.py` with `TREE_FORM=corrx0`
`TREE_SLICED=24576,40960,49152,57344` and the v8_shared2 environment, journaled compiles, then the tables
(`v8/build_dec8ts_tables.sh`) and the server manifest (`v8/dec8_srv_manifest.py`)). About 4 h on the M6 with the
builder Python. These scripts carry the development machine's absolute paths (table in
[README: Layout](../README.md#layout)); adjust them to your layout.

## 9. Known gaps

- The research scripts other than `hf/serve_v8_96k.sh`, `pipe/gpu_tables_kit.py`, `pipe/build_gpu_tables.sh` and
  `pipe/engine/build_engines.sh` still use the development machine's absolute paths.
- Tested only on the M6 that made it; nobody has yet run steps 3 to 7 on a second Mac.
- After a 96K request a later small one keeps one 96K KV set resident (about 2.8 GB) until the next 96K request
  replaces it; bounded, not a leak (`results/22nov/memleak/README.md`).
