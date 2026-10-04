# ANEMLL Forge

**A research project for inference of large dense language models on the M6 Apple Neural Engine.** ANEMLL Forge shares inference code, quantization and conversion workflows, experiments, and lessons about Core AI, Core ML, compiler behavior, and ANE architecture.

The first model is an independent ANEMLL adaptation of **[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)**, developed by the **Qwen Team / Alibaba Cloud**. The release runs a **Core AI target with a Swift bridge and the matching Core AI DFlash2 speculative drafter**. Both target and drafter are included in the model bundle. Core ML conversion code and research findings remain available for reproduction and learning.

## Hardware and software

- **M6 is the main development and performance target.** M5, M5 Pro, and M5 Max are also supported; expect roughly half the M6 throughput (about 2× slower) for comparable ANE workloads. This is approximate guidance, not a matched benchmark across every chip and context size.
- **macOS 27 with a compatible Xcode 27 / Core AI SDK.** The bridge was checked with Apple Swift 6.4. Older macOS/SDK versions without `CoreAI` cannot build this runtime.
- **Python 3.11** for inference; **Node.js 22.19 or newer** and npm for optional Pi coding sessions.
- **32 GB or more unified memory is recommended.** Start with 16K context. The paired bundle is approximately 15 GB; allow additional disk space for downloads and compilation caches (11–14 GB was observed for a cold target compilation).

The first start of a model on a given macOS build compiles every package for the ANE once (about 20 to 25 minutes for the full target on M6); later starts load from the cache in seconds. Startup prints `[ANE compile]` lines: how many packages still need compiling, an estimate, per-package progress with the time left, and build options that compile faster ([measured times and a sample readout](docs/SERVER.md#first-start-compile-time-and-readout)). Stopping is safe at any point (Ctrl-C, or `scripts/qwen38_server.sh stop`): finished packages stay cached and the next start resumes. To compile ahead without starting the server, run `python forge.py compile --build "$FORGE_BUNDLE/coreai"` with the same Python as the server (the cache is keyed by macOS build and Python: its bundle identifier, such as `org.python.python` for Homebrew's framework Python, else its executable name); add `--force` to drop that Python's cached specializations of the build and recompile every package. If a load crashes on a cached package, `forge.py compile` and `forge.py serve` purge that package and retry once. M5-family users should read the [cold-compilation issue and cache pre-warming workaround](docs/SESSION_LESSONS.md#release-preparation-diagnostic-m5m5-max-cold-compile-crash-bonded-vs-non-bonded-september-29) if the compiler reports a topological-sort failure. The runtime selects the ANE bonded compile mode by SoC generation: `1` on the M5 family (H17) and `2` on M6 and newer (H18+); pre-M5 chips are unsupported and fail before loading. An explicit `MPSGRAPH_ANE_BONDED_COMPILE_MODE` still overrides the policy, and startup and `/health` report the effective mode. See the [ANE compile mode policy](docs/ANE_COMPILE_MODE_POLICY.md). [Environment details](docs/ENVIRONMENT.md) distinguish prepared-bundle inference from the separate conversion toolchain.

## 1. Set up the inference environment

Install Python 3.11 and the compatible Xcode toolchain, then run:

```sh
git clone https://github.com/Anemll/anemll-forge.git
cd anemll-forge

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-inference.txt

python forge.py doctor
xcrun swiftc --version
bash coreai/swift_bridge/build.sh
```

The requirements file provides starting version pins for inference from prepared assets. The Core AI Swift path does not require installing the Python Core AI build SDK. Conversion and quantization have additional dependencies, including research patches described in [ENVIRONMENT.md](docs/ENVIRONMENT.md). An import check is not full-model or clean-environment validation; see [validation records](docs/VALIDATION.md).

## 2. Download the model and drafter

Model files: **[anemll/anemll-forge-qwen3.8-27B](https://huggingface.co/anemll/anemll-forge-qwen3.8-27B/tree/main)** — [Core AI target](https://huggingface.co/anemll/anemll-forge-qwen3.8-27B/tree/main/coreai), [DFlash2 drafter](https://huggingface.co/anemll/anemll-forge-qwen3.8-27B/tree/main/drafter), and [tokenizer/config/embeddings](https://huggingface.co/anemll/anemll-forge-qwen3.8-27B/tree/main/model).

**Model update (4 October 2026):** the published target packages ([revision `1192a9c`](https://huggingface.co/anemll/anemll-forge-qwen3.8-27B/tree/1192a9c83d1b3e7ad76602ed6ec6d05e6852cad2)) use the [faster exact ANE graph](#faster-ane-graph-conversion-default) and a V8-only cache: FP16 keys, INT8 values. The target weights, output head, tokenizer/embeddings and tested Core AI DFlash2 drafter are unchanged. The first start compiles the new packages once (about 23 minutes on M6). Prepared context entries are **8K, 16K, 32K, 48K and 64K**. The previous revision [`cd7dfc6`](https://huggingface.co/anemll/anemll-forge-qwen3.8-27B/tree/cd7dfc605ccad091b961f7788939c30d01c3793e) has selectable FP16/V8 packages with the earlier graph; older revisions are FP16-only. The server startup line and `/health` report which graph and cache format are loaded.

Use the helper to download the complete matching pair and verify its file inventory:

```sh
export FORGE_BUNDLE="$HOME/Models/anemll-forge-qwen3.8-27B"

python forge.py download \
  --repo anemll/anemll-forge-qwen3.8-27B \
  --revision main --runtime coreai --output "$FORGE_BUNDLE"

python forge.py quick-test \
  --bundle "$FORGE_BUNDLE" --runtime coreai --check-only
```

The helper resolves `main` to a Hub commit, downloads the target, drafter, tokenizer/config, embeddings, and license/provenance documents, and verifies hashes. Use a full Hub commit instead of `main` for reproducible runs. If access requires authentication, use `hf auth login` with your own account. Keep the bundle layout intact; original BF16 weights and Core ML model packages are unnecessary for this inference path. See the [download and bundle guide](docs/HUGGING_FACE.md).

When upgrading an existing download, choose a **new output directory**, for example `export FORGE_BUNDLE="$HOME/Models/anemll-forge-qwen3.8-27B-v8"`, before running the download and quick-test commands. The helper rejects an output directory containing a different release inventory. A runtime flag cannot add V8 inputs to an older FP16-only model package.

## 3. Run a quick inference test

Before starting a persistent server, test the target/drafter pair:

```sh
python forge.py quick-test \
  --bundle "$FORGE_BUNDLE" --runtime coreai --ctx 16384 \
  --prompt "The capital of France is" --tokens 16 \
  --report "$HOME/forge-smoke-report.json"
```

This loads both models and exercises speculative propose/verify/accept generation. The integrity-only check above does not run inference. A successful short smoke test checks basic execution; it does not establish sustained throughput or model quality.

## 4. Start the server and test the API

```sh
python forge.py serve --runtime coreai \
  --model "$FORGE_BUNDLE/model" --build "$FORGE_BUNDLE/coreai" \
  --ctx 16384 --host 127.0.0.1 --port 8765
```

The server discovers the matching sibling `drafter/` and enables DFlash2 by default. `--build` names the build directory; it does not rebuild anything. To force a fresh ANE compile of every target and drafter package (for example after changing the compile mode, or when a cached package misbehaves), recompile first, then start the server as above:

```sh
python forge.py compile --build "$FORGE_BUNDLE/coreai" --force
```

Wait for the serving message, then test from another terminal:

```sh
curl --fail http://127.0.0.1:8765/health
curl --fail http://127.0.0.1:8765/v1/models
curl --fail http://127.0.0.1:8765/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-27b-ane","messages":[{"role":"user","content":"Explain vector quantization in one sentence."}],"max_tokens":64,"stream":false,"chat_template_kwargs":{"enable_thinking":false}}'
```

Main startup options ([all options, defaults and request fields](docs/SERVER.md)):

| Option | `forge.py serve` | Wrapper variable | Default |
| --- | --- | --- | --- |
| Context cap (an entry in the build's manifest) | `--ctx` | `CTX` | `16384` |
| KV cache format | `--kv-cache-dtype` | `KV_CACHE_DTYPE` | `auto` (the build's default) |
| Speculative drafter | `--draft PATH`, or `--plain` to disable | `DRAFT` | the bundle's `drafter/` |
| Listen address and port | `--host`, `--port` | `BIND_HOST`, `PORT` | `127.0.0.1`, `8765` |

Requests choose their generation settings. Defaults: thinking off (`chat_template_kwargs.enable_thinking`); temperature 0.7, top_p 0.8, top_k 20 without thinking and 1.0, 0.95, 20 with it, following the Qwen3.8 model card; `max_tokens` 4096; `presence_penalty` 0. Speculative decoding keeps the target's sampling distribution. The card's `presence_penalty` 1.5 for non-thinking chat penalizes every repeated token and is usually unhelpful for code ([DFlash2 sampling plan](docs/research/DFLASH2_SAMPLING_PLAN.md)). Server-wide defaults such as `--think`, `--max-tokens` and `--presence` are flags of `scripts/qwen38_server.py`, not of `forge.py serve`; see [changing a server-wide default](docs/SERVER.md#changing-a-server-wide-default).

The API supports chat completions, streaming, and tool calls. It has no authentication and defaults to loopback. Use one inference client at a time; run the standalone smoke test before the server, rather than concurrently with it. Stop with Ctrl-C. `--plain` is a slower target-only diagnostic; it is not the recommended coding-session configuration.

`--ctx` limits the available fixed-shape context ladder. The runtime selects larger prepared ANE entries as needed, resizes and copies the cached KV prefix, and preserves recurrent state. See [speculative decoding and context expansion](docs/SPECULATIVE_DECODING.md#how-ane-context-expansion-is-implemented) for capacity, memory, and compilation costs.

### Optional wrapper and larger contexts

`scripts/qwen38_server.sh` wraps the same `forge.py serve` command with `start`, `stop`, `restart`, `status`, `check`, and `log` subcommands. Before loading models or stopping a server for restart, it validates the checkpoint, context, and target/drafter pairing and checks that the Swift bridge library exists, can load, and has the expected ABI. If the bridge check fails, run `bash coreai/swift_bridge/build.sh` from this checkout. The release server requires the Swift bridge for the target and drafter.

The wrapper reads `FORGE_BUNDLE` (default `~/Models/anemll-forge-qwen3.8-27B`), `MODEL`/`BUILD`, `CTX` (a number or `16K`/`64K`), `KV_CACHE_DTYPE`, `PORT`, `BIND_HOST`, `DRAFT`/`DRAFTER`, `PY`, `LOG`/`PIDFILE`, `PI_SYNC`/`PI_DIR` and `START_WAIT_S` from the environment ([defaults](docs/SERVER.md#startup-options)):

```sh
scripts/qwen38_server.sh start              # CTX defaults to 16384
CTX=64K scripts/qwen38_server.sh restart    # grow the ladder up to the 64K entry
scripts/qwen38_server.sh status             # pid + /health
scripts/qwen38_server.sh log                # follow the server log
```

Startup prints the loaded graph (`target graph: GDN_FAST=… ATT_BLOCK=… | build <path>`) and the KV format, then reports completed target chunks and elapsed time. On a first start it also shows every `[ANE compile]` line: packages still to compile, the estimate, per-package progress and time left, and options that compile faster. The wrapper keeps watching until the server is serving (`START_WAIT_S`, default 7200 seconds). Ctrl-C only stops watching: the server keeps starting in the background, already compiled packages stay cached, and a later `start` resumes the compile. Use `status` and `log` to check readiness, then sync the Pi profile with the command below if needed. `python forge.py compile --build "$BUILD"` compiles a build without starting the server; add `--force` to recompile every package.

The wrapper manages only verified processes recorded in its PID file and their matching server children. Separate instances need distinct ports and `ANEMLL_FORGE_STATE` directories (or `PIDFILE`/`LOG` paths). Stop a foreground `forge.py serve` session with Ctrl-C. `restart` keeps the previous log as `<log>.prev`.

The prepared manifest advertises 8K–64K entries; the largest usable capacity is 65,472 rows. A 64K session used roughly 25 GB for the target plus drafter in the recorded M6 experiments; validate memory on your setup and run the smoke test before starting a persistent server.

### INT8 V cache and FP16 fallback

The Core AI server supports **FP16 K with INT8 V** through `--kv-cache-dtype v8`, or `KV_CACHE_DTYPE=v8` in the wrapper. This requires target packages exported with V8 inputs and token/head scales; older FP16-only bundles cannot be switched by a runtime flag. `auto` (default) reads the selected build's manifest, while explicit `fp16` or `v8` rejects a mismatched build before loading or stopping a server. `/health` reports the active cache format. V8 remains a research option with bounded quality validation.

A plain conversion from this branch now produces V8-only packages (`_kvv8`). A `--kv-cache-dtype both` conversion contains both formats in shared-weight packages and defaults to V8; it compiles about 1.5 times longer on first start. The server's `auto` mode follows the manifest's declared default, so no precision override is needed for a new V8-default build. Select `fp16` or `v8` explicitly at startup using the same `BUILD` path; conversion can retain an FP16 default with `--kv-cache-default fp16`. Existing FP16-only bundles remain compatible. Changing formats requires a restart with an empty cache; existing prompt values are never reinterpreted in a different format.

```sh
# With the updated selectable bundle, auto uses its V8 default.
CTX=64K scripts/qwen38_server.sh restart
curl --fail http://127.0.0.1:8765/health  # kv_cache_dtype should be "v8"

# Select the FP16-cache baseline from the same bundle.
KV_CACHE_DTYPE=fp16 CTX=64K scripts/qwen38_server.sh restart
```

For a separately converted target, set `BUILD="$V8_BUILD"` and `DRAFT="$FORGE_BUNDLE/drafter/dflash2_lut4_gptq.aimodel"`. The matching tokenizer/model assets and drafter are still required.

Prefill, decode and DFlash2 verification emit FP16 new rows. The host compresses only committed V rows; keys stay FP16. Growth copies V codes and their scales together, and restore uses the same position masks and recurrent-state snapshots. Paired full-server prefill and three-repeat decode measurements completed at 8K, 16K, 32K, 48K and 64K on M6. On one synthetic workload, observed decode throughput changes ranged from −3.1% at 8K to +23.9% at 64K, including speculative acceptance differences. Compiled-model KL-512 was nearly unchanged on a short 64-sequence trace; long-context quality remains unmeasured. See [V8 conversion, measured throughput and validation](docs/KV_CACHE_V8.md).

### Faster ANE graph (conversion default)

Two exact graph changes make the target faster on the M6 ANE without changing weights, cache formats or the runtime. `GDN_FAST=1` rewrites the Gated DeltaNet core: a closed-form triangular solve for the chunkwise delta rule's UT transform (background: [DeltaNet Explained, Part II](https://sustcsonglin.github.io/blog/2024/deltanet-2/)), a native depthwise conv1d and one fewer state matmul per prefill sub-chunk. `ATT_BLOCK=2048` / `ATT_BLOCK_PREFILL=4096` compute attention over the KV history in narrower tiles. Both are the converter defaults, together with a V8-only cache, and the published packages use them since revision `1192a9c`; `GDN_FAST=0 ATT_BLOCK=16384` rebuilds the release graph.

Measured on M6 against the release graph (complete target and head, V8 cache):

| Context entry | Verify call | 64-row prefill call |
| --- | --- | --- |
| 8K | 113.1 → 91.5 ms (−19%) | 284.5 → 202.8 ms (−29%) |
| 32K | 144.8 → 108.1 ms (−25%) | 412.6 → 285.1 ms (−31%) |
| 64K | 190.3 → 131.5 ms (−31%) | 603.4 → 394.1 ms (−35%) |

With DFlash2, server cold prefill was 26 to 41% faster and decode 21 to 25% faster at 32K to 64K on a synthetic coding workload (measured on a two-format build with the same changes; the default build's per-call times are lower still). Compiled KL-512 against BF16 is unchanged (mean 0.18427 → 0.18417), and a 64K long-context check matches the release graph (perplexity 5.0686 → 5.0673). The first start compiles in about 23 minutes, like the release graph. Check what a server loaded with its startup line or `curl -s localhost:8765/health` (`target_graph`). Details, method and limits: [M6 compute acceleration](docs/research/M6_COMPUTE_ACCELERATION_2026-10-03.md). An 8-row verifier remains the fastest end to end ([verifier block length](docs/verifier_len.md)).

## 5. Apply the Pi configuration patches

The supplied Pi changes configure the local provider, Qwen thinking/tool replay, sampling, and 16K context compaction. **Pi 0.87.1 already supports these settings; no Pi source-code patch is required for this version.** Install that version into a dedicated directory:

```sh
export FORGE_PI_ROOT="$HOME/.local/share/anemll-forge/pi"
npm install --prefix "$FORGE_PI_ROOT" @earendil-works/pi-coding-agent@0.87.1
export PATH="$FORGE_PI_ROOT/node_modules/.bin:$PATH"
export PI_MODULES="$FORGE_PI_ROOT/node_modules"

node examples/pi/validate_pi.mjs
```

Apply the supplied configuration to a new profile. `mkdir` intentionally fails if that profile already exists; choose a new directory or merge changes into an existing profile yourself:

```sh
export PI_CODING_AGENT_DIR="$HOME/.pi-forge"
mkdir "$PI_CODING_AGENT_DIR"
cp examples/pi/models.json examples/pi/settings.json "$PI_CODING_AGENT_DIR/"
export PI_LIVE_THROUGHPUT_DIR="$PI_CODING_AGENT_DIR"

pi --offline --list-models qwen38
```

If the server runs with a larger `--ctx`, update the declared window and compaction point so Pi sends at most what fits. `scripts/qwen38_pi_config.py` edits only the `ane-qwen38/qwen38-27b-ane` entry of a profile (backing up the previous files as `*.prev-qwen38`):

```sh
python scripts/qwen38_pi_config.py --ctx 65536 --pi-dir "$PI_CODING_AGENT_DIR" --build coreai
```

`--pi-dir` defaults to `~/.pi/agent`. The helper accepts the prepared 8K/16K/24K/32K/48K/64K windows, validates both profile files before writing, preserves unrelated settings, and uses atomic file replacements with backups and rollback on write failure. It sets `contextWindow`, `maxTokens` (`clamp(ctx/4, 2048, 16384)`) and the per-model compaction override (`reserveTokens`, `keepRecentTokens`). Reopen `/model` in Pi to reload models.json, and restart Pi after changing compaction settings. `scripts/qwen38_server.sh` runs this sync after startup reaches the serving message unless `PI_SYNC=0`.

With the server running, enter the project you want Pi to edit and start a fresh session:

```sh
cd /path/to/your/coding-project
pi --offline --provider ane-qwen38 --model qwen38-27b-ane --thinking off
```

Start with a small edit and check its diff/tests. The profile declares **16,384 total context tokens and up to 4,096 output tokens**; instructions, tools, retained reasoning, and output share that context. Use low thinking when needed. See [Pi setup, thinking budgets, and compaction](docs/PI_CODING.md), including the optional live throughput extension. `--offline` disables Pi startup network operations, not requests to the local server or network access by coding tools.

For Droid, see the [ANE Qwen reasoning configuration example](docs/DROID_CODING.md). Its fixed High preset targets a 64K server and must be merged with existing Droid settings.

## Research, evaluation, and limitations

The repository includes mixed-bit GPTQ, scalar/vector LUT quantization, per-channel scaling, online rotations, low-rank corrections, calibration and KL evaluation, Core ML/Core AI conversion, and the speculative runtime. It also preserves failed experiments and compiler/ANE findings from the original research, with source provenance recorded in [docs/provenance.json](docs/provenance.json).

KL divergence is the current distributional-fidelity evaluation; it does not establish coding or reasoning capability. An [M6 NIAH retrieval pilot](docs/research/NIAH_PILOT_2026-10-03.md) passed exact match at four paired FP16-KV/V8 placements, but has one trial per case and is not an official RULER score. M6 full-server prefill and decode measurements for the V8 option are recorded in the [KV-cache quantization research report](docs/research/KV_CACHE_QUANTIZATION_2026-10-02.md), and those for the faster graph in the [M6 compute acceleration report](docs/research/M6_COMPUTE_ACCELERATION_2026-10-03.md), including context sizes, sampling, acceptance, timing boundaries and validation limits. Historical measurements are labeled separately from downloaded-release validation. Treat this as research software and review the documented numerical, placement, compilation, and memory limitations.

- [Quantization overview and implementation](docs/QUANTIZATION.md)
- [Quantization → conversion → inference workflows](docs/WORKFLOW.md)
- [DFlash2 integration and correctness](docs/SPECULATIVE_DECODING.md)
- [V8 cache setup and conversion](docs/KV_CACHE_V8.md) and [KV-cache quantization research trace](docs/research/KV_CACHE_QUANTIZATION_2026-10-02.md)
- [M6 compute acceleration](docs/research/M6_COMPUTE_ACCELERATION_2026-10-03.md) (faster DeltaNet and attention graph, compile cost), [verifier block length](docs/verifier_len.md) and the [DFlash2 sampling plan](docs/research/DFLASH2_SAMPLING_PLAN.md)
- [Techniques and limitations](docs/TECHNIQUES.md)
- [Session lessons and troubleshooting](docs/SESSION_LESSONS.md)
- [Quality benchmark plan](docs/BENCHMARK_PLAN.md)
- [Serving-session performance analysis](docs/PERFORMANCE_SESSION.md)
- [Experiment guide](docs/EXPERIMENTS.md) and [M3U pipeline archive](pipelines/m3u/README.md)
- [Release checklist](docs/RELEASE.md) and [validation records](docs/VALIDATION.md)

## Contributing

Bug reports, documentation improvements, reproducible experiments, and focused pull requests are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for setup, validation, benchmark reporting, and handling sensitive trace data.

## License and attribution

Independently authored ANEMLL Forge code and documentation are licensed under **[MIT](LICENSE)**. Third-party code and assets retain their applicable upstream licenses and notices.

The **Qwen-derived model artifacts follow Alibaba Cloud / Qwen's upstream Apache 2.0 license and redistribution requirements**; they are not relicensed under MIT. The bundle preserves the [Qwen license](release/huggingface/LICENSE), [attribution notice](release/huggingface/NOTICE), and [modification record](release/huggingface/MODIFICATIONS.md). The DFlash2 drafter carries its own upstream license and notices. Read [model attribution and redistribution requirements](docs/ATTRIBUTION.md). This is an independent ANEMLL project; no endorsement by Qwen, Alibaba Cloud, the drafter authors, or Apple is claimed.
