# ANEMLL Forge

**A research project for inference of large dense language models on the M6 Apple Neural Engine.** ANEMLL Forge shares inference code, quantization and conversion workflows, experiments, and lessons about Core AI, Core ML, compiler behavior, and ANE architecture.

The first model is an independent ANEMLL adaptation of **[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)**, developed by the **Qwen Team / Alibaba Cloud**. The release runs a **Core AI target with a Swift bridge and the matching Core AI DFlash2 speculative drafter**. Both target and drafter are included in the model bundle. Core ML conversion code and research findings remain available for reproduction and learning.

## Hardware and software

- **M6 is the main development and performance target.** M5, M5 Pro, and M5 Max are also supported; expect roughly half the M6 throughput (about 2× slower) for comparable ANE workloads. This is approximate guidance, not a matched benchmark across every chip and context size.
- **macOS 27 with a compatible Xcode 27 / Core AI SDK.** The bridge was checked with Apple Swift 6.4. Older macOS/SDK versions without `CoreAI` cannot build this runtime.
- **Python 3.11** for inference; **Node.js 22.19 or newer** and npm for optional Pi coding sessions.
- **32 GB or more unified memory is recommended.** Start with 16K context. The paired bundle is approximately 15 GB; allow additional disk space for downloads and compilation caches (11–14 GB was observed for a cold target compilation).

First-use compilation can take minutes. M5-family users should read the [cold-compilation issue and cache pre-warming workaround](docs/SESSION_LESSONS.md#release-preparation-diagnostic-m5m5-max-cold-compile-crash-bonded-vs-non-bonded-september-29) if the compiler reports a topological-sort failure. [Environment details](docs/ENVIRONMENT.md) distinguish prepared-bundle inference from the separate conversion toolchain.

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

The server discovers the matching sibling `drafter/` and enables DFlash2 by default. Wait for the serving message, then test from another terminal:

```sh
curl --fail http://127.0.0.1:8765/health
curl --fail http://127.0.0.1:8765/v1/models
curl --fail http://127.0.0.1:8765/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-27b-ane","messages":[{"role":"user","content":"Explain vector quantization in one sentence."}],"max_tokens":64,"stream":false,"chat_template_kwargs":{"enable_thinking":false}}'
```

The API supports chat completions, streaming, and tool calls. It has no authentication and defaults to loopback. Use one inference client at a time; run the standalone smoke test before the server, rather than concurrently with it. Stop with Ctrl-C. `--plain` is a slower target-only diagnostic; it is not the recommended coding-session configuration.

`--ctx` limits the available fixed-shape context ladder. The runtime selects larger prepared ANE entries as needed, resizes and copies the cached KV prefix, and preserves recurrent state. See [speculative decoding and context expansion](docs/SPECULATIVE_DECODING.md#how-ane-context-expansion-is-implemented) for capacity, memory, and compilation costs.

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

With the server running, enter the project you want Pi to edit and start a fresh session:

```sh
cd /path/to/your/coding-project
pi --offline --provider ane-qwen38 --model qwen38-27b-ane --thinking off
```

Start with a small edit and check its diff/tests. The profile declares **16,384 total context tokens and up to 4,096 output tokens**; instructions, tools, retained reasoning, and output share that context. Use low thinking when needed. See [Pi setup, thinking budgets, and compaction](docs/PI_CODING.md), including the optional live throughput extension. `--offline` disables Pi startup network operations, not requests to the local server or network access by coding tools.

## Research, evaluation, and limitations

The repository includes mixed-bit GPTQ, scalar/vector LUT quantization, per-channel scaling, online rotations, low-rank corrections, calibration and KL evaluation, Core ML/Core AI conversion, and the speculative runtime. It also preserves failed experiments and compiler/ANE findings from `ane-vector-lut`.

KL divergence is the current model-fidelity evaluation; it does not establish coding, reasoning, or long-context capability. ANE benchmarks will be added with exact artifacts, hardware, context, generation settings, and measurement methods. Historical measurements are labeled separately from downloaded-release validation. Treat this as research software and review the documented numerical, placement, compilation, and memory limitations.

- [Quantization overview and implementation](docs/QUANTIZATION.md)
- [Quantization → conversion → inference workflows](docs/WORKFLOW.md)
- [DFlash2 integration and correctness](docs/SPECULATIVE_DECODING.md)
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
