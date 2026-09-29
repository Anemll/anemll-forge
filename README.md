# ANEMLL Forge

Prepare, quantize, convert, and run language models on Apple's Neural Engine—and explain what we learned along the way.

**Private release preparation. Not yet a validated public release.** The initial target is **[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)** by the **Qwen Team**, targeting **M6 ANE**. Original model copyright: **Alibaba Cloud**, licensed under **Apache 2.0**. ANEMLL provides an independent quantization, conversion and runtime adaptation. The source uses the `qwen3_5` architecture. See [Qwen attribution and pinned source evidence](docs/ATTRIBUTION.md).

The release runtime is **Core AI with the Swift bridge**. Hugging Face will hold Core AI chunk/head packages and matching config, tokenizer and embedding assets. Core ML packages are not part of the planned release upload; Core ML source and findings remain included for conversion, experiments and learning.

Planned Hugging Face repository: **`anemll/anemll-forge-qwen3.8-27B`**. The model card and license/notice files are prepared under [release/huggingface/](release/huggingface/); this repository does not create or publish the Hub model automatically.

This first port contains working-tree research source from `ane-vector-lut`, including its previously untracked Qwen and Core AI work. It includes no model weights, compiled binaries, private calibration sessions, or generation logs. Historical performance and quality results are retained as reported measurements; they have not been reproduced by this port.

## Start here

- [Workflow: quantization → conversion → inference](docs/WORKFLOW.md)
- [Quantization: basic overview and detailed implementation](docs/QUANTIZATION.md)
- [Quality benchmark plan and low-bit comparisons](docs/BENCHMARK_PLAN.md)
- [Core AI release bundle: Hugging Face upload, download and smoke test](docs/HUGGING_FACE.md)
- [Qwen attribution and redistribution requirements](docs/ATTRIBUTION.md)
- [Techniques, findings, and limitations](docs/TECHNIQUES.md)
- [Lessons recovered from the research session](docs/SESSION_LESSONS.md)
- [Experiment navigation and reproduction gaps](docs/EXPERIMENTS.md)
- [Original M3U pipeline archive](pipelines/m3u/README.md)
- [Two-hour serving session performance and interpretation](docs/PERFORMANCE_SESSION.md)
- [Environment and dependency provenance](docs/ENVIRONMENT.md)
- [Release preparation and outstanding work](docs/RELEASE.md)
- [Import provenance](docs/provenance.json)
- [What has and has not been validated](docs/VALIDATION.md)

```sh
python forge.py doctor
python forge.py --help
```

`forge.py` adds explicit input/output paths and a documented Core ML numerical recipe. `scripts/` retains the original module names so references and experiments remain recognizable. Direct script entry points are research interfaces; some still require local artifacts or tools described in the experiment guide.

## Included

- **Inference:** Core ML chunk runtime, chat CLI, local HTTP server, Core AI runtime and Swift bridge, optional DFlash2 research implementation.
- **Quantization:** scalar/vector LUTs, per-channel scaling, GPTQ, online Hadamard rotations, sensitivity planning, block reconstruction, low-rank corrections, calibration generation and KL evaluation.
- **Conversion:** Core ML MIL graphs and Core AI conversion, including lazy DeltaNet state handling and host-managed KV caches.
- **Learning material:** successful and failed experiments, numerical debugging, memory behavior, compiler placement, and the distinction between observations and architectural hypotheses.

The local HTTP server has no authentication; the port defaults to loopback. Optional speculation requires separate drafter assets and is not enabled by the launcher.

The Qwen-derived model artifacts carry the upstream Apache 2.0 license. A license for independently authored ANEMLL source code has not been selected yet. See [licensing and release gates](docs/RELEASE.md) before distribution.
