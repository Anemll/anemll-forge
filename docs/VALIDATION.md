# Initial port validation — 2026-09-29

Latest downloaded-release check: [release 0.2 on the base M5, October 10](VALIDATION_M5_RELEASE_0_2.md), including cold compilation, 8K/16K server profiles, ANE placement and streaming API checks.

Passed:

- Parsed all 107 Python files with `ast.parse` (no model imports or execution).
- Eight standard-library launcher tests: explicit paths including spaces, final numerical recipe, existing-output protection, unsupported model shape rejection, loopback serving, checkpoint-specific embedding cache, dry-run behavior, quantization input/tag checks, supported Core AI contexts and ambiguous Core ML manifest rejection.
- CPU GPTQ/export round trips for vector 2×16 + per-channel scale, scalar LUT4 + per-channel scale and INT8 per-channel. Reconstructed exported weights match the quantized tensors within fp16 export tolerance on a small synthetic fixture.
- Server CLI help and Core ML model-module import in the existing research environment. The imported builder reports `MLP_SILU=tanh`.
- Compared source hashes for the imported research snapshot: source bytes unchanged by this port. Provenance now covers 130 source/document/data/dependency-snapshot files including the later synchronization.
- Retrieved all 28 M3U pipeline scripts, verified local SHA256 values against remote originals, and passed `zsh -n` for all of them; no pipeline was executed.
- Basic known credential-pattern scan of source/document files: no matches. This is not a comprehensive secret or provenance audit.

The import emits compatibility warnings: installed scikit-learn 1.9.1 exceeds coremltools' supported conversion range, and installed PyTorch 2.14.0 is newer than its tested 2.8.0 version. The tested quantization helper uses sklearn directly and passed its small fixture; that does not establish converter compatibility.

Not run: full 27B quantization/conversion, ANE compilation/placement, Swift/Core AI bridge compilation, model inference, historical statistical sampling experiment, long-run memory/quality tests, or the two-hour session analysis from raw logs. Historical timings and quality metrics remain reported findings.

Commands:

```sh
python -m unittest discover -s tests -p test_launcher.py -v
# In the compatible ML environment:
python -m unittest discover -s tests -p test_quantization.py -v
python scripts/qwen38_server.py --help
PYTHONPATH=scripts python -c 'import qwen38_ane_model as m; print(m.C.MLP_SILU)'
```

## Hugging Face bundle tooling — 2026-09-29

The updated suite passed **35 tests** in the existing research environment; all **111 Python files** parsed. The same coremltools dependency-version warnings remain. These tests use small fixtures and mocked Hub/runtime interfaces, not uploaded weights or full 27B inference.

- Fifteen portable bundle tests cover both runtime layouts, SHA256 corruption detection, complete ordered layer coverage, context/verify-width checks, embedding headers, hostile paths, compiled-package overrides, pinned download revisions, selected components and safe reports.
- Eight mocked inference tests use real NumPy with fake tokenizer/runtime modules: bounded greedy generation, early stop, finite logits, correct vocabulary shape, visible output, prompt capacity, token IDs, Core AI context selection and bridge path selection.
- Two embedding-loader tests use small real NumPy arrays and the actual checkpoint class. Prepared embeddings load read-only through mmap without an original weight index or shards; wrong dtype, shape or storage order fails.
- Nine launcher tests include selecting a published embedding table; the existing CPU quantization/export roundtrip test also passes.

```sh
# Portable tooling only:
python -m unittest discover -s tests -p test_launcher.py -v
python -m unittest discover -s tests -p test_hf_release.py -v
# In the compatible ML environment, including mocked inference:
python -m unittest discover -s tests -v
```

No real Hugging Face download, upload, full-model smoke test, or new ANE placement/performance test was run. Run the [bundle workflow](HUGGING_FACE.md) after the weights are uploaded; a real hardware PASS is still outstanding.

## Attribution and upload preparation — 2026-09-29

The updated suite passed **41 tests** in the existing research environment. Six additional bundle tests verify the default HF destination, mandatory license/source documents, document hashing/download selection, symlink rejection and per-file modification notices. All five prepared release documents are included in the verified download inventory.

The exact copied Qwen LICENSE matches the pinned upstream bytes and preserves `Copyright 2026 Alibaba Cloud`. Ten local checkpoint documents/config/tokenizer files match official upstream hashes at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; all 18 shard cache etags match upstream SHA256 metadata. The large shard bytes were not rehashed. [QWEN_SOURCE.json](../release/huggingface/QWEN_SOURCE.json) records this evidence and remaining artifact-lineage limits.

A separate upload agent created and verified the HF target `anemll/anemll-forge-qwen3.8-27B`, staged the Core AI source packages plus embeddings and tokenizer/config assets, and passed `release-manifest` and `quick-test --check-only` on **65 inventoried files / 13,184,004,271 bytes**. Upload was then started; this staging check does not confirm completed transfer or downloaded-artifact inference. The model card describes a research project for the M6 Apple Neural Engine, with KL as the current evaluation and future ANE benchmarks left unclaimed. Full hardware inference remains outstanding.

## Expanded release scope: DFlash2 required

The intended release and performance benchmark include the matching Core AI DFlash2 drafter. Earlier target-only staging counts, integrity checks and plain-generation tests above remain historical records; they do not validate this expanded bundle. Completion requires paired-artifact inventory/download checks and actual speculative execution, acceptance/stop/cache/context tests and measured target-plus-drafter memory and latency. No new hardware result is asserted by this documentation update. See [SPECULATIVE_DECODING.md](SPECULATIVE_DECODING.md).

## DFlash2 release integration checks (2026-09-29)

The applied release launcher, inventory/downloader and speculative smoke helpers passed **56 focused standard-library tests** in the existing research environment. All **112 Python sources** parsed, local current-guide links resolved, and `git diff --check` passed. Coverage includes default Core AI pairing, explicit plain diagnostics, selector extraction/header checks, target/head/tap associations, finite tensor contracts, proposed-prefix rejection and commit/cap/stop boundaries. Mocked inference tests do not establish actual ANE placement or performance.

The real staged paired bundle passed manifest generation and the default speculative integrity-only check: **74 inventoried files / 15,074,603,597 bytes** before `release.json` itself. This checks file hashes, shapes and pairing; `inference_run` remains false for that check. Main-model artifacts were unchanged from the earlier target-only upload. The separate Terminal-Bench pilot uses a recorded runtime snapshot and live deployed artifacts, not a full downloaded-bundle validation; its results and environment limitations must be reported separately.

The Pi 0.87.1 integration validator passed configuration schema checks, off/low/medium/high request payload checks, reasoning/tool replay and context clamping against the installed Pi modules. All requests used mocked fetch; no live model call or coding-task success is claimed by that check.

## Context-switch lifetime repair — 2026-09-29

Four mocked-native buffer ownership tests passed after removing the Python buffer's self-retaining NumPy view. A guarded live test then showed that this fix alone was insufficient; native IOSurface autorelease retention remained. Tiny no-model probes isolated the required pool scopes to `cai_buffer_create` and `cai_buffer_address`. The Swift bridge with both scopes compiled using Apple Swift 6.4 targeting macOS 27, and the native allocation/release probe passed without a Python-side pool.

The corrected bridge and historical runtime snapshot then passed a bounded full-target-plus-GPTQ-DFlash2 probe on M6: **32 transitions between 8K and 16K**, exact preservation of 72 KV rows, identical target logits, finite drafter outputs and no surviving old buffer owners. Process physical footprint grew about **4.3 MB** across the completed roundtrips; the transient sampled peak was about **4.21 GB**. This is process accounting, not whole-system ANE/model memory. [Exact results and build provenance](results/coreai_buffer_lifetime_2026-09-29.json) retain the earlier failed diagnostic as well. No multi-hour serving validation, downloaded-bundle validation or new capability score is claimed. The benchmark remains paused at the owner's request.

```sh
python -m unittest discover -s tests -p test_bridge_ownership.py -v
# Required after pulling the Python and native source fixes:
bash coreai/swift_bridge/build.sh
```

## Public research guide and inference setup — 2026-09-29

The README now documents environment setup, the paired model download, bridge build, speculative smoke test, server/API checks, and Pi configuration. Current release guides and the local model-card template no longer describe a private beta. Independently authored ANEMLL code uses MIT; the Qwen and drafter licenses/notices remain separate.

Documentation validation checked **80 local links**, including fragments, **28 shell blocks** with `bash -n`, and **five README Forge commands** against the actual argument parser. The example chat-completion JSON parsed successfully and uses the server's documented model ID. All nine starting dependency pins are published on PyPI and accept Python 3.11 where `requires_python` metadata is specified. `git diff --check` passed.

The Pi 0.87.1 mock validator passed off/low/medium/high request mapping, reasoning/tool replay and the context clamp. The installed model-config, OpenAI-completions, simple-options and transcript modules were byte-identical to the published npm packages. This integration therefore needs the supplied configuration changes, not a Pi source-code patch.

The prepared-bundle host modules (`qwen38_ane_model`, `qwen38_coreai_model`, and `dflash2_coreai_drafter`) imported with the public coremltools 9.0 wheel substituted into the existing research environment. Converter warnings about newer PyTorch/scikit-learn remain. This was an import-only check: no new model, server, upload or hardware benchmark was run. A fresh-environment install and full-model test with the new inference pins remain outstanding.

## V-only INT8 KV cache and selectable packages — 2026-10-02

The portable suite ran 115 tests in the existing inference environment: **114 passed**, and the GPTQ vector-codebook roundtrip required scikit-learn, which is deliberately absent there. That remaining test passed separately in the quantization environment for its vector, scalar-LUT and INT8 fixtures. No dependency was added to inference for this check.

V8 coverage includes accepted counts 0/1/7/8, valid prefill rows, exact key copies, untouched rejected tails, stored FP16 scales, manifest/physical-function selection, dtype rejection, code/scale growth and snapshot restore. Launcher/wrapper tests verify that a cache-format mismatch fails before a restart stops a running server. Placement-audit tests select only the requested format from an intact selectable manifest. These fixture checks do not load native models.

Separate recorded native M6 checks completed before PR preparation: all five 8K/16K/32K/48K/64K FP16/V8 pairs, three cached decode repeats per mode/context, finite logits, within-mode reply equality, strict selected-function cached-placement audits, and owned server exits. The completed prototype prefill sweep and 64-sequence KL-512 pass remain distinct from the final selectable export. Both final selectors were bit-exact against their prototype on the first three teacher-forced sequences (2,299 positions); this is not a complete rerun of the 64-sequence KL trace.

The [dated research trace](research/KV_CACHE_QUANTIZATION_2026-10-02.md) preserves aggregate measurements, timing boundaries, numerical caveats and remaining work. The V8 manifest default was changed after evaluation without changing model packages or the explicitly selected benchmark modes. Logical cache-buffer saving is not measured resident-memory saving. Physical bonded-cluster allocation, broader capability/long-context quality and clean-environment validation remain outstanding. Heavy model files and raw prompts/traces remain outside the source repository.

```sh
# Inference environment:
python -m unittest discover -s tests -v
# Quantization environment, with scikit-learn:
python -m unittest discover -s tests -p test_quantization.py -v
```

## Faster ANE graph, new converter defaults and first-start guidance (2026-10-03)

Host suite in the inference environment: **180 tests, 179 passed**. The remaining failure, `test_mil_skeleton_without_coremltools_does_not_build`, assumes coremltools is absent and fails the same way on the unchanged tree. New host tests: [test_gdn_fast.py](../tests/test_gdn_fast.py) (the Neumann-product inverse, the `GDN_FAST` graph against the release graph for verify and prefill, and 8-, 4- and 3-row verify blocks with partial acceptances against an FP64 token-by-token gated delta rule; a deliberate 1e-3 corruption of the fast solve fails it), [test_coreai_compile_guide.py](../tests/test_coreai_compile_guide.py) (estimates against measured compiles, hints, cache detection, heartbeat, error propagation, Ctrl-C exit with code 130) and [test_target_graph.py](../tests/test_target_graph.py) (startup and `/health` report of the graph settings).

Native M6 checks, macOS 27.0.1 (26A434), recorded in [M6_COMPUTE_ACCELERATION_2026-10-03.md](research/M6_COMPUTE_ACCELERATION_2026-10-03.md): strict cached-graph audits of all 16 chunks and the head fully on the ANE with compile mode 2; compiled KL-512 on the 64-sequence BF16 trace for the two-format candidate (release rerun bit-identical to the V8 study; candidate mean KL 0.184168 against 0.184270; direct KL between the builds 3.4e-5 nats; not rerun on the default V8-only build); a 64K-entry long-context comparison after a 64,000-token prefill for both the candidate and the default build (perplexity within 0.03% of release, direct KL about 5e-5); ten paired full-server cases with DFlash2 at 8K to 64K (strict audits, within-build reply equality, owned server exits); full-chain timing of the release, two-format candidate and default V8-only builds; cold compile of the default build through `forge.py compile` (22 min 48 s) and a warm server start through the wrapper.

Published as Hugging Face revision `1192a9c83d1b3e7ad76602ed6ec6d05e6852cad2` (4 October 2026). Before upload, the staged bundle passed `release-manifest`, `quick-test --check-only` and a full speculative `quick-test` run with the public Forge source `6e0a85c` (16 tokens, coherent output). After upload, all 74 inventory files at that revision matched their recorded sizes and SHA-256 (LFS) or git blob hashes through the Hub API.

SoC compile-mode policy (4 October 2026): implemented with mocked-detection unit tests; **not** validated end-to-end. The only positive runtime evidence is one successful base-M5 `forge.py serve` with `MPSGRAPH_ANE_BONDED_COMPILE_MODE=1` (maintainer). M5 Pro/Max are a maintainer directive without an in-tree reproduction; M6 remains on mode 2. See [ANE compile mode policy](ANE_COMPILE_MODE_POLICY.md).

Not covered: a clean-environment install, M5-family hardware, a full `forge.py download` of the published revision, long-run memory and repeated context transitions on the new graph, sampled (temperature above 0) quality, and broader capability or retrieval evaluation.
