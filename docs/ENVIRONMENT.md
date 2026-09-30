# Environment setup and provenance

## Prepared Core AI bundle inference

Follow the [README setup](../README.md#1-set-up-the-inference-environment): Python 3.11, a virtual environment, [requirements-inference.txt](../requirements-inference.txt), and the compatible macOS 27 / Xcode 27 Core AI toolchain. Compile the Swift bridge with `bash coreai/swift_bridge/build.sh`. The prepared target and DFlash2 drafter use the Swift bridge; the Python Core AI build SDK is not required for this inference path.

The requirements file pins the observed host packages and uses public `coremltools==9.0`. An import-only check of the target/drafter host modules passed with the published 9.0 wheel substituted into the existing research environment. No model was loaded. This does not establish full generation with that wheel or a fresh environment, and transitive dependencies are not locked. Coremltools still warns that the observed PyTorch version is outside its tested conversion range; keep conversion validation separate.

For optional Pi sessions, the guide pins `@earendil-works/pi-coding-agent@0.87.1` with Node.js 22.19 or newer. The installed model-config, OpenAI-completions, simple-options, and transcript modules matched the published 0.87.1 package bytes. Apply the supplied configuration files; no Pi source patch is required for the checked integration. See [PI_CODING.md](PI_CODING.md).

## Research and conversion environment

The installed research environment inspected on 2026-09-29 reports Python 3.11.16, torch 2.14.0, NumPy 2.4.6, SciPy 1.17.1, safetensors 0.8.0, **coremltools 9.1.dev1**, transformers 5.17.0, tokenizers 0.23.2, scikit-learn 1.9.1 and ml_dtypes 0.6.0.

These are observations of an existing environment, not a verified public lockfile. In particular, the notebooks mention custom coremltools FP8 support. A version string does not capture local patches or a Git revision. Record the source and revision of that dependency before promising a clean installation.

Local source inspection found coremltools branch `fp8-ane-support` at base commit `db4dd46b64dcaa4c636c8360b5485c18828140d8`, with 28 modified tracked files (552 insertions / 63 deletions in the inspected unstaged diff) and 13 untracked source/test/example files. These counts describe a working tree, not a releasable version. The untracked files include `_fp8_compile.py` and iOS26 operations. [dependency-provenance.json](dependency-provenance.json) records per-file hashes at inspection; no custom dependency source or compiled binary is bundled by this port. The patch set still needs review, attribution, packaging and clean-environment verification.

The full observed Core AI requirements list is retained as a historical snapshot in [coreai-requirements-observed.txt](history/coreai-requirements-observed.txt); it is not an endorsed public installation lockfile.

The historical vector-LUT notes report M6, macOS 27.0 (26A428), Xcode 27.2 and coremltools 9.0. Those values describe those earlier measurements, not necessarily the current installed environment.

The source Core AI build requirements report coreai-core 1.0.0b2, coreai-opt 0.2.1, coreai-torch 0.4.2 and torch 2.11.0 in a separate Python 3.13 environment. Do not merge that stack blindly into the Core ML runtime environment. Availability and compatibility of those SDK/packages for public users still need validation.

The code uses NumPy, PyTorch, SciPy, safetensors, ml_dtypes and coremltools for the Core ML build path. Calibration/reference generation also uses transformers/tokenizers; k-means needs scikit-learn. Core AI and Swift require their own compatible toolchain. The launcher, bundle manifest/integrity commands and their portable tests use the standard library. Download additionally needs `huggingface_hub`. The inference smoke test needs tokenizers plus the existing ML runtime imports; Core AI's Swift bridge avoids the Python Core AI SDK but still imports the Core ML model module and its dependencies.

No full model conversion was performed during this first port. Run `python forge.py doctor` to capture versions without importing ML frameworks. Full clean-environment validation remains a release task; do not infer it from successful imports in the original research venv.

## DFlash2 release runtime

The fast release loads a Core AI DFlash2 drafter through the same Swift bridge. It needs package metadata, checkpoint configuration and predecessor/successor selector codebooks, plus the target's matching embeddings. The host uses PyTorch for candidate selection and fixes its CPU thread count to one; NumPy and safetensors remain runtime dependencies. A target-only smoke environment does not validate the complete bundle. See [SPECULATIVE_DECODING.md](SPECULATIVE_DECODING.md) for scheduling settings and warm-up requirements.
