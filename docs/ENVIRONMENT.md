# Environment provenance

The installed research environment inspected on 2026-09-29 reports Python 3.11.16, torch 2.14.0, NumPy 2.4.6, SciPy 1.17.1, safetensors 0.8.0, **coremltools 9.1.dev1**, transformers 5.17.0, tokenizers 0.23.2, scikit-learn 1.9.1 and ml_dtypes 0.6.0.

These are observations of an existing environment, not a verified public lockfile. In particular, the notebooks mention custom coremltools FP8 support. A version string does not capture local patches or a Git revision. Record the source and revision of that dependency before promising a clean installation.

Local source inspection found coremltools branch `fp8-ane-support` at base commit `db4dd46b64dcaa4c636c8360b5485c18828140d8`, with 28 modified tracked files (552 insertions / 63 deletions in the inspected unstaged diff) and 13 untracked source/test/example files. These counts describe a working tree, not a releasable version. The untracked files include `_fp8_compile.py` and iOS26 operations. [dependency-provenance.json](dependency-provenance.json) records per-file hashes at inspection; no custom dependency source or compiled binary is bundled by this port. The patch set still needs review, attribution, packaging and clean-environment verification.

The full observed Core AI requirements list is retained as a historical snapshot in [coreai-requirements-observed.txt](history/coreai-requirements-observed.txt); it is not an endorsed public installation lockfile.

The historical vector-LUT notes report M6, macOS 27.0 (26A428), Xcode 27.2 and coremltools 9.0. Those values describe those earlier measurements, not necessarily the current installed environment.

The source Core AI build requirements report coreai-core 1.0.0b2, coreai-opt 0.2.1, coreai-torch 0.4.2 and torch 2.11.0 in a separate Python 3.13 environment. Do not merge that stack blindly into the Core ML runtime environment. Availability and compatibility of those SDK/packages for public users still need validation.

The code uses NumPy, PyTorch, SciPy, safetensors, ml_dtypes and coremltools for the Core ML build path. Calibration/reference generation also uses transformers/tokenizers; k-means needs scikit-learn. Core AI and Swift require their own compatible toolchain. The Python-only launcher and its fast tests use the standard library.

No dependency installation or full model conversion was performed during this first port. Run `python forge.py doctor` to capture versions without importing ML frameworks. A supported clean-environment recipe is a release gate, not something to infer from successful imports in the original research venv.
