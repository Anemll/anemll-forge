# Initial port validation — 2026-09-29

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

## Attribution and private upload preparation — 2026-09-29

The updated suite passed **41 tests** in the existing research environment. Six additional bundle tests verify the default HF destination, mandatory license/source documents, document hashing/download selection, symlink rejection and per-file modification notices. All five prepared release documents are included in the verified download inventory.

The exact copied Qwen LICENSE matches the pinned upstream bytes and preserves `Copyright 2026 Alibaba Cloud`. Ten local checkpoint documents/config/tokenizer files match official upstream hashes at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; all 18 shard cache etags match upstream SHA256 metadata. The large shard bytes were not rehashed. [QWEN_SOURCE.json](../release/huggingface/QWEN_SOURCE.json) records this evidence and remaining artifact-lineage limits.

A separate upload agent created and verified the private HF target `anemll/anemll-forge-qwen3.8-27B`, staged the Core AI source packages plus embeddings and tokenizer/config assets, and passed `release-manifest` and `quick-test --check-only` on **65 inventoried files / 13,184,004,271 bytes**. Upload was then started; this staging check does not confirm completed transfer or downloaded-artifact inference. The model card describes a research project for the M6 Apple Neural Engine, with KL as the current evaluation and future ANE benchmarks left unclaimed. Full hardware inference remains outstanding.
