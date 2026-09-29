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
