# Contributing to ANEMLL Forge

Contributions to inference, quantization, conversion, documentation, and reproducible ANE experiments are welcome. Discuss substantial architecture changes in an issue before implementing them. Keep communication respectful and technical.

## Get started

Follow the [README setup](README.md#1-set-up-the-inference-environment). Use a branch for your change and keep it focused. Conversion and quantization need the additional toolchains described in [ENVIRONMENT.md](docs/ENVIRONMENT.md). Model weights and generated results belong outside the source checkout.

## Report a bug or an experiment

Open an [issue](https://github.com/Anemll/anemll-forge/issues) with the expected and actual behavior, a minimal reproduction, and relevant sanitized diagnostics. Include:

- Chip, unified memory, macOS, Xcode/Swift, Python, and dependency versions (`python forge.py doctor`).
- Forge commit and model-bundle revision, target/drafter identity, context limit, and command/environment settings.
- A synthetic or releasable prompt, generation/sampling/thinking settings, and whether DFlash2 was enabled.

For performance changes, report emitted tok/s, prefill or TTFT, context/prompt/output sizes, speculative acceptance where available, memory, warm-up, sample count, and measurement method. Compare like-for-like settings. For quality results, specify the corpus, reference model, evaluator, and truncation/timeout policy. KL is fidelity evidence, not a complete capability benchmark. Distinguish measurements from hypotheses and projections.

## Submit a pull request

1. Explain the problem, resulting behavior, and how to reproduce or validate the change.
2. Preserve the original upstream notices and document any adapted source or new dependency.
3. Update the relevant guide when setup, interfaces, numerical behavior, or limitations change.
4. Run checks appropriate to the change and state what was tested, including any missing hardware validation.

Useful portable checks from the checkout are:

```sh
python -m unittest discover -s tests -p test_launcher.py -v
python -m unittest discover -s tests -p test_hf_release.py -v
git diff --check
```

These fixture tests do not load the full model or prove ANE placement. Native, numerical, or performance changes need relevant hardware checks when available; provide commands and evidence. Documentation-only changes generally need link and command checks, not new tests. See [VALIDATION.md](docs/VALIDATION.md).

## Keep sensitive data out of contributions

Do not commit or attach credentials, account configuration, personal paths, unpublished weights, or conversation/session-derived calibration and trace payloads without a content/provenance review. Use synthetic examples and sanitized diagnostics. **Token IDs remain decodable text; NPY/NPZ files are not anonymization.** Feature traces can also include full token sequences. Review server logs for decoded prompt excerpts before sharing them, and keep generated JSON/JSONL reports outside the checkout.

If you discover exposed credentials or personal data, do not repost the payload in a public issue; contact a maintainer without the sensitive content. Only contribute code and data you have the right to distribute.

## Licensing

Independently authored contributions to Forge are submitted under the repository's [MIT license](LICENSE), unless an applicable third-party license is clearly identified and preserved. Qwen-derived model artifacts retain Alibaba Cloud / Qwen's Apache 2.0 license; the DFlash2 assets retain their separate upstream license and notices. Contributing to Forge does not change those terms. Read [ATTRIBUTION.md](docs/ATTRIBUTION.md).
