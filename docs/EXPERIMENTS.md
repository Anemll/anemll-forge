# Experiment map

Original module names are retained to keep the research trail navigable. Historical Markdown notes stay at the repository root except for the original vector-LUT README. Paths identifying the author's machine were generalized. Notes contain historical commands and some external references; they are not all standalone tutorials yet.

## Quantization and quality

- `scripts/qwen3_lut_common.py`: shared LUT formats, k-means, rotations and GPTQ.
- `scripts/qwen38_gptq_27b.py`: full checkpoint export and sensitivity sweeps.
- `scripts/qwen38_plan.py` and `qwen38_plan_indomain.py`: bit allocation from measured sensitivity.
- `scripts/qwen38_plan_mixr.py`: measured mixer-to-MLP byte trade behind the deployed `mix25in_mixr_lr64mix` export; [exact historical pipelines](../pipelines/m3u/README.md).
- `scripts/qwen38_calib_gen.py`: self-generated calibration; `qwen38_calib_pi.py`: explicit user-selected session importer.
- `scripts/qwen38_blockrecon.py` and `qwen38_lowrank_export.py`: reconstruction and low-rank correction.
- `scripts/qwen38_lowrank_aw.py`, `qwen38_lowrank_dyn.py`: activation-weighted factors and dynamic rank allocation; `qwen38_wikippl.py`: export evaluation on the quantizer's WikiText windows.
- `scripts/qwen38_kl.py`, `qwen38_quant_ablation.py`, `qwen38_ane_trace_ppl.py`: quality evaluation and attribution.
- [Quantization notebook](../QUANTIZATION_NOTES.md): include failed and pending runs as such, not as successful results.
- [Initial scalar sensitivity prior](../VQ2BIT_SENSITIVITY.md) and accompanying JSON/CSVs/tiers: kept for attribution and historical comparison. Its early recommendation to compress late MLP layers conflicts with later in-domain VQ results; do not present it as the final allocation recipe.

## Inference and numerics

- `scripts/qwen38_ane_chunk.py`: MIL graph construction and numerical workarounds.
- `scripts/qwen38_ane_model.py`: checkpoint access, model assembly, host state, runtime.
- `scripts/qwen38_decode_ref.py`, `dflash2_target_ref.py`: reference implementations.
- `scripts/qwen38_divergence.py` and `qwen38_ane_capture.py`: error localization.
- `scripts/qwen38_server.py` and `qwen38_chat.py`: serving and chat.
- [DeltaNet numerical notebook](../ANE_DELTANET_NUMERICS.md): read through the final recipe; earlier MLP scaling was superseded.

## Core AI and speculative decoding

- `coreai/qwen38_coreai_build.py`: graph mirror, LUT injection, multiple entries, blocked attention.
- `scripts/qwen38_coreai_model.py`: Python/Swift runtime selection, buffers, context changes and cache handling.
- `coreai/swift_bridge/CoreAIBridge.swift` and `coreai_bridge.py`: native bridge source and Python wrapper.
- `scripts/dflash2_*` and `coreai/dflash2_*`: optional drafter work.
- `scripts/qwen38_spec_unit_test.py`: a CPU statistical sampler test (large sample count, not a fast unit test).
- [Core AI notebook](../COREAI_PORT_NOTES.md) and [DFlash2 notebook](../DFLASH2_ANE_PLAN.md).

## Reproduction gaps retained explicitly

- `bench_vector_lut.py` imports external `bench_stacked` and `bench_sparsity` helpers; these were not in the source tree.
- Some probes require `ane_mil_bench`, `time_models`, private API tools, or source from the sibling `fp8-mlp-metal41-bench` repository.
- `qwen38_decode_ref.py` standalone validation expects extracted layer test tensors; the full builder uses its own checkpoint loader.
- Drafter reference parity can require a separate `dflash` source checkout and checkpoint.
- Historical `tests/*.py` cited under model artifact directories were not in this source repository. They need to be recovered or replaced before the corresponding claims are independently reproducible.
- Quantized exports, token traces, numeric arrays, compiled packages, and private session data were deliberately not imported as source assets. The deployed [mixr bit plan](../configs/quantization/mix25in_mixr.json), retrieval provenance and export-header inspection are now included; headers do not establish tensor-payload or compiled-package lineage.
- The local service manager and personal pi configuration editor were excluded. Use the foreground launcher; it does not stop existing servers or edit agent settings.
- Research probes may overwrite their own output directories or clear selected compiler caches. They are not part of the fast test suite and are not automatically run.

The per-file source hashes in [provenance.json](provenance.json) identify the source working-tree bytes before editing. They matter because much of this work was untracked at the source HEAD.
