# Experiment map

Original module names are retained to keep the research trail navigable. Historical Markdown notes stay at the repository root except for the original vector-LUT README. Runnable source paths now resolve within Forge; the [localized research pipelines](../pipelines/m3u/README.md) configure external weights, datasets and outputs. Historical measurements have not been rerun during localization.

## Quantization and quality

- `scripts/qwen3_lut_common.py`: shared LUT formats, k-means, rotations and GPTQ.
- `scripts/qwen38_gptq_27b.py`: full checkpoint export and sensitivity sweeps.
- `scripts/qwen38_plan.py` and `qwen38_plan_indomain.py`: bit allocation from measured sensitivity.
- `scripts/qwen38_plan_mixr.py`: measured mixer-to-MLP byte trade behind the deployed `mix25in_mixr_lr64mix` export; [localized historical pipelines and original hashes](../pipelines/m3u/README.md).
- `scripts/qwen38_calib_gen.py`: self-generated calibration; `qwen38_calib_pi.py`: explicit user-selected session importer.
- `scripts/qwen38_blockrecon.py` and `qwen38_lowrank_export.py`: reconstruction and low-rank correction.
- `scripts/qwen38_lowrank_aw.py`, `qwen38_lowrank_dyn.py`: activation-weighted factors and dynamic rank allocation; `qwen38_wikippl.py`: export evaluation on the quantizer's WikiText windows.
- `scripts/qwen38_kl.py`, `qwen38_quant_ablation.py`, `qwen38_ane_trace_ppl.py`: quality evaluation and attribution.
- [Quantization notebook](../QUANTIZATION_NOTES.md): include failed and pending runs as such, not as successful results.
- [Initial scalar sensitivity prior](../VQ2BIT_SENSITIVITY.md) and accompanying JSON/CSVs/tiers: kept for attribution and historical comparison. Its early recommendation to compress late MLP layers conflicts with later in-domain VQ results; do not present it as the final allocation recipe.

## ANE compute dtypes (research)

- [M6 ANE compute notes](research/M6_ANE_COMPUTE_2026-10-02.md): ranked INT8-INT8 / FP8-FP8 attention options versus the current V8 dequant-to-FP16 path. Distinguishes measured prior results from inferred ranks. Not a claimed 2x.
- `scripts/m6_attn_compute.py`: host recipes, V8 score-scale identity, E4M3 240/448, and a Core ML skeleton. [M6 runbook](research/M6_ATTN_COMPUTE_README.md).
- [M6 compute feasibility](research/M6_COMPUTE_FEASIBILITY_2026-10-02.md): Amdahl / traffic leverage model for how much an attention-compute speedup can move end-to-end, with an on-device measurement gate.
- `scripts/m6_compute_roofline.py`: pure-Python Amdahl and byte-traffic model behind the feasibility doc.
- [M6 compute acceleration](research/M6_COMPUTE_ACCELERATION_2026-10-03.md): measured on the M6. `GDN_FAST=1` (exact DeltaNet core rewrites) and attention history tiles (`ATT_BLOCK=2048` for verify, `ATT_BLOCK_PREFILL=4096` for prefill) cut full-target verify 19 to 31% and prefill 29 to 35% with unchanged quality, at about the release graph's cold compile time. Builder defaults in `coreai/qwen38_coreai_build.py` since 3 October; `GDN_FAST=0 ATT_BLOCK=16384` restores the release graph.
- On-device tools: `scripts/m6_entry_sweep.py` (per-entry and full-chain timing, context fits), `m6_layer_ablation.py` (per-part cost of a real chunk), `m6_gdn_bench.py` and `m6_attn_bench.py` (GDN core and attention core variants with FP32 checks and placement), `m6_chunk_ab.py` (two builds of one chunk), `m6_kl512_eval.py` (compiled KL-512 gate), `m6_long_ctx_eval.py` (64K prefill plus teacher forcing, build-to-build KL), `m6_server_bench.py` (owned full-server prefill/decode case). Host test: `tests/test_gdn_fast.py`.
- [DFlash2 sampling plan](research/DFLASH2_SAMPLING_PLAN.md): temperature sampling with the drafter's own distribution (exact speculative sampling), baseline measurement matrix, exactness tests and presence-penalty notes. Plan only.
- [Verifier block length](verifier_len.md): 8-, 4- and 3-row verifiers measured on the optimized target (`scripts/m6_verify_len.py`, `m6_verify_len_report.py`); 8 rows stays fastest end to end.
- [Jeff / Unsloth decision models on the ANE](research/JEFF_DECISION_ANE.md): Path B convert/smoke for Jeff (Qwen3.5-0.8B readout) on Core AI. `forge.py jeff-convert` / `jeff-smoke`; prefill-only FP16 or light INT8; no 27B VQ/DFlash2. Host tests in `tests/test_jeff_coreai.py`. ANE compile is the Mac spike.

## Inference and numerics

- `scripts/qwen38_ane_chunk.py`: MIL graph construction and numerical workarounds.
- `scripts/qwen38_ane_model.py`: checkpoint access, model assembly, host state, runtime.
- `scripts/qwen38_decode_ref.py`, `dflash2_target_ref.py`: reference implementations.
- `scripts/qwen38_divergence.py` and `qwen38_ane_capture.py`: error localization.
- `scripts/qwen38_server.py` and `qwen38_chat.py`: serving and chat.
- `scripts/qwen38_server.sh` and `qwen38_pi_config.py`: start/stop/status wrapper and Pi context/compaction sync.
- [DeltaNet numerical notebook](../ANE_DELTANET_NUMERICS.md): read through the final recipe; earlier MLP scaling was superseded.

## Core AI and speculative decoding

- `coreai/qwen38_coreai_build.py`: graph mirror, LUT injection, multiple entries, blocked attention.
- `scripts/qwen38_coreai_model.py`: Python/Swift runtime selection, buffers, context changes and cache handling.
- `coreai/swift_bridge/CoreAIBridge.swift` and `coreai_bridge.py`: native bridge source and Python wrapper.
- `scripts/dflash2_*` and `coreai/dflash2_*`: drafter conversion, runtime and experiments. Matching DFlash2 is required for the intended fast release; alternative quantizers and probes remain research paths. See [the current speculative-decoding guide](SPECULATIVE_DECODING.md).
- `scripts/qwen38_spec_unit_test.py`: a CPU statistical sampler test (large sample count, not a fast unit test).
- [Core AI notebook](../COREAI_PORT_NOTES.md) and [DFlash2 notebook](../DFLASH2_ANE_PLAN.md).

## Local helpers and remaining reproduction gaps

- [bench_vector_lut.py](../scripts/bench_vector_lut.py) uses [coreai_bench_helpers.py](../scripts/coreai_bench_helpers.py) locally. Weight-sharing, output-pool, KV-slice/writer, attention-entry, overhead-slope and build-merge probes are included under [coreai/probes](../coreai/probes/coreai_entry_share.py); imported source hashes are in [LOCAL_EXPERIMENT_IMPORTS.json](../provenance/LOCAL_EXPERIMENT_IMPORTS.json).
- The [Core ML timer](../tools/coreml/README.md) is included with its Apple license. Historical direct-ANE measurements used a separate private `ane_mil_bench` tool that is not distributed; these are historical evidence, not a runnable Forge recipe or equivalent to public Core ML timing. Private API surveys remain unavailable.
- `qwen38_decode_ref.py` standalone validation expects extracted layer test tensors; the full builder uses its own checkpoint loader.
- Drafter reference parity uses the [vendored DFlash reference](../vendor/dflash_reference/README.md) by default, with its original MIT notice. Checkpoints remain separate inputs.
- The four historical context tests named in the Core AI notebook (`op_limit_test.py`, `decode_ctx_test.py`, `transition_test.py`, `drafter_gap_test.py`) are unavailable. They need to be recovered or replaced before the corresponding claims are independently reproducible; current bridge validation and paired smoke tests cover different checks.
- Quantized exports, token traces, numeric arrays, compiled packages, and private session data were deliberately not imported as source assets. The deployed [mixr bit plan](../configs/quantization/mix25in_mixr.json), retrieval provenance and export-header inspection are now included; headers do not establish tensor-payload or compiled-package lineage.
- The local service manager and personal pi configuration editor were excluded. Use the foreground launcher; it does not stop existing servers or edit agent settings.
- Research probes may overwrite their own output directories or clear selected compiler caches. They are not part of the fast test suite and are not automatically run.

The per-file source hashes in [provenance.json](provenance.json) identify the source working-tree bytes before editing. They matter because much of this work was untracked at the source HEAD.

The bridge shell workflows `validate_full_model.sh`, `run_full_bench.sh` and probe `run_mode.sh` measure target parity, target-only chunk timing and single-chunk memory respectively. They do not validate the complete DFlash2 speculative release; use the paired smoke and evaluation workflow in [SPECULATIVE_DECODING.md](SPECULATIVE_DECODING.md).
