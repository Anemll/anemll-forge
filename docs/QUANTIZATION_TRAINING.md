# Quantization training and source provenance

This checkout includes the GPTQ and frozen-index QAT implementation used by the documented release 0.2 method, its corpus generator, and the matching export/evaluation code. The [quantization guide](QUANTIZATION.md) explains the method and historical results. The commands here describe how to run the method with your own data; they do not promise bit-identical recreation of the published weights.

## Included source

| Stage | Implementation | Contract |
| --- | --- | --- |
| Centroid fitting and GPTQ helpers | [qwen3_lut_common.py](../scripts/qwen3_lut_common.py) | Per-output-row RMS scales; activation-weighted scalar/vector fitting; seed/thread controls; GPTQ error feedback |
| Sequential GPTQ | [qwen38_gptq_27b.py](../scripts/qwen38_gptq_27b.py) | Mixer rotations; rank-64 factors fitted inside the sequential pass; exports codes, tables, scales and rotation metadata |
| Training prompts | [release list](../configs/kl/qat_corpus_prompts.json), [candidate builder](../configs/kl/qat_corpus_build_prompts.py) | Checked list of 527 prompt entries; separate seed-7 builder produces 560 candidates before filtering |
| Corpus generation | [qwen38_corpus_gen.py](../scripts/qwen38_corpus_gen.py) | BF16 responses with thinking; sequence-level train/validation split; packed token rows and metadata |
| Teacher cache and QAT | [qwen38_qat_kl.py](../scripts/qwen38_qat_kl.py) | Frozen LUT indices and INT8 codes; trainable tables, per-channel scales, mixer factors and head |
| Materialization | [qwen38_overlay_materialize.py](../scripts/qwen38_overlay_materialize.py) | Replaces tensors from an overlay while preserving codes and metadata in a self-contained export |
| Reference evaluation | [qwen38_kl.py](../scripts/qwen38_kl.py) | Applies QAT overlays and mixer rotations; reports storage both excluding and including correction factors |

The scripts were recovered from the committed `quant-kl-research` snapshot `86096c053e4960c4fc37efa4913c793cc32492cd`. [Source provenance](quantization-source-provenance.json) records the original file digests. The research commit is not included in this public checkout, so those original digests require the archived source to verify; they are not hashes of the modified public files. Uncommitted changes from the research checkout were not imported. This is a source recovery, not proof that the historical Colab job ran exactly that complete checkout. The public commit containing this port should be cited by publications; the research branch name alone is not a stable source citation.

The port makes output paths portable, refuses existing QAT/materialization output directories, resumes corpus generation after the last completed sequence, and resets evaluator factor-byte accounting for each evaluation. The prompt builder writes a new candidate file rather than overwriting the checked release list. The list is preserved as recovered, including repeated entries (527 entries, 515 distinct prompt strings). Rebuilding candidates does not reproduce the historical filtering step.

Other research switches remain available but are off in the release recipe: residual-stream folding (`ROT_DIR`), asymmetric calibration (`ASYM`), index refresh (`FIXED_CB`), tail-weighted losses, long-row QAT and training norms or small DeltaNet projections. The later uncommitted experiments are excluded. `qwen38_blockrecon.py` is the older per-layer MSE experiment; it is not the release's end-to-end KL trainer.

## Environment and data

Use the separate [quantization dependencies](../requirements-quantization.txt) and read [environment provenance](ENVIRONMENT.md#quantization-training-source--2026-10-13). The recorded fitting run used a Colab G4 with an RTX PRO 6000 Blackwell and 96 GB GPU memory. The dependency list is not a validated fresh CUDA lockfile. CPU fixtures below do not require Transformers, model weights, Core AI or the ANE.

GPTQ and reference evaluation select MPS when available, otherwise CUDA, otherwise CPU; `DEVICE` overrides that selection. Initial centroid fitting and low-rank SVD fitting run on CPU. GPTQ calibration and error compensation run on `DEVICE`; FP64 Hessian factorization uses that CUDA device or CPU for CPU/MPS runs. `DEVICE=cpu` keeps factorization on CPU even when CUDA is visible. QAT defaults to CUDA when available, otherwise CPU; the full DeltaNet training path was exercised on CUDA, not validated end to end on MPS. CUDA memory logging is emitted only for CUDA runs.

Run commands from the repository root in a compatible training environment. Set paths to existing inputs and keep generated artifacts outside the source checkout:

```sh
export MODEL="/path/to/Qwen3.8-27B"
export WORK="/path/to/quantization-work"
export DATA="$WORK/data"
export WIKI="$WORK/wikitext"
mkdir -p "$DATA" "$WORK/runs"
```

`MODEL` must contain the BF16 checkpoint, configuration and tokenizer. The recorded checkpoint revision is identified in [QUANTIZATION.md](QUANTIZATION.md). `WIKI` contains `wiki2_train.txt` and `wiki2_test.txt`; GPTQ prepares its `qwen38_train_ids.npy` and test token cache when needed. The recipe also needs `calib_chat_ids.npy` and `calib_pi_ids.npy`, each an array of 1,024-token rows. [Chat calibration generation](../scripts/qwen38_calib_gen.py) and [session rendering](../scripts/qwen38_calib_pi.py) document their inputs.

The original session-derived calibration rows and cached development/selection/held-out traces are not distributed. Token arrays retain the original text and are not anonymized. Use independently prepared data you can share, and record that this defines a new training experiment. Prepare disjoint reference traces with `qwen38_kl.py generate` and `reference` (`PROMPTS_FILE` selects a JSON list of strings). This differs from `CORPUS_PROMPTS`, which reads objects with `p`, `tools` and `cat` fields; do not pass the QAT corpus file as `PROMPTS_FILE`. Keep the checkpoint-selection trace separate from final evaluation. A new run also needs recorded package versions, seeds, commands and input/trace identities.

## Run the release 0.2 method

Use a fresh shell or clear optional research overrides before running this recipe. The release uses `BASIS=online`, `LOAD_DTYPE=bf16`, `ASYM=0`, no `ROT_DIR`/`FIXED_CB`, and no extra QAT loss or long-row options.

### 1 Fit GPTQ weights

```sh
OUT="$WORK/runs" TAG=release02_gptq EXPORT=1 METHOD=gptq BASIS=online LOAD_DTYPE=bf16 \
PLAN=configs/quantization/mix25in_vq3pA.json \
MIXER="LUT4 per-tensor + pcs" KV_FMT="INT8 per-channel" HEAD="LUT4 per-tensor + pcs" \
AW=1 BASELINE=0 ASYM=0 LR_RANK=64 LR_PARTS=gdn,attn LR_ITERS=1 NCAL=128 \
MIX_ROT=readers+writers KMEANS_SEED=1 KMEANS_THREADS=1 KMEANS_SAMPLE=300000 KMEANS_INIT=3 \
CAL_MIX="$DATA/calib_chat_ids.npy:36,$DATA/calib_pi_ids.npy:48" \
  python scripts/qwen38_gptq_27b.py
export EXPORT_DIR="$WORK/runs/export/release02_gptq"
```

All 192 MLP matrices use `vector 2x64 + pcs`. Mixer LUTs use the plan's allocation, while full-attention K/V projection weights use their own INT8 per-channel scales. The head uses scalar LUT4 with per-channel scales. Low-rank factors remain FP16 and act in the original input basis. The shared codebook and row scales stay fixed within each GPTQ call; the outer low-rank iterations can refit them between calls.

The default converter preserves the exported K/V INT8 codes as compile-time weights; `QCONV_INT8=0` is the older FP16-expansion path. Runtime INT8 scores, the V8 cache and M6 FP8 softmax are separate from weight fitting; see [accelerated attention](QUANTIZATION.md#accelerated-attention-on-the-ane).

### 2 Generate the QAT corpus

```sh
CORPUS_PROMPTS=configs/kl/qat_corpus_prompts.json CAL_OUT="$DATA/corpus_ids.npy" \
SEQ=1024 MAX_NEW=1792 GEN_BATCH=64 VAL_FRAC=0.05 SEED=1234 \
  python scripts/qwen38_corpus_gen.py
```

The generator writes training rows first, then validation rows; the JSON sidecar records their counts. `TRAIN=...:meta` reads that boundary. The historical run produced 637 training rows and 673 total rows; different generated responses can change those counts. The partial cache can resume a stopped run, but a resumed random-generation stream is not guaranteed to match an uninterrupted run. Keep prompts and settings unchanged when resuming; use a new output path for a different corpus.

To explore a new corpus, generate candidates separately:

```sh
python configs/kl/qat_corpus_build_prompts.py --output "$DATA/new_candidates.json"
```

Perform overlap and duplicate checks against your own evaluation sets before using new candidates. This command does not filter them. The checked release list is used directly by the release recipe.

The checked list has 141 code, 130 tool-use, 126 multilingual, 50 math/reasoning and 80 rare-token entries. The builder's percentage targets do not describe this actual mix; the 33 removed candidates comprise 30 math and three code entries. Duplicates exist before filtering too (560 candidates, 548 distinct strings).

### 3 Cache the teacher and check initial parity

```sh
export TEACHER="$DATA/qat_teacher"
export TRAIN="$DATA/corpus_ids.npy:meta,$DATA/calib_pi_ids.npy:44,$DATA/calib_chat_ids.npy:32"
export WIKI_ROWS=44 HOLD=4 SEQ=1024 TOPK=256 BS=2 TEACHER_DTYPE=fp16
python scripts/qwen38_qat_kl.py teacher

# Supply your independently prepared development and selection traces.
export DEV_TRACE="$DATA/kl_dev"
export SEL_TRACE="$DATA/kl_selection"
TRACE="$DEV_TRACE" TAG=release02_gptq python scripts/qwen38_kl.py eval
CHECK_BASE=release02_gptq CHECK_SEQ=32 python scripts/qwen38_qat_kl.py check
```

The teacher command holds out four rows from each of four sources (corpus, Pi, chat and WikiText), for 16 monitor rows when all are available. `:meta` sets the corpus training boundary; `HOLD` limits how many rows after that boundary enter the monitor. The check compares reconstructed weights and projections, including mixer rotation metadata and factors, then compares initial trace KL with the GPTQ evaluator. It loads the full model; the CPU fixture tests below are a separate, smaller check.

The historical cache rounds teacher log-probabilities to FP16, then computes loss in FP32. That rounding introduces a small numerical bias and can give slightly nonzero KL even for an identical underlying distribution. The compiled KL-512 protocol is also different from this top-256 training target. For a new precision experiment, set `TEACHER_DTYPE=fp32` and use a new `TEACHER` directory when generating the cache. This avoids the FP16 cache-rounding step and doubles the log-probability array storage; it does not change existing caches. Both short- and long-row teacher modes support it. `teacher.json` records the chosen precision, and the training log records the actual loaded cache dtypes. FP16 remains the historical default. Report the chosen cache and evaluation precision with new results.

### 4 Train continuous parameters and materialize

```sh
export OUT_DIR="$WORK/runs/export/release02_qat"
TRAIN_PARTS=lut,scale,head,lr STEPS=600 BS=2 EVAL_EVERY=50 WARMUP=20 \
LR_LUT=2e-4 LR_SCALE=2e-4 LR_LRF=2e-5 LR_HEAD=1e-4 DEV_SEQ=32 \
  python scripts/qwen38_qat_kl.py train

python scripts/qwen38_overlay_materialize.py \
  "$OUT_DIR" "$WORK/runs/export/release02_full"

EXPORT_DIR="$WORK/runs/export/release02_full" TRACE="$DEV_TRACE" TAG=release02_full \
  python scripts/qwen38_kl.py eval
```

QAT keeps indices and INT8 codes fixed while training LUT values, scales (including INT8 K/V scales), mixer factors, and the head's table/scales. Scales use `s = s0 * exp(log_s)`; values, scales and factors are rounded to FP16 on export. `SEL_TRACE` chooses the best checkpoint; `DEV_TRACE` is a monitor. QAT and materialization require new output directories. An overlay links back to its source export; keep that source until materialization is complete.

Convert the full export with [the release conversion workflow](WORKFLOW.md), then validate the compiled model. Rebuild the drafter with the same target LM head. The materialized 129 weight files are conversion inputs, not a directly loadable Transformers checkpoint. This source update does not replace released model packages.

## Numerical fixes after source recovery

The source recovery is recorded separately from the following numerical changes in Git history. The INT8 rounder now clamps its scale after FP16 rounding to at least the smallest positive FP16 value. Previously, the `1e-12` floor rounded back to zero, so an all-zero row could produce NaNs. Ordinary positive representable scales are unchanged. This is a new edge-case fix; the released weights were not scanned or retrained to measure its impact.

The optional FP32 teacher-cache mode above is a new precision control. It preserves the historical FP16 default and does not silently change the targets of an existing run.

Review fixes also correct the step-0 hold-out log when a selection trace is set and restrict CUDA memory reporting to CUDA runs. Hessian factorization now follows the requested compute device instead of CUDA visibility; CPU runs on CUDA-capable hosts can therefore differ numerically from earlier runs that silently used cuSOLVER.

Two optional research controls have explicit contracts: `PRE_GAIN_HESSIAN=1` captures one extra forward before the current layer is quantized, using inputs propagated through earlier quantized layers, and removes its hooks before other forwards. Saved metadata labels that stream and records one calibration pass's token count. `LOSS_TAIL` with `LONG_EVERY` is rejected before model loading because the long-row implementation supports only token-weighted mean KL, not a combined tail objective. Both controls remain off in the release recipe.

## Validation and remaining limits

Small CPU checks cover vector/scalar/INT8 export round trips, centroid gradients, trainable scales across MLP/mixer/K/V/head paths, rotated projections with low-rank factors, GPTQ factor propagation, materialization, and corpus metadata:

```sh
KMEANS_THREADS=1 python -m unittest discover -s tests -p 'test_quantization*.py' -v
git diff --check
```

These checks validate numerical and file-format contracts on synthetic fixtures. They do not reproduce the 27B training run, verify a clean CUDA installation, establish ANE placement, or rerun the paper's quality and performance measurements. Preserve the historical results as reported measurements until those larger checks are rerun with identified inputs and software revisions.
