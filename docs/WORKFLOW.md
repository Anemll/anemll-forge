# First port workflow

For the planned **Core AI M6 release with matching DFlash2 speculative decoding**, start with [the Core AI Hugging Face bundle guide](HUGGING_FACE.md). The Core ML build/inference steps below preserve the research baseline for conversion and experiments; they do not require uploading Core ML model packages.

Run commands from the repository root. Paths below are examples to replace, not bundled assets. `--dry-run` validates basic inputs and prints the command without loading models. Full builds can use substantial memory and disk space; this port has not rerun them.

## 1. Prepare the environment and checkpoint

Read [ENVIRONMENT.md](ENVIRONMENT.md). The historical installation includes a development build of coremltools, so a clean public dependency recipe is still outstanding. Use a known compatible environment initially, then record `python forge.py doctor` for each experiment.

Set `MODEL` to the original sharded checkpoint containing `config.json`, `model.safetensors.index.json`, tokenizer files, and weights. The expected text configuration has 64 layers and hidden size 5120. Checkpoint identity must be pinned before release.

```sh
export MODEL=/path/to/Qwen3.8-27B
export WIKI=/path/to/wikitext
export RUNS=/path/to/new-quantization-run
export BUILDS=/path/to/new-coreml-build
```

`WIKI` must contain `wiki2_train.txt` and `wiki2_test.txt`; token caches are written there. Keep calibration and evaluation data separate. Historical private pi sessions are not distributed. `qwen38_calib_gen.py` provides self-generated calibration; the pi importer now requires an explicit `SESSIONS_GLOB` supplied by the user.

## 2. Quantize

```sh
python forge.py quantize --model "$MODEL" --wiki "$WIKI" --output "$RUNS" --tag qwen38-27b-vq2
```

This is a reproducible *starting configuration*, not a claim to reproduce the best historical export: MLP vector 2×16 + per-channel scale, scalar LUT4 mixers/head, INT8 K/V projection weights, GPTQ and online rotations. The Core AI runtime's KV cache is FP16. The recovered [historical mixed-bit plan](../configs/quantization/mix25in_mixr.json) is included; calibrated export tensors and private calibration rows are not. See [the quantization guide](QUANTIZATION.md) for the exact allocation and reproduction limits. Pass `--plan /path/to/plan.json` for a generated plan.

Outputs are in `$RUNS/export/qwen38-27b-vq2`. The source quantizer also evaluates WikiText perplexity. Tune `NCAL`, `NEVAL`, `SEQ`, `DEVICE`, `AW` and `CAL_MIX` through environment variables when reproducing experiments; record their values.

For sensitivity and quality work, use `qwen38_plan.py`, `qwen38_plan_indomain.py`, `qwen38_kl.py`, `qwen38_blockrecon.py` and `qwen38_lowrank_export.py`. Their module docstrings describe their research inputs. Block reconstruction needs a distinct `OUT_DIR`, never the input export. KL generation/reference/eval are separate steps, and the reference must use the same token trace as the candidate.

### Historical deployed mixr recipe

The newly recovered `qwen38_plan_mixr.py` and [M3U pipeline archive](../pipelines/m3u/README.md) document the later deployed `mix25in_mixr_lr64mix` path. Given the original KL band measurements, sensitivity sweep and starting plan, generate the plan with explicit paths:

```sh
python scripts/qwen38_plan_mixr.py --kl-dir /path/to/kl \
  --sweep /path/to/sweep_mlp.json --plan /path/to/plan_indomain.json --out /path/to/plan_mixr.json
AW=1 NCAL=48 CAL_MIX='/path/to/calib_chat_ids.npy:16,/path/to/calib_pi_ids.npy:16' \
  python forge.py quantize --model "$MODEL" --wiki "$WIKI" --output "$RUNS" \
  --plan /path/to/plan_mixr.json --tag mix25in_mixr
MODEL="$MODEL" EXPORT_DIR="$RUNS/export/mix25in_mixr" OUT_DIR="$RUNS/export/mix25in_mixr_lr64mix" \
  LR_RANK=64 PARTS=gdn,attn python scripts/qwen38_lowrank_export.py
MODEL="$MODEL" TRACE=/path/to/kl EXPORT_DIR="$RUNS/export/mix25in_mixr_lr64mix" \
  TAG=mix25in_mixr_lr64mix python scripts/qwen38_kl.py eval
```

The historical 48-row calibration used 16 WikiText rows plus 16 chat and 16 pi rows. Replacing private rows creates a new experiment, not an exact historical reproduction. The scripts, commands and [original plan with provenance](../configs/quantization/mix25in_mixr.provenance.json) are included; original calibration and evaluation data remain outside the repository. Use `--plan configs/quantization/mix25in_mixr.json` to use the recovered allocation. Verify quality against the stored baseline before converting or pairing a drafter.

## 3. Convert the Core ML baseline

```sh
python forge.py convert --model "$MODEL" --export "$RUNS/export/qwen38-27b-vq2" --output "$BUILDS" --ctx 16384
```

The launcher fixes the `ane7i` numerical recipe: tanh SiLU in DeltaNet and MLP, stable softplus already in the builder, `GDN_SQ=16`, `GDN_SV=64`, no MLP down-input scaling, 16 chunks of four layers, verify width eight, read-only KV inputs. It requires a new output destination. Legacy `build_v3` is the function name that produces these **v4** manifests; this is not a version mismatch.

The build is stored under `$BUILDS/qwen38-27b-vq2`. Compilation alone does not prove ANE placement or accuracy. Check compute plans, same-weight reference parity, recurrent-state behavior, and teacher-forced quality before drawing performance conclusions.

## 4. Run the Core ML diagnostic baseline

```sh
python forge.py chat --model "$MODEL" --build "$BUILDS/qwen38-27b-vq2" --ctx 16384 --no-think --prompt 'Explain vector quantization.'
python forge.py serve --runtime coreml --plain --model "$MODEL" --build "$BUILDS/qwen38-27b-vq2" --ctx 16384
```

The `chat` command remains the plain Core ML research interface. The example server explicitly selects Core ML and `--plain`; the main release serving command below uses Core AI plus DFlash2. The server binds to `127.0.0.1:8765`; stop with Ctrl-C. No personal agent configuration is changed. A prepared `$MODEL/embed_tokens_fp16.npy` takes precedence; inference then needs no original checkpoint shards or index. Otherwise the embedding table is cached under `$MODEL/.anemll-forge/`; the checkpoint directory must be writable, or invoke the underlying script with an explicit `EMBED_NPY` in a writable location. See [the Hugging Face bundle workflow](HUGGING_FACE.md) for downloading prepared artifacts and running an integrity/inference smoke test.

## 5. Convert and pair the Core AI release

The port preserves the latest Core AI build/runtime and Swift bridge. Use a separate compatible Core AI environment. Build the bridge with `bash coreai/swift_bridge/build.sh` using the selected Xcode (`DEVELOPER_DIR` if needed). No prebuilt dylib is included.

```sh
MODEL="$MODEL" EXPORT_DIR="$RUNS/export/qwen38-27b-vq2" OUT=/path/to/coreai-builds \
  python coreai/qwen38_coreai_build.py all --ctx 8192,16384 --pctx 8192,16384
python forge.py serve --runtime coreai --model "$MODEL" --build /path/to/coreai-builds/qwen38-27b-vq2 --ctx 16384 \
  --draft /path/to/matching-drafter/dflash2_lut4_gptq.aimodel --drafter /path/to/matching-drafter
```

A freshly quantized target needs its own matched and validated drafter/head pairing; the prepared `mix25in_mixr_lr64mix` drafter is not automatically compatible with the example export above. For the ready-made pair, use the downloaded bundle workflow instead. The drafter conversion source is `coreai/dflash2_coreai_build.py`; pin `DRAFT_EXPORT`, `HEAD_EXPORT`, `DRAFTER`, numerical settings and metadata. Read [SPECULATIVE_DECODING.md](SPECULATIVE_DECODING.md) before building or changing that pair.

These Core AI commands remain unverified in a clean environment. Start with a single `chunk 0-3` build. The current runtime defaults to compile mode 2 and removes incompatible cache specializations for the selected package before loading; inspect this behavior before changing modes on an existing installation. The Python binding has documented long-run allocation problems; the Swift bridge was the later research solution.

## Validation before release

Start with CPU tests, then one chunk, then the full model: finite outputs, relative L2 and norms (not just cosine), Core ML CPU vs ANE on the same graph, PyTorch same-weight parity, teacher-forced perplexity/KL, context transitions, and prolonged speculative generation with the matching drafter. Include accepted-prefix/stop/cap behavior, cache restoration and separately labeled plain diagnostics. Record hardware, OS/Xcode, dependencies, checkpoint/export hashes, numerical settings, placement, warm-up policy, latency distribution, memory baseline, and sample counts. Keep quantization error separate from runtime numerical error.
