# Jeff / Unsloth decision models on the ANE

- **Scope:** run Jeff (Qwen3.5-0.8B decision head) on the Apple Neural Engine through a prefill-only Core AI path.
- **Status:** Path B runs end to end on an M5 Max (7 October 2026). `jeff-base` v1.3 FP16 converts in 27 s and compiles in 79 s. All six chunks and the readout head are cached fully on the ANE. A Jeff prompt prefills in 64 ms (up to 256 tokens), 254 ms (1,008 tokens) and 508 ms (2,018 tokens), and the option probabilities track the PyTorch FP32 reference within the FP16 noise band. See [M5 Max results](#m5-max-results-7-october-2026). The live-last prefix cache snapshots GDN, conv and KV after the shared prefix and prefills only the changing suffix. With a 64-row entry that suffix is one call: about 18 ms, 54–56 decisions/s, against 2.1 decisions/s for a cold 2,018-token prefill. See [Live-last prefix cache](#live-last-prefix-cache). A 1,024-row and a 2,048-row entry share weights with the 256-row entry and stay fully on the ANE; a 256-row chain is still the faster cold prefill (233 ms vs 305 ms at 1,008 tokens, 473 ms vs 610 ms at 2,018 tokens). See [Wide prefill entries](#wide-prefill-entries). A fixed-shape sweep from 256 through 2,048 rows stays fully on the ANE. Rows/s drops at 512 (about 4,150 to 3,660) and again, less, at 1,024 (to about 3,400). See [Prefill width cliff](#prefill-width-cliff). Not done: LoRA adapters, ANE temperature/ECE fit, a serving route.
- **Branch:** `cursor/jeff-decision-ane-85f5`.

Evidence labels: **Source-verified** (this checkout), **External doc** (Jeff / Unsloth / Qwen cards, not fetched as weights), **Inferred**.

## What Jeff is

Jeff is a small **decision** model, not a chat generator. You send a situation (`state`) and a list of questions with named options; one forward pass returns a calibrated probability for every option. There is no generated text to parse. **External doc:** [mstrasser/jeff-base](https://huggingface.co/mstrasser/jeff-base) (v1.3), [mstrasser/Jeff-Qwen3.5-0.8B](https://huggingface.co/mstrasser/Jeff-Qwen3.5-0.8B) (v1.2 zero-shot), [firelex/jeff](https://github.com/firelex/jeff), [Unsloth FastDecisionModel](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth).

Shared serving shape (Jev / SystemOne / Clef / Jeff / Unsloth):

| Piece | Role |
| --- | --- |
| Backbone | Qwen3.5 text stack; Jeff-0.8B is a fine-tune of `Qwen/Qwen3.5-0.8B` |
| Prompt | Instructions + questions/options + changing `state` |
| Head | Scores the supplied options; does not emit next-token text |
| Calibration | Fitted temperature so stated confidence tracks hit rate (ECE) |
| API | `state` + `questions` (`choice` / `noul` / `score`), not `/v1/chat/completions` |

Jeff v1.3 (`jeff-base`) is adapter-first: LoRA adapters on a fixed base, **live-last** prompt (fixed instructions/options first, changing state last) so the prefix can be cached. Checkpoint extras: `readout.safetensors` (trained readout over answer codes A, B, … AA, …) and `decision_config.json` (codes, token ids, temperature, `prompt_layout: live-last`). **External doc.** v1.1+ Qwen Jeff models accept up to 254 options.

Unsloth `FastDecisionModel` trains a **Clef-style** head: one prompt of input + every question/option; a small head reads the backbone hidden states over those spans and scores options together. Same non-generative contract, different readout tensor than Jeff’s answer-code head. Both are in scope for an ANE spike.

Published Apple Silicon numbers are **MLX**, not Core AI: about 28 ms/decision on M4 Max for the 0.8B card, and ~0.25 s mean including adapters/prompt on the Jeff v1.3 comparison. Treat those as an MLX baseline, not an ANE forecast. **External doc.**

## Hypothesis check

The working hypothesis was: *most Qwen converter / Core AI graph code is reusable because the backbone is still Qwen3.5 dense; the new piece is the decision head and a different serve API.*

**Partly right, wrong about “dense.”**

Qwen3.5-0.8B is the **same hybrid family** as Forge’s Qwen3.8-27B: Gated DeltaNet layers plus gated full attention on a 3:1 cadence, `qwen3_5` / `qwen3_5_text` config. It is not a dense all-attention Qwen3. **External doc:** Qwen3.5-0.8B card (`6 × (3 × GDN → FFN, 1 × gated attention → FFN)`), Ollama `qwen3.5:0.8b` `layer_types`.

The older in-repo **Qwen3-0.6B** path (`scripts/qwen3_lut_common.py`) *is* dense all-attention. That is a different architecture. Its LUT/GPTQ helpers are reusable; its reference forward is **not** a Jeff graph template.

What actually holds:

- Reuse the **hybrid GDN + attention + SwiGLU** Core AI graphs if they are retargeted to 0.8B shapes.
- Do **not** assume `forge.py convert` / `serve` will accept Jeff today (they gate 64 × 5120).
- The genuinely new work is the decision/readout graph, a prefill-only serve path, prompt layout, and calibration — plus dropping DFlash2 / next-token decode for this model.

## Shape gap (must retarget)

| | Forge today (Qwen3.8-27B) | Jeff backbone (Qwen3.5-0.8B) |
| --- | --- | --- |
| Layers | 64 (48 GDN + 16 attn) | 24 (18 GDN + 6 attn) |
| Hidden / MLP | 5120 / 17408 | 1024 / 3584 |
| Full attention | 24 Q / 4 KV, `head_dim` 256 | 8 Q / 2 KV, `head_dim` 256 |
| GDN | config-driven (`linear_num_*`) | 16 QK + 16 V, dim 128 |
| Vocab | 248320, **untied** `lm_head` | 248320, **tied** embeddings |
| Layer pattern | `full_attention_interval` 4 | same 3:1 (`layer_types`) |
| Native context | 64K prepared ladder | 262K card; decisions are short |
| Weight prefix | `model.language_model.*` | same HF text prefix (confirm on download) |

27B figures: **Source-verified** ([QUANTIZATION_NOTES.md](../../QUANTIZATION_NOTES.md), [docs/QUANTIZATION.md](../QUANTIZATION.md)). 0.8B figures: **External doc.**

`coreai/qwen38_coreai_build.py` and `scripts/qwen38_ane_chunk.py` already read mixer/attention widths from `CFG` / `layer_types`. **Source-verified.** The *launcher* does not: `forge.py` `prepare()` and `scripts/hf_release.py` reject anything other than 64 layers and hidden 5120. Chunk plans, DFlash2 taps `[5, 19, 33, 47, 61]`, and the vocab LUT4 `Head` are 27B-serving choices.

## Inventory (Qwen-touching entry points)

| Stage | Path | What it does for Qwen | Jeff fit |
| --- | --- | --- | --- |
| Launcher | [`forge.py`](../../forge.py) | 27B: `quantize` / `convert` / `serve` / `chat` (still 64 × 5120). Jeff: `jeff-convert` / `jeff-smoke` (hybrid + readout only). Core AI compile is shared and does not require a drafter. | 27B gate unchanged. Jeff commands bypass it. |
| Bundle check | [`scripts/hf_release.py`](../../scripts/hf_release.py) | Same 64 × 5120 + vocab check; `embed_tokens_fp16.npy` | Same. |
| 0.6B-era LUT | [`scripts/qwen3_lut_common.py`](../../scripts/qwen3_lut_common.py) | Dense Qwen3 fp32 ref (default `~/Models/Qwen3-0.6B`), tokenizer, Hadamard, LUT/GPTQ | Formats yes; graph no. |
| 0.6B-era MLP | [`scripts/qwen3_mlp_gptq_model.py`](../../scripts/qwen3_mlp_gptq_model.py) | Sequential GPTQ of MLPs, attention left fp32 | Lesson: small dense models did not need 27B VQ. |
| 27B quant | [`scripts/qwen38_gptq_27b.py`](../../scripts/qwen38_gptq_27b.py) | Mixed-bit GPTQ + VQ export | Skip for 0.8B first spike. |
| Core ML graph | [`scripts/qwen38_ane_chunk.py`](../../scripts/qwen38_ane_chunk.py), [`scripts/qwen38_ane_model.py`](../../scripts/qwen38_ane_model.py) | GDN/attn/MLP MIL; host embedding; `build_head` = final RMSNorm + vocab `lm_head` | Backbone primitives reusable; vocab head is the wrong output. |
| Core AI export | [`coreai/qwen38_coreai_build.py`](../../coreai/qwen38_coreai_build.py) | Multifunction chunks (`v8_<ctx>k`, `p64_<ctx>k`) + `head_T8.aimodel`; LUT inject; GDN_FAST / ATT_BLOCK | Best reuse: 6 × 4-layer chunks (or fewer), config-sized tensors, **no vocab head**. |
| Runtime | [`scripts/qwen38_coreai_model.py`](../../scripts/qwen38_coreai_model.py) | AneQwen3 API: prefill / decode / verify / snapshot / restore; Swift bridge | Prefill + snapshot stay; decode/verify/DFlash2 are unused for Jeff. |
| Serve | [`scripts/qwen38_server.py`](../../scripts/qwen38_server.py), [docs/SERVER.md](../SERVER.md) | OpenAI `POST /v1/chat/completions`, thinking, tools, speculative loop | Wrong contract. New SystemOne-style route. Prefix-cache + GDN snapshot is the reusable idea. |
| Drafter | [`coreai/dflash2_coreai_build.py`](../../coreai/dflash2_coreai_build.py), `scripts/dflash2_*` | Target `lm_head` + draft/verify | Out of scope for a decision model. |

Packaging patterns that are model-size-agnostic: shared-weight multifunction packages, `manifest.json` context ladder, ANE compile cache + `[ANE compile]` readout, bonded compile mode ([docs/ANE_COMPILE_MODE_POLICY.md](../ANE_COMPILE_MODE_POLICY.md)), Swift bridge IOSurface views, host `embed_tokens_fp16.npy` lookup.

## Reusable

**Backbone graphs (Inferred, high confidence given config-driven widths).** GDN lazy-commit, gated full attention (RoPE with `partial_rotary_factor`, Q/K RMSNorm, GQA), SwiGLU + tanh-SiLU, scale-free RMSNorm (`1 + weight`), conv1d short conv, overflow-safe softplus. These are the 27B lessons that exist *because* Qwen3.5 is hybrid, not despite it. **Source-verified** builders: `qwen38_ane_chunk.py`, `qwen38_coreai_build.py` `LayerW` / `GDNW` / `AttnW`. ANE numerics notes: [ANE_DELTANET_NUMERICS.md](../../ANE_DELTANET_NUMERICS.md), [COREAI_PORT_NOTES.md](../../COREAI_PORT_NOTES.md).

**Tokenizer / embeddings.** Qwen tokenizer files + host FP16 embedding table. 0.8B embeddings are ~248320 × 1024 × 2 ≈ 508 MB, vs ~2.5 GB at hidden 5120. Tied embeddings mean there is no separate 27B-style `lm_head.weight` unless the decision readout replaces it.

**Core AI packaging.** One `.aimodel` per chunk, multiple prefill/context entries sharing weights, compile-once cache, Swift vs Python runtime split. A 0.8B FP16 (or INT8) target should be a **small** package set: six 4-layer chunks, or one/two packages if ANEC size allows. 27B needed 16 chunks for resident-weight and compile limits, not because the math requires it.

**Prefix reuse.** `qwen38_server.py` already snapshots Gated DeltaNet / conv state at prompt boundaries so a follow-up that shares a prefix can restore. That is the live-last mechanism Jeff wants: cache instructions+options, prefill only the new `state`. **Source-verified** (server docstring and GDN snapshot). Independent Jeff requests must still `reset()`; recurrent state must not leak across unrelated decisions. **External doc:** Jeff llama.cpp notes say the same.

**Small-model conversion lessons (0.6B era).** `qwen3_lut_common.py` defaulted to `Qwen3-0.6B` and treated embeddings as tied with `lm_head`. LUT4 / INT8 / FP8 formats and GPTQ helpers still apply if a later spike wants light palettization. **Source-verified.** The 0.6B reference has **no** `linear_attn` / `layer_types`. Do not convert Jeff through that forward.

**Skip heavy VQ/GPTQ for 0.8B.** ~0.85B × 2 bytes ≈ 1.6 GB FP16; INT8 ≈ 0.8 GB. The 27B mix (`vector 2x16`, rank-64 residuals, 85-minute GPTQ) exists to fit 27B on 32 GB. **Inferred:** first Jeff spike should stay FP16 dense or INT8 per-channel.

## New

1. **Decision / readout graph.** Replace `Head` (final RMSNorm + LUT4 vocab `lm_head`, 8 rows → 248320 logits) with either Jeff’s `readout.safetensors` over answer-code positions, or a Clef/Unsloth span scorer. Then: gather N option scores, divide by the fitted temperature, softmax. Do not compile a 248k-way LM head just to throw it away.

2. **Non-autoregressive serve.** One prefill of the rendered prompt, optional prefix restore, readout, JSON probabilities. No token loop, no thinking budget, no DFlash2, no `/v1/chat/completions`. New route shaped like SystemOne / `jeff-serve` (`state`, `questions`, per-option probs).

3. **Prompt layout.** Build the Jeff / Unsloth string (v1.3 live-last: fixed block first, changing field last), apply the Qwen chat template with thinking **off**. Option keys must not be bare numbers. **External doc.**

4. **Calibration.** Jeff ships a temperature per model (and per GGUF format). An ANE FP16 or INT8 build needs its own fit on holdout rows; do not reuse the MLX or Q4_K_M number. Unsloth `FastDecisionModel.calibrate` is the same idea.

5. **Launcher / config.** Relax or bypass the 64 × 5120 gate; read `layer_types` and 24-layer plans; do not require a drafter pair.

6. **Optional later:** LoRA adapter load (v1.3), multiple questions per request, Unsloth Clef head vs Jeff readout A/B.

## Risks

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Treating 0.8B as “dense Qwen3” | Would send Jeff through `qwen3_lut_common.Qwen3` and drop GDN | Use `layer_types` / the 27B hybrid graph, retargeted |
| Hardcoded 27B serving | `forge.py`, `hf_release.py`, 16-chunk plan, DFlash2 taps, vocab head | New entry or explicit 0.8B overrides; do not silently reuse `serve` |
| Width / RoPE / GDN dim drift | 8 vs 24 Q heads; 16×128 vs 27B GDN; `partial_rotary_factor` 0.25 | Drive every tensor from `text_config`; parity vs MLX/PyTorch on a short prompt |
| Weight key / multimodal wrapper | 0.8B card includes a vision tower; Jeff may still ship `processor_config.json` | Confirm `model.language_model.*` after download; ignore vision for text Jeff |
| MLX ≠ Core AI | firelex `JEFF_BACKEND=mlx` is the supported Mac path | Core AI is a new backend: compile placement, fp16 numerics, ECE on-device |
| Recurrent leak | GDN state reused across unrelated decisions corrupts probs | `reset()` per request; snapshot only for live-last prefix |
| Calibration after quant | INT8/LUT shifts option logits | Fit temperature on the compiled ANE graph |
| Clef head vs Jeff readout | Unsloth span head ≠ answer-code readout | Spike Jeff readout first (published files); keep Clef as a second head package |
| Prefill entry shape | 27B uses T=8 verify + T=64 prefill | One 256-row prefill entry, chained for longer prompts. A 1,024-row entry was slower and partly on the GPU (measured below) |

## Implementation (Path B)

| Piece | Path | Role |
| --- | --- | --- |
| Checkpoint + prompt | [`coreai/jeff_coreai.py`](../../coreai/jeff_coreai.py) | Hybrid config, `JeffCheckpoint` (bf16 safetensors, `language_model.*` keys, vision tower skipped), 255×hidden `readout.safetensors`, host softmax. `decision_messages` / `prompt_ids`: an exact port of Jeff's prompt (system message, `Question:` / `State:` / `Options:` / `Latest:`, chat template with `enable_thinking=False`). Answer codes come from `decision_config.json` (A..Z, then single-token pairs: `BQ` is skipped, so index 68 is `BR`) |
| Core AI export | [`coreai/jeff_coreai_build.py`](../../coreai/jeff_coreai_build.py) | Rebinds `qwen38_coreai_build` widths to Jeff `text_config`; FP16 or light INT8; **no** LUT/VQ/palettize; prefill-only entries; `head_readout.aimodel` |
| Runtime | [`coreai/jeff_coreai_runtime.py`](../../coreai/jeff_coreai_runtime.py) | IOSurface `NDArray` I/O. Prompts longer than the entry run as chained TP-row calls: DeltaNet conv/recurrent state carried, every call's `k/v_new` rows written to the KV cache at its position, history mask opened up to it. T=1 readout on the last prompt row |
| Reference | [`scripts/jeff_reference.py`](../../scripts/jeff_reference.py) | Jeff's PyTorch forward (`Qwen3_5Model` FP32/bf16 + readout / temperature). Checks the prompt port against upstream `jeff/model.py` token for token. `--magnitudes` (DeltaNet FP16 headroom), `--chunk-dump` (per-chunk ANE error) |
| CLI | [`scripts/jeff_coreai_convert.py`](../../scripts/jeff_coreai_convert.py), [`scripts/jeff_coreai_smoke.py`](../../scripts/jeff_coreai_smoke.py) | Convert / host + Core AI smoke; `--cases` parity (max \|dp\|, KL, argmax, hidden cosine), `--bench`, `--dump` |
| Launcher | [`forge.py`](../../forge.py) `jeff-convert`, `jeff-smoke` | Bypasses the 27B 64 × 5120 gate **only** on these commands. `COREAI_PYTHON` names the Core AI SDK interpreter for convert, `--build` smoke and `compile` of a `jeff-decision` build |

The 27B `convert` / `serve` / `hf_release` checks are unchanged. DFlash2 is not built. `--quant int8` is per-channel INT8 on large projections only (not GPTQ/VQ).

Host smoke steps [`qwen38_decode_ref.DecodeLayer`](../../scripts/qwen38_decode_ref.py) (config-driven GDN + gated attention). It matches Jeff's PyTorch FP32 forward to 2e-6 in probability, so it is a portable golden for fixture tests. ANE uses tanh-SiLU, FP16 and the Core AI graph; it is compared against the FP32 reference, not expected to be bit-identical.

## Convert / compile / smoke (M5 Max)

Weights stay on the Mac. Default layout: `/Users/anemll/Models/jeff/jeff-base-v1.3` (`config.json`, `model.safetensors`, `readout.safetensors`, `decision_config.json`, tokenizer). Do not download them in a cloud agent.

Two interpreters on the M5 Max:

- **Forge `.venv`** (Python 3.11): transformers, for the launcher, the reference and prompt token ids.
- **Core AI SDK venv** (Python 3.12): `coreai-torch`, `coreai-opt`, `coreai.runtime`, for convert, compile and `--build` smoke. Here that is `CoreAI-experiments/experiments/coreai_gemm_bench/.venv`.

`COREAI_PYTHON` points `forge.py` at the SDK venv. Compile and smoke must run in the same interpreter: the ANE cache is keyed by the loading Python's identity. A framework Python caches as `org.python.python`; the SDK venv's caches as `python`. Compiling a `jeff-decision` build therefore also uses `COREAI_PYTHON`.

```sh
export COREAI_PYTHON=/Users/anemll/SourceRelease/GITHUB/ML_playground/CoreAI-experiments/experiments/coreai_gemm_bench/.venv/bin/python
M=/Users/anemll/Models/jeff/jeff-base-v1.3

# 1) Plan only (any OS; no SDK)
python forge.py jeff-convert --model $M --output /Users/anemll/Models/jeff-coreai \
  --ctx 2048 --prefill 256 --quant fp16 --dry-run

# 2) Export FP16 backbone + 255-way readout (macOS + Core AI SDK; runs in $COREAI_PYTHON)
python forge.py jeff-convert --model $M --output /Users/anemll/Models/jeff-coreai \
  --ctx 2048 --prefill 256 --quant fp16          # light INT8 projections: --quant int8

# 3) Specialize for this Mac's ANE in $COREAI_PYTHON (no drafter)
python forge.py compile --build /Users/anemll/Models/jeff-coreai/coreai

# 4) Golden cases: exact Jeff prompt ids + PyTorch FP32 probabilities (forge .venv; checks the port vs upstream)
python scripts/jeff_reference.py --model $M --jeff-src /Users/anemll/Models/jeff/jeff-src/src \
  --out /Users/anemll/Models/jeff/spike/parity/cases.json

# 5) Core AI prefill + readout on the ANE, parity vs the cases, timing (runs in $COREAI_PYTHON)
python forge.py jeff-smoke --model $M --build /Users/anemll/Models/jeff-coreai/coreai \
  --cases /Users/anemll/Models/jeff/spike/parity/cases.json --bench 5 \
  --out /Users/anemll/Models/jeff/spike/parity/coreai_fp16_p256.json
#    add --host for the DecodeLayer reference; --row row.json or --state/--options for one Jeff row

# 6) Placement audit of the cached specializations the smoke used
python coreai/inspect_coreai_cache.py --model-dir /Users/anemll/Models/jeff-coreai/coreai \
  --executable python --strict

# 7) Where the FP16 error enters: per-chunk ANE outputs vs HF layers, DeltaNet magnitudes
$COREAI_PYTHON scripts/jeff_coreai_smoke.py --model $M --build /Users/anemll/Models/jeff-coreai/coreai \
  --cases /Users/anemll/Models/jeff/spike/parity/cases.json --dump /Users/anemll/Models/jeff/spike/parity/dump_p256
python scripts/jeff_reference.py --model $M --rows rows.json --magnitudes \
  --chunk-dump /Users/anemll/Models/jeff/spike/parity/dump_p256 --out cases_layers.json
```

`--prefill` must be a multiple of 8 and greater than 8 (GDN sub-chunk). `--ctx` must be ≥ `--prefill`. A prompt may be up to the entry's KV rows (`pkv_len`, 2,048 here): the runtime chains `ceil(n / prefill)` calls. Output layout:

```
jeff-coreai/
  model/     config, tokenizer, decision_config, embed_tokens_fp16.npy, readout
  coreai/    manifest.json, chunk_L00-03.aimodel, …, head_readout.aimodel
```

`manifest.json` has `"kind": "jeff-decision"`, `"dflash2": false`, FP16 KV, and a single `p{prefill}_{ctx}k` entry per chunk. First `compile` / smoke load compiles each package for the ANE once.

Pass/fail for the Mac spike: convert writes packages; compile stays on the ANE (placement audit `fully_ane`); Jeff's exact prompt; Core AI option probabilities within the FP16 noise band of the FP32 reference.

## M5 Max results (7 October 2026)

Apple M5 Max, macOS 27.2 (26B5091g), Xcode 27.2 (27B5019j), ANE bonded compile mode 1 (M5 policy). Checkpoint `jeff-base` v1.3 (`decision_config` step 1113, temperature 1.0752, `prompt_layout: live-last`). Core AI SDK: coreai-core 1.0.0b2, coreai-torch 0.4.2, coreai-opt 0.2.1, torch 2.11. Reference: transformers 5.17, torch 2.14, PyTorch FP32 CPU (the `chunk_gated_delta_rule` reference kernel). Logs and JSON are in `/Users/anemll/Models/jeff/spike/{logs,parity}/` on that Mac.

### Convert and compile

| Build | Convert | Compile (`forge.py compile`) | Packages | ANE placement |
| --- | --- | --- | --- | --- |
| FP16, `p256_2k` (recommended) | 26.5 s, 4.7 GB peak RSS | 79 s (13 s per 4-layer chunk) | 6 × 159 MB chunks + 0.5 MB head; 505 MB host embedding | `fully_ane`: every entry one ANE region, no GPU region, mode 1 |
| FP16, `p1024_2k` | 79 s | 6 min 07 s (61 s per chunk) | 6 × 161 MB | **`gpu_regions_present`**: every chunk has an ANE region and a GPU region; head `fully_ane` |
| FP16, `p256_2k`, `GDN_SQ=64` | 26.6 s | 80 s | same | not audited (numerics sweep) |
| INT8 per-channel, `p256_2k` | 26.6 s | 79 s | 6 × 80 MB | `fully_ane` |

A warm load from the cache takes 0.1 s.

### Prefill on the ANE

Real Jeff prompts (built-in support-chat rows of `jeff_reference.py`, live-last, chat template), median of 5 warm runs. The 255-way readout head adds 0.34 ms.

| Prompt tokens | Options | `p256_2k` calls | `p256_2k` prefill | tok/s | `p1024_2k` prefill |
| --- | --- | --- | --- | --- | --- |
| 175 | 3 | 1 | 63.4 ms | 2,760 | 318.5 ms |
| 238 | 5 | 1 | 64.2 ms | 3,705 | 319.6 ms |
| 1,008 | 30 | 4 | 254.3 ms | 3,964 | 319.7 ms (1 call) |
| 2,018 | 100 | 8 | 508.3 ms | 3,970 | 639.4 ms (2 calls) |

A 256-row call costs 63 ms whether full or not (6 chunks); a prompt costs about 63 ms × `ceil(n / 256)`. The single 1,024-row entry is slower per token than four chained 256-row calls and partly runs on the GPU, so `--prefill 256` is the build to use. Per-call overhead below 256 rows was not measured (no `p64` / `p128` build).

### Parity against Jeff's PyTorch FP32 forward

Same token ids (port checked against upstream `jeff/model.py` on every row; unit test `test_prompt_ids_match_upstream_jeff`). Probabilities are `softmax(readout[:n] @ h / T)`.

| Prompt | FP32 answer (p) | ANE answer (p) | max \|dp\| | KL(FP32 ‖ ANE) | hidden cosine | bf16 max \|dp\| / KL / argmax |
| --- | --- | --- | --- | --- | --- | --- |
| 175 tok, 3 opt | B 0.4639 | B 0.4564 | 0.0075 | 1.1e-4 | 0.999975 | 0.0089 / 2.4e-4 / same |
| 238 tok, 5 opt | C 0.4525 | C 0.4537 | 0.0012 | 4.6e-6 | 0.999974 | 0.0092 / 2.8e-4 / same |
| 1,008 tok, 30 opt | B 0.2794 | B 0.2749 | 0.0091 | 6.6e-4 | 0.999913 | 0.0023 / 1.3e-4 / **flips** (A) |
| 2,018 tok, 100 opt | A 0.2460 | **B** 0.2433 | 0.0131 | 8.3e-4 | 0.999879 | 0.0023 / 1.6e-4 / same |

- **Host DecodeLayer** vs FP32 (175 and 238 tokens): max \|dp\| 1.1e-6 and 1.9e-6, KL ≈ 1e-11.
- **The one ANE argmax flip** is a near tie: FP32 has A 0.2460 against B 0.2448, a 0.0012 margin under the ANE's 0.013 error. bf16, the precision Jeff serves in on GPU / MLX, flips a different near tie (the 1,008-token row).
- **The readout head is not the error source.** The same readout applied on the host to the ANE hidden state gives the same distribution (`host_head` in the JSON).
- **The ANE build is inside bf16's absolute band** (max \|dp\| ≤ 0.013 vs ≤ 0.009). Its KL is about 3–5× bf16's on the 1K–2K prompts.

**INT8** (`--quant int8`, per-channel on the large projections) halves the packages at the same speed (63.2 / 63.3 / 252.0 / 504.7 ms). Its error is 2–3× FP16's:

- max \|dp\|: 0.007 / 0.011 / 0.019 / 0.022
- KL: 9.7e-5 / 3.2e-4 / 2.2e-3 / 2.8e-3
- hidden cosine: at least 0.99957
- argmax: same four answers as FP16, including the near-tie flip

FP16 stays the default: a 0.8B model has no memory pressure, and INT8 buys no speed.

### W8A8

`--quant int8` is weight-only. Chunk `L00-03` has 25 `coreai.blockwise_shift_scale` ops (INT8 weights, per-output-channel f16 scales) and each one returns f16. The following `coreai.conv2d` is f16 × f16. That chunk has 34 conv2d, 0 `quantize`, 0 `dequantize`, and 0 `f8E4M3`. The same counts hold for the other five chunks. The multiply-adds in that MIL are FP16, which is why the end-to-end speed matches the FP16 build.

A fused INT8 multiply-add on this M5 is a constant-scale `quantize` / `dequantize` pair (zero point 0) on the activation, with the weight a compile-time INT8 constant. An isolated conv, 512 rows × 4096, stack of 2, all fully on the ANE:

| Graph | Call | TOPS | vs torch |
| --- | ---: | ---: | --- |
| Per-tensor activation scale | 1.68 ms | 20.5 | within one bin |
| Per-channel scale inside `quantize` | 2.31 ms | 14.9 | up to 11 bins |
| Per-channel absolute max folded into the weight, then step 1/127 | 1.86 ms | 18.5 | max \|d\| 0.003 |

The direct per-channel `quantize` stays on the ANE and is the slower kernel. The folded form keeps the scalar pair. MIL still types the conv as f16 × f16; the fusion happens in the ANE compiler. `/Library/Caches/com.apple.aned` is mode 700, so the HWX kernel format was not read. FP8 e4m3 is rejected on this M5 (`E4M3 not supported as kernel format on this architecture`, the probe places on the GPU). There is no FP8 Jeff build.

One absolute max for a whole tensor is a poor scale here. On the 238-token prompt the layer-0 down-projection input has median absolute value 0.014 and max 3.94. That per-tensor build (`/Users/anemll/Models/jeff-coreai-w8a8`) is fully on the ANE and 51 ms per 256-row call, and the answers are wrong.

`--quant w8a8` now records a per-channel absolute max on the 238 / 1,008 / 2,018-token prompts, folds it into the INT8 weight, and quantizes the divided activation at step 1/127. Query and key skip the output quantize: an output quantize in front of RoPE makes this M5's ANEC abort with "Must be connected" and the whole chunk leaves the ANE. Every chunk's MIL: 39 scalar `quantize` to si8, 39 `dequantize` to f16, 25 `blockwise_shift_scale`, 34 `conv2d`, 0 fp8. Shared input quantizes (q/k/v, gate/up) are commoned, which is why 39 is less than one pair per projection. Build `/Users/anemll/Models/jeff-coreai-w8a8pc`, 84 MB per chunk, `fully_ane` (`mps.fullyPlacedOnANE`, `mps.noGPUActivity`).

Same harness as the rows above (median of 5). `jeff_serve` was resident under 1% CPU and no other ANE compile was running.

| Build | MB/chunk | 256-row call | rows/s | 238 tok | 1,008 tok | 2,018 tok |
| --- | ---: | ---: | ---: | --- | --- | --- |
| FP16 `w256` | 166.4 | 60.2 ms | 4,250 | 60.2 ms, KL 4.6e-6, \|Δp\| 0.0012, C = C | 240 ms, 6.6e-4, 0.0091, B = B | 482 ms, 8.3e-4, 0.013, B vs A |
| INT8 weight-only | 84.1 | 59.7 ms | 4,288 | 59.5 ms, 3.2e-4, 0.011, C = C | 240 ms, 2.2e-3, 0.019, B = B | 477 ms, 2.8e-3, 0.022, B vs A |
| W8A8 per-tensor | 83.8 | 50.9 ms | 5,027 | 51.2 ms, 0.50, 0.29, D vs C | 208 ms, 1.24, 0.26, D vs B | 417 ms, 1.14, 0.22, I vs A |
| W8A8 folded per-channel | 84.0 | 58.2 ms | 4,398 | 58.4 ms, 9.3e-3, 0.047, C = C | 232 ms, 0.084, 0.11, B = B | 463 ms, 0.135, 0.11, A = A |

The folded build picks the HF top answer on all three prompts, including the 2,018-token near-tie. Its KL is about 20× to 100× the FP16 build, and the 256-row call is 58 ms against 60 ms. The chunk is dominated by the DeltaNet and attention elementwise work, so the 18 TOPS kernel does not show up end to end. FP16 stays the serve default.

The 1.3 base without an adapter is uncertain on these rows (top probability 0.25–0.46). Parity on adapter-served prompts, where Jeff is confident, is still to be measured.

### Where the FP16 error comes from

`--chunk-dump` compares every chunk's ANE output with the HF layer output on all rows:

| After layer | rel L2 (175 tok) | rel L2 (2,018 tok) | last-row rel L2 (2,018 tok) | max \|x\| |
| --- | --- | --- | --- | --- |
| 3 | 0.019 | 0.018 | 0.024 | 1 |
| 7 | 0.018 | 0.019 | 0.019 | 2 |
| 11 | 0.021 | 0.025 | 0.019 | 1 |
| 15 | 0.019 | 0.021 | 0.018 | 3 |
| 19 | 0.014 | 0.016 | 0.016 | 7 |
| 23 | 0.010 | 0.012 | 0.009 | 27 |

The error enters in the first four layers, where the residual stream is about 1, and shrinks as the residual grows. It grows only mildly with length (last row after layer 23: 0.005 → 0.008 → 0.009 at 175 / 1,008 / 2,018 tokens). Chaining eight calls adds no step.

DeltaNet magnitudes at 2,018 tokens (`--magnitudes`), in the ANE graph's scaled units (`GDN_SQ` 16 × `GDN_SV` 64 = 1,024 on q·S, 64 on the state):

- **Output:** max 477 (layer 0), so at least 137× FP16 headroom.
- **State:** max 692 (layer 4), at least 94× headroom.
- **Low tail:** the 1st percentile of \|q·S\| is at least 9.4e-5 (layer 1), just above the FP16 normal minimum 6.1e-5.

A `GDN_SQ=64` rebuild moved no chunk error (layer 3 last row 0.0268 vs 0.0264). Final KL was 4.9e-5 / 4.1e-5 / 7.2e-4 / 7.4e-4 against 1.1e-4 / 4.6e-6 / 6.6e-4 / 8.3e-4: noise, not a fix. The 27B scales are adequate for Jeff and the default is unchanged. The early-layer error is ordinary FP16 rounding on small activations; next suspects are the layer-0..3 projections and RMSNorm, not overflow.

### Gaps and next steps

1. **LoRA adapters.** v1.3 is a base for adapters (per-adapter LoRA, readout and temperature). Merging an adapter into the FP16 weights before `jeff-convert` is the simplest path. A shared base with runtime LoRA is not built.
2. **ANE temperature / ECE.** The FP32 temperature is reused. Fit it on holdout rows with the compiled build.
3. **Longer prompts.** Jeff trains at up to 8,192 tokens. Build `--ctx 8192` (same 256-row entry, 8,192-row KV cache) and recheck placement and parity.
4. **Serving route.** A SystemOne-style `state` / `questions` endpoint around `JeffCoreAI.prepare_prefix` / `decide(handle, suffix)`.

## Live-last prefix cache

`decision_config.json` sets `prompt_layout: live-last`: the system message, question, instructions, options and every state field except the last come first, and the changing field is rendered after `Latest:`. Gated DeltaNet state cannot be rewound, so a snapshot is valid only for the exact token ids a prefill call committed.

`JeffCoreAI.prepare_prefix(token_ids, n_options=)` runs that prefix (reusing the longest cached strict prefix, which is a previous chunk boundary or prefix end) and stores, per layer, the GDN conv / recurrent / pending state, the attention KV rows `[0, pos)` and the position. The key is the prefix token ids. `JeffCoreAI.decide(handle, suffix_ids)` restores that snapshot and prefills only the suffix: the last field, the closing instruction and the generation-prompt tail. A cold `decide(token_ids, n_options)` is unchanged.

`split_live_last` returns `prefix` and `suffix` whose concatenation is `prompt_ids`. On the Qwen tokenizer the cut after `Latest:\n` is a token boundary (`271, 30938, 25, 198`), so two decisions that differ only in the last field share the prefix ids exactly.

The Jeff server's hook is the same store: `runtime.prefix_cache.lookup(token_ids)` returns the longest snapshot whose token ids are a prefix of the prompt, and `decide(token_ids, n_options, prefix=snapshot)` resumes there. `enable_prefix_cache(live_mark)` records every committed call, and splits a cold prefill at that `Latest:` mark so the next decision restores the shared prefix rather than a 256-token boundary. `capture_state()` is what `store` keeps.

```python
split = split_live_last(model, row)          # prefix, suffix, ids
handle = runtime.prepare_prefix(split["prefix"], n_options=n)
out = runtime.decide(handle, split["suffix"])  # probabilities, restore_ms, suffix_ms, total_ms
```

A short suffix still occupies a whole prefill call. `--prefill-extra 64` (and, if wanted, `32`) compiles those widths into the same packages as the 256-row entry. Weights stay shared: six chunks are still 159 MB each. Without measured call times the suffix uses the smallest width that can hold it in one call; `measure_prefill_calls` / `set_prefill_costs` picks the width whose call count times measured milliseconds is smallest. The 256-row entry stays the one used to build a long prefix.

Calls use `libcoreai_bridge.dylib` (`coreai/swift_bridge/build.sh`) when it is present. Outputs are bound once and DeltaNet state ping-pongs between two IOSurface sets. The Python binding allocates a new output surface on every call; that pool is not reclaimed and the process aborts (`NDArray+Pool.swift`, a 100 KB allocate) after a long bench. `COREAI_BRIDGE=0` forces the Python binding.

### M5 Max, 7 October 2026

Build `/Users/anemll/Models/jeff-coreai-prefix/coreai`: `p32_2k`, `p64_2k`, `p256_2k`, context 2,048, bonded mode 1. `inspect_coreai_cache.py --strict` reports `fully_ane` for every entry (one ANE region each, no GPU region). An idle 27B Core AI server was resident on port 8766; a `jeff_serve.py` process was alive during convert/compile and had exited before this timing run. The 256-row call is 58.4 ms, the same as the single-width build (57.8 ms).

One call, median of three: **p32 15.1 ms, p64 17.9 ms, p256 58.4 ms**. Every measured suffix is 36–88 tokens, so the planner never picks p32: a 41-token suffix is one p64 call (17.9 ms), not two p32 calls (30 ms). An 85-token Tetris suffix is two p64 calls (36 ms), which beats one p256 call (58 ms). p64 is the width to ship beside 256.

Cache versus a cold prefill of the same ids. The suffix is its own call, so the FP16 reduction order differs from a cold prefill that fuses the prefix tail with the suffix. Worst max |Δp| is 0.0026, under the 0.005 gate, and the argmax matches on every row. Repeats, and a cold prefill after cached decisions, are bit-identical (max |Δp| 0). Median of five decisions. JSON: `/Users/anemll/Models/jeff/spike/parity/prefix_cache_p32_p64_p256.json`.

| Case | Tokens | Prefix | Suffix | Suffix entries | Cached | Cold | max \|Δp\| vs cold | max \|Δp\| vs HF FP32 |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| readme, 3 options | 175 | 134 | 41 | p64 | 17.7 ms, 56.5/s | 58.6 ms, 17.1/s | 0.0021 | 0.0070 |
| 5 options | 238 | 197 | 41 | p64 | 17.9 ms, 55.9/s | 59.4 ms, 16.8/s | 0.0018 | 0.0009 |
| 30 options | 1,008 | 967 | 41 | p64 | 17.8 ms, 56.2/s | 234 ms, 4.3/s | 0.0024 | 0.0067 |
| 100 options | 2,018 | 1,977 | 41 | p64 | 18.6 ms, 53.7/s | 467 ms, 2.1/s | 0.0026 | 0.0157 |
| 100 options, message changed | 2,023 | 1,977 | 46 | p64 | 18.3 ms, 54.6/s | 468 ms, 2.1/s | 0.0019 | 0.0038 |
| Snake A | 170 | 134 | 36 | p64 | 18.0 ms, 55.6/s | 58.1 ms, 17.2/s | 0.0014 | 0.0033 |
| Snake B | 171 | 134 | 37 | p64 | 17.8 ms, 56.1/s | 58.2 ms, 17.2/s | 0.0011 | 0.0035 |
| Tetris A | 242 | 157 | 85 | p64+p64 | 35.7 ms, 28.0/s | 58.4 ms, 17.1/s | 0.0010 | 0.0059 |
| Tetris B | 245 | 157 | 88 | p64+p64 | 35.8 ms, 27.9/s | 58.2 ms, 17.2/s | 0.0022 | 0.0047 |

Argmax matches HF FP32 on every row except the 2,018-token 100-option prompt, where both the cached path and the cold path answer B and FP32 does not. That is the published near-tie (FP32 A 0.2460 vs B 0.2448); cold vs FP32 max |Δp| is 0.013. Snake A/B both answer A, Tetris A/B both answer D, and the second 100-option message answers D on both paths. Restoring the snapshot is 0.3 ms (short prefix) to 0.7 ms (1,977 tokens). A prefix longer than 256 tokens reuses the first 256-token snapshot (`chunk_reuse` 256 on the 1,008- and 2,018-token rows).

`scripts/jeff_prefix_bench.py prepare` writes Snake, Tetris and the published parity rows (including a second ~2K-token, 100-option message) with the split and HF FP32 probabilities. `run` checks cache against a cold prefill and against that reference, and reports decisions/sec.

## Wide prefill entries

`jeff-convert --prefill 256 --prefill-extra 1024,2048` puts `p256_2k`, `p1024_2k` and `p2048_2k` in one package per chunk. Weights are shared. Each chunk is 179.5 MB (the 32+64+256 package is 159 MB). KV history stays 2,048 rows, the same buffer for every width. Build: `/Users/anemll/Models/jeff-coreai-wide/coreai`.

The earlier 1,024-row-only package (`jeff-coreai-p1024`) compiled in about 6 minutes (61 s per chunk) and left one GPU region. Its graph contains one `mps.tile` and the string `Unsupported mps.tile op for this ANE architecture`. The producer is the causal mask in `AttnW.forward`: a `(T, T)` fp16 mask passed through `repeat(grp, 1)` with `grp = 4`. At 256 rows the compiler emits no `mps.tile`. Jeff's Gated DeltaNet head repeat has factor 1 (16 key heads, 16 value heads), so that path is a reshape.

The mask is now added in `(group, T, T)` layout, which broadcasts the `(T, T)` constant. The factor-1 GDN repeat is omitted. KV stays fp16. The triangular mask is still an fp16 constant. `inspect_coreai_cache.py --strict` then reports `fully_ane` for `p256_2k`, `p1024_2k`, `p2048_2k` and the readout, bonded mode 1, three ANE regions per chunk, `mps.tile` count 0, no GPU region and no unsupported-op string. A standalone SDPA of the same shape (history 2,048, head dim 256) also lands entirely on the ANE at both widths. The existing split-softmax path and the 4,096-row prefill tile were left unchanged.

Export of the six chunks and the head took 4 min 31 s. ANE specialization of one three-entry chunk took 4 min 55 s (layers 0–3) and 5 min 03–09 s for each of the other five chunks (30 min 16 s for the six chunks). The readout loaded from cache in under a second.

One fixed-shape call, median of five, Swift bridge: **p256 58.5 ms, p1024 307 ms, p2048 604 ms**. The published prompts are 1,008 and 2,018 tokens, so the wide call is one partial entry (the extra rows stay invalid) and the chain is four or eight 256-row calls. Median of five prefills. JSON: `/Users/anemll/Models/jeff/spike/parity/wide_prefill.json`. An idle 27B server was resident at 0% CPU. `jeff_serve.py` was not running during this timing.

| Prompt | Tokens | Plan | Prefill | KL vs HF FP32 | max \|Δp\| vs HF | Top answer |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| 30 options | 1,008 | 1×1024 | 305 ms | 4.2e-4 | 0.0061 | B, same as HF |
| 30 options | 1,008 | 4×256 | 233 ms | 6.6e-4 | 0.0091 | B, same as HF |
| 100 options | 2,018 | 1×2048 | 610 ms | 9.1e-4 | 0.0140 | B, HF says A |
| 100 options | 2,018 | 8×256 | 473 ms | 8.3e-4 | 0.0131 | B, HF says A |

Wide versus the 256-row chain on the same ids: max |Δp| 0.0031 (1,008 tokens) and 0.0026 (2,018 tokens), same top answer on both. The 2,018-token row is the published near-tie (FP32 A 0.2460 vs B 0.2448); both ANE plans answer B, as the cold 256-row path did before. The previous GPU-spill 1,024-row call was 319 ms, and two of those calls covered 2,018 tokens in 639 ms. Fully on the ANE, one 1,024-row call is 305 ms and one 2,048-row call is 610 ms. Both are slower than the 256-row chain (233 ms and 473 ms; the published chain figures were 254 ms and 508 ms). A cold prefill uses the largest entry, so this build's cold path is the slower one. The prefix-cache build (`p64` beside `p256`) stays the one to serve.

`scripts/jeff_wide_prefill.py` runs the two plans against the stored HF FP32 probabilities.

## Prefill width cliff

Each width is its own package (`jeff-coreai-w<T>`, ctx 2048, fp16, one prefill entry). KV history is 2,048 rows at every width (`pkv_len` 2048). Compile time is the `forge.py compile` wall clock for the six chunks and the readout. Placement is the compile-cache audit (`mps.fullyPlacedOnANE`, `mps.noGPUActivity`, GPU/CPU region names, `mps.tile` count, unsupported-op strings). The call time is the median of five full-shape calls (eight live tokens, the rest masked; the ANE still runs the whole entry). Rows/s is the width divided by that call. Parity is one prompt, `t1024_30opt` (1,008 tokens, 30 options), against the stored HF FP32 probabilities. A width narrower than 1,008 chains that entry; a wider width is one partial call. JSON: `/Users/anemll/Models/jeff/spike/parity/width_sweep.json`. `jeff_serve.py` was not running during these compiles.

| Width | Query rows | Compile | Placement | Call | rows/s | KL vs HF | max \|Δp\| | Top |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 256 | 1,024 | 122 s | fully ANE | 57.8 ms | 4428 | 6.6e-4 | 0.0091 | B, same as HF |
| 384 | 1,536 | 115 s | fully ANE | 92.1 ms | 4170 | 8.4e-4 | 0.0109 | B, same as HF |
| 448 | 1,792 | 145 s | fully ANE | 109.7 ms | 4085 | 4.6e-4 | 0.0065 | B, same as HF |
| 480 | 1,920 | 140 s | fully ANE | 115.7 ms | 4148 | 4.5e-4 | 0.0048 | B, same as HF |
| 512 | 2,048 | 152 s | fully ANE | 139.8 ms | 3663 | 4.5e-4 | 0.0063 | B, same as HF |
| 640 | 2,560 | 195 s | fully ANE | 174.3 ms | 3671 | 4.0e-4 | 0.0050 | B, same as HF |
| 768 | 3,072 | 236 s | fully ANE | 211.1 ms | 3638 | 5.3e-4 | 0.0070 | B, same as HF |
| 896 | 3,584 | 309 s | fully ANE | 251.8 ms | 3559 | 5.6e-4 | 0.0069 | B, same as HF |
| 960 | 3,840 | 345 s | fully ANE | 271.7 ms | 3534 | 5.4e-4 | 0.0066 | B, same as HF |
| 1024 | 4,096 | 379 s | fully ANE | 304.7 ms | 3361 | 4.2e-4 | 0.0061 | B, same as HF |
| 1536 | 6,144 | 716 s | fully ANE | 449.3 ms | 3419 | 4.2e-4 | 0.0061 | B, same as HF |
| 2048 | 8,192 | 1,328 s | fully ANE | 603.7 ms | 3393 | 4.2e-4 | 0.0061 | B, same as HF |

Every graph is fully on the ANE: no GPU region, no CPU region, `mps.tile` count 0, no unsupported-op string. Package size grows from 166.4 MB per chunk at 256 rows to 176.1 MB at 2,048. Compile time grows smoothly (about 20 s per chunk at 256, 25 s at 512, 63 s at 1,024, 221 s at 2,048) and no width failed to compile.

```
rows/s
4428 | 256
4170 | 384
4148 |   480
4085 | 448
3663 |     512
3671 |       640
3638 |         768
3559 |           896
3534 |             960
3361 |               1024
3419 |                  1536
3393 |                     2048
```

The rows/s cliff is the **512-row** entry. 480 rows still run at 4,148 rows/s (call 116 ms). 512 rows drop to 3,663 rows/s (call 140 ms). Width grows 6.7% and the call grows 21%. From 512 through 960, latency scales with width and rows/s sits near 3,530–3,670. A second, smaller step is the **1,024-row** entry: 960 rows are still 3,534 rows/s (call 272 ms) and 1,024 rows are 3,361 rows/s (call 305 ms). From 1,024 through 2,048, rows/s stays near 3,400 and latency scales with width.

At 512 the attention score tensor's query axis reaches 2,048. Jeff attention has 2 KV heads and 4 query groups, so queries are `(2, 4T, 256)`. History scores are `(2, 4T, 2048)` and the within-block scores are `(2, 4T, T)`; softmax runs over the concatenation, width `2048 + T`. That query axis is 1,920 at 480 rows and 2,048 at 512. The same width is where the GDN prefill unrolls 64 sub-chunks of 8 (60 sub-chunks at 480). The sub-chunk stays 8. Query rows equal 32 times the GDN block count, so the two counts step at the same widths and this sweep does not separate them. The second step is the same pair of counts at the next power of two: query axis 4,096 and 128 GDN sub-chunks, at 1,024 rows (3,840 query rows and 120 sub-chunks at 960).

KV history width is 2,048 at every point in the table, so it is not what moves at either step. The causal mask is a `(T, T)` fp16 broadcast. The earlier `repeat` path emitted `mps.tile` and one GPU region at 1,024 rows; after the broadcast, `mps.tile` stays 0 through 2,048 rows.

The 256-row chain is still the faster cold prefill of a real prompt (233 ms vs 305 ms at 1,008 tokens, 473 ms vs 610 ms at 2,018 tokens). The prefix-cache build (`p64` beside `p256`) stays the one to serve.

The spike's Core ML readout head in `/Users/anemll/Models/jeff/spike/coreml/` is not used here. Its normalization multiplies `amax` back in and is scale-incorrect; the Core AI head uses `rms_hidden`. The live-last layout comes from `decision_config.json` (the external feasibility note's "state-first" was wrong for v1.3).

## What was verified where

- **Linux cloud VM:** implemented Path B and ran the unit tests (no Core AI SDK, no Jeff weights).
- **M5 Max (this section):** convert, compile, placement audit, prefill timing and FP32 parity with the local weights. Unit tests: `python -m unittest tests.test_jeff_coreai tests.test_launcher tests.test_checkpoint` (34 tests, including the upstream token-id check). The prefix-cache tests (`tests.test_jeff_prefix_cache`) cover the `Latest:` cut, Snake/Tetris shared prefixes and the ~2K-token split; the M5 bench is the table above. The wide-entry placement and the 1,008 / 2,018-token timings are in [Wide prefill entries](#wide-prefill-entries). The per-width placement, call time and rows/s sweep is in [Prefill width cliff](#prefill-width-cliff).

## See also

- [docs/EXPERIMENTS.md](../EXPERIMENTS.md) — experiment map
- [docs/SERVER.md](../SERVER.md) — current generative API
- [COREAI_PORT_NOTES.md](../../COREAI_PORT_NOTES.md) — Core AI port
- [QUANTIZATION_NOTES.md](../../QUANTIZATION_NOTES.md) — 27B hybrid layout
