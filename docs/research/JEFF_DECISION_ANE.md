# Jeff / Unsloth decision models on the ANE

- **Scope:** run Jeff (Qwen3.5-0.8B decision head) on the Apple Neural Engine through a prefill-only Core AI path.
- **Status:** Path B implemented on this branch. Config-driven hybrid graphs, 255-way readout instead of the vocab `lm_head`, `forge.py jeff-convert` / `jeff-smoke` (no 64 × 5120 gate). No Hugging Face weights were downloaded in the cloud agent. No ANE compile or timing here. Nothing below is a measured tokens/s claim.
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
| Prefill entry shape | 27B uses T=8 verify + T=64 prefill | Decision serve only needs a prefill entry (start with 512–2048 rows) |

## Implementation (Path B)

| Piece | Path | Role |
| --- | --- | --- |
| Checkpoint + readout | [`coreai/jeff_coreai.py`](../../coreai/jeff_coreai.py) | Hybrid config, `JeffCheckpoint`, 255×hidden `readout.safetensors`, live-last prompt, host softmax |
| Core AI export | [`coreai/jeff_coreai_build.py`](../../coreai/jeff_coreai_build.py) | Rebinds `qwen38_coreai_build` widths to Jeff `text_config`; FP16 or light INT8; **no** LUT/VQ/palettize; prefill-only entries; `head_readout.aimodel` |
| Runtime | [`coreai/jeff_coreai_runtime.py`](../../coreai/jeff_coreai_runtime.py) | One prefill through every chunk, T=1 readout on the last valid row |
| CLI | [`scripts/jeff_coreai_convert.py`](../../scripts/jeff_coreai_convert.py), [`scripts/jeff_coreai_smoke.py`](../../scripts/jeff_coreai_smoke.py) | Convert / host+Core AI smoke |
| Launcher | [`forge.py`](../../forge.py) `jeff-convert`, `jeff-smoke` | Bypasses the 27B 64 × 5120 gate **only** on these commands |

The 27B `convert` / `serve` / `hf_release` checks are unchanged. DFlash2 is not built. `--quant int8` is per-channel INT8 on large projections only (not GPTQ/VQ).

Host smoke steps [`qwen38_decode_ref.DecodeLayer`](../../scripts/qwen38_decode_ref.py) (config-driven GDN + gated attention). That is the portable correctness path. ANE uses tanh-SiLU and the Core AI graph; treat host vs ANE as a later parity check, not as bit-identical.

## Convert / compile / smoke (M5 Max)

Weights stay on the Mac. Default layout: `/Users/anemll/Models/jeff/jeff-base-v1.3` (`config.json`, `model.safetensors`, `readout.safetensors`, `decision_config.json`, tokenizer). Do not download them in a cloud agent.

Conversion SDK: Core AI Python stack from [ENVIRONMENT.md](../ENVIRONMENT.md) (`coreai-torch`, `coreai-opt`). Inference compile uses the same Python as `forge.py serve` so the ANE cache key matches.

```sh
# 1) Plan only (any OS; no SDK)
python forge.py jeff-convert \
  --model /Users/anemll/Models/jeff/jeff-base-v1.3 \
  --output /Users/anemll/Models/jeff-coreai \
  --ctx 2048 --prefill 256 --quant fp16 --dry-run

# 2) Export FP16 backbone + 255-way readout (macOS + Core AI SDK)
python forge.py jeff-convert \
  --model /Users/anemll/Models/jeff/jeff-base-v1.3 \
  --output /Users/anemll/Models/jeff-coreai \
  --ctx 2048 --prefill 256 --quant fp16

# Light INT8 projections (optional; still no GPTQ/VQ):
#   ... --quant int8

# 3) Specialize for this Mac's ANE (no drafter)
python forge.py compile --build /Users/anemll/Models/jeff-coreai/coreai

# 4) Host prefill + readout (works without Core AI; needs torch + the checkpoint)
python forge.py jeff-smoke \
  --model /Users/anemll/Models/jeff/jeff-base-v1.3 \
  --state "The disk on db-02 is 97 percent full and still growing." \
  --options page,wait,ignore

# 5) Same prompt through the compiled Core AI package
python forge.py jeff-smoke \
  --model /Users/anemll/Models/jeff/jeff-base-v1.3 \
  --build /Users/anemll/Models/jeff-coreai/coreai \
  --state "The disk on db-02 is 97 percent full and still growing." \
  --options page,wait,ignore
```

`--prefill` must be a multiple of 8 and greater than 8 (GDN sub-chunk). `--ctx` must be ≥ `--prefill`. Output layout:

```
jeff-coreai/
  model/     config, tokenizer, decision_config, embed_tokens_fp16.npy, readout
  coreai/    manifest.json, chunk_L00-03.aimodel, …, head_readout.aimodel
```

`manifest.json` has `"kind": "jeff-decision"`, `"dflash2": false`, FP16 KV, and a single `p{prefill}_{ctx}k` entry per chunk. First `compile` / smoke load compiles each package for the ANE once.

Pass/fail for the Mac spike: convert writes packages; compile stays on the ANE; host smoke returns option probabilities; Core AI smoke on the same prompt returns a distribution (compare argmax, then ECE later).

## Recommended first spike (M5 Max / M6, no 27B VQ)

Goal: prove ANE **compile + one-forward prefill** for a 0.8B-class hybrid, not product serving. The convert/smoke commands above are that spike. Remaining after a green Mac compile: time 256/512/1024/2048 prefill, compare option order vs MLX `jeff-serve`, fit ANE temperature/ECE, then live-last prefix cache and a SystemOne route. Adapters and Unsloth Clef head stay later.

## What this VM verified

- Implemented the Jeff convert/smoke path and ran `python3 -m unittest tests.test_jeff_coreai tests.test_launcher` (no Core AI SDK, no Jeff weights).
- Checked public Jeff / Unsloth / Qwen3.5-0.8B docs for task, head, and architecture. Did not download weights.
- Did not run Core AI export, ANE compile, or on-device timing.

## See also

- [docs/EXPERIMENTS.md](../EXPERIMENTS.md) — experiment map
- [docs/SERVER.md](../SERVER.md) — current generative API
- [COREAI_PORT_NOTES.md](../../COREAI_PORT_NOTES.md) — Core AI port
- [QUANTIZATION_NOTES.md](../../QUANTIZATION_NOTES.md) — 27B hybrid layout
