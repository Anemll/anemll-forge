# Jeff / Unsloth decision models on the ANE

- **Scope:** can ANEMLL Forge run Jeff (Qwen3.5-0.8B decision head) and Unsloth-style Clef heads on the Apple Neural Engine through the existing Core AI path?
- **Status:** source inventory plus external-doc check. No Hugging Face weights were downloaded here. No ANE compile or timing. Nothing below is a measured tokens/s or millisecond claim.
- **Branch:** `cursor/jeff-decision-ane-85f5`. Analysis only; no converter or server code.

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
| Launcher | [`forge.py`](../../forge.py) | `quantize` → `qwen38_gptq_27b.py`; `convert` → `qwen38_ane_model.py build_v3`; `serve`/`chat` → `qwen38_server.py` / `qwen38_chat.py`; Core AI compile | Hard-gates 64 × 5120. Needs a Jeff command or a relaxed gate. |
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

## Recommended first spike (M5 Max / M6, no 27B VQ)

Goal: prove ANE **compile + one-forward prefill** for a 0.8B-class hybrid, not product serving.

1. On the Mac, download `mstrasser/jeff-base` revision `v1.3` (or `Jeff-Qwen3.5-0.8B` v1.2 if zero-shot is enough). Do not pull the 27B bundle for this. Record `config.json` `text_config` (layers, hidden, `layer_types`, GDN/attn heads).
2. Export **backbone only**, FP16 dense (or INT8 per-channel if FP16 compile is ugly): 24 layers, host embedding table, **no** vocab `lm_head`. Reuse `qwen38_coreai_build.py` with a 24-layer plan (`0-3,…,20-23` or fewer packages). Stub the head as RMSNorm + a tiny linear over hidden (identity or zeros) so the package links.
3. Compile for ANE (`forge.py compile` pattern or `coreai_compile.py`). Record package count, compile time, bonded mode, and whether every function stays on the ANE (existing placement probes).
4. Time **prefill only** at 256 / 512 / 1024 / 2048 tokens. That *is* the decision cost. Do not start a decode loop.
5. Wire Jeff readout as a second increment: gather answer-code hidden or logits, apply `decision_config.json` temperature, softmax. Compare option order vs MLX `jeff-serve` on a handful of synthetic `choice` rows. Then fit ECE.
6. Only after that: live-last prefix cache via existing GDN snapshot, then a `/v1/systemone`-style stub. Adapters and Unsloth Clef head come later.

Pass/fail for the spike: packages compile onto the ANE; one prefill of a ~512-token Jeff prompt returns; latency is in the same ballpark as the published MLX 0.8B figure (tens of milliseconds, not 27B prefill seconds). Quality and calibration are the next gate, not this one.

## What this VM verified

- Read the Forge Qwen entry points listed above on `main` (`e671153`).
- Checked public Jeff / Unsloth / Qwen3.5-0.8B docs for task, head, and architecture. Did not download weights.
- Did not run `forge.py`, Core AI, or any compile.

## See also

- [docs/EXPERIMENTS.md](../EXPERIMENTS.md) — experiment map
- [docs/SERVER.md](../SERVER.md) — current generative API
- [COREAI_PORT_NOTES.md](../../COREAI_PORT_NOTES.md) — Core AI port
- [QUANTIZATION_NOTES.md](../../QUANTIZATION_NOTES.md) — 27B hybrid layout
