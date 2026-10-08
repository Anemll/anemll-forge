# Stock Qwen3.5-0.8B on the Apple Neural Engine

Baseline for the Colab Unsloth run: the stock decoder and tied LM head, not Jeff's 255-way readout.

## Source

- Weights: `Qwen/Qwen3.5-0.8B` revision `2fc06364715b967f1860aea9cf38778875588b17` (the revision in Jeff v1.3 `base_model`).
- Unsloth `FastDecisionModel` example loads `unsloth/Qwen3.5-*`. For this size that repo is `unsloth/Qwen3.5-0.8B` revision `23c69c53358a07516b5827588b3fdb12ae78fd65`.
- Safetensors SHA-256 of both repos: `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` (1,746,942,600 bytes). The weight files match. Chat-template text can still differ; this run uses the Qwen snapshot.
- Local copy: `/Users/anemll/Models/qwen35-0.8b-stock` (hardlink of the existing snapshot). Build: `/Users/anemll/Models/qwen35-0.8b-stock-coreai`.

## Build

Same backbone as `jeff-coreai`: 6 chunks of 4 layers, FP16, prefill entry `p256_2k`, context 2048, `SILU=tanh`, `GDN_FAST=1`, `ATT_BLOCK=2048`, `ATT_BLOCK_PREFILL=4096`.
Tied embeddings: no `lm_head` tensor. The head is final RMSNorm plus `embed_tokens`, split into 16 convs of 15,520 rows (248,320 / 16) so each output channel count stays under 16,384.
Token embedding lookup stays on the host, as in the Jeff runtime. There is no separate verify/decode graph; a new token reuses `p256_2k`.
Inference uses the Swift bridge. Each tensor is one IOSurface, bound once and reused. The Python Core AI runtime allocates a new IOSurface per call and runs out of surfaces on this eval.

## Placement

Every backbone chunk and every LM-head slice is `fully_ane`: one ANE region and zero GPU regions.

| Package | Status | ANE regions | GPU regions |
| --- | --- | ---: | ---: |
| `chunk_L00-03.aimodel` | fully_ane | 1 | 0 |
| `chunk_L04-07.aimodel` | fully_ane | 1 | 0 |
| `chunk_L08-11.aimodel` | fully_ane | 1 | 0 |
| `chunk_L12-15.aimodel` | fully_ane | 1 | 0 |
| `chunk_L16-19.aimodel` | fully_ane | 1 | 0 |
| `chunk_L20-23.aimodel` | fully_ane | 1 | 0 |
| `head_lm_00.aimodel` | fully_ane | 1 | 0 |
| `head_lm_01.aimodel` | fully_ane | 1 | 0 |
| `head_lm_02.aimodel` | fully_ane | 1 | 0 |
| `head_lm_03.aimodel` | fully_ane | 1 | 0 |
| `head_lm_04.aimodel` | fully_ane | 1 | 0 |
| `head_lm_05.aimodel` | fully_ane | 1 | 0 |
| `head_lm_06.aimodel` | fully_ane | 1 | 0 |
| `head_lm_07.aimodel` | fully_ane | 1 | 0 |
| `head_lm_08.aimodel` | fully_ane | 1 | 0 |
| `head_lm_09.aimodel` | fully_ane | 1 | 0 |
| `head_lm_10.aimodel` | fully_ane | 1 | 0 |
| `head_lm_11.aimodel` | fully_ane | 1 | 0 |
| `head_lm_12.aimodel` | fully_ane | 1 | 0 |
| `head_lm_13.aimodel` | fully_ane | 1 | 0 |
| `head_lm_14.aimodel` | fully_ane | 1 | 0 |
| `head_lm_15.aimodel` | fully_ane | 1 | 0 |

## Parity vs Hugging Face torch fp32 (CPU)

Per-position top-1 is argmax agreement of the full vocabulary at every prompt position. KL is KL(fp32 ‖ ANE) on the final position. Cosine is the last backbone row before the final RMSNorm.

| Prompt | Tokens | Per-position top-1 | Final top-1 | KL(fp32 ‖ ANE) | Last-hidden cosine |
| --- | ---: | ---: | --- | ---: | ---: |
| capital | 30 | 96.7% | match | 1.843e-03 | 0.999473 |
| sum | 33 | 100.0% | match | 9.963e-04 | 0.999689 |
| colors | 21 | 90.5% | match | 5.735e-04 | 0.999821 |
| story | 38 | 100.0% | match | 2.101e-04 | 0.999785 |
| snake-sample | 179 | 98.3% | match | 3.161e-04 | 0.999792 |

## Latency

Load 0.35 s. Prompt `capital` (30 tokens).

| Measurement | ms |
| --- | ---: |
| Cold prefill (first call after load) | 66.61 |
| Cached prefill (median of 5) | 58.73 |
| Prefix hit (same prompt, no backbone calls) | 0.59 |
| Decode backbone, one new token on `p256_2k` (median of 8) | 58.4 |
| Decode LM head, 16 slices (median of 8) | 5.51 |

Each decode step is one new token on the p256_2k prefill entry plus 16 LM-head slices.

Cached prefill calls (ms): 58.87, 58.73, 58.75, 58.68, 58.68.

## Greedy smoke

Prompt `sum`, up to 24 new tokens, argmax, stop on eos.

```
42<|im_end|>
```

## Zero-shot Snake

256 rows from `/Users/anemll/Models/jeff-snake-data/heldout.jsonl`. Chat prompt contains the rules, the board, and the four options. The score is the LM-head logit of each option word.

| Scorer | Accuracy |
| --- | ---: |
| Torch fp32, bare tokens `up/down/left/right` | 20.7% |
| Torch fp32, leading-space tokens | 20.7% |
| ANE, bare tokens | 20.7% |
| ANE, leading-space tokens | 20.7% |
| ANE bare vs torch bare (same prediction) | 100.0% |

Torch snake correct 53 / 256.
Torch bare-token prediction counts: up 0, down 0, left 0, right 256.
