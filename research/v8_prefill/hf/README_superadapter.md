---
base_model:
  - Qwen/Qwen3.8-27B
  - harness-hub/Qwen3.8-27B-super-adapter
base_model_relation: merge
license: apache-2.0
pipeline_tag: text-generation
library_name: anemll-forge
tags:
  - coreai
  - ane
  - metal
  - quantized
  - lora
  - agent
  - coding-agent
  - pi
  - speculative-decoding
  - dflash2
  - tree-verification
  - gpu-ane-prefill
  - hybrid
  - long-context
---

# ANEMLL Forge · Qwen3.8-27B + harness super-adapter · hybrid Neural Engine + GPU build, 96K context (research preview)

> **Research work, not a release.** This repository is an experimental research build. It needs System Integrity
> Protection disabled and a one-time root permission, runs only on an M6 with macOS 27.3, and is tested only on the M6
> that made it. The same build without the adapter:
> [anemll/anemll-forge-qwen3.8-27B-V8-96K](https://huggingface.co/anemll/anemll-forge-qwen3.8-27B-V8-96K); the
> supported release: [anemll/anemll-forge-qwen3.8-27B-192K](https://huggingface.co/anemll/anemll-forge-qwen3.8-27B-192K).

**Start here:**

- **Build, deploy and test:** [DEPLOY.md](DEPLOY.md), section 8b, and [AGENTS.md](AGENTS.md) for coding agents: every
  step of the V8-96K build with `VARIANT=sa` (this build, its GPU-table kit and its smoke checksum).
- **How the adapter was added, and every result:**
  [results/24nov/adapter/README.md](https://github.com/Anemll/anemll-forge/blob/v8-gpu-ane-prefill/research/v8_prefill/results/24nov/adapter/README.md)
  on the research branch (the same file is in this repository under `code/anemll-forge/research/v8_prefill/`).
- **Scripts:** [research/v8_prefill/adapter/](https://github.com/Anemll/anemll-forge/tree/v8-gpu-ane-prefill/research/v8_prefill/adapter)
  (export, 8K test build, 96K A/B).
- **The base build and its research:** the V8-96K card, the tree-verification write-up
  [TREE_VERIFY.md](code/anemll-forge/research/v8_prefill/TREE_VERIFY.md) and the article
  [One model, two accelerators](https://github.com/Anemll/anemll-forge/blob/v8-gpu-ane-prefill/research/v8_prefill/ARTICLE_ANE_GPU_M6.md).

**What this is.** The V8-96K tree build (release 0.2 weights, Neural Engine + GPU prefill, DFlash2 drafter, tree8
verification, sliced KV widths, 96K context) with the harness-hub **super-adapter** added: a LoRA trained with
reinforcement learning inside the coding agents [pi](https://github.com/earendil-works/pi), Hermes Agent and OpenCode
([harness-hub/Qwen3.8-27B-super-adapter](https://huggingface.co/harness-hub/Qwen3.8-27B-super-adapter), blog
[Harness adapters at scale](https://huggingface.co/blog/harness-adapters-at-scale)). Its authors report, with thinking on
and BF16 weights, equal or higher pass rates on SWE-bench Pro and Terminal-Bench 2.1 under all three agents and 19-48 %
fewer tool calls per task.

- **No re-quantization.** The 3-bit weights are the release's, bit for bit. The adapter runs as exact fp16 low-rank
  side paths next to them (`y += a (b x)`, 0.89 GiB): rank 96 on every MLP (gate / up / down, 64 layers) and, on the
  16 attention layers, appended to the release's own rank-64 corrections (rank 160, zero-padded to 192 for the GPU's
  64-row tiles). Gated DeltaNet, embeddings and the output head are untouched, as in the adapter.
- **Build:** [coreai/m6-96k-dec8ts-sa/](coreai/m6-96k-dec8ts-sa) (24 Nov), the same 8 packages x 81 entry points as
  m6-96k-dec8ts; [gpu/w_dec8ts_sa/](gpu/w_dec8ts_sa), its GPU-table kit.

ANEMLL independently converts and quantizes [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B); credit for
the target belongs to the **Qwen Team and Alibaba Cloud** (Copyright 2026 Alibaba Cloud, Apache-2.0). The adapter is
the work of the **harness-hub** team (Apache-2.0, revision `ADAPTER_SHA`, recorded in
[adapter/ADAPTER_SOURCE.json](adapter/ADAPTER_SOURCE.json)); ANEMLL converted its factors without training them. The
drafter derives from [ProCreations/Ternary-Bonsai-2-27B-DFlash2](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2)
(z-lab DFlash2, Prism Bonsai provenance in its notice). No endorsement by these projects or Apple is claimed.

## How the adapter was added

`research/v8_prefill/adapter/adapter_export.py` writes a new export: each adapted matrix gets `lr_a = 2 B` and
`lr_b = A` (PEFT scale alpha / r = 192 / 96) in the original weight basis, the convention of the export's own low-rank
factors; the builder moves `b` into the rotated MLP / attention basis. Checks on layer 3 against the BF16 checkpoint:

| matrix | adapter change / weight | release quantization error | adapted - base export vs the adapter's change | builder path vs (W + delta) x |
|---|---|---|---|---|
| MLP gate / up / down | 0.10-0.19 % | 21-24 % | 3e-4 | exact (0.00000) |
| attention q / k / o | 0.12 % | 36 % / 1 % (INT8) / 42 % | 3e-4 | exact |

The adapter moves each weight by 0.1-0.2 % while the 3-bit quantization error is 21-42 %: merging it into the
weights and re-quantizing would bury it, so it stays a separate exact path. The tree build's chunked MLP
(`v8/chunkmlp.py`) and the GPU prefill's low-rank kernel (`pipe/engine/gpu_v8_f8q.swift`, rank now read from the
tables, identical output at rank 64) were extended for it; without an adapter both produce the base build unchanged
(smoke `29aef2ec9db18a3f`).

## Measured (M6, 32 GB, macOS 27.3, 24 Nov)

**Accuracy, 8K Core AI test build** (KL-512 harness: 64 sequences, 40,023 positions, teacher forcing against BF16):

| build | mean KL vs BF16 | top-1 agreement | perplexity (BF16 2.136) |
|---|---|---|---|
| release, no adapter | 0.0518 | 91.89 % | 2.131 |
| + super-adapter | 0.0533 | 91.67 % | 2.131 |

The adapter keeps the distance to BF16 and the perplexity, and changes the next token at 1.4 % of positions of this
generic (non-agent) trace (KL between the two builds 0.0016).

**Speed, 96K tree build against the same build without the adapter** (test port, greedy, 320 tokens, tree8 + slices,
one fresh server per arm; decode tok/s token-weighted):

| set | without | with adapter | change | tokens per verify | ms per verify cycle |
|---|---|---|---|---|---|
| realistic 2K (9 prompts) | 45.48 | 40.40 | -11.2 % | 4.86 / 4.75 | 107 / 117 |
| realistic 32K (9) | 47.81 | 42.32 | -11.5 % | 5.59 / 5.38 | 117 / 127 |
| realistic 96K (3) | 52.30 | 48.47 | -7.3 % | 6.74 / 6.73 | 129 / 139 |
| agent prompts with tools (9) | 56.41 | 49.60 | -12.1 % | 5.66 / 5.53 | 100 / 111 |
| long agent session, 8K to 55K (19) | 50.83 | 48.22 | -5.1 % | 4.57 / 4.88 | 90 / 101 |

The cost is the fp16 side path read on every verify cycle (+8-11 %; prefill +7-17 % time to first token) and a
drafter trained on the base model (-1 to -9 % tokens per verify on short prompts, equal at 96K). Peak footprint
11,264 MB in both arms.

**Behaviour.** In single-turn replays 21 of 51 replies were identical; where they differ, the adapter skips
exploratory tool calls (answers "no changes needed" instead of a no-op edit, edits directly instead of grepping first,
once reports "done" without checking). In Pi (thinking medium, sampled), on the voxel-pagoda prompt of the V8-96K
tests: 10.5 minutes, 4 model turns and 16.9K tokens against 82 minutes (33 minutes of model time), 33 turns and 55K
tokens without the adapter, at the same decode speed; its page renders a complete pagoda where the run without the
adapter left the roofs broken. Its first reply printed the file instead of saving it (one nudge), and in a Tetris task
it reported done before seeing a hidden start screen. One sample each at temperature 1.0: suggestive, not a benchmark.
Screenshots and logs: the results README.

## Download

```
hf download anemll/anemll-forge-qwen3.8-27B-V8-96K-SuperAdapter --local-dir ~/Models/anemll-forge-qwen3.8-27B-V8-96K-SuperAdapter
export VARIANT=sa   # then DEPLOY.md / AGENTS.md as for V8-96K
```

About 15 GB, plus Core AI's compile cache and the system's compiled programs on the internal disk (DEPLOY.md,
section 3).

## Files

- [coreai/m6-96k-dec8ts-sa/](coreai/m6-96k-dec8ts-sa): 8 packages of 8 layers, 81 entry points each (the V8-96K
  set: verify and prefill per context, hybrid prefill parts, SPLIT+GPU, tree verify `t8_*`, sliced `v8s_*` / `t8s_*`),
  with the adapter's fp16 factors; the output head (the release's, unchanged) and the server manifest.
- [gpu/w_dec8ts_sa/](gpu/w_dec8ts_sa): the GPU-table kit (`kit.json` with every sha256), attention corrections at rank 192.
- [adapter/](adapter): the upstream `adapter_config.json`, its pinned source (`ADAPTER_SOURCE.json`) and the export's
  record of every converted matrix (`adapter_export.json`). The adapter's own weights are not redistributed here.
- [drafter/all3/](drafter/all3), [model/](model), [config.json](config.json): as in V8-96K and release 0.2.1.
- [code/anemll-forge/](code/anemll-forge): the source snapshot (MIT) of the research branch
  [`v8-gpu-ane-prefill`](https://github.com/Anemll/anemll-forge/tree/v8-gpu-ane-prefill/research/v8_prefill) at commit
  `COMMIT`.
- [DEPLOY.md](DEPLOY.md), [AGENTS.md](AGENTS.md), [MODIFICATIONS.md](MODIFICATIONS.md), [NOTICE](NOTICE), [LICENSE](LICENSE).

## Status and limitations

- Research preview with the V8-96K requirements: SIP off and a root permission (the GPU reads the weights from the
  Neural Engine's compiled programs), M6 only, macOS 27.3 for the shipped tables (26C5047e), otherwise a table rebuild
  from the public release export plus [adapter/](adapter) and the upstream adapter (`adapter_export.py`).
- The drafter is the base model's; a drafter distilled on the adapted model would recover the lower acceptance.
- Not evaluated on SWE-bench Pro or Terminal-Bench with this quantized build; the upstream numbers are BF16 with vLLM.
  The server caps thinking (medium 6,144, xhigh 12,288 tokens); the upstream thinking-on runs allowed 32,768 per call.
- The adapter's three stacked rank-32 parts share nearly the same input subspace (their `A` factors differ by 4-9 %),
  so a rank-32 SVD keeps 99.3-99.9 % of it: a planned lighter build at about a third of the side-path cost.
