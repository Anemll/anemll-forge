---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
pipeline_tag: text-generation
tags:
  - coreai
  - ane
  - quantized
---

# ANEMLL Forge · Qwen3.8-27B for ANE

**Research project for the M6 Apple Neural Engine.** ANEMLL Forge explores quantization, conversion and inference with Core AI and a Swift bridge, and shares the experiments and known limitations. The repository is private during release preparation; final bundle provenance and complete hardware validation are still pending.

ANEMLL independently quantizes and converts [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for text generation on the Apple Neural Engine, targeting M6 through Core AI and the ANEMLL Forge Swift bridge. Credit for the original model belongs to the Qwen Team; the upstream weights carry **Copyright 2026 Alibaba Cloud**.

## Intended contents and scope

The planned runtime bundle contains Core AI model packages, an FP16 embedding table, and the upstream configuration and tokenizer assets. It supports text generation; vision, multi-token prediction (MTP), and a speculative drafter are outside the default bundle.

The deployed research recipe uses GPTQ with mixed two-bit/four-bit quantization, per-channel scaling, online rotations, and rank-64 residual corrections. The exact exported tensors, context entries, and artifact hashes must be recorded in the staged bundle before release. No throughput, accuracy, or memory benchmark is claimed by this card.

The prepared Core AI build is identified as `mix25in_mixr_lr64mix`. Its manifest advertises 8,192, 16,384, 24,576, 32,768, 49,152 and 65,536-token entries; the largest entry has a 65,472-row KV capacity. These are artifact entry sizes, not claims that long-context quality or performance has been validated.

## Source and modifications

- Upstream model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
- Pinned upstream revision: [`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0).
- Original model license: [Apache License 2.0, as supplied upstream](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/LICENSE).
- Read [MODIFICATIONS.md](MODIFICATIONS.md) for the transformation notice and [NOTICE](NOTICE) for attribution.

The model-weight license is Apache-2.0. ANEMLL Forge source-code licensing is a separate matter and remains unresolved during preparation. The model-card license does not grant a license to the inference or conversion repository. Preserve the upstream license and relevant notices when redistributing model derivatives.

This is an independent ANEMLL conversion, with no claimed endorsement by the Qwen Team, Alibaba Cloud, or Apple.

## Evaluation status

**Current evaluation: KL divergence only.** Quality evaluation compares the quantized model with the upstream BF16 reference on the same teacher-forced token trace. A versioned KL report tied to the exact released artifacts will be linked here when available; historical research-session values are not a newly validated result for a downloaded bundle.

ANE benchmarks will be added when measured. Planned measurements include prefill and decode throughput/latency, time to first token, behavior across context lengths, memory use, and power/thermal behavior. Each result will identify the artifact revision, hardware, OS/toolchain, numerical settings, workload and measurement method. No ANE benchmark results are claimed by this card yet.

## Download and quick test

Run these commands from an ANEMLL Forge checkout in an existing compatible research environment, after the bundle has been uploaded. The source is [Anemll/anemll-forge](https://github.com/Anemll/anemll-forge); access may remain private during preparation. A clean public dependency recipe is still being validated.

Replace `HF_COMMIT` with the **actual uploaded bundle commit**, not the upstream checkpoint revision. Private downloads require your existing authorized Hugging Face login. Install `huggingface_hub` in the environment if it is absent.

```sh
python forge.py download \
  --repo anemll/anemll-forge-qwen3.8-27B \
  --revision HF_COMMIT \
  --runtime coreai \
  --output ./models/qwen3.8-27b-ane

python forge.py quick-test \
  --bundle ./models/qwen3.8-27b-ane \
  --runtime coreai --check-only

bash coreai/swift_bridge/build.sh

python forge.py quick-test \
  --bundle ./models/qwen3.8-27b-ane \
  --runtime coreai --tokens 16
```

The integrity check verifies the selected runtime and model assets without loading the model. Inference needs compatible macOS, hardware, SDK/runtime dependencies, and a successfully built Swift bridge. The default test uses the bundle's smallest declared context. It checks short greedy execution and finite logits; it does not establish model quality, sustained performance, or ANE placement.

The runtime bundle is intended to run without downloading the original BF16 checkpoint shards. Requantization and conversion are separate workflows requiring the appropriate original weights and research dependencies.

## Before release

Finish artifact staging, record export provenance and contexts, verify the per-file inventory and modification notices, and complete hardware checks. Retain observed limitations and experiment history in ANEMLL Forge. Publication and visibility changes require the owner's approval; preparing this card does not upload or publish anything.
