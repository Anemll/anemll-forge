---
base_model:
  - Qwen/Qwen3.8-27B
  - ProCreations/Ternary-Bonsai-2-27B-DFlash2
base_model_relation: quantized
license: apache-2.0
pipeline_tag: text-generation
tags:
  - coreai
  - ane
  - quantized
  - speculative-decoding
  - dflash2
---

# ANEMLL Forge · Qwen3.8-27B for ANE

**Research project for inference of large dense models on the M6 Apple Neural Engine.** ANEMLL Forge explores quantization, conversion and speculative inference with Core AI and a Swift bridge, and shares experiments and known limitations. M6 is the main target; M5, M5 Pro and M5 Max are also supported, with roughly half the M6 throughput as approximate guidance rather than a matched benchmark. See the [known M5 cold-compilation workaround](https://github.com/Anemll/anemll-forge/blob/main/docs/SESSION_LESSONS.md). Complete artifact provenance and broader hardware validation remain research tasks.

The default runtime pairs the converted Qwen target with its matching **Core AI DFlash2 speculative drafter**. The drafter is required for the deployed speculative workflow and its performance measurements. The target verifies an anchor plus seven proposed tokens in each T=8 cycle; acceptance determines how many tokens are emitted.

ANEMLL independently quantizes and converts [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for text generation on the Apple Neural Engine. Credit for the original target belongs to the Qwen Team; its upstream weights carry **Copyright 2026 Alibaba Cloud**. The drafter derives from [ProCreations/Ternary-Bonsai-2-27B-DFlash2](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2), with the z-lab DFlash2 and Prism Bonsai provenance preserved in its upstream notice.

## Bundle contents and scope

The bundle includes Core AI target chunks and head, an FP16 embedding table, matching tokenizer/configuration assets, and the paired drafter. Vision and multi-token prediction (MTP) are outside this text-generation bundle.

The target build is `mix25in_mixr_lr64mix`. Its recipe uses mixed two-bit/four-bit GPTQ, per-channel scaling, online rotations and rank-64 residual corrections. The manifest advertises 8,192, 16,384, 24,576, 32,768, 49,152 and 65,536-token entries; the largest entry has a 65,472-row KV capacity. These are artifact entry sizes, not validated long-context quality or performance results.

The matching drafter is `drafter/dflash2_lut4_gptq.aimodel`, with its same-stem JSON sidecar. It uses the `q7_cal` LUT4 GPTQ export, the mixr target's head and mask-row scale 0.7. Its five layers use a 2,048-row context ring and target feature taps at layers 5, 19, 33, 47 and 61. `drafter/config.json` and `drafter/selector.safetensors` are required host-side assets. The compact selector file contains only the two original BF16 codebook tables, with tensor values preserved exactly. The original full drafter checkpoint is unnecessary for this prepared inference bundle.

The drafter source was adapted to Bonsai features before this ANEMLL conversion. Pairing it with the Qwen target and its matching head does not itself establish speculative acceptance or speed. Exact file hashes and associations are recorded in `release.json` and [drafter/DFLASH2_SOURCE.json](drafter/DFLASH2_SOURCE.json).

## Source and modifications

- Qwen target: [`Qwen/Qwen3.8-27B` at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0), with its original Apache-2.0 license preserved in [LICENSE](LICENSE).
- Drafter: [`ProCreations/Ternary-Bonsai-2-27B-DFlash2` at `4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b`](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2/tree/4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b), with its original [LICENSE](drafter/LICENSE) and [NOTICE](drafter/NOTICE) preserved unchanged.
- The drafter source records donor [`z-lab/Qwen3.8-27B-DFlash2` at `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/tree/50307d4c4cde6860d4eee73e2547cd786fe8e8a4) and training target [`prism-ml/Ternary-Bonsai-2-27B-gguf` at `6ed5e12bf84b7a63069882c91dd9e9218647d17b`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/tree/6ed5e12bf84b7a63069882c91dd9e9218647d17b).
- Read [MODIFICATIONS.md](MODIFICATIONS.md) and ANEMLL's [NOTICE](NOTICE) for conversion and packaging changes.

The complete local BF16 drafter checkpoint was rehashed and matched the pinned upstream LFS SHA256. Compact selector extraction was checked for identical tensor bytes. The Core AI body has not been independently reconstructed from that checkpoint and the GPTQ export; its head/calibration associations currently come from the source sidecar and export metadata. Preserve this distinction when reproducing the build.

The model-weight license is Apache-2.0. Independently authored ANEMLL Forge code and documentation use the separate [MIT license](https://github.com/Anemll/anemll-forge/blob/main/LICENSE). This model-card license applies to the model artifacts and does not replace the licenses of third-party code. Preserve all applicable upstream licenses and notices when redistributing derivatives. No endorsement by Qwen, Alibaba Cloud, ProCreations, z-lab, Prism ML or Apple is claimed.

## Evaluation status

**Current quality evaluation: KL divergence only.** It compares the quantized target with the upstream BF16 reference on the same teacher-forced token trace. A versioned report tied to the exact released artifacts will be linked when available. Historical session results are not newly validated results for this downloaded pair.

ANE benchmarks will be added when measured. They will identify both target and drafter revisions, hardware, OS/toolchain, numerical settings, workload and measurement method. Planned measurements include speculative prefill/decode latency and throughput, accepted drafts per cycle, emitted tokens per verifier call, time to first token, context behavior, memory, and power/thermal behavior. No new speed, acceptance, quality or memory result is claimed by this card.

## Download and quick test

Follow the [ANEMLL Forge README](https://github.com/Anemll/anemll-forge#1-set-up-the-inference-environment) for environment setup, bridge compilation, Pi configuration and API testing. Browse the [model files](https://huggingface.co/anemll/anemll-forge-qwen3.8-27B/tree/main), including the [Core AI target](https://huggingface.co/anemll/anemll-forge-qwen3.8-27B/tree/main/coreai) and [DFlash2 drafter](https://huggingface.co/anemll/anemll-forge-qwen3.8-27B/tree/main/drafter).

Replace `HF_COMMIT` with the actual **paired bundle commit**, not the upstream checkpoint revision or the earlier target-only upload. If access requires authentication, use `hf auth login` with your own account. Install the inference dependencies from the Forge README.

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

python forge.py serve \
  --model ./models/qwen3.8-27b-ane/model \
  --build ./models/qwen3.8-27b-ane/coreai \
  --runtime coreai --ctx 16384
```

Default Core AI download includes the drafter, its configuration, selector tables and source notices. Serving discovers this drafter beside the target build; quick-test runs the speculative propose/verify/accept/commit path by default. `--plain` selects an explicit target-only diagnostic. The integrity-only check validates hashes, tensor structure and pairing without loading either model. Inference requires compatible macOS/hardware/SDK dependencies and the built Swift bridge. The smoke test does not establish sustained performance, output quality, long-context quality or ANE placement.

The prepared runtime needs neither original BF16 checkpoint. Requantization and conversion remain separate workflows requiring the original weights and research dependencies.

## Research limitations and reproduction

Finish export reconstruction/provenance review, verify the inventory and notices, and complete hardware checks for the target/drafter pair. Retain observed limitations and experiment history in ANEMLL Forge. See Forge's release checklist and validation records for the current evidence and remaining work.
