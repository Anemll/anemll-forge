---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
pipeline_tag: text-generation
library_name: anemll-forge
tags:
  - coreai
  - ane
  - metal
  - quantized
  - speculative-decoding
  - dflash2
  - kv-cache
  - long-context
  - gpu-ane-prefill
  - hybrid
---

# ANEMLL Forge · Qwen3.8-27B · hybrid Neural Engine + GPU build, 96K context (research preview)

**Research build that runs one large dense model on the M6's Apple Neural Engine and GPU at the same time.** It carries
the **release 0.2 weights** of [anemll/anemll-forge-qwen3.8-27B](https://huggingface.co/anemll/anemll-forge-qwen3.8-27B)
with a context of up to **98,304 tokens (96K)**, rebuilt for the hybrid schedule (research name V8):

- **Prefill:** every FFN runs on the Neural Engine and every token mixer (Gated DeltaNet and attention) on the GPU, both
  at once, sharing one copy of the weights: the GPU reads them straight from the Neural Engine's compiled programs.
- **Decode:** speculative, with the included Core AI DFlash2 drafter (8-row verification on the Neural Engine). Past
  64K tokens the verification's attention over the KV history runs on the GPU between Neural Engine calls
  (SPLIT+GPU); the Neural Engine programs are the same for every context length.

ANEMLL independently converts and quantizes [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Credit for
the target belongs to the **Qwen Team and Alibaba Cloud**, with upstream **Copyright 2026 Alibaba Cloud** and
Apache-2.0 licensing. The paired drafter derives from
[ProCreations/Ternary-Bonsai-2-27B-DFlash2](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2), whose
upstream notice preserves the z-lab DFlash2 and Prism Bonsai provenance. No endorsement by these projects or Apple is
claimed.

## Status

This is a research preview, not a release:

- **It needs System Integrity Protection off and a one-time root permission.** Core AI is not designed to run one
  model on the Neural Engine and the GPU at the same time: it gives the GPU no way to use the weights the Neural Engine
  already holds. Apple may support this in the future; for research, the GPU maps the weights inside the compiled
  Neural Engine programs (`model.hwx` in `/Library/Caches/com.apple.aned/`, 10.8 GiB, read-only, no copy). That cache
  is root-only (one `sudo chmod +a` grants read access) and carries macOS's `datavault` flag, which SIP enforces even
  against root. Disabling SIP lowers the security of the whole Mac: use a dedicated research machine. Details:
  [Requirements: root and SIP](code/anemll-forge/research/v8_prefill/README.md#requirements-root-and-sip).
- The code is included: [code/anemll-forge/](code/anemll-forge) is a snapshot of the ANEMLL Forge research branch
  [`v8-gpu-ane-prefill`](https://github.com/Anemll/anemll-forge/tree/v8-gpu-ane-prefill/research/v8_prefill) at
  commit `COMMIT` (the runtime, the Core AI builder and Swift bridge, `forge.py`, and in
  [research/v8_prefill/](code/anemll-forge/research/v8_prefill) the server, the GPU engine and decode attention, the
  GPU-table prep, the build scripts and the notes). The release 0.2 server alone does not load these packages (they
  use a segmented KV cache).
- The research scripts run with absolute paths of the development machine (listed in
  [research/v8_prefill/README.md](code/anemll-forge/research/v8_prefill/README.md)); adjust them to your layout.
- The GPU engine's tables are generated on each machine after its first compile, from that machine's compiled Neural
  Engine programs: `prep_engine.py` and `prep_kv8.py` read them with the `hwx_parsing` tool of
  [freedomtan/coreml_to_ane_hwx](https://github.com/freedomtan/coreml_to_ane_hwx) (BSD-3-Clause, built with `make`)
  and also read the release export, which is not part of this repository yet. Until a portable table kit is added,
  running the hybrid on another machine needs that export.
- **M6 only** (FP8 attention), 32 GB of memory, macOS 27.3.

## Measured (M6, 32 GB, macOS 27.3)

The server at 96K context, DFlash2 drafter, greedy decoding, thinking off, the GPU otherwise idle. Times to the first
streamed token at the client. A follow-up is the second turn of the same conversation (the model's own answer plus 300
new tokens); the server continues from its saved state and prefills only the new text (about 320 tokens). Prefill and
follow-ups are from the 16 Nov run; decode is from the 17 Nov rerun with SPLIT+GPU, which left prefill unchanged within
run noise.

| prompt | cold prefill | follow-up | decode cycle | decode, cold turn |
|---|---|---|---|---|
| 2K | 586 tok/s (3.6 s) | 1.04 s | 107 ms | 36.0 tok/s |
| 8K | 658 tok/s (12.5 s) | 1.15 s | 111 ms | 23.2 tok/s |
| 16K | 620 tok/s (26.5 s) | 1.90 s | 119 ms | 20.9 tok/s |
| 32K | 541 tok/s (61 s) | 1.75 s | 133 ms | 20.0 tok/s |
| 64K | 420 tok/s (156 s) | 2.36 s | 141 ms | 20.6 tok/s |
| 95K | 347 tok/s (274 s) | 2.92 s | 149 ms | 20.4 tok/s |

The same model's Neural-Engine-only packages prefill about 300 tok/s at 2K to 8K and 180 at 96K. Decode speed is the
cycle time over the accepted draft tokens (2.4 to 3.7 per cycle on these cold turns, more when an answer repeats
earlier text). SPLIT+GPU took the decode cycle from 153 to 141 ms at 64K and from 155 to 149 ms at 95K; below 64K it is
off and the cycle is unchanged.

How busy each unit is (share of wall time): in prefill both work at once, the Neural Engine 98% and the GPU 95 to 97%
at 6K, the Neural Engine 72% and the GPU 96% at 64K, where the GPU's attention is the bottleneck. In decode they take
turns: up to 64K the Neural Engine does all of it; at 95K the GPU takes about 21% of each cycle.

A comparison with splash 1.3.2 on the same Mac, the decisions behind the design and every rejected variant:
[the article](code/anemll-forge/research/v8_prefill/ARTICLE_ANE_GPU_M6.md) and
[PREFILL_RESEARCH.md](code/anemll-forge/research/v8_prefill/PREFILL_RESEARCH.md).

## Download

```
hf download anemll/anemll-forge-qwen3.8-27B-V8-96K --local-dir ~/Models/anemll-forge-qwen3.8-27B-V8-96K
```

Download the whole repository: the server needs every folder below. Then follow the
[Build chain](code/anemll-forge/research/v8_prefill/README.md#build-chain) (compile, GPU tables, GPU libraries) and
[Run](code/anemll-forge/research/v8_prefill/README.md#run) steps.

## Files

- [coreai/m6-96k-dec8/](coreai/m6-96k-dec8): 8 packages of 8 layers (verify, prefill, hybrid and SPLIT+GPU entry
  points on one copy of the weights), the output head and the server manifest.
- [drafter/all3/](drafter/all3): the DFlash2 drafter package with configuration, selector table, the target head's
  norm gain, licenses and provenance (as in release 0.2.1).
- [model/](model): tokenizer, configuration and FP16 host embedding table, byte-identical to release 0.2.
- [config.json](config.json): copy of `model/config.json` for Hub discovery.
- [code/anemll-forge/](code/anemll-forge): the source snapshot (MIT), see Status.

Disk: about 14 GB to download, plus Core AI's compile cache and the system's compiled programs.

## Limitations

- Research preview: SIP off and a root permission are required, the included scripts use the development machine's
  paths, and the GPU-table step needs the release export (see Status).
- Quality: the weights and their quantization are those of release 0.2 (KL and perplexity measured there, up to 64K).
  The hybrid prefill computes the same model on the GPU and the Neural Engine; its logits differ from the
  Neural-Engine-only path by a small, length-dependent KL (last-token KL 0.010 at 2K, 0.058 at 64K). SPLIT+GPU's
  verification matches the Neural-Engine-only one closely on real text (KL 0.0005 to 0.054, equal or lower negative
  log-likelihood).
- One set of synthetic prompts per length is not a general benchmark; decode speed depends on the text.

## Attribution and licenses

- Target: [`Qwen/Qwen3.8-27B` at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0). Original Apache-2.0 [LICENSE](LICENSE) and ANEMLL-added [NOTICE](NOTICE) are included; [QWEN_SOURCE.json](QWEN_SOURCE.json) records source checks and remaining provenance limitations.
- Drafter: [`ProCreations/Ternary-Bonsai-2-27B-DFlash2` at `4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b`](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2/tree/4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b). Its original `LICENSE` and `NOTICE` are included unchanged in the drafter folder.

[MODIFICATIONS.md](MODIFICATIONS.md) describes the converted derivatives and the graph changes, including this build's.

**Model-derived assets follow their upstream Apache-2.0 licenses and notices. ANEMLL Forge source code and
documentation are MIT-licensed.** Preserve the applicable model licenses, copyright and modification notices when
redistributing derivatives.
