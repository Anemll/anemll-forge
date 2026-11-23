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
  - tree-verification
---

# ANEMLL Forge · Qwen3.8-27B · hybrid Neural Engine + GPU build, 96K context (research preview)

> **Research work, not a release.** This repository is an experimental research build. It needs System Integrity
> Protection disabled and a one-time root permission (see [Status](#status)), runs only on an M6 with macOS 27.3, and
> is not yet push-button to reproduce (the scripts carry the development machine's paths; see Status). For the supported release use
> [anemll/anemll-forge-qwen3.8-27B-192K](https://huggingface.co/anemll/anemll-forge-qwen3.8-27B-192K).

**Start here:**

- **Build, deploy and test this build:** [DEPLOY.md](DEPLOY.md) (the same file as
  `code/anemll-forge/research/v8_prefill/hf/DEPLOY.md`).
- **For coding agents:** [AGENTS.md](AGENTS.md): rules and the staged script `hf/agent_deploy.sh` (check, download,
  venv, bridge, compile, engines, tables, serve, smoke) with exit codes, including when a person must act.
- **Tree verification for hybrid ANE + GPU on Apple M6, simply:** [TREE_VERIFY_SIMPLE.html](code/anemll-forge/research/v8_prefill/TREE_VERIFY_SIMPLE.html)
  (plain language, illustrated; open it in a browser), the short companion to the paper.
- **Tree verification write-up:** [TREE_VERIFY.md](code/anemll-forge/research/v8_prefill/TREE_VERIFY.md): the tree8
  drafter / verifier on the Neural Engine and GPU, what is new, measurements, what did not work.
- **Article:** [One model, two accelerators](https://github.com/Anemll/anemll-forge/blob/v8-gpu-ane-prefill/research/v8_prefill/ARTICLE_ANE_GPU_M6.md): why, results against splash 1.3.2, how
  prefill and decode were optimized, the decisions and the research environment restrictions.
- **Research branch:** [`v8-gpu-ane-prefill`, research/v8_prefill](https://github.com/Anemll/anemll-forge/tree/v8-gpu-ane-prefill/research/v8_prefill): code, notes, benchmarks and every result.
- **Deployment:** [the research README](https://github.com/Anemll/anemll-forge/blob/v8-gpu-ane-prefill/research/v8_prefill/README.md): [requirements (root and SIP)](https://github.com/Anemll/anemll-forge/blob/v8-gpu-ane-prefill/research/v8_prefill/README.md#requirements-root-and-sip),
  [build chain](https://github.com/Anemll/anemll-forge/blob/v8-gpu-ane-prefill/research/v8_prefill/README.md#build-chain) (compile, GPU tables, GPU libraries) and [run](https://github.com/Anemll/anemll-forge/blob/v8-gpu-ane-prefill/research/v8_prefill/README.md#run).
- **Research notes:** [PREFILL_RESEARCH.md](https://github.com/Anemll/anemll-forge/blob/v8-gpu-ane-prefill/research/v8_prefill/PREFILL_RESEARCH.md) (what was tried, kept and rejected) and
  [DECODE_GPU_ATTENTION.md](https://github.com/Anemll/anemll-forge/blob/v8-gpu-ane-prefill/research/v8_prefill/pipe/unit/DECODE_GPU_ATTENTION.md) (the decode split).

**Research build that runs one large dense model on the M6's Apple Neural Engine and GPU at the same time.** It carries
the **release 0.2 weights** of [anemll/anemll-forge-qwen3.8-27B](https://huggingface.co/anemll/anemll-forge-qwen3.8-27B)
with a context of up to **98,304 tokens (96K)**, rebuilt for the hybrid schedule (research name V8):

- **Prefill:** every FFN runs on the Neural Engine and every token mixer (Gated DeltaNet and attention) on the GPU, both
  at once, sharing one copy of the weights: the GPU reads them straight from the Neural Engine's compiled programs.
- **Decode:** speculative, with the included Core AI DFlash2 drafter. Since 21 Nov the verification is a tree
  (tree8: the drafter's best 7 continuations in one 8-row Neural Engine call, for greedy and sampled requests); since
  22 Nov sliced verify entries at 24K / 40K / 48K / 56K attend only over the KV prefix in use. At the 96K entry the
  verification's attention over the KV history runs on the GPU between Neural Engine calls (SPLIT+GPU).
- **Build:** [coreai/m6-96k-dec8ts/](coreai/m6-96k-dec8ts) (22 Nov), 8 packages with every entry point on one copy of
  the weights, and [gpu/w_dec8ts/](gpu/w_dec8ts), a portable kit of the GPU engine's tables.

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
- The deployment path of this build is portable: [DEPLOY.md](DEPLOY.md) with `hf/serve_v8_96k.sh` (the served
  configuration, every path a variable), `pipe/engine/build_engines.sh` (GPU engine dylibs) and
  `pipe/gpu_tables_kit.py` / `pipe/build_gpu_tables.sh` (GPU tables). The other research scripts run with absolute
  paths of the development machine (listed in
  [research/v8_prefill/README.md](code/anemll-forge/research/v8_prefill/README.md)); adjust them to your layout.
- The GPU engine's tables point into this Mac's compiled Neural Engine programs. [gpu/w_dec8ts/](gpu/w_dec8ts) ships
  the tables of the development M6 with a sha256 of each compiled program's weight section: after the first compile,
  `gpu_tables_kit.py install` checks the local programs against them and installs the tables in seconds (on the M6 the
  installed tables are byte-identical to the served ones). If the programs differ (another macOS build or compiler),
  `build_gpu_tables.sh` rebuilds them from the public release 0.2 export,
  [anemll/anemll-quantized-qwen3.8-27b-for-CoreAI](https://huggingface.co/anemll/anemll-quantized-qwen3.8-27b-for-CoreAI)
  (`export/release_vq3pA_mixh_s600_k1_mat/` and the small tensors in `model/`), with the builder environment (torch
  and the Core AI Python tools).
- **M6 only** (FP8 attention), 32 GB of memory, macOS 27.3 (the kit's tables: build 26C5047e). Tested only on the
  M6 that made it.

## Tree verification

Decoding is speculative: the DFlash2 drafter proposes a block of 7 tokens per cycle on the Neural Engine (16
candidates per position), and one 8-row target call verifies them. Tree verification (tree8, served since 21 Nov)
keeps those 8 rows but fills them with a tree instead of one chain:

- **The tree.** Row 0 is the anchor; the other 7 rows are the drafter's most probable continuations, chosen best first
  on path probability (up to 4 children per node). Each row attends to the KV history and to its own ancestors only.
  A draft with a single path runs on the served chain entry; only a fork pays for the tree entry (ancestor masks and
  the `corrx0` DeltaNet short-convolution correction, which keeps chain replies bit-identical: 45/45).
- **The accepted path** is committed through the existing chain routine: the selected DeltaNet pending state and
  convolution rows are packed, the new KV rows compacted and the drafter's features gathered in path order. Sampled
  requests (Pi, temperature 1.0) accept child by child with the chain's rejection rule.
- **Across both engines.** At the 96K entry the verification's attention runs on the GPU between Neural Engine calls,
  and the Metal kernel applies the same ancestor mask.
- **Sliced KV widths (22 Nov).** Verify entries at 24K / 40K / 48K / 56K bind the KV segments of the next larger
  entry and attend only over the prefix in use (logits identical on 471 checked calls).

| measurement (M6, client tok/s) | result |
|---|---|
| greedy, 2K to 64K rungs | +1.9% to +6.5% over the chain (2K: 45.80 against 42.74) |
| sampled (Pi-like), two 36-request blocks | +8.5% and +10.2% pooled; tool-agent subset -1.6% / -3.9% |
| sliced widths, 19-turn Pi-like session (8.6K to 55.5K) | +3.4% pooled (+2.5% without one slow baseline seed; +5.7% at 32K to 57K), 152 of 152 replies identical |

Wider trees do not pay on the Neural Engine: 16 or 32 verify rows cost +26-39% or +87-116% per call, and timing probes
with replay put the best wider tree only 1.6% ahead at 8K. Also measured and not kept: per-cycle tree / chain choice, prompt
lookup for Pi, JetSpec++ (projected 12-46% slower on the serial schedule), drafting ahead during the verify, and
rescoring the tree. Switches: `TREE=0` serves the chain, `RUNG_SLICE=` (empty) the build without slicing. Write-up,
prior art and every result: [TREE_VERIFY.md](code/anemll-forge/research/v8_prefill/TREE_VERIFY.md); deployment: [DEPLOY.md](DEPLOY.md).

## Measured: prefill and chain decode (M6, 32 GB, macOS 27.3, 16 to 17 Nov)

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

For this build only (about 15 GB), and the steps after it, see [DEPLOY.md](DEPLOY.md) (sections 3 to 7: download,
compile, GPU engine, GPU tables, serve, smoke test with a fixed output checksum). The earlier builds follow the
research README: [requirements (root and SIP)](https://github.com/Anemll/anemll-forge/blob/v8-gpu-ane-prefill/research/v8_prefill/README.md#requirements-root-and-sip),
[build chain](https://github.com/Anemll/anemll-forge/blob/v8-gpu-ane-prefill/research/v8_prefill/README.md#build-chain) (compile, GPU tables, GPU libraries) and [run](https://github.com/Anemll/anemll-forge/blob/v8-gpu-ane-prefill/research/v8_prefill/README.md#run) (the same
file is in this repository at `code/anemll-forge/research/v8_prefill/README.md`).

## Files

- [coreai/m6-96k-dec8ts/](coreai/m6-96k-dec8ts) (22 Nov, served): 8 packages of 8 layers, 81 entry points each on
  one copy of the weights (verify and prefill per context, hybrid prefill parts, SPLIT+GPU `dva` / `dvba` / `dvb`,
  tree verify `t8_*`, sliced verify `v8s_*` / `t8s_*`), the output head and the server manifest.
- [gpu/w_dec8ts/](gpu/w_dec8ts): the GPU-table kit (1344 files, 0.9 GB, `kit.json` with every sha256).
- [DEPLOY.md](DEPLOY.md): build, deploy and test.
- [coreai/m6-96k-dec8/](coreai/m6-96k-dec8) (17 Nov) and [coreai/m6-96k-v8/](coreai/m6-96k-v8) (16 Nov): the earlier
  chain-only builds, kept for comparison.
- [drafter/all3/](drafter/all3): the DFlash2 drafter package with configuration, selector table, the target head's
  norm gain, licenses and provenance (as in release 0.2.1).
- [model/](model): tokenizer, configuration and FP16 host embedding table, byte-identical to release 0.2.
- [config.json](config.json): copy of `model/config.json` for Hub discovery.
- [code/anemll-forge/](code/anemll-forge): the source snapshot (MIT), see Status.

Disk: about 15 GB for this build (DEPLOY.md, section 3), about 35 GB for the whole repository with the two earlier
builds, plus Core AI's compile cache and the system's compiled programs (about 25 GB on the internal disk).

## Limitations

- Research preview: SIP off and a root permission are required; outside the deployment path of DEPLOY.md the
  included scripts use the development machine's paths; a table rebuild (when the kit does not match) needs the public
  release export and the builder environment (see Status). Nobody has yet deployed it on a second Mac.
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
