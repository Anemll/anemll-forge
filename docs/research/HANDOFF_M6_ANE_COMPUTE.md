# Handoff: M6 ANE INT8-INT8 / FP8-FP8 compute research

Read this file first. It is the single handoff for the next agent. Do not reopen the LLKVApprox / KVA branch or PR #1.

## Update, 3 October 2026: measured on the M6 (branch `accelrate-compute`)

Steps 1 and 2 below were run on device. Results and method: [M6_COMPUTE_ACCELERATION_2026-10-03.md](M6_COMPUTE_ACCELERATION_2026-10-03.md).

- **Attention share, measured** (release V8 target, full 16-chunk chain): verify `101.5 ms + 1.38 ms/K`, prefill `248 ms + 5.42 ms/K`. Context-dependent share of verify is 10 / 18 / 31 / 47% at 8K / 16K / 32K / 64K, about twice the byte model, because history attention is op bound (about 36 GB/s of K/V at 64K), not bandwidth bound.
- **Gated DeltaNet core** was a third of a 64-row prefill chunk call. The row-by-row triangular solve was half of that. `GDN_FAST=1` (exact Neumann-product inverse written as broadcast multiply-reduce, native depthwise conv1d, merged state matmuls) halves the GDN core on the ANE.
- **Attention tiles:** `ATT_BLOCK=2048` (release 16384) cuts the attention core 14 to 38%.
- **Full target, V8:** verify -12% (8K) to -26% (64K), prefill -16% to -28%; KL-512 mean 0.18427 to 0.18417, direct KL between builds 3.4e-5; fully on the ANE. Cost: about 5x longer one-time cold compile.
- **Full server with DFlash2 (V8):** cold prefill +26% (8K) to +41% (64K); decode +21 to +25% at 32K to 64K with identical replies; 64K long-context perplexity and direct KL unchanged.
- **Closed after measurement:** W8A8 (LUT weights times INT8 activations fails ANEC and falls to the GPU; dense INT8 is weight-bound and slower than the LUT path), INT8 / FP8 attention MACs (the history path is dominated by passes over scores, not MACs), Kronecker Hadamard (slower), transposes (not the cost), larger GDN sub-chunks (slower), shorter verifiers ([../verifier_len.md](../verifier_len.md): 8 rows stays fastest end to end).
- **Defaults since 3 October:** the converter builds V8-only packages with `GDN_FAST=1 ATT_BLOCK=2048 ATT_BLOCK_PREFILL=4096`. That build verifies 19 to 31% faster and prefills 29 to 35% faster than the release graph and compiles cold in about 23 minutes (narrow prefill tiles had made it 5 times longer). First starts print guided `[ANE compile]` progress; `python forge.py compile --build <dir>` compiles ahead.
- **Next:** [DFLASH2_SAMPLING_PLAN.md](DFLASH2_SAMPLING_PLAN.md) (exact speculative sampling with the drafter distribution), a weight-free attention program shared by all 16 layers if compile time must drop further, and the remaining validation listed in [../VALIDATION.md](../VALIDATION.md).

The original VM-era handoff follows unchanged.

## Status

- Local branch only: `cursor/m6-ane-compute-feasibility-e49d`
- Commits on top of `main` (`6e0a85c`):
  - `74c40b8` research notes plus attention-dtype host harness
  - `cee9988` Amdahl / traffic leverage model
  - this handoff commit, if present
- Not pushed. No pull request. Do not `git push`, force-push, or open/update a PR until the user explicitly says to.
- Parent research branch `cursor/m6-ane-compute-research-e49d` stops at `74c40b8`. Continue from the feasibility branch, not from that one.
- VM was Linux x86_64. No M6, no `coremltools`, no `torch`, no model weights. Nothing here is an ANE timing.

## Question

Can Forge approach about 2x compute throughput on M6 ANE versus the current Core AI baseline (mixed-bit LUT weights, selectable V8 KV)?

## Answer so far

V8 attention is the right first kernel to inspect, and it does not already run INT8-INT8 or FP8-FP8 compute.

In `coreai/qwen38_coreai_build.py` class `AttnW`:

- Historical V is INT8 plus an FP16 token/head scale.
- The graph calls `torch.ops.coreai.dequantize` with scale `1/128` and `output_dtype=float16`.
- QK is an FP16 matmul. PV is an FP16 matmul.
- Dynamic `scale * 128` is applied to unnormalized exp scores so the softmax denominator stays unscaled.
- Attention K/V projection weights are INT8 per-channel, then multiplied into FP16 inside `QConv` before the conv.

Name trap: entry `v8_16k` means an 8-token DFlash2 verify at 16K context. KV format `v8` means FP16 keys and INT8 values. They are different.

A 2x attention kernel is not a 2x forward unless attention is already most of the forward. Modeled from the repo's own byte counts (`docs/PERFORMANCE_SESSION.md`, about 10.61 GB weights per verifier forward, 64 KiB FP16 KV per token, 48.125 KiB V8):

| Context | KV share of forward bytes, FP16 | V8 |
| --- | ---: | ---: |
| 8K | 0.048 | 0.037 |
| 16K | 0.092 | 0.071 |
| 32K | 0.168 | 0.132 |
| 64K | 0.288 | 0.233 |

Amdahl: end-to-end = `1 / ((1 - f) + f / s)` where `f` is the attention time fraction and `s` is the kernel speedup. To reach 2x end-to-end, `f` must exceed 0.5. At `f = 0.25` and `s = 2`, end-to-end is about 1.14x. At `f = 0.25` and an infinitely fast kernel, end-to-end is 1.33x. The attention time fraction has not been measured. Do not invent it.

Prior V8 serving numbers (measured on M6, already in the repo, not rerun here) peaked at +23.87% decode at the 64K entry. That study says it does not prove integer attention MAC execution. Bonded compile mode 2 is already the Core AI default and matched mode 0 latency on the tested chunk while using less memory. No public source in this pass named a distinct bonded FP8 compute format. Forge's bonded flag is `MPSGRAPH_ANE_BONDED_COMPILE_MODE`.

## Ranked options

Kernel ceilings are inferred best cases if the hardware executes the path. They are not timings.

| Rank | Option | Ceiling | Quality | Hook |
| --- | --- | --- | --- | --- |
| 1 | INT8-INT8 QK and PV, both operands, INT32 accum | High only if native INT8 MAC; else none | Medium | V8 dequant in `AttnW` |
| 2 | FP8-FP8 E4M3 both operands (240 ANE weight max, or 448 E4M3FN) | High only if native FP8 MAC | Medium-high | Weight FP8 only. FP8 LUT values were promoted to FP16 at compile time |
| 3 | INT8 keys plus current V8 | Bandwidth, about another quarter of KV payload | Medium-high | `quantize_values` is V-only |
| 4 | Fused or blocked attention | Low-medium | Low | `scripts/qwen38_attn_chunk_probe.py`, `ATT_BLOCK=16384` |
| 5 | Zero-skip on masked KV | Low-medium if the ANE skips zeros | Low | `scripts/qwen38_zeroskip_test.py` |
| 6 | Lower-precision GDN compute | Medium, and 48 of 64 layers so Amdahl leverage can beat attention | High | LUT4 mixers, `GDN_SQ` / `GDN_SV` |
| 7 | Compile mode / bonded flag | None versus mode 2 | Low | Already default |
| 8 | Winograd | None for 1x1 convs and matmuls | n/a | No hook, no in-repo code |

Dense INT8 or FP8 weights are slower for decode than the current LUT path because they move more bytes. Per-group vector LUTs leave the ANE. Do not revive those as a decode 2x plan.

## Files

| Path | Role |
| --- | --- |
| `docs/research/M6_ANE_COMPUTE_2026-10-02.md` | Sources, graph facts, ranked list. Labels measured vs inferred |
| `docs/research/M6_COMPUTE_FEASIBILITY_2026-10-02.md` | Amdahl gate and byte table |
| `docs/research/M6_ATTN_COMPUTE_README.md` | How to run both harnesses, including on an M6 |
| `scripts/m6_attn_compute.py` | Host recipes and a Core ML skeleton |
| `scripts/m6_compute_roofline.py` | Pure-Python leverage model |
| `tests/test_m6_attn_compute.py` | 18 tests |
| `tests/test_m6_compute_roofline.py` | 21 tests |
| `docs/EXPERIMENTS.md` | Index pointers only |

Visual summary for a human (not in git): canvas `M6 ANE compute research`.

## Commands that already passed on the VM

```sh
python3 -m unittest tests/test_m6_attn_compute.py tests/test_m6_compute_roofline.py tests/test_qwen38_kv_cache.py
python3 scripts/m6_attn_compute.py compare --ctx 64 --queries 8 --seed 0
python3 scripts/m6_compute_roofline.py required --target 2.0
python3 scripts/m6_compute_roofline.py traffic --ctx 8192 16384 32768 65536
```

Host relative RMSE versus an FP32 reference on one synthetic draw (ctx 64, T=8, seed 0) was about 0 for `fp16_baseline`, 0.0069 for `v8_current`, 0.0105 for `int8_int8_attn`, and 0.0309 for `fp8_fp8_attn`. That is numeric error, not speed.

## What the next agent should do

On an M6, in order. Stop when a gate fails.

1. Measure the share of forward time spent in attention compute at 8K, 16K, 32K, and 64K, for prefill and for decode, on the current V8 build. Separate it from projections, MLP, GDN, head, drafter, and host cache work.
2. Run `python3 scripts/m6_compute_roofline.py options --attention-fraction <measured>`. If an infinite kernel still misses the goal, attention dtype cannot deliver a 2x forward by itself. Look at GDN compute only with the quality warning above.
3. If the ceiling is still interesting, compile the attention core from `scripts/m6_attn_compute.py mil-skeleton --build` and time it with `ane-bench` on device. Record `MLComputePlan` placement, compile mode 2 and 0, and the executed dtype (fused INT8 or FP8 MAC, dequant-then-FP16, or CPU fallback). Placement text alone is not the dtype.
4. Only after a real kernel gain, run a paired full-server prefill and decode in the style of `docs/research/KV_CACHE_QUANTIZATION_2026-10-02.md`. Keep arithmetic changes (stable softmax vs stock softmax) as a separate control: `--kv-cache-dtype fp16 --stable-attention`.

## Constraints

- No em dashes in new prose.
- Do not claim a measured 2x.
- Do not invent timings.
- Do not merge.
- New tests stay `unittest` and must run without Core ML.
- Imports stay at module top.
- Qwen3.8-27B attribution stays with the Qwen Team / Alibaba Cloud (Apache 2.0). ANEMLL work is independent conversion and ANE research.

## Sources already used

- Forge: `AttnW`, `QConv`, `scripts/qwen38_kv_cache.py`, `docs/KV_CACHE_V8.md`, `RESULTS_M6_INT8.md`, `COREAI_PORT_NOTES.md`, `docs/history/VECTOR_LUT_README.md`, `QUANTIZATION_NOTES.md`
- Apple / coremltools: quantization overview, performance, and API pages for W8A8 on A17 Pro / M4-class Neural Engine (`linear_quantize_activations` plus `linear_quantize_weights`). Those pages are vision-model docs, not this Qwen graph and not M6.
