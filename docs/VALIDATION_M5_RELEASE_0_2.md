# Release 0.2 validation on Apple M5 — 2026-10-10

## Environment and model

- MacBook Pro, Apple M5 (10 CPU cores), 32 GB unified memory; macOS 27.2, build `26B5101f`.
- Python 3.11.14, Core AI authoring package `coreai-core==1.0.0b3`, existing Swift bridge passed native load/ABI checks.
- Source main was current with remote at `5513390`; fixes and update checker committed as `64da0ce`.
- Downloaded `anemll/anemll-forge-qwen3.8-27B` at resolved Hub commit `440f4e366b43d0f9bbccd79e695b159c4a94d1a0`.
- Complete target/drafter bundle: **75 verified files, 15,628,752,803 bytes**. Download verification and `forge.py quick-test --check-only` both passed.
- Target export `release_vq3pA_mixh_s600_k1_mat`, release **0.2**, V8 cache with transposed FP16 keys and INT8 values. M5 function extraction selected `ATT_INT8MM=s8,s8b` (INT8 scores; FP16 softmax/PV), bonded compile mode **1**. Runtime reports `soc_class=m5`, release `0.2` and the actual derived build directory.

The M5 build was extracted in **42 seconds**, adding about **10.6 GB** of target packages. First compilation/loading of the 18 packages took **27m16s** after extraction, with **28m01s** from benchmark launch to a serving server. All 16 target chunks compiled successfully, typically in 97–114 seconds each; the head loaded in 3 seconds and the drafter compiled in 10 seconds. No topological-sort workaround or compiler retry was needed. The following 16K server started from cache in **6.04 seconds**.

## Whole-server profile

One cold request and three cached repeats at each context, after a tiny warmup. Public synthetic testing notes plus a Python `clamp` task; greedy sampling, thinking off, 256 output tokens, DFlash2 with a 3 ms draft gap. Prefill uses the cold request's server timer; decode is the median of the three cached requests. Target, drafter, host sampling and speculative verification are included in decode.

| Context cap | Prompt tokens | Cold prefill | Cached decode median | Cached decode samples | Draft acceptance |
| --- | ---: | ---: | ---: | --- | ---: |
| 8K | 7,673 | **129.09 tok/s** (59.44 s) | **18.00 tok/s** | 19.73, 18.00, 17.99 | 76.79% |
| 16K | 15,865 | **127.82 tok/s** (124.12 s) | **19.19 tok/s** | 19.19, 19.27, 18.74 | 76.79% |

Both cases passed:

- All four replies were byte-identical within each case and contained the requested `clamp` function. The 256-token cap truncates the full requested test suite; this is a performance fixture, not a code-quality score.
- Prefill and final decode logits were finite.
- Cold requests reused zero prompt tokens; cached repeats reused all but seven prompt tokens.
- Correct active context entry and successful speculative state reuse. The 16K run expanded from 8K at position 8,150 and copied the KV prefix in **123 ms**.
- Strict cached-graph audit: **all 18 packages `fully_ane`**, return code 0. This inspects cached graph placement, not a hardware power trace.
- Every owned benchmark server exited when its run completed.

Target verification accounted for roughly **86–91%** of cached decode time; drafting was about **28–39 ms per cycle**. These are task-dependent speculative results: a separate short prose streaming request generated 128 tokens in **15.02 seconds (8.52 tok/s)**, with **24.32%** draft acceptance and **2.72 tokens per verify cycle**, versus 6.4 on the synthetic coding fixture. Do not present 18–19 tok/s as universal chat throughput.

### Memory and measurement limits

Peak owned-process RSS was **1.87 GiB** during the cold 8K run and **1.31 GiB** during the warm 16K run. RSS excludes substantial driver/ANE/IOSurface allocations and is **not total model memory**. During 16K inference the system reported 14% memory free and about 28.5 GiB swap in use. After all test servers stopped, memory free returned to 74%, while swap remained allocated. These are whole-machine snapshots with other apps present; they cannot attribute all memory or swap to the model. This was a short profile on the user's active Mac, without power measurement, thermal isolation, extended stability testing, or BF16 quality evaluation.

## Live API and regression checks

A separate 16K server passed live checks for:

- `/health?t=1` and `/v1/models?t=1`, release/chip/compile-mode reporting, and allowed-origin status reads.
- Nonstream chat: `2 + 2` returned `4`.
- Streaming content, usage event and `[DONE]`; no CORS response headers on completions.
- `decode_live` visible during generation and reset to `null` after completion.
- Clean shutdown of the owned test server.

Full source suite: **232 passed, 13 skipped, 120 subtests passed**. Focused regressions also cover clearing live decode after a failed generation, three-bit vector-LUT export reconstruction, preserved rotation sidecars in new and cached M5 builds, and status URLs with query strings. CPU converter-switch tests no longer require a local checkpoint; the missing-Core-ML test explicitly simulates an absent package.

`check_update_model.py` passed six portable tests and live checks against both inventories: the October 4 release reports **56 changed target/drafter files**, while the downloaded 0.2 bundle reports `up_to_date`. The checker downloads only release metadata, pins reads to a resolved Hub commit, reports documentation-only changes separately, and prints a complete-pair download command with a fresh destination. It does not verify installed file contents or replace them.

## Reproduction and local evidence

```sh
python scripts/check_update_model.py --bundle "$FORGE_BUNDLE"
python forge.py quick-test --bundle "$FORGE_BUNDLE" --runtime coreai --check-only

ANEMLL_FORGE_STATE="$PWD/artifacts/m5-v02/state" python scripts/m6_server_bench.py \
  --build "$FORGE_BUNDLE/coreai" --bundle "$FORGE_BUNDLE" --ctx 8192 \
  --out "$PWD/artifacts/m5-v02/bench-8k.json"

ANEMLL_FORGE_STATE="$PWD/artifacts/m5-v02/state" python scripts/m6_server_bench.py \
  --build "$FORGE_BUNDLE/coreai" --bundle "$FORGE_BUNDLE" --ctx 16384 \
  --out "$PWD/artifacts/m5-v02/bench-16k.json"
```

Use fresh output filenames for reruns; the benchmark preserves existing records. Local raw JSON, placement audits, server logs, API responses and memory snapshots are under `artifacts/m5-v02/` (ignored by Git). Downloaded models are under `models/qwen38-release-0.2/`; neither model assets nor compilation caches are committed. The 8K run began before the loaded-numerics reporting fix; its saved record was annotated afterward from the actual loaded manifest, retaining the original source numerics.
