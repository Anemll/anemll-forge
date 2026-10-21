# Late-Layer KV Approximation (LLKVApprox / CED-style) for Qwen3.8-27B — integration + measurement

Spinoff research integration of the "project the late-half KV / mixer state from a mid-stack
residual" idea for the anemll-forge Qwen3.8-27B hybrid, following
[alimpfard's recipe](https://huggingface.co/alimpfard/qwen3.8-27b-kv-approximation) and the
[@kis Qwen3-8B LLKVApprox](https://x.com/kis/status/2098185646749909306) /
[DeepSeek CED](https://arxiv.org/abs/2609.19969) line of work. Code and usage:
[`pipelines/kva/`](pipelines/kva/README.md).

> **Scope and honesty.** This run was authored in an environment with **no macOS, no MLX, no Apple
> GPU/ANE, and no Qwen3.8-27B checkpoint** (details in [Blocked](#blocked-for-a-real-qwen3827b-benchmark)).
> A full on/off benchmark of the real model is therefore blocked. What *is* delivered and verified:
> a hardware-independent NumPy prototype that implements the recipe end-to-end, a validated projector
> I/O contract against the real config dims, a bit-exact-tail correctness invariant, and prefill
> speedups measured two ways (analytic matmul-MACs and synthetic-config NumPy wall-time). No
> Apple-hardware numbers are claimed, and no peer-reviewed Qwen3.8-27B CED paper is claimed to exist.

## Hardware / software for the measurements below

- **Host:** Linux `6.12.94` x86-64, glibc 2.39 (cloud CI VM; **not** Apple Silicon).
- **Software:** Python 3.12, NumPy 2.4.4 only. No torch, transformers, coremltools, MLX, or Core AI.
- The NumPy wall-times are a **mechanism demonstration on CPU**, dominated by the sequential
  per-token Gated-DeltaNet Python loop. They are **not** representative of M3U GPU / M6 ANE
  throughput. The hardware-relevant figure is the analytic matmul-MAC ratio.

## Architecture verified against the repo (not invented)

Qwen3.8-27B (`qwen3_5`, hybrid Gated-DeltaNet + GQA), reconstructed from forge source into
[`pipelines/kva/qwen3_5_27b_config.reference.json`](pipelines/kva/qwen3_5_27b_config.reference.json)
(each number's provenance is in that file):

| property | value | source |
| --- | --- | --- |
| layers / hidden / MLP / vocab | 64 / 5120 / 17408 / 248320 | `QUANTIZATION_NOTES.md`, `forge.py:63` |
| layout | 48 GDN (`linear_attention`) + 16 GQA (`full_attention`), interval 4 (layers 3,7,…,63) | `QUANTIZATION_NOTES.md:16` |
| GQA heads | 24 Q / 4 KV, head_dim 256, q_proj emits q+gate (2·hd) | `QUANTIZATION_NOTES.md:16`, `scripts/qwen38_decode_ref.py` |
| GDN | nv=48, nk=16, dk=dv=128, conv kernel 4 | `scripts/qwen38_ane_chunk.py:4`, `DFLASH2_ANE_PLAN.md:134` |
| GDN conv_dim | 10240 = 2·16·128 + 48·128 | `ANE_DELTANET_NUMERICS.md:27` |
| GDN in_proj | qkv 10240, z 6144, a 48, b 48 (×5120) | `scripts/qwen38_ane_chunk.py:216-219` |

The GDN conv + delta recurrence and GQA (partial RoPE, per-head output gate) math in
`kva_qwen38.py` is a line-by-line NumPy mirror of `scripts/qwen38_decode_ref.py`.

## The recipe as implemented

`prefill_kva` (in `kva_qwen38.py`):

1. **Exact early half** — layers `[0, split)` (`split = L/2 = 32` by default) run exactly for all
   prompt tokens, producing the split residual `H_32[t]` and the early layers' own KV / GDN state.
2. **Projected late half, approximated region** — for prompt positions `[0, T − tail_exact)` a thin
   per-late-layer linear projector maps `H_32` to the late mixer inputs:
   - late GDN: predict `in_proj_qkv / in_proj_a / in_proj_b` (+`z` for the output gate), then run
     the **exact cheap conv + delta scan** to advance the recurrent/conv state;
   - late GQA: predict `k_proj / v_proj` (pre-norm, pre-RoPE), apply exact `k_norm` + RoPE, then the
     **exact K/V write**.
   The late MLP, out-proj/o-proj and residual propagation are **skipped** for these positions — that
   is where prefill time is saved.
3. **Exact tail** — the last `tail_exact` tokens run the full late stack, so the generation boundary
   and the final logits are exact.

Decode then proceeds over all 64 layers using the (mostly approximated) late state.

### Correctness invariant (verified)

With `tail_exact == T` there is no approximation, so KVA must equal the full model. Measured
relative error of the final-position hidden state: **0.0** (bit-exact) — see
`tests/test_kva_prototype.py::test_full_exact_tail_matches_full_model`. This confirms the exact-tail
path and the early/late split reproduce the full model; only the projected region differs.

## Measured prefill speedup

### Analytic matmul-MACs at the real Qwen3.8-27B dims (hardware-independent)

`python pipelines/kva/kva_qwen38.py --account --full-dims --prompt-len T --kva-tail-exact 256`

| prompt T | off (GMAC) | on (GMAC) | **speedup** |
| ---: | ---: | ---: | ---: |
| 512 | 12552 | 9972 | **1.26×** |
| 2048 | 50518 | 32322 | **1.56×** |
| 4096 | 101861 | 62482 | **1.63×** |

The 2k/4k figures land in alimpfard's reported **~1.5–1.6× GPU prefill** range. Short prompts gain
less because the fixed 256-token exact tail is a larger fraction of them.

Tail knob at T=2048 (`--kva-tail-exact`):

| tail_exact | 0 | 256 | 1024 | 2048 |
| ---: | ---: | ---: | ---: | ---: |
| speedup | 1.70× | 1.56× | 1.26× | 1.00× (= full model) |

### Measured NumPy wall-time on a synthetic config (CPU mechanism demo)

`--compare --scale-hidden 768 --scale-layers 16 --kva-tail-exact 64` (12 GDN + 4 GQA, split 8).
JSON in [`pipelines/kva/measured/`](pipelines/kva/measured/).

| prompt T | off (ms) | on (ms) | wall speedup | analytic MACs |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 664 | 530 | 1.25× | 1.48× |
| 512 | 1868 | 1251 | 1.49× | 1.60× |
| 1024 | 6526 | 4000 | 1.63× | 1.68× |

Measured wall-time tracks the analytic ratio and converges toward it as the prompt grows and the
matmul (MLP) work dominates the fixed per-token scan overhead.

## Quality (continuation PPL / greedy agreement) — not meaningful here

`--compare` also reports greedy agreement vs exact and continuation PPL. On this prototype the base
weights are **random** (no checkpoint), so there is no linguistic signal for a projector to recover;
the agreement numbers (25–100% across configs/seeds) and PPL are **not a quality signal** and are
reported only to exercise the decode path. A least-squares projector fit (`--fit-projector`) is
included as a synthetic demonstration of the training plumbing, not a Qwen3.8-27B result. Real
quality evaluation (alimpfard reports ~+4.5% PPL) requires the frozen checkpoint + a trained
projector on Apple Silicon — see below.

## Smoke behavior (tail on/off)

The chat / tool-ish / code-ish smoke reduces, in this no-checkpoint setting, to the exact-tail knob:
`tail_exact = 0` maximizes approximation (1.70× MACs) while `tail_exact = T` is bit-exact with the
full model (1.00×, rel-err 0.0). Agentic / tool / code prompts, where the most recent tokens matter
most, should use a non-zero `--kva-tail-exact` (alimpfard recommends 1–4k); the knob trades the
measured speedup above for tail fidelity.

## Reproduce

```sh
# projector I/O contract against the real Qwen3.8-27B dims (no weights, no checkpoint)
python pipelines/kva/kva_qwen38.py --shapes --full-dims

# hardware-independent prefill speedup from the recipe
python pipelines/kva/kva_qwen38.py --account --full-dims --prompt-len 2048 --kva-tail-exact 256

# measured on/off prefill + decode on a fast synthetic config
python pipelines/kva/kva_qwen38.py --compare --prompt-len 512 --scale-hidden 768 --scale-layers 16 \
  --kva-tail-exact 64 --fit-projector

# shape + correctness tests
python -m unittest tests.test_kva_prototype
```

## Blocked for a real Qwen3.8-27B benchmark

A real on/off benchmark needs, none of which is present in this environment:

1. **The Qwen3.8-27B checkpoint** (safetensors + `config.json`). Searched `/Volumes/*`, `Qwen3.8*`
   and `*wen3*/config.json`; none found. Provide a path and run with `--model DIR` (it already reads
   the real `text_config`).
2. **A weight loader** mapping the checkpoint's per-layer tensors into the dict the forward functions
   consume (names documented in `init_weights`). Small, mechanical; deferred because there is nothing
   to load here.
3. **A projector artifact** — alimpfard's weights *if license-compatible*, or a projector trained
   against frozen-base `H_32 → late mixer-input` targets. Training needs the checkpoint + GPU.
4. **Apple Silicon (M3U GPU / M6 ANE)** and the Core AI / MLX toolchain for the real throughput and
   PPL numbers. This VM is Linux x86-64.

## Next steps toward ANE (M6)

Not measured; design notes only, consistent with the forge ANE findings:

- **Early-exit the late `mf*` chunks during prefill.** The 16×4-layer Core AI/Core ML chunking
  (`scripts/qwen38_ane_chunk.py`, `coreai/qwen38_coreai_build.py`) already isolates layer groups.
  Late chunks (32–63) would run a *projector head + the cheap GDN scan / GQA K/V write* function
  instead of the full mixer+MLP, writing the same host-owned DeltaNet buffers (`GDN_IO`,
  `DFLASH2_ANE_PLAN.md`) and KV cache the decode path already reads.
- **Hybrid GDN + GQA state** is already host-owned and context-resizable (the speculative/context-
  expansion machinery in `docs/SPECULATIVE_DECODING.md`), so an approximated prefill can populate it
  and hand off to exact decode unchanged.
- **Numerics caution:** the late projector outputs feed the same fp16-sensitive GDN scan that needed
  the `SILU=tanh` + `GDN_SQ/GDN_SV` scaling fixes (`ANE_DELTANET_NUMERICS.md`). A projector trained
  in fp32 must be validated through those ANE fixes; the small near-zero conv/scan activations are
  exactly where the ANE silu/subnormal errors bite.
- Do **not** claim ANE speedups until measured on-device with exact artifacts, hardware, context, and
  generation settings (per the repo's benchmark policy).
