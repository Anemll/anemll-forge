# Late-Layer KV Approximation (LLKVApprox / CED-style) for Qwen3.8-27B

A spinoff research prototype of the "project the late-half KV / mixer state from a mid-stack
residual" idea, targeting the anemll-forge Qwen3.8-27B hybrid (48 Gated-DeltaNet + 16 GQA
layers) on Apple Silicon (M3U GPU/MLX prototype → M6 ANE path).

This prototype **adapts published work rather than reinventing it**:

- **DeepSeek-V4.1-Flash CED** — decoder global KV projected from an encoder mid-state `H_{L/2}`
  ([arxiv 2609.19969](https://arxiv.org/abs/2609.19969)).
- **@kis LLKVApprox on Qwen3-8B** — ~½ prefill, frozen base + a trained projector
  ([tweet](https://x.com/kis/status/2098185646749909306),
  [blog 1](https://nowokay.hatenablog.com/entry/2026/09/10/203534),
  [blog 2](https://nowokay.hatenablog.com/entry/2026/09/11/120001),
  [projector](https://huggingface.co/kishida/Q3-8B-KVA-Projector),
  [demo](https://kishida.github.io/webdemos/llkvapprox/)).
- **alimpfard's hybrid Qwen3.8-27B KV-approximation** — the recipe this prototype follows
  ([model card](https://huggingface.co/alimpfard/qwen3.8-27b-kv-approximation)):
  prefill layers 0–31 exact; a projector maps the layer-31 residual to the late mixer inputs for
  layers 32–63; late GDN predicts `in_proj_qkv / in_proj_a / in_proj_b` then runs the exact cheap
  conv+delta scan; late GQA predicts `k_proj / v_proj` (pre-norm, pre-RoPE) then the exact K/V
  write; an exact tail of the last 1–4k tokens is recommended for agentic fidelity. Reported
  ~1.5–1.6× prefill on GPU, ~+4.5% PPL (llama.cpp kva fork; not ANE yet).

> This is **research software and a mechanism prototype**, not a trained model or a benchmark of
> Qwen3.8-27B. See the constraints section and
> [`RESULTS_KVA.md`](../../RESULTS_KVA.md) for exactly what was and was not measured.

## What is here

| file | purpose |
| --- | --- |
| `kva_qwen38.py` | NumPy reference of the Qwen3.8 decoder (RMSNorm, GDN conv+delta scan, GQA with partial RoPE and the per-head output gate) plus the KVA on/off prefill paths, the projector I/O contract, exact-tail flag, timing and analytic MAC accounting. Mirrors `scripts/qwen38_decode_ref.py`. |
| `qwen3_5_27b_config.reference.json` | Architecture dims reconstructed from this repo so the prototype and its tests run **without the checkpoint**. Provenance for every number is in the file. At runtime `--model` reads the real `text_config` and overrides it. |
| `measured/` | JSON reports from the measured synthetic runs quoted in `RESULTS_KVA.md`. |
| `../../tests/test_kva_prototype.py` | Dry-run I/O shape + correctness tests (pure NumPy, no checkpoint). |

## Why NumPy / why a prototype

The release path is Core AI / Core ML on Apple Silicon (M3U → M6 ANE). That toolchain, the
Qwen3.8-27B checkpoint, and Apple GPU/ANE are **not present on Linux/CI**, where this was authored
(no macOS, no MLX, no weights — see `RESULTS_KVA.md` for the exact blocker list). So the prototype
is dependency-light and hardware-independent: it validates the projector I/O contract and the
prefill plumbing, and measures the *structural* compute reduction, on any machine. The absolute
NumPy wall-times are **not** representative of GPU/ANE throughput; the analytic matmul-MAC ratio is
the hardware-relevant figure.

## Flags

```
--kva on|off              apply LLKVApprox (on) or run the full model (off)
--kva-tail-exact N        trailing prompt tokens processed exactly through the late layers
--kva-split K             exact early-layer count (default num_hidden_layers//2 = layer 32)
--prompt-len T            synthetic prompt length
--model DIR               read the real Qwen3.8-27B text_config from DIR/config.json
--full-dims               use the real Qwen3.8-27B dims (heavy for prefill; fine for --shapes/--account)
--scale-hidden / --scale-layers   synthetic size for fast CPU timing
--fit-projector           least-squares fit the projector (synthetic demonstration of the recipe)
--compare                 run BOTH on and off; report prefill speedup + greedy agreement + PPL
--shapes                  dry-run: print the projector I/O + state shapes, then exit
--account                 dry-run: print the analytic matmul-MAC KVA speedup, then exit
--json PATH               write a JSON report
```

## Quick start (runs anywhere, pure NumPy)

Projector I/O contract against the real Qwen3.8-27B dims (no weights, no checkpoint):

```sh
python pipelines/kva/kva_qwen38.py --shapes --full-dims
```

Hardware-independent prefill speedup from the recipe (no weights):

```sh
python pipelines/kva/kva_qwen38.py --account --full-dims --prompt-len 2048 --kva-tail-exact 256
```

Measured on/off prefill + decode on a fast synthetic config:

```sh
python pipelines/kva/kva_qwen38.py --compare --prompt-len 512 --scale-hidden 768 --scale-layers 16 \
  --kva-tail-exact 64 --fit-projector
```

Shape + correctness tests:

```sh
python -m unittest tests.test_kva_prototype
```

## Using a real frozen Qwen3.8-27B + a published projector

With the checkpoint reachable, `--model DIR` reads the real `text_config` (overriding the reference
dims). The remaining work to run a true on/off benchmark is wiring a weight loader and a trained
projector; both are blocked in this environment and tracked in `RESULTS_KVA.md`:

1. Load frozen Qwen3.8-27B weights (safetensors) into the per-layer dict the forward functions
   expect (`init_weights` documents the exact checkpoint-relative names).
2. Load projector weights into the per-late-layer head dict (`projector_output_spec` is the
   contract). alimpfard's artifact is the recommended source **if license-compatible**; otherwise
   train only the projector against frozen-base mid-residual → late mixer-input targets.
3. Run `--compare --model DIR` at 512 / 2k / 4k and record prefill tok/s, decode tok/s, continuation
   PPL and greedy agreement vs exact.

The GDN/GQA math here is a line-by-line NumPy mirror of `scripts/qwen38_decode_ref.py`, so a Torch
or MLX port that reuses the forge modules will match it; the KVA control flow (`prefill_kva`) is the
only new piece.

## Constraints honored

- No claim that peer-reviewed Qwen3.8-27B CED papers exist.
- No invented benchmark numbers. The only speedups reported are (a) analytic matmul-MAC ratios and
  (b) measured NumPy wall-times on an explicitly synthetic config; neither is an Apple GPU/ANE
  benchmark.
- Frozen base preferred; the projector is the only trainable part, and only a synthetic
  least-squares demo is included (`--fit-projector`) because the real checkpoint/compute is absent.
