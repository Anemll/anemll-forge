# Running the M6 attention-compute prototype

Host-side recipes and a Core ML skeleton for the top candidate in [M6_ANE_COMPUTE_2026-10-02.md](M6_ANE_COMPUTE_2026-10-02.md): keep V8-style INT8 values, and test whether QK and PV can run as INT8-INT8 (or FP8-FP8) compute instead of dequantizing to FP16.

This is not a measured 2x result. NumPy times on any machine are host arithmetic only.

## On this Linux checkout (no M6)

From the repository root, with NumPy:

```sh
python3 -m unittest tests/test_m6_attn_compute.py -v

python3 scripts/m6_attn_compute.py options
python3 scripts/m6_attn_compute.py plan
python3 scripts/m6_attn_compute.py compare --ctx 128 --queries 8 --seed 0
python3 scripts/m6_attn_compute.py host --recipe v8_current --ctx 256 --queries 8
python3 scripts/m6_attn_compute.py host --recipe int8_int8_attn --ctx 256 --queries 8
python3 scripts/m6_attn_compute.py ranges --ctx 256
python3 scripts/m6_attn_compute.py mil-skeleton --out /tmp/m6_attn_mil_plan.json
```

`mil-skeleton` writes a plan JSON. It builds an `.mlpackage` only if `coremltools` is importable. This VM does not have `coremltools` or an ANE.

## On an Apple M6 (required for ANE numbers)

Need: macOS 27, Xcode / coremltools matching [ENVIRONMENT.md](../ENVIRONMENT.md), Python 3.11+, and enough unified memory to compile a small attention graph (weights are not required for the synthetic core).

1. Run the host commands above first. They check the V8 score-scale identity and the INT8/FP8 recipes without touching the ANE.
2. Install the conversion extras and compile the skeleton:

```sh
python3 scripts/m6_attn_compute.py mil-skeleton --build --out "$HOME/forge-m6-attn-compute"
```

That writes `plan.json` plus, when the build succeeds, one compiled model per recipe that `coremltools` can lower (`fp16_baseline`, `v8_current`, and an INT8-input dequant graph). Treat a failed INT8-INT8 or FP8-FP8 compile as a result: the compiler refused the compute path.

3. Time on ANE only, same shapes as Qwen3.8 full attention (4 KV heads, 6 query heads per KV head, head dim 256, T=8):

```sh
python3 scripts/m6_attn_compute.py ane-bench \
  --models "$HOME/forge-m6-attn-compute" \
  --ctx 8192,16384,32768,65536 \
  --queries 8 --warmup 10 --rounds 50
```

`ane-bench` refuses to run unless `coremltools` loads a compiled model. Record median ms, `MLComputePlan` unit, and whether every convolution/matmul is `NeuralEngine`.

4. Optional: rebuild one real four-layer chunk with the current V8 graph (`--kv-cache-dtype v8`) and compare single-layer T=8/T=64 latency to the synthetic core. Conversion still needs the original checkpoint and export. See [KV_CACHE_V8.md](../KV_CACHE_V8.md).

## What to write down on the M6

For each recipe and context:

- Compile success or the exact ANEC / MPSGraph error.
- Placement: ANE / GPU / CPU, compile mode (`MPSGRAPH_ANE_BONDED_COMPILE_MODE=2` and `0`).
- Median latency and input payload bytes (the harness prints the logical byte counts).
- Output relative RMSE versus the host FP16 reference on a short synthetic draw (the harness prints this for host recipes; repeat after `predict`).
- Any evidence of executed dtype (compute-plan op types, not just input storage).

Do not combine those kernel numbers with the full-server V8 tokens/s table. Do not claim 2x unless a paired ANE measurement of the same graph shows it.

## Companion: end-to-end leverage model

Before timing a kernel on an M6, bound the payoff with the pure-Python leverage model (runs anywhere, no coremltools):

```sh
python3 -m unittest tests/test_m6_compute_roofline.py -v

python3 scripts/m6_compute_roofline.py traffic --ctx 8192 16384 32768 65536
python3 scripts/m6_compute_roofline.py required --target 2.0
python3 scripts/m6_compute_roofline.py amdahl --fraction 0.25 --kernel-speedup 2.0
# After measuring the attention time fraction on an M6:
python3 scripts/m6_compute_roofline.py options --attention-fraction 0.3
```

See [M6_COMPUTE_FEASIBILITY_2026-10-02.md](M6_COMPUTE_FEASIBILITY_2026-10-02.md). The `options` command needs an attention time fraction measured on device; the default is a placeholder, not a result.

## Flags

| Flag | Meaning |
| --- | --- |
| `--recipe` | `fp16_baseline`, `v8_current` (Forge today), `int8_int8_attn` (top candidate), `fp8_fp8_attn` |
| `--fp8-max` | `240` (ANE weight E4M3 convention) or `448` (E4M3FN max) |
| `--ctx` / `--queries` | history length and query rows (default T=8) |
| `--qk-dtype` / `--pv-dtype` | override recipe operands: `fp16`, `int8`, `fp8` |
| `--accum` | `fp16`, `fp32`, `int32` |
