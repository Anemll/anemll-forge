# M6 ANE compute feasibility: how much can attention-dtype buy end-to-end

- **Scope:** continuation of [M6_ANE_COMPUTE_2026-10-02.md](M6_ANE_COMPUTE_2026-10-02.md). Same target: Apple M6 ANE / Core ML / Core AI compute in ANEMLL Forge, with a ~2x compute-throughput goal versus the current baseline path.
- **Question answered here:** suppose INT8-INT8 or FP8-FP8 attention really runs faster on the M6 ANE. How much would end-to-end prefill/decode actually move, and what has to be measured first for the goal to be reachable?
- **Status:** analysis plus a pure-Python leverage model ([`scripts/m6_compute_roofline.py`](../../scripts/m6_compute_roofline.py)). Runs on this Linux VM. No M6, Core ML, Core AI, or weights here. No ANE timing was collected.

Evidence labels match the companion doc: **Source-verified**, **Measured (prior)**, **Inferred**, **External doc**, and a new **Modeled** for arithmetic this module produces from documented bytes. Nothing below is a measured 2x.

## Why a leverage model, not another kernel

The companion doc established, **Source-verified**, that Forge's V8 attention dequantizes INT8 V to FP16 and runs QK/PV as FP16, and that no INT8-INT8 or FP8-FP8 attention MAC runs today. The natural next step is to build that kernel and time it. Before spending M6 time, this doc bounds the payoff so the on-device experiment has a pass/fail gate instead of a hope.

The lever is Amdahl's law. If attention compute takes a fraction `f` of a forward and a new kernel speeds that phase by `s`, the end-to-end speedup is:

```
end_to_end = 1 / ((1 - f) + f / s)
```

A 2x attention kernel (`s = 2`) is not a 2x forward unless attention is most of the forward.

## What the model says

All numbers in this section are **Modeled** from documented bytes in [PERFORMANCE_SESSION.md](../PERFORMANCE_SESSION.md) and the V8 trace. They are arithmetic, not measured time. Reproduce with `python3 scripts/m6_compute_roofline.py`.

### Byte traffic: KV is a minority of the forward

The documented traffic model is about 10.61 GB of weights per verifier forward, plus KV history of 64 KiB per token (FP16) or 48.125 KiB (V8). The KV share of modeled forward bytes:

| Context | KV share, FP16 cache | KV share, V8 cache |
| --- | --- | --- |
| 8K | 0.048 | 0.037 |
| 16K | 0.092 | 0.071 |
| 32K | 0.168 | 0.132 |
| 64K | 0.288 | 0.233 |

**Modeled.** Even at 64K, KV traffic is under a third of the bytes, and V8 already removed part of it. A change that only touches KV bytes (for example K8 on top of V8) has a small bandwidth ceiling. This is a **bandwidth** share; attention **compute** is separate and not captured by byte counting. Decode in the imported notes is described as weight-bandwidth bound, which is exactly the regime where an attention-compute speedup helps least. **Measured (prior)** that decode looks bandwidth bound; **Inferred** that attention-compute changes are therefore low-leverage for short-context decode.

### Amdahl: the gate for a 2x forward

To reach a target end-to-end speedup `T`, the sped-up kernel must occupy more than `1 - 1/T` of forward time. For `T = 2`, that is more than 50%. **Modeled** outputs:

| Attention time fraction (measured on M6) | Kernel speedup needed for 2x end-to-end |
| --- | --- |
| 0.10 | unreachable at any finite speedup |
| 0.30 | unreachable at any finite speedup |
| 0.50 | unreachable (needs an infinitely fast kernel) |
| 0.70 | about 3.5x |

And the other direction, how much a fixed kernel speedup buys:

| Attention fraction | `s = 2` kernel | `s = inf` kernel (attention free) |
| --- | --- | --- |
| 0.25 | 1.14x end-to-end | 1.33x end-to-end |
| 0.50 | 1.33x end-to-end | 2.00x end-to-end |

**Modeled.** The reading: a 2x attention kernel only yields a 2x forward if attention is already at least half the forward time and the kernel is better than 2x. For the compute goal to be reachable from attention alone, the measured attention fraction has to be high. That is the single most important on-device number, and it is currently unknown for this model.

## Consequences for the ranked options

Combining the companion doc's ranks with this leverage model (**Inferred** unless noted):

1. **INT8-INT8 attention (rank 1).** Still the first experiment, because it is the only full-attention path and Forge already inserts the dequant that is the insertion point. But its end-to-end value is gated by the attention fraction. Likely meaningful in **prefill and long-context** (more attention compute), likely small for short-context decode (bandwidth bound). Measure the fraction before promising a forward-level gain.
2. **FP8-FP8 attention (rank 2).** Same leverage gate, plus the extra risk that the compiler casts E4M3 to FP16, which Forge already observed for FP8 LUT values. **Measured (prior)** that FP8 LUT values were promoted to FP16 at compile time.
3. **K8 on top of V8 (rank 3).** A bandwidth play. The table above caps it: KV is less than a third of bytes even at 64K, and V8 took part already. Do not expect a compute 2x from it.
4. **Mixer (GDN) low-precision compute (rank 6).** GDN is 48 of 64 layers, so its time fraction can be larger than attention's. If a mixer-compute speedup is real, its Amdahl leverage is higher than attention's, but the recurrent-quality risk is **High** per the imported DeltaNet results. **Measured (prior)** that 4-bit DeltaNet cost KL 0.170 before the rank-64 rescue.
5. **Fused/blocked attention, zero-skip, compile flags, Winograd (ranks 4,5,7,8).** None are a 2x MAC. Compile mode 2 is already default and was the same speed as mode 0. **Measured (prior).**

Net: a ~2x **compute** result on the M6 is most plausible as a per-kernel prefill/long-context attention number, not an end-to-end decode number, and only if the measured attention fraction is high. The honest framing stays: research, with a measurement gate, not a promised 2x.

## On-device decision gate

Run these in order on an M6. Stop and reconsider if an early gate fails.

1. **Measure the attention time fraction** of a forward at 8K/16K/32K/64K for prefill and decode, with the current V8 build. Isolate attention-core time from projections, MLP, GDN, head, and host work. Without this number, no end-to-end claim is defensible.
2. Feed that fraction into `python3 scripts/m6_compute_roofline.py options --attention-fraction <f>` to get the modeled end-to-end ceiling per option. If the ceiling is below the goal even at an infinite kernel speedup, the option cannot reach it from attention alone.
3. Only if the ceiling is promising, build and time the INT8-INT8 attention core from [`scripts/m6_attn_compute.py`](../../scripts/m6_attn_compute.py) (`mil-skeleton --build`, then `ane-bench`). Record executed dtype and placement, not just storage.
4. Compare the per-kernel speedup to the modeled ceiling. Report both. Never multiply the host NumPy error budget or the modeled ceiling into a tokens/s claim.

## What this VM verified

- The leverage model runs and its outputs match hand arithmetic (`tests/test_m6_compute_roofline.py`): Amdahl values, required-fraction and required-speedup solvers, the KV-share table, and the V8 versus FP16 KV byte constants.
- Reproduced the companion harness still passes (`tests/test_m6_attn_compute.py`).
- Byte constants trace to documented repository figures. No new measurement was performed or invented.

## What still needs an M6

- The attention time fraction (gate 1). Everything downstream depends on it.
- The executed attention dtype on M6 (fused INT8 MAC versus dequant-to-FP16 versus CPU fallback).
- Per-kernel INT8-INT8 and FP8-FP8 latency at each context, prefill and decode.
- Only then, a paired full-server prefill/decode comparison in the style of the V8 study.

## Attribution

Qwen3.8-27B is developed by the Qwen Team; original model copyright belongs to Alibaba Cloud (Apache 2.0). ANEMLL provides independent quantization, conversion, and ANE research. Apple and coremltools material is cited as documentation, not as this project's measurement.
