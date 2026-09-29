> Historical research notes imported on 2026-09-29. Paths were generalized; results were not rerun. Later sections may supersede earlier findings. See the current [release workflow](docs/WORKFLOW.md).

# M6 INT8 vector LUT tests, compared with M5 Max (2026-09-25)

The same models and timing harness as [RESULTS_M5_MAX_INT8.md](RESULTS_M5_MAX_INT8.md), run on an
**Apple M6** (`Mac18,5`, ANE arch `h18g`), macOS 27.0 `26A428`. Models were built with
[scripts/lut_fp8_coreml.py](scripts/lut_fp8_coreml.py) (`BALANCED_LUT=1`, iOS 26 MIL target) and timed
with [scripts/time_coreml_pair.swift](scripts/time_coreml_pair.swift), which initializes every input
element: 10 warmup calls, then five rounds of 50 synchronous Core ML predictions, alternating S=8 and
S=16 first. All inputs and activations are FP16; **INT8** describes the LUT values or dense weight
codes. Every model produced cosine 1.00000 against its quantized-weight reference and `MLComputePlan`
assigned every convolution to `NeuralEngine`. Raw data:
[m6_lut_timing.tsv](results/m6_lut_timing.tsv), [m6_lut_placement.tsv](results/m6_lut_placement.tsv),
[m6_parallel_timing.tsv](results/m6_parallel_timing.tsv),
[m6_parallel_placement.tsv](results/m6_parallel_placement.tsv).

```sh
SEQ="dense int8_dense int8_s4 int8_s3 int8_s2 int8_s1 int8_v2n6 int8_v2n4 int8_v4n6 int8_v4n4 int8_v8n4 int8_v16n4"
for s in 8 16; do BALANCED_LUT=1 C=4096 S=$s python lut_fp8_coreml.py $SEQ; done
BALANCED_LUT=1 PARALLEL_BRANCHES=1 C=2048 S=32 python lut_fp8_coreml.py $SEQ
swiftc -O time_coreml_pair.swift -o time_coreml_pair
./time_coreml_pair int8_v2n4 lut_fp8_coreml/int8_v2n4_C4096_S8_balanced.mlmodelc \
    lut_fp8_coreml/int8_v2n4_C4096_S16_balanced.mlmodelc 4096
./time_coreml_pair --single int8_v2n4 lut_fp8_coreml/int8_v2n4_C2048_S32_balanced_parallel.mlmodelc 2048
```

## Sequential chains (4096→4096 1×1 conv, per-layer slope)

`(median(S16) − median(S8)) / 8`, which removes the fixed per-call cost (~0.25 ms).

| Weights | bits/w | Compression vs FP16 | M5 Max ms/layer | M6 ms/layer | M6 vs FP16 | M6 vs INT8 dense | M6 vs M5 Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 dense | 16 | 1× | 0.211 | 0.218 | 1.00× | 0.47× | 0.97× |
| INT8 dense | 8 | 2× | 0.109 | 0.103 | 2.13× | 1.00× | 1.06× |
| INT8 LUT s4 | 4 | 4× | 0.108 | 0.045 | 4.9× | 2.3× | 2.4× |
| INT8 LUT s3 | 3 | 5.3× | – | 0.050 | 4.3× | 2.0× | – |
| INT8 LUT 2×64 (6-bit idx) | 3 | 5.3× | 0.110 | 0.049 | 4.4× | 2.1× | 2.2× |
| INT8 LUT s2 | 2 | 8× | – | 0.028 | 7.7× | 3.6× | – |
| INT8 LUT 2×16 | 2 | 8× | 0.112 | 0.028 | 7.7× | 3.6× | 4.0× |
| INT8 LUT 4×64 (6-bit idx) | 1.5 | 10.7× | 0.110 | 0.024 | 9.2× | 4.3× | 4.7× |
| INT8 LUT s1 | 1 | 16× | – | 0.026 | 8.4× | 4.0× | – |
| INT8 LUT 4×16 | 1 | 16× | 0.111 | 0.026 | 8.5× | 4.0× | 4.3× |
| INT8 LUT 8×16 | 0.5 | 32× | 0.111 | 0.023 | 9.6× | 4.5× | 4.9× |
| INT8 LUT 16×16 | 0.25 | 64× | 0.125 | 0.030 | 7.2× | 3.4× | 4.1× |
| FP16 LUT s4 / 2×16 / 4×16 | | | 0.112 / 0.109 / 0.111 | 0.055 / 0.025 / 0.026 | | | |
| FP8 dense / FP8 LUT s4 / 2×16 / 4×16 / 8×16 | | | – | 0.118 / 0.053 / 0.023 / 0.026 / 0.026 | | | |

## Parallel weight streaming (32 × 2048² branches on one input, summed)

134,217,728 distinct weights per call; includes the fixed per-call cost and 31 adds.

| Weights | bits/w | Nominal MB | M5 Max ms/call | M6 ms/call | M6 vs FP16 | M6 vs INT8 dense |
|---|---:|---:|---:|---:|---:|---:|
| FP16 dense | 16 | 268.4 | 1.852 | 1.924 | 1.00× | 0.61× |
| INT8 dense | 8 | 134.2 | 1.049 | 1.173 | 1.64× | 1.00× |
| INT8 LUT s4 | 4 | 67.1 | 1.036 | 0.678 | 2.84× | 1.73× |
| INT8 LUT 2×64 | 3 | 50.3 | 1.056 | 0.558 | 3.45× | 2.10× |
| INT8 LUT 2×16 | 2 | 33.6 | 1.039 | **0.448** | **4.29×** | **2.62×** |
| INT8 LUT 4×64 | 1.5 | 25.2 | 1.066 | 0.608 | 3.16× | 1.93× |
| INT8 LUT 4×16 | 1 | 16.8 | 1.072 | 0.615 | 3.13× | 1.91× |
| INT8 LUT 8×16 | 0.5 | 8.4 | 1.036 | 0.616 | 3.12× | 1.91× |
| INT8 LUT 16×16 | 0.25 | 4.2 | 1.051 | 0.639 | 3.01× | 1.84× |
| FP8 dense / FP8 LUT s4 / 2×16 / 8×16 | | | – | 1.126 / 0.690 / 0.445 / 0.621 | | |

## Findings

- **Dense weights stream at the same rate on both chips** (~155–165 GB/s); INT8 dense halves FP16
  time on both.
- **The M6 decodes LUT weights from the compressed indices, much faster than the M5 Max.** On the M5
  Max every LUT format stays at the INT8-dense rate (~0.15 T weights/s), so compression below 8 bits
  gives no speedup. On the M6, time falls with index bits down to about 2 bits/weight, then flattens at
  ~0.023–0.028 ms/layer (~0.6–0.7 T weights/s): up to ~9.6× FP16 dense and ~4.5× INT8 dense, 4–5×
  the M5 Max. If the M6 expanded the LUTs to dense weights, they would run at dense speed.
- **Vector LUTs are native on the ANE, but not faster than scalar LUTs at equal bits.** Scalar 2-bit
  (0.028) matches vector 2×16 (0.028); scalar 1-bit (0.026) matches vector 4×16 (0.026). The speedup
  comes from the smaller index stream; the vector form buys accuracy per bit (16 four-weight codewords
  at 1 bit/weight instead of a 2-entry table) and sub-1-bit rates, at the scalar speed.
- **Non-power-of-two index widths appear to be padded.** 3-bit scalar is no faster than 4-bit; 6-bit
  indices behave like 8-bit (2×64 runs like 4 bits/w, 4×64 like 2 bits/w). Inferred from timing only.
- **The LUT value type does not matter on the M6:** INT8, FP16 and FP8 LUTs run at the same speed.
- **Parallel branches:** 2×16 is best (~2.6× INT8 dense). The wider vectors sit at ~0.61 ms, so this
  shape has another limit besides index bytes.

In short: the M6 ANE has hardware-accelerated palettized (LUT) weight decompression that reads
compressed indices at ~0.7 T weights/s, and it runs vector LUTs (≤256 values) natively at the speed of
scalar LUTs with the same index bits. These conclusions come from wall-clock timing and placement;
no hardware counter or compiled weight format was inspected.

## Balanced LUTs

`BALANCED_LUT=1` builds each codebook from ± pairs and scales it to RMS 1/√C. A plain random codebook
of 16 entries has a nonzero mean, which every weight inherits; that rank-one component grows the
signal ~0.25·√C per layer (~16× at C=4096) and overflows FP16 within a few layers. Balancing only
changes values, not shapes or index widths: the earlier unbalanced M6 FP8 timings in the README agree
with the balanced reruns above.
