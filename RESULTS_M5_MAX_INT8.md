> Historical research notes imported on 2026-09-29. Paths were generalized; results were not rerun. Later sections may supersede earlier findings. See the current [release workflow](docs/WORKFLOW.md).

# M5 Max INT8 vector LUT tests (2026-09-25)

This is the INT8 counterpart to the M6 FP8 bandwidth table in the README,
measured on this MacBook Pro (`Mac17,6`, M5 Max, macOS 27.2 `26B5091g`).
It uses Core ML with the iOS 26 MIL target and coremltools 9.0. All inputs and
activations are FP16. **INT8** describes the LUT values or dense weight codes,
not INT8 activations.

## Sequential chains

Each model is a chain of 4096→4096 1×1 convolutions on a 4×4 input. There are
16,777,216 weights per layer. Every reported 8-layer and 16-layer model
produced a finite output with cosine 1.00000 against its quantized-weight
reference; `MLComputePlan` assigned every convolution to `NeuralEngine`. For
each model size, the Swift harness made 10 warmup calls, then five rounds of
50 timed, synchronous Core ML predictions. It alternated whether S=8 or S=16
ran first. The displayed per-layer time is `(median(S16) − median(S8)) / 8`,
which removes most per-call overhead. The raw round medians are in
[results/m5_max_lut_timing.tsv](results/m5_max_lut_timing.tsv); per-model
placement and cosine results are in
[results/m5_max_lut_placement.tsv](results/m5_max_lut_placement.tsv).

The Swift harness initializes **every** input element to a deterministic FP16
value. Earlier timings taken with only one element initialized are superseded
by the values below.

| Weights | bits/w | Compression vs FP16 | ms/layer | vs FP16 | vs INT8 dense | Effective rate |
|---|---:|---:|---:|---:|---:|---:|
| FP16 dense | 16 | 1× | 0.211 | 1.00× | 0.52× | 159 GB/s |
| INT8 dense | 8 | 2× | 0.109 | 1.93× | 1.00× | 154 GB/s |
| INT8 LUT s4 | 4 | 4× | 0.108 | 1.95× | 1.01× | 0.155 T weights/s |
| INT8 LUT 2×64 (6-bit idx) | 3 | 5.3× | 0.110 | 1.92× | 1.00× | 0.153 T weights/s |
| INT8 LUT 2×16 | 2 | 8× | 0.112 | 1.89× | 0.98× | 0.150 T weights/s |
| INT8 LUT 4×64 (6-bit idx) | 1.5 | 10.7× | 0.110 | 1.91× | 0.99× | 0.152 T weights/s |
| INT8 LUT 4×16 | 1 | 16× | 0.111 | 1.90× | 0.98× | 0.151 T weights/s |
| INT8 LUT 8×16 | 0.5 | 32× | 0.111 | 1.90× | 0.99× | 0.152 T weights/s |
| INT8 LUT 16×16 | 0.25 | 64× | 0.125 | 1.69× | 0.88× | 0.135 T weights/s |

`bits/w` is index bits divided by vector width for LUT rows. Compression is
the nominal weight-storage ratio, excluding the small codebook, scale, and
package overhead. Dense rows show nominal stored-weight bandwidth. LUT rows
show weights processed per second; this is **not** a measured hardware decoder
counter. INT8 LUTs from 4 to 0.5 bits/weight all take about 0.108–0.112
ms/layer on this Mac. They do not get faster as the encoded indices shrink.
The same-shape FP16 LUT controls take 0.112, 0.109, and 0.111 ms/layer for
scalar 4-bit, vector 2×16, and vector 4×16 respectively, so the plateau is
not specific to INT8 LUT values.

The 2048-channel vector 2×16 control gives 0.0259 ms/layer with FP16 LUT
values and 0.0270 ms/layer with INT8 LUT values. INT8 time grows 4.14× when
the weights per layer grow 4× to 4096 channels. This is consistent with a
roughly 0.15 T weights/s limit for the sequential Core ML LUT path.

## Parallel weight-streaming workload

The sequential result alone does not test every bandwidth-bound workflow.
The parallel model applies 32 distinct 2048×2048 1×1 weights to the **same**
4×4 input and sums the outputs. Each prediction has 134,217,728 distinct
coefficients, with reuse only across the 4×4 positions within each branch.
Timing used 10 warmups and five rounds of 50 calls. Every iOS 26 model produced
a finite output with cosine 1.00000, and `MLComputePlan` assigned all 32
convolutions and 31 additions to `NeuralEngine`. Raw timings and placement are in
[results/m5_max_parallel_timing.tsv](results/m5_max_parallel_timing.tsv) and
[results/m5_max_parallel_placement.tsv](results/m5_max_parallel_placement.tsv).

| Weights | bits/w | Nominal stored MB | ms/call | vs FP16 | vs INT8 dense | Effective rate |
|---|---:|---:|---:|---:|---:|---:|
| FP16 dense | 16 | 268.4 | 1.852 | 1.00× | 0.57× | 145 GB/s |
| INT8 dense | 8 | 134.2 | 1.049 | 1.77× | 1.00× | 128 GB/s |
| INT8 LUT s4 | 4 | 67.1 | 1.036 | 1.79× | 1.01× | 0.130 T weights/s |
| INT8 LUT 2×64 | 3 | 50.3 | 1.056 | 1.75× | 0.99× | 0.127 T weights/s |
| INT8 LUT 2×16 | 2 | 33.6 | 1.039 | 1.78× | 1.01× | 0.129 T weights/s |
| INT8 LUT 4×64 | 1.5 | 25.2 | 1.066 | 1.74× | 0.98× | 0.126 T weights/s |
| INT8 LUT 4×16 | 1 | 16.8 | 1.072 | 1.73× | 0.98× | 0.125 T weights/s |
| INT8 LUT 8×16 | 0.5 | 8.4 | 1.036 | 1.79× | 1.01× | 0.130 T weights/s |
| INT8 LUT 16×16 | 0.25 | 4.2 | 1.051 | 1.76× | 1.00× | 0.128 T weights/s |

The dense rows correspond to 145 and 128 GB/s if their nominal stored weights
stream once, consistent with a weight-bandwidth gain from FP16 to INT8 dense.
The much smaller LUT index streams do **not** reduce call time
below INT8 dense in this Core ML path. The effective LUT weight rate stays
near 0.125–0.130 T weights/s. The iOS 18 route gives the same pattern:
1.845 ms for FP16 dense and 1.040 ms for INT8 vector 4×16; all 32 convs
were assigned to `NeuralEngine`, and the INT8 output cosine was 0.9999994.

These timings identify a weight-count-related limit for the tested LUT paths,
not its exact cause. Preferred placement and wall-clock time cannot establish
whether the ANE decodes indices natively, whether the compiler expands weights,
or which hardware resource dominates. The M6 FP8 results in the README are a
different chip and toolchain; their speedups should not be assigned to this
M5 Max.

The initial unbalanced random LUT chains overflowed FP16 by layer 7–8 for
three formats. The reported LUT models use `BALANCED_LUT=1`, which builds each
codebook from positive/negative pairs and normalizes its variance. This
produced finite outputs through 16 layers. The original M6 results were not
changed; balanced artifacts have a `_balanced` filename suffix. Parallel
artifacts also have a `_parallel` suffix. The code for these models is
[scripts/lut_fp8_coreml.py](scripts/lut_fp8_coreml.py), and the timing harness
is [scripts/time_coreml_pair.swift](scripts/time_coreml_pair.swift).
