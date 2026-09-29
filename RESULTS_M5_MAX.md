> Historical research notes imported on 2026-09-29. Paths were generalized; results were not rerun. Later sections may supersede earlier findings. See the current [release workflow](docs/WORKFLOW.md).

# M5 Max vector LUT validation (2026-09-25)

Machine: MacBook Pro `Mac17,6`, Apple M5 Max, ANE target `h17c`, macOS 27.2
(`26B5091g`). Python 3.12.13 with coremltools 9.0, coreai-opt 0.2.1,
coreai-torch 0.4.2, and PyTorch 2.11.0. The CoreAI scripts used the existing
environment and helper modules in `../fp8-mlp-metal41-bench/coreai`.

The tested vector format was `cluster_dim=2`, `n_bits=4`: 16 LUT entries, two
FP16 or INT8 values per entry, two index bits per weight. All tests used
512 input and output channels, a 1×1 convolution, FP16 input, and per-tensor
LUTs. Models were compiled and executed on this Mac. The CoreAI timings are
50 warmed calls after five warmups, so they exclude export and compilation.

| Ordered check | Result | Placement and output evidence |
|---|---|---|
| CoreAI FP16 LUT, four convolutions, 4×4 spatial | **Pass**; 0.271 ms median | Cache manifest has `mps.fullyPlacedOnANE` and `mps.noGPUActivity`; cosine vs FP32 PyTorch reference 1.0000. |
| Core ML INT8 LUT, one convolution, 32×32 spatial | **Pass** | `MLComputePlan` assigns the convolution to `NeuralEngine`; cosine vs FP32 reference 1.00000. The FP16 LUT control with the same shape also passed. |
| CoreAI INT8 LUT, four convolutions, 4×4 spatial | **No full ANE placement**; 7.938 ms median | Output cosine 0.9999, but the cache manifest lacks `mps.fullyPlacedOnANE`; an ANE region is present. |

For a fresh one-convolution CoreAI INT8 compile, setting
`MPSGRAPH_PRINT_ANE_PLACEMENT_ANALYSIS=1` reported six runtime layers:
four ANE and two GPU. It listed `mps.multiply`, `mps.permute`, `mps.identity`,
and `mps.conv_2d` among operations that could not be placed on ANE. Thus the
CoreAI INT8 model produces a correct output but does **not** run the complete
vector-LUT convolution on the ANE. This one-layer diagnostic measured 2.280 ms.

The exported CoreAI INT8 IR applies `coreai.blockwise_shift_scale` **after**
`coreai.lut_to_dense`, while the working Core ML model applies
`constexpr_blockwise_shift_scale` to the INT8 LUT **before**
`constexpr_lut_to_dense`. That graph difference is a plausible reason for
the placement difference; it is not established as the sole cause.

Reproduction, from this repository using the existing environment:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../fp8-mlp-metal41-bench/coreai \
  ../fp8-mlp-metal41-bench/coreai/.venv/bin/python scripts/bench_vector_lut.py \
  v2n4 --channels=512 --hw=4 --stack=4 --compute=ane --lut-dtype=fp16

PYTHONDONTWRITEBYTECODE=1 \
  ../fp8-mlp-metal41-bench/coreai/.venv/bin/python scripts/lut_rules.py \
  base_v2n4 int8lut_v2n4

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../fp8-mlp-metal41-bench/coreai \
  ../fp8-mlp-metal41-bench/coreai/.venv/bin/python scripts/bench_vector_lut.py \
  v2n4 --channels=512 --hw=4 --stack=4 --compute=ane --lut-dtype=int8
```

CoreAI needs access to its user cache and ANE compiler services; a sandboxed
attempt failed to load its specialization with `AIModelCacheError error 0`.
The reported runs were made with native access. These checks establish model
execution and reported placement on this OS and toolchain; they do not prove
instruction-level ANE decode behavior or generalize to other models.
