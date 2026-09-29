# Techniques and limitations

This guide interprets the imported experiments, not a new hardware validation. Dates, chip, tensor shapes, compiler, and runtime matter. Treat placement boundaries as observations for the tested stacks, not permanent architectural guarantees.

## Compression that preserves the ANE path

Vector LUT quantization stores one index for several weights along output channels. A 2×16 codebook uses a four-bit index for two weights: nominally two bits per weight, excluding codebook/scales. The research found vectors up to width 16 and codebooks up to 256 total values on the tested paths. Per-group vector LUTs and vectors along input channels lost ANE placement. Scalar per-group LUTs had different behavior.

At fixed index width, wider vectors mainly improve accuracy per bit or reduce storage. The reported M6 bandwidth benefit saturated below roughly two bits per weight; M5 Max did not show the same decode speedups. Do not equate compression ratio with end-to-end speedup. [Original vector-LUT investigation](history/VECTOR_LUT_README.md), [M6 comparison](../RESULTS_M6_INT8.md).

Per-output-channel scaling normalizes weights before fitting a shared codebook. GPTQ uses calibration activations to compensate rounding error. Online block Hadamard rotations spread difficult input directions, but their placement and cost must be measured with the full graph. Mixed-bit plans allocate more precision to sensitive matrices; WikiText and in-domain chat/code sensitivity are not interchangeable. [Quantization notebook](../QUANTIZATION_NOTES.md).

## Accuracy is more than weight SNR

The early experiments found that MLP quantization dominated aggregate error, especially in later layers. Low-rank residual correction helped DeltaNet projections much more than the two-bit MLP. Block reconstruction keeps indices fixed and trains LUT entries/scales, optionally low-rank factors, against a reference stream with held-out selection.

These are candidate techniques, not universal prescriptions. The historical best export, bit plan, calibration rows, and reference trace are not included here. A generic all-two-bit MLP run does not reproduce the published mixed-bit measurements. Maintain distinct calibration and evaluation datasets and report tail errors and top-1 agreement alongside mean KL/perplexity.

## Numerical bugs can hide behind fluent output

The DeltaNet investigation isolated errors by feeding each chunk the correct reference input and exposing intermediate tensors. Cosine similarity alone missed magnitude errors; a fluent whole-model output did not establish correctness.

- Native fp16 softplus overflowed on the tested ANE path. Stable `relu(x) + log(1 + exp(-abs(x)))` avoids that expression's overflow.
- Native SiLU had substantial relative error for small activations. The retained recipe uses `0.5*x*(1+tanh(x/2))` in DeltaNet and MLP. Writing `x*sigmoid(x)` alone was fused back to native SiLU by the compiler.
- Small recurrent-state products encountered fp16 subnormal loss. Scaling q and v, with the corresponding squared factor in normalization epsilon, improved parity on the tested model.
- MLP down-projection input scaling was **rejected**: static and dynamic variants overflowed or corrupted outputs. Do not copy earlier notebook recommendations without reading the later results.

The imported notes report `ane7i` trace perplexity 2.426 against a same-weight PyTorch value of 2.428, and top-1 agreement 0.965 on a separate 2K-token row. These are different evaluations, not a single accuracy score. [Numerical investigation and final recipe](../ANE_DELTANET_NUMERICS.md).

## Graph compilation and state management

The runtime uses chunked execution, lazy DeltaNet updates, and host-managed KV buffers. A successful build or `CPU_AND_NE` setting does not establish ANE placement: the framework can fall back. Capture placement and check timing/memory behavior.

Core ML and Core AI use different lowering pipelines. The Core AI port replaced a chained inverse formulation with forward substitution, changed RMSNorm scaling, and retained KV writes on the host because particular in-graph update expressions lost placement. Large attention windows used blocked softmax with a global maximum and combined numerator/denominator; independently normalized blocks would be incorrect.

Core ML multifunction packages reduced disk duplication without achieving the same runtime weight sharing observed in Core AI. Later Core AI measurements used multiple verify/prefill/context entries in one package, a native Swift bridge with reusable buffers, and a compiler mode selecting a bonded variant. These techniques depend on SDK/runtime behavior and must be revalidated on supported public tooling. [Core AI notebook](../COREAI_PORT_NOTES.md), [multifunction memory investigation](../FEEDBACK_ANE_MULTIFUNCTION_MEMORY.md).

## Observations versus architectural hypotheses

Placement checks, numerical comparisons, and latency measurements support claims about the tested execution path. Patent diagrams and compiler strings help form hypotheses; they do not identify the exact physical implementation in a chip. In particular, a measured 256-value codebook boundary is not proof of a 256-byte SRAM. Preserve this distinction in public explanations.

Keep failed experiments when they isolate a useful constraint, but record the version, reproduction command, expected failure, and whether later work superseded it. Never promote a CPU fallback parity result into evidence of ANE numerical correctness.
