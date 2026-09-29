# Lessons recovered from the CoreML-fp8 research session

Curated from the owner's authorized local session covering September 23–29, 2026. This is a technical synthesis of historical session statements and local source inspection, **not a rerun of the experiments**. Raw conversation, private prompts, calibration sessions, tool outputs, and internal session identifiers are not included. The separate [201-run session analysis](PERFORMANCE_SESSION.md) describes a different reporting window.

## What the final design became

The session's final reported target is `mix25in_mixr_lr64mix`, a mixed-precision export: 38 MLP layers use nominal two-bit vector LUTs and 26 use scalar LUT4; early mixer layers 0–23 use vector LUTs and later mixers use LUT4, with rank-64 residual correction. Attention K/V stays INT8, the head LUT4, embeddings on the host. This is not an all-two-bit model. Online Hadamard rotation supports MLP quantization.

The reported mixer-to-MLP reallocation comparison changed in-domain mean KL from 0.1952 to 0.1852 and top-1 agreement from 84.6% to 86.1%, at 9.062 to 9.069 GiB. WikiText perplexity was essentially unchanged: 7.021 to 7.029. These workload-dependent results favor measuring the application's actual distribution rather than relying on a single proxy dataset. Without the residual factors, the new export's reported WikiText perplexity was 8.68.

The deployed factors were a plain SVD of the weight error, not proof of deployment of every activation-weighted or dynamic-rank experiment now in the repository. The original 48×1024 calibration rows mixed WikiText, self-generated chat and private coding-agent data. Replacing those rows creates a new reproducible public experiment; it does not recreate the exact old export.

See [archived pipeline sequence](../pipelines/m3u/README.md), [quantization notes](../QUANTIZATION_NOTES.md), and the explicit-path [workflow](WORKFLOW.md).

## Compiler behavior is not a hardware specification

The FP8 work predates and is distinct from the main mixed-LUT Qwen recipe. The local coremltools changes span MIL FP8 types and serialization, C++ blob storage, newer quantize/dequantize operations, optimization APIs, tests and examples. An experimental native-FP8 compilation workaround compiles an int8-labeled intermediate and restores FP8 metadata. That workaround is not equivalent to stock Xcode accepting the original graph.

The session distinguishes native FP8 dense weights, FP8-valued LUT tables and low-bit LUT indices. Reported numerical range differences between Core ML and Core AI must stay attached to the representation and route tested. FP8 external array binding was another separate limitation; internal graph support does not imply runtime I/O support.

An early attribution of FP8 ResNet slowdown to an unavoidable M6 hardware limitation was superseded by compiler-input replay: MIL and ANECIR routes produced different timings for the test. This localizes the observed difference to the compilation route, while proprietary lowering details remain unknown. Those ResNet numbers should not be used as Qwen speed predictions.

Vector-LUT placement and timing are consistent with compressed execution, but latency slopes and compiler strings do not identify exact physical decode circuitry. Preserve the experimentally observed 256-value boundary and equal-index-width controls; label decode-path and padding explanations as inferences unless supported by additional evidence.

## Numerics: isolate the first wrong tensor

Fluent output and good final cosine concealed large recurrent-layer errors. Stable softplus, tanh-form SiLU, and the matched DeltaNet scaling/epsilon changes produced the largest documented runtime-quality recovery. Promising early-layer MLP down-input scaling failed on later layers and was dropped. An obvious algebraic replacement can also be optimized back into the problematic native operation.

The final Core ML recipe and the rejected variants are preserved in [ANE_DELTANET_NUMERICS.md](../ANE_DELTANET_NUMERICS.md). The early statement that MLP SiLU was unaffected was later superseded by the MLP measurements. Tiny graphs running on CPU are not evidence that the same operations behave correctly on ANE.

## Runtime memory: three different problems

1. **Core ML function residency:** shared package weights did not imply one resident weight copy across loaded functions.
2. **Core AI Python outputs:** allocation/pooling failure was isolated to the Python binding in the recorded tests. A Swift bridge with preallocated output views later passed 5,000 small-model calls and a reported 2,000-call full-model run with flat memory. These are finite tests, not a universal leak-free guarantee.
3. **Compiled variants:** much of the Core AI program overhead was attributed to bonded/nonbonded variants. Mode 2 reduced it in the tested stack. The setting is undocumented and version-sensitive; the attempted analogous Core ML bonded-only mode caused timeouts and is not part of the supported recipe.

Report source-package bytes, compiled program bytes, scratch, KV storage and system wired memory separately. A small package can require a much larger runtime allocation. Larger Qwen Core AI chunks also failed despite working Core ML counterparts; that is evidence about a particular graph/toolchain, not a proven universal maximum number of attention layers or ANE weight bytes.

## Context scaling: final measurements supersede projections

The later Core AI design uses 16 four-layer chunks, verify T=8 and prefill T=64, with shared-weight entries at 8K, 16K, 24K, 32K, 48K and 64K. Eight candidate rows do not guarantee eight emitted tokens: the drafter proposes seven candidates next to an anchor and acceptance determines output per cycle.

On the reported 512-token coding-prompt runs, sampled rates across those six contexts were 32.3, 30.0, 25.5, 23.8, 20.6 and 18.3 tokens/s. Measured greedy endpoints were 35.2 and 20.8 tokens/s; some intermediate greedy values in the conversation were estimates. At 64K, measured target prefill was 80 tokens/s, below the earlier 97-token/s projection.

Reported target-plus-drafter wired memory above idle was 19.3 GB at 8K and 22.6 GB at 64K, with about 25.4 GB total system wired at 64K. The usable 64K entry holds 65,472 positions, with further headroom required for speculative writes. Growing context reportedly matched pinned-context greedy output; one 32.6K-token prompt prefills in 207 seconds growing versus 327 seconds pinned to 48K.

The 8K/64K perplexities 2.1986/2.1983 covered only eight sequences. They must not be compared directly with the earlier full-trace 2.4148 score or described as a full 64K quality evaluation. Short live requests reaching higher peak rates are also different workloads from the sustained 201-run serving summary.

## Artifact provenance can beat an apparently correct reference

The suspected drafter-head numerical bug was a stale target-head mismatch. The reference reused the same stale default and therefore appeared to validate the incorrect artifact. Record target export and head identity in drafter metadata and compare them at load time. Equal outputs are insufficient if both paths consume the wrong weights.

The final reported drafter retained earlier calibration, used the mixr target head and mask scale 0.7; a later recalibration candidate was not deployed after a reported marginal gain. The archive includes the experiment, not evidence that every output was used in production.

## Serving policy changes generation too

Aggressive DRY penalties suppressed legitimate repeated code syntax; the default was changed to off. This supersedes any blanket explanation that all repetition/coding corruption came solely from quantization. Earlier speculative/plain comparisons also included a wrong-build selection mistake. Preserve these experimental confounders.

A 3 ms gap after target verification removed observed drafter stalls in a small interleaved A/B test; rare verifier stalls remained. The server implements this mitigation, but its mechanism and portability are not established.

Thinking budgets were later enforced by the server to reserve answer space. A reported 128-request window contained 122 tool-call completions, six normal stops, no truncations/errors/loop-guard stops and eight budget closures. That is a separate window from the pasted 201-run analysis. Forced thinking closure is an explicit generation policy, not transparent model behavior.

## Release consequences

- Pin the checkpoint, target/head/drafter identities, calibration recipe, compiler source and patches, and runtime settings.
- Make same-weight stage parity, ANE placement, long-run memory and context transitions release tests.
- Retain failed experiments with dates and supersession notes; distinguish measured results, projections and hypotheses.
- Keep optional FP8 research tooling separate from the requirements for serving prebuilt mixed-LUT artifacts.
- Recover raw benchmark records before advertising hardware ceilings or resolving the 10.61 GB verifier-byte discrepancy. This session does not resolve that discrepancy.
