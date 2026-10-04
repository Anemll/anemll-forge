# Lessons recovered from the CoreML-fp8 research session

Curated from the owner's authorized local session covering September 23–29, 2026. This is a technical synthesis of historical session statements and local source inspection, **not a rerun of the experiments**. Raw conversation, private prompts, calibration sessions, tool outputs, and internal session identifiers are not included. The separate [201-run session analysis](PERFORMANCE_SESSION.md) describes a different reporting window.

## What the final design became

The session's final reported target is `mix25in_mixr_lr64mix`, a mixed-precision export: 38 MLP layers use nominal two-bit vector LUTs and 26 use scalar LUT4; early mixer layers 0–23 use vector LUTs and later mixers use LUT4, with rank-64 residual correction. Attention K/V projection weights use the INT8 export quantizer format; the Core AI runtime's KV cache is FP16. The head uses LUT4 and embeddings are on the host. This is not an all-two-bit model. Online Hadamard rotation supports MLP quantization. The recovered plan and export headers now confirm the allocation; see [the quantization guide](QUANTIZATION.md).

The reported mixer-to-MLP reallocation comparison changed in-domain mean KL from 0.1952 to 0.1852 and top-1 agreement from 84.6% to 86.1%, at 9.062 to 9.069 GiB. This evaluator uses teacher top-256 plus a combined tail bucket over prompt and answer positions, with a token-weighted mean in nats; it is not full-vocabulary KL. WikiText perplexity was essentially unchanged: 7.021 to 7.029. These workload-dependent results favor measuring the application's actual distribution rather than relying on a single proxy dataset. Without the residual factors, the new export's reported WikiText perplexity was 8.68.

A subsequent check of existing M3U results on September 29 confirmed mean KL **0.18519121 → 0.18550856 (+0.1714%)** when expanding the teacher partition from top-256 to top-512 on the identical 40,023-position trace. Top-1 agreement and trace perplexity were unchanged; median/p99 were unchanged only at the displayed precision. Both BF16 controls reported zero KL. See the [exact result and provenance record](results/kl_topk_comparison_2026-09-29.json) and [evaluation limitations](QUANTIZATION.md#interpret-kl-carefully). This check does not establish compiled Core AI quality or reproduce Mirai's protocol.

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

## Runtime memory: different problems need different mitigations

1. **Core ML function residency:** shared package weights did not imply one resident weight copy across loaded functions.
2. **Core AI Python outputs:** allocation/pooling failure was isolated to the Python binding in the recorded tests. A Swift bridge with preallocated output views later passed 5,000 small-model calls and a reported 2,000-call full-model run with flat memory. These are finite tests, not a universal leak-free guarantee.
3. **Compiled variants:** much of the Core AI program overhead was attributed to bonded/nonbonded variants. Mode 2 reduced it in the tested stack. The setting is undocumented and version-sensitive; the attempted analogous Core ML bonded-only mode caused timeouts and is not part of the supported recipe.
4. **Compiled-cache availability:** near-full-disk load failures were repaired by purging the affected package cache and recompiling. Purgeable ANE program-cache eviction was suspected. This load-time failure is distinct from repeated eviction or residency pressure during generation; neither the observed scheduling stalls nor system swap counters establish that diagnosis. The current [inference troubleshooting guide](SPECULATIVE_DECODING.md#eviction-memory-pressure-and-stalled-calls) explains the implemented retry and how to isolate slow calls.
5. **Context-switch buffer ownership:** the September 29 Terminal-Bench server using snapshot `89db7ac` was killed by macOS for low swap after 160 switches between 8K and 16K. Discarded KV allocations total about 120 GiB, consistent with the very large process footprint. Tiny tests isolated both a Python `Buffer → ndarray → Buffer` cycle and native IOSurface autorelease retention. `.np` now returns a view on access without caching it on its owner; `cai_buffer_create` and `cai_buffer_address` drain scoped autorelease pools without ending the buffer's lifetime. Updating only Python or pooling only creation was insufficient; rebuild the native bridge too. Four lifetime tests cover final release, surviving views/slices and repeated replacements. Sustained model-level transition validation remains a separate check. The earlier flat-memory call tests did not establish safety of repeated context resizing. See [diagnosis and deployment steps](SPECULATIVE_DECODING.md#eviction-memory-pressure-and-stalled-calls).

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

## Release-preparation diagnostic: cached GPU fallback (September 29)

A separate synthetic per-binding probe found a 5.7-second stall in the first target chunk at both 8K and 16K. Its cached specialization placed all 12 entries on GPU, although the requested compute was ANE, compile mode was `2`, the top-level cache field said no GPU adapter, and live driver logs confirmed bonded requests on both ANE units. Those requests verified the other active ANE work, not placement of every target chunk.

The bad cache's creation time matched a failed sandbox launch. The source package was unchanged. After preserving only that specialization and recompiling with permitted hardware access, the chunk's prefill took 22.7–24.7 ms and verification 6.0–7.7 ms; whole target verifier plans took 106–141 ms in the bounded synthetic probe. Stopping the local benchmark VM alone had not repaired the stall. These measurements diagnose placement and do not establish release throughput or quality. The evidence supports a sandbox-created GPU specialization being reused; it does not demonstrate general ANE eviction or a permanent hardware limitation.

The lesson is to combine per-binding timing with per-entry placement evidence (`mps.fullyPlacedOnANE`, `mps.noGPUActivity`, ANE/GPU region symbols). Preferred compute, cache mode, package metadata and bonded driver activity each answer a narrower question. Preserve and repair a verified affected cache, then recheck placement before timing; see [the troubleshooting procedure](SPECULATIVE_DECODING.md#cached-gpu-placement-can-look-like-an-ane-stall).

## Release-preparation diagnostic: M5/M5 Max cold-compile crash, bonded vs non-bonded (September 29)

A first end-to-end download-and-serve pass on an **Apple M5** (not the M3/T6031 the vector-LUT/GOC-width probe above targets) surfaced a distinct failure from a cold `~/Library/Caches/coreai-cache`: `forge.py quick-test --ctx 16384` and `forge.py serve`, which load all 16 four-layer chunks plus head plus the DFlash2 drafter inside `CoreAIQwenBridge.__init__`, repeatedly hit `Error Domain=com.apple.appleneuralengine.compiler ... "_ANECompiler : ANECCompile() FAILED" ... "Compiler internal error: Couldn't do topological sort"` on most chunks' first compile attempt. The single-op capability probe above (`ane_caps_smoke.py`) is unaffected and in fact shows this M5 accepting vector-LUT and >16384-channel GOC cases the M3/T6031 probe rejected — a genuinely different capability profile, not a regression.

Switching `MPSGRAPH_ANE_BONDED_COMPILE_MODE` from the code's default `2` (bonded) to `0` (non-bonded) did **not** avoid the crash: a full cold run in mode 0 still produced 32 topological-sort failures and, notably, consumed the ~45 GB of freed disk faster than mode 2 before a fatal `LLVM ERROR: IO failure on output stream: No space left on device` ended it. Compile mode is not the variable that matters here; do not expect a bonded/non-bonded switch to fix this class of crash on this chip.

Isolated reproduction attempts all succeeded with zero errors, in mode 2: a single chunk package alone (smallest and largest chunk tried), the head package alone, the drafter package alone, all 16 chunks + head loaded sequentially in one process, and — critically — all 16 chunks + head compiled cold *and kept resident simultaneously* (not garbage-collected between loads, matching what `CoreAIQwenBridge` actually does). None of these manual repros reproduced the crash, so holding many compiled programs resident and compiling many packages sequentially are each independently ruled out as the trigger. The crash was only ever observed inside the real `quick-test`/`serve` invocation path; the exact additional condition (buffer/KV-cache allocation interleaved with per-chunk loading in that class's `__init__`, or an intermittent compiler condition) is not established.

**Working mitigation**: pre-warm the compile cache by loading each package individually first — e.g. `coreai/.venv/bin/python coreai/qwen38_coreai_greedy.py probe <package.aimodel>` once per chunk/head/drafter file — before running `quick-test`/`serve`. Every isolated load this way succeeded; once cached, the real pipeline then loads from cache (sub-second per chunk) and runs correctly. Confirmed working after pre-warming: `quick-test --ctx 16384` speculative generation (drafter engaged, coherent output, finite logits) and `forge.py serve` answering a real chat completion through the OpenAI-compatible API.

Disk discipline matters here: a single top-to-bottom cold compile of all 16 chunks + head used roughly 11–14 GB of `~/Library/Caches/coreai-cache`; a crash-looping run left unattended can exhaust disk outright (`LLVM ERROR: ... No space left on device` was observed directly). Repeatedly interrupting a compiling run mid-flight (e.g. via a wrapper's process timeout) wastes the partial compile and does not reliably avoid the disk cost of the next attempt. Clear `coreai-cache` before retrying after a disk-exhaustion crash.

**Update (2026-10-04):** `MPSGRAPH_ANE_BONDED_COMPILE_MODE=1` made the real `serve` path compile and start on an Apple M5 base, so the bonded mode *is* generation-specific after all; the September 29 mode-`0`/`2` failures above are preserved as that record. The runtime now selects the mode by SoC generation — `1` on the M5 family (H17), `2` on M6 and newer (H18+), fail on pre-M5 — through one helper (`scripts/ane_compile_mode.py`) used by `forge.py serve`/`compile`, `coreai_compile.py` and the Core AI runtime. An explicit `MPSGRAPH_ANE_BONDED_COMPILE_MODE` still wins. See [ANE compile mode policy](ANE_COMPILE_MODE_POLICY.md). M5 Pro/Max are a maintainer directive, not yet reproduced in-tree.

## Compute acceleration on M6: what the compiler taught (October 3)

- **Measure the split before choosing a kernel.** A byte model put KV traffic at 23% of a 64K verify; timing the context ladder showed attention at 47%, because the history path was op bound (about 36 GB/s of K/V), not bandwidth bound. INT8 or FP8 MACs would not have touched the dominant cost.
- **Op count is the cost for small graphs.** The DeltaNet core had little arithmetic but cost as much as an MLP layer at 64 rows. Replacing 7 serial row updates with a closed-form inverse halved it.
- **Matmul chains between computed tensors fail ANEC.** The compiler aborts with an internal error and the program falls back to the GPU. The same products written as broadcast multiply plus reduce compile and stay on the ANE.
- **A cost model is a hint.** anemll-profile's Core ML cost model ranked transposes first; removing them in Core AI saved 3% at 64 rows and slowed 8 rows.
- **Compile time grows faster than the program.** Narrow attention tiles made a chunk compile 5 times longer; three quarters of it came from the 64-row prefill entries, which gained nothing below 4096-wide tiles. Writing the tiles as one batched op did not help: the compiler expands it into the same per-tile work. The same tiles compiled in a standalone program cost about one fifteenth as much as inside a chunk.
- **There are two compile caches.** Deleting a package's Core AI cache entry redoes only the MPSGraph stage; the system ANE service also keeps compiled programs it has seen. Time true cold compiles with packages that are new to the system (for example a negligible constant change).
- **Guide long first starts.** A cold compile is now announced with an estimate, per-package progress and time left, and can be interrupted safely; see [M6 compute acceleration](research/M6_COMPUTE_ACCELERATION_2026-10-03.md).

## Current release interpretation

DFlash2 is a required part of the intended fast Core AI release, not an optional side experiment. Preserve the historical findings above, including unsuccessful recalibration and stale-head checks, while using [the current pairing guide](SPECULATIVE_DECODING.md) for release assets and flags. The earlier observations are not new validation of the assembled download.
