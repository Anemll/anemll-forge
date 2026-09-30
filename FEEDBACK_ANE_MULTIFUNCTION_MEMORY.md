> Historical research notes imported on 2026-09-29. Paths were generalized; results were not rerun. Later sections may supersede earlier findings. See the current [release workflow](docs/WORKFLOW.md).

# Feedback draft: multifunction Core ML models duplicate weights in ANE memory

Status: reproduced on M6 / macOS 27 (Xcode 27.2), 2026-09-26. To file with Apple (Feedback Assistant, Core ML).

## Summary

The functions of a multifunction ML Program share one weight blob on disk (`ct.utils.save_multifunction`
deduplicates), but when two functions are loaded for the Neural Engine, each one wires its own copy of the weights.
This happens both with one `MLModelAsset` shared by both loads and with two `MLModel(contentsOf:)` loads. For an
LLM that needs several shapes of the same weights (prompt vs decode rows, or several context lengths), resident
memory is N times the weights. That rules out keeping more than one shape of a large model loaded.

## Minimal repro

Model: one decoder chunk of Qwen3.8-27B (4 layers, palettized LUT weights, 0.69 GB on disk), two functions that
differ only in the KV-cache input length (`ctx2048`, `ctx8192`; identical weights, deduplicated).

- Build: `scripts/qwen38_mf_share_test.py` (writes `mftest/chunk_L00-03_mf_2048_8192.mlmodelc`).
- Load test: `scripts/mf_asset_share.swift`:
  ```
  # From the Forge root; ANE_OUT is your prepared Core ML build directory.
  ANE_OUT="$ANE_OUT" CTXS=2048,8192 python scripts/qwen38_mf_share_test.py
  swiftc -O -parse-as-library scripts/mf_asset_share.swift -o /tmp/forge-mf-asset-share
  /tmp/forge-mf-asset-share "$ANE_OUT/mftest/chunk_L00-03_mf_2048_8192.mlmodelc" ctx2048 ctx8192 asset
  /tmp/forge-mf-asset-share "$ANE_OUT/mftest/chunk_L00-03_mf_2048_8192.mlmodelc" ctx2048 ctx8192 separate
  ```
  Wired memory is read with `host_statistics64` (`wire_count`) after each load and first prediction;
  compute units `.cpuAndNeuralEngine`.

## Measured (wired memory added)

| file / load path | after load of fn 1 | after load of fn 2 | after first prediction of both |
| --- | ---: | ---: | ---: |
| multifunction (0.69 GB), one `MLModelAsset` | +1.05 GB | +2.09 GB | +2.09 GB |
| multifunction (0.69 GB), `MLModel(contentsOf:)` x 2 | +0.65 GB | +1.68 GB | +1.77 GB |
| multifunction (0.69 GB), coremltools `CompiledMLModel` x 2 | +1.31 GB | +2.11 GB | +2.19 GB |
| two single-function files (0.69 GB each), coremltools | +0.79 GB | +1.83 GB | +1.92 GB |

Expected: the second function of the same weight blob adds only its own program (small), not another ~0.8-1.0 GB.

### Same model through Core ML and Core AI (M6, macOS 27)

Toy: 16 x Conv2d(4096, 4096) FP16 (537 MB) + attention over K / V inputs. Four entry points over the same weights:
`s2k`, `s8k`, `s16k` (8 rows, K/V length 2048 / 8192 / 16384) and `p64_s2k` (64 rows). Both packages are 537 MB on
disk. Both run fully on the ANE (Core AI manifest: `mps.fullyPlacedOnANE`, `mps.noGPUActivity`).

| entries loaded | Core ML multifunction (`.mlmodelc`) | Core AI entry points (`.aimodel`) |
| --- | ---: | ---: |
| s2k | +0.54 GB | +0.50 GB |
| + s8k | +1.00 GB | +0.50 GB |
| + s16k | +1.55 GB | +0.50 GB |
| + p64_s2k | +1.75 GB | +0.50 GB |
| call time s2k / s8k / s16k / p64 | 5.10 / 6.99 / 7.64 / 5.16 ms | 4.77 / 4.93 / 4.99 / 4.84 ms |

Core AI keeps one resident copy of the weights for all entry points; Core ML wires another copy for every function.
(Core ML timings include numpy input copies of the K/V inputs.)
Scripts: `scripts/coreml_entry_share.py`, `coreai/probes/coreai_entry_share.py ladder`.

Confirmed with a real layer of the target (Qwen3.8-27B layer 3: gated attention over KV-cache inputs + 4-bit LUT MLP,
208 MB): one `.aimodel` with entry points for KV length 2K / 8K / 16K: using all three instead of one adds 0.09 GB
(one weight copy would be 0.2 GB), all fully on the ANE, outputs identical to the torch reference (cos 1.0000) at
every length. And a full 4-layer chunk (3 Gated DeltaNet + 1 attention, 758 MB, entries for KV 2K and 8K): first
entry +1.14 GB wired, second entry +0.04 GB; outputs match the Core ML chunk (y cos 0.993-0.997).

Apple's own on-device model does the same as Core AI: the AFM draft model's package ships one ANE binary
(`MPSGraph/mpsExecutable.mpsgraphpackage/binary_0.hwx`, 263 MB for a ~300M-parameter model) for all 17 of its
`extend_{context}_{rows}` functions (context 64-4096, 8 or 64 rows), i.e. one weight copy.

Full model: one set of 16 such chunks (11 GB on disk) wires ~20 GB at 64K context; a second set (another context
length or a prompt-length function) does not fit in 32 GB, and alternating between two loaded sets stalls
(hundreds of ms per call, probably eviction). Reloading a set takes 60-80 s even when the compiled programs are cached.

## Questions for Apple

- Core AI entry points and the AFM package share one weight copy on the ANE; Core ML multifunction models do not.
  Can Core ML multifunction models (same MIL, same deduplicated weight blob) get the same single-binary /
  shared-weight compilation on the ANE?
- Do enumerated input shapes in one Core ML function compile to one ANE program with shared weights?
- Until then, is Core AI the recommended path for LLMs that need several shapes (prefill / decode, context
  lengths) on the ANE?


## Second issue: Core AI runtime output pool grows until a fatal allocation error (2026-09-27, M6, macOS 27)

Core AI entry points DO share weights (above), which makes Core AI the only way to serve several shapes of one LLM on
the ANE. But a long-running Core AI inference loop dies: every call allocates new output NDArrays (12 per chunk
call for our model), the runtime's output pool keeps growing (per-chunk call time 8.7 -> 12.9 ms, wired memory
+0.5 GB over 600 calls), and after ~850 function calls the process aborts:

    NDArray+Pool.swift:77: Fatal error: Failed to allocate storage for NDArray

- Not Python object lifetime: gc every call with a frozen heap changes nothing (`coreai_leak_probe.py`).
- No API to supply output buffers (Core ML has `outputBackings`), so a decoder that calls 17 programs per token
  (~50 tokens here) cannot run.
- Repro: `coreai/probes/coreai_leak_probe.py` (one chunk, loop of calls, prints wired memory and
  per-call time until the fatal error). Model: Qwen3.8-27B chunk (4 layers) exported with coreai-torch 0.4.2,
  coreai-core 1.0.0b2 runtime.

Questions: is output-buffer reuse (caller-provided outputs, or a pool that recycles released outputs) planned for the
Core AI runtime? Is there a supported pattern for autoregressive loops with many calls?


## Third issue: ANE silu accuracy near zero and fp16 subnormals (2026-09-27, M6, macOS 27)

The same compiled ML Program gives different results on CPU_ONLY and CPU_AND_NE. Found in the Gated DeltaNet layers
of Qwen3.8-27B, where the ANE output was ~50% wrong while the model still produced fluent text.

- `silu` on the ANE has ~1e-3 **absolute** error for inputs near 0 (measured on an (8, 10240) fp16 tensor with >99% of
  its values in [-0.5, 0.5]: mean |error| 0.0011 against mean |silu(x)| 0.008-0.01, i.e. ~12% relative; the CPU is
  exact to 0.08%). `x * sigmoid(x)` cannot be used as a workaround because `mil_backend::fuse_activation_silu` turns
  it back into `silu`; `0.5 x (1 + tanh(x / 2))` is accurate (0.1%).
- fp16 subnormal values (< 6.1e-5) lose their precision on the ANE: a matmul output with 61% subnormal values was
  20% off (0.25% after scaling the inputs into the normal range).
- Repro: one-layer chunk with extra outputs, `UNITS=CPU_ONLY` vs `CPU_AND_NE` (scripts:
  `qwen38_ane_capture.py`, `DBG_MIXER_IN=1 DBG_TAPS=gdn` builds); details in ANE_DELTANET_NUMERICS.md.

Questions: is the ANE silu (and other activation) accuracy near 0 documented? Are subnormals flushed to zero on the
ANE? Could the silu fusion pass keep an accurate form when the target is the ANE?

## Fourth issue: kernel panic in the ANE driver after a large Core AI program fails (2026-09-27 23:00, M6, 26A428)

- A 12-layer Qwen3.8 chunk (1.67 GB, Core AI entry points v8_24k and p64_24k, coreai-torch 0.4.2 source package)
  specialized on load, but every call failed with `MPSGraphDelegateError.ndxRuntimeError("inferValue failed: Error, no
  procedureInfo entry found corresponding to the entry function: v8_24k_...")`, reproducibly.
- Loading an 8-layer chunk (1.1 GB, same entry points) in a new process right after panicked the kernel:
  `panic(cpu 7 ...): Break 0x0800 instruction exception from kernel. Panic (by design)`, panicked task `aned`,
  backtrace in `com.apple.driver.AppleH16ANEInterface(10.19.2)`. Earlier the same day a watchdog reset happened during a
  Core AI compile of 32K / 64K attention entries.
- 4-layer chunks (0.55-0.72 GB) of the same model with the same entry points run fine (thousands of calls).
Questions: is there a documented size / complexity limit for a Core AI (MPSGraph) ANE program? A user-space program
should fail with an error, not panic the kernel.
Also: Xcode-beta's coreai-build writes MPSGraph package 7.1.2 `.aimodelc`; macOS 27.0 (26A428) reads up to 7.0.80 and
the load segfaults the process instead of failing with a version error.
- Reproduced 2026-09-28 00:43 on a freshly booted M6: loading an 8-layer chunk (1.1 GB, a single entry point v8_24k,
  nothing else loaded, no earlier failure) panicked the kernel again. 4-layer chunks (0.52-0.72 GB) of the same model
  and entry points are stable.
