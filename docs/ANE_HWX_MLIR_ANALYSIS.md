# What the ANE compiler made: HWX and MLIR analysis for 8-bit and fused graphs

A guide for agents and developers changing quantized (INT8 / FP8) or fused graphs for the M6 Neural Engine through
Core AI: how to see the program the ANE actually runs, what to look for, and how to validate a change. Timing alone
misleads (zeros, wrong scales and failed fusions all change speed); the compiled program and real-data checks do not.
The measurements behind every rule here are in
[M6 compute acceleration, follow-up 5 October](research/M6_COMPUTE_ACCELERATION_2026-10-03.md#follow-up-5-october-compiled-ane-programs-and-8-bit-attention-measured).

## Requirements

- **SIP disabled.** aned keeps compiled programs in `/Library/Caches/com.apple.aned` (and `com.apple.aneuserd`).
  Both directories carry the `datavault` flag (`ls -ldO`), and data vaults are enforced by System Integrity
  Protection: with SIP enabled, only entitled system processes can open them, root included. The tools here were
  used with SIP disabled (`csrutil status`); disabling SIP lowers system security, so do this on a research machine
  only.
- **Admin rights once**, to add a read ACL for your user (the directories are owned by root / `_neuralengine`, mode
  700). After that the inspector runs without sudo, and new cache entries inherit the ACL.
- **Without either:** the MLIR level (section 3) needs neither, and isolated Core ML MIL kernels can be compiled to
  HWX directly with `mil_to_hwx` from the same repository (`-a h18g` for M6), which writes the HWX to its own
  output directory. That shows how the ANE compiler treats a construction, but not the program of a Core AI package.

## Quick start

```sh
# once: the HWX parser (macOS, Xcode command-line tools)
git clone https://github.com/freedomtan/coreml_to_ane_hwx && make -C coreml_to_ane_hwx/hwx_dump
export HWX_PARSING=$PWD/coreml_to_ane_hwx/hwx_dump/hwx_parsing
# once, SIP disabled: read access to aned's cache (root-owned data vault); only adds an inherited read ACL
sudo chmod -R +a "$USER allow list,search,read,readattr,readextattr,readsecurity,file_inherit,directory_inherit" \
    /Library/Caches/com.apple.aned /Library/Caches/com.apple.aneuserd

# a package must have been loaded (compiled) once on this macOS build, with the bonded mode the runtime uses
MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 <coreai venv>/bin/python -c "import sys; sys.path.insert(0,'coreai/swift_bridge'); \
    import coreai_bridge as B; B.Model('<package>.aimodel', compute='ane')"

python scripts/m6_hwx_inspect.py map <package>.aimodel              # where its compiled program is
python scripts/m6_hwx_inspect.py roles <package>.aimodel            # attention matmuls by role and operand format
python scripts/m6_hwx_inspect.py summary --ne-dims <package>.aimodel
python scripts/m6_hwx_inspect.py pseudo <package>.aimodel --stream 2 --first 390 --count 30
```

## 1. Where the compiled program is

A Core AI package (`.aimodel`) holds the program as MLIR bytecode (`main.mlirb`). The first load compiles it:
Core AI writes an MLIR graph with ANE regions to `~/Library/Caches/coreai-cache/<OS build>/<executable>/<main.hash
hex>/`, and aned compiles the ANE regions into an ANE program (HWX) in its own cache:

```
package main.hash -> coreai-cache/.../model.aimodelx/**/manifest.plist -> "ANERegionsHash" {h18g: "<a>_<b>"}
                  -> /Library/Caches/com.apple.aned/<OS build>/ModelAssetsCache/-_unsigned/<a>/<b>/model.hwx
```

`m6_hwx_inspect.py map` follows that chain. Requirements and pitfalls:

- **Compile mode.** The Core AI cache does not key on `MPSGRAPH_ANE_BONDED_COMPILE_MODE`. A package first loaded
  without mode 2 keeps that program for later mode-2 loads (one attention core: 2.2x the cycles). Always load with
  the mode the runtime uses (2 on M6); `python forge.py compile --force` rebuilds a build's cache.
- **Stale entries.** If aned's cache is cleaned (or the disk fills and aned writes empty entries), Core AI entries
  point at programs that no longer exist and loads fail with `nilError`. Move the dangling Core AI entry aside (do
  not delete blindly) and load again; the server retries target chunks this way but not the drafter.
- **Disk.** Research compiles fill the internal disk quickly (both caches together reached 325 GB in one day). Check
  `df -h /System/Volumes/Data` before long pipelines; archive Core AI entries of research packages to another disk.

## 2. Reading one task

An HWX holds one or more streams of tasks (a multi-function package compiled with bonded mode 2 has several,
including the bonded compiler's variants). A task is not an instruction: it is a register block configuring a fixed pipeline (DMA in, the NE
multiply-add array or the PE planar engine, on-chip L2 or DMA out). There is nothing to decompile; read the fields:

| Field (hwx_parsing) | Meaning | What to look for |
| --- | --- | --- |
| `MacCfg TaskType` | 0 = NE (multiply-add), 4 / 3 = elementwise with reduction, 5 / 6 = elementwise | which engine does the step |
| `InDim` / `OutDim ... Type=` (`Src2Type`) | operand and result formats: `float16`, `int8`, `uint8`, `e4m3` | 8-bit where intended |
| `KernelCfg Fmt=` / `Pal=` | the NE kernel operand format; `Pal=1(4bit)` is a hardware palette (our vector LUT, 2 bits per weight) | `int8` / `uint8` / `e4m3` kernels |
| `Src1DMAConfig` / `Src2DMAConfig En=1` | operand read from DRAM | DRAM reads of large tensors |
| `DstDMAConfig En=1` | result written to DRAM (`DstComp ... Lossy=1`: compressed write) | DRAM round trips of score-sized tensors |
| `L2_Result` without Dst DMA | result stays on chip (L2) | the cheap case |
| `MacCfg ... OutTrans=1` | output transposed | layout passes (key-tile transposes) |
| `PE Config Op=` / `PE PreScale` / `Pool=` | planar op (Add, Mul, None), a folded scale, a reduction (max, sum) | folded constants, reductions |
| `ExeCycles` | the compiler's static cycle estimate | relative cost only; not a measurement |

`m6_hwx_inspect.py` turns these into views:

- `summary` groups tasks by engine, task type, operand / kernel / result format and DRAM traffic, with static cycles;
  `--ne-dims` lists NE tasks by tensor shape (QK reads the 256-wide key tile, PV writes the score rows).
- `roles` classifies attention NE tasks (projections, history QK, history PV, key-tile transpose) with their formats
  and shares, plus PE elementwise and reductions. The first view to open for an attention change.
- `pseudo` prints one line per task: `NE4 y:int8[C48 H1 W2048]@dram = mac(x:float16[C256 H1 W2048]@l2, kern:fp16)`
  (NE with 4 engines, 8-bit result written to DRAM, FP16 input from L2, FP16 kernel operand). `T` marks an output
  transpose, `comp` a compressed write. `W0` appears where the parser reports no width; in these attention shapes it is the
  256-wide head dimension (inferred from the matching QK task).

## 3. The MLIR level

What the converter produced, before the ANE compiler:

```python
from coreai.authoring.asset import AIModelAsset
text = str(AIModelAsset.load("<package>.aimodel").program)   # coreai dialect, e.g. coreai.quantize ... -> ui8
```

Check here that every intended pair is present with the right dtypes (`si8`, `ui8`, `f8E4M3FN`), zero points and
scales, before blaming the compiler. The text parses back (`ir.Module.parse` under the Core AI context, then the
private `AIProgram._from_mlir_module`, then `save_asset`) and compiles to a byte-identical HWX, so the graph can be
edited or written by hand at this level (for example to try ops with no torch path, such as
`coreai.symmetric_quantization_statistics`). The MPSGraph `mps` stage in between cannot be printed with these bindings.

## 4. 8-bit rules (INT8, UINT8, FP8) and their HWX signatures

API: `torch.ops.coreai.quantize(x, scale, dtype, zero_point=None, minval=None, axis=0)` and
`torch.ops.coreai.dequantize(...)` (coreai-torch `_compression/custom_layers.py`; types int8, uint8, int4, uint4,
fp8_e4m3fn, fp8_e5m2; a vector scale with `axis` is per channel). Apple documents no fusion rules; these are measured.

- **Every operand of an 8-bit matmul is written as quantize, then dequantize**, FP8 as INT8, and an operand that
  arrives as INT8 (cache codes) too: `dequantize`, `quantize`, `dequantize` (with Core AI 0.4.2 the extra pair on an
  INT8 input compiles to the same program; write it anyway).
- **Constant weights** are compile-time INT8 constants (`constexpr_blockwise_shift_scale`, or the `quantize_weights`
  pass) feeding the conv / matmul directly.
- **Scales must fill the code range.** A step that is too coarse makes most codes zero: high error and a speedup
  that comes from zero skipping. Test data must be dense (no runs of zeros).
- **FP8 on M6:** keep FP8 operand values at or below about 100. With values near 200 (e4m3 holds 448) the device
  computed INT8 x FP8 PV wrongly for one head of a real layer; we scale FP8 operands to at most 64. FP8 needs a
  per-chip option (the M5 ANE has no FP8).

| Construction (pair on the FP16 operand) | HWX signature | Result |
| --- | --- | --- |
| no pair | NE `in=int8 kern=fp16` (INT8 read directly, no conversion task) | INT8 x FP16 |
| INT8 / UINT8, zero point 0, constant scale | NE `in=int8 kern=int8` / `kern=uint8`; producer task `out=int8` / `uint8` | 8-bit multiply-adds, 1-byte traffic |
| FP8 e4m3, constant scale | NE `kern=e4m3`, cycles equal to FP16; producer `out=e4m3` | 1-byte traffic, no multiply-add gain |
| zero point -128, or `minval` mode | explicit `EW in=int8 -> out=float16` tasks writing to DRAM, NE `fp16 x fp16` | no fusion; slower than no pair |
| runtime per-axis scale | no ANE program (region on the GPU, `E3B0C442...` empty-hash region) | rejected |
| FP8 e5m2 | compiles; the call crashes | unusable |

Fused passes: a pair right after an op fuses into that op's output (QK writes `int8` scores; the exp task writes
`uint8` / `e4m3` weights), and a pair before an op fuses into its input (a PE pass reads `int8` with the dequantize in
its scale). A pair separated from the producer by another op (for example a per-token scale multiply) only helps
that first read; place the pair on the tensor the expensive passes read.

## 5. Attention: what the program looks like

There is no softmax task. `torch.softmax` lowers to `coreai.softmax`, which compiles to the same chain as a
hand-written softmax: scale and mask (PE `Add` with `PreScale`), max and sum (reductions, `TaskType=4`, `Pool=2`),
exp (an NE task with an elementwise tile shape: the NE activation stage), division / combine (PE). The PE's own
nonlinear field (`NLMode`) stays 0. Per history tile of the production graph, several of these passes write the
score-sized tensor to DRAM in FP16. In the 8-bit form C2 (`ATT_INT8MM=s8,s8b,sm8,pvf8`) one verify tile reads:

```
EW  y:float16[C2048 H1 W0]@l2 T   = (None)(a:float16[...]@dram)                       # FP16 key-tile transpose (V8 keys)
NE4 y:int8[C48 H1 W2048]@dram     = mac(x:float16[C256 H1 W2048]@l2, kern:fp16)      # QK writes INT8 scores
PE  y:int8[C4 H24 W2048]@dram     = none+scale(a:int8[...]@dram, ...)                 # mask pass, INT8 in and out
PE  y:float16[C4 H4 W512]@dram    = none+scale reduce(a:int8[...]@dram, ...)          # tile max reads INT8
NE2 y:e4m3[C4 H24 W2048]@dram     = mac(x:float16[...]@l2, kern:fp16)                # exp writes FP8 weights
EW  y:float16[C4 H4 W512]@dram    = EW w/ Reduction ...(a:e4m3[...]@dram)             # softmax sum reads FP8
PE  y:e4m3[C4 H48 W1024]@dram     = add(a:e4m3[...]@dram, b:float16@l2)               # value-scale fold, FP8
NE4 y:float16[C48 H1 W0]@dram     = mac(x:int8[C2048 H1 W0]@dram, kern:e4m3)         # PV: INT8 values x FP8 weights
```

What to look for in an attention change:

- shares from `roles`: in the FP16 core prefill, softmax elementwise and reductions are about 62% of cycles, QK and
  PV 28%, key transposes 6% (19% of verify);
- which score-sized tensors still cross DRAM in FP16 (in C2: only small vectors and the FP16 key transpose);
- new layout passes an 8-bit form adds (a UINT8 / FP8 transpose of the weights before PV costs about 1,000 cycles at
  32K);
- whether a "8-bit" matmul is really 8-bit (`kern=int8` / `uint8` / `e4m3`), not silently FP16.

## 6. Validation ladder for a quantized or fused change

Each rung catches what the previous one cannot; do not skip the long-context rungs.

1. **Host simulation** of the exact graph with the 8-bit rounding (`scripts/m6_long_ctx_attn.py` for `kv8` cores;
   `scripts/m6_attn_core_check.py` for V8): expected error, zero / underflow fractions.
2. **Compiled program**: `m6_hwx_inspect.py roles` on a core package; the intended matmuls 8-bit, no unexpected
   FP16 conversion tasks.
3. **Core timing** on an idle machine, variants interleaved, two or more runs (any CPU job, including a host
   simulation, distorts ANE timings).
4. **Real data on the device**: capture real layer inputs (`scripts/m6_capture_attn_inputs.py`) and compare device
   against host simulation (`scripts/m6_attn_core_check.py --per-head`). Random data never produced the FP8 fault;
   real layer 63 did.
5. **Real-data accuracy over all layers** on the host (`scripts/m6_attn_logit_stats.py`): score ranges, per-layer
   error (read the layer mean, not the pooled figure, which layer 63 dominates).
6. **Chunk A/B** (`scripts/m6_chunk_ab.py`, `--visible`, `--unused-scale`): speed. Its random inputs and residual
   output hide attention errors; do not use it as a quality check.
7. **Model quality**: KL-512 (`scripts/m6_kl512_eval.py`) and the long-context evals
   (`scripts/m6_long_ctx_eval.py`): verify path `--prefill 64 --eval 4096`, `--prefill 7600 --eval 512`, and 64K.
   KL-512 sequences fit in one history tile; it rated a broken form lossless.
8. **Server benchmark** (`scripts/m6_compare_bench.py`) for the user-facing numbers.

## 7. Debugging a numerical problem

- **Find the position pattern**: per-position KL from two `m6_long_ctx_eval.py` runs (a jump at a tile boundary such
  as position 2,048 points at the history tiles).
- **Find the layer**: `scripts/m6_hybrid_build.py` builds that take some chunks from the candidate and the rest from
  the baseline (no recompile); halve the chunk set with the verify-path eval until one chunk carries the difference.
- **Find the operation**: capture that layer's real inputs, run `m6_attn_core_check.py` with each form alone and
  `--per-head`; a form whose device output departs from its host simulation is a device / compiler issue.
- **Probe the cause**: `--vscale` (value magnitude), builder switches such as `ATT_PF8_UNIT` (FP8 range); the FP8
  fault moved with the FP8 scale and not with the value scale.

## 8. Reference

Builder research switches (environment, recorded in each chunk's manifest numerics):

| Switch | Meaning |
| --- | --- |
| `ATT_INT8MM` | 8-bit attention forms, comma-separated. V8: `s8` (INT8 pair on the QK output), `s8b` (after the mask), `t8` (`s - tile max`), `sm8` (FP8 exp output and the softmax sum from it), `pvtu` / `pvf8` (UINT8 / FP8 PV weights per tile), `s8r` (scores relative to the block's row maximum). M6 form C2: `s8,s8b,sm8,pvf8` |
| `ATT_INT8MM_BY_LAYER` | per-layer override, `63:s8,s8b;59:` (empty: production attention) |
| `ATT_S8_UNIT`, `ATT_S8B_UNIT` | INT8 score steps (default 1/4) |
| `ATT_PF8_UNIT` | FP8 scale of `sm8` / `pvf8` (default 1/64) |
| `QCONV_INT8` | INT8 export weights as compile-time INT8 constants (default on) |
| `ATT_TILE_DEQUANT` | per-tile cache dequantize (no effect: the compiler reads INT8 tiles directly) |

Tools: `m6_hwx_inspect.py` (HWX views), `m6_attn_core_check.py` (device vs host on real inputs),
`m6_capture_attn_inputs.py` (real layer inputs), `m6_hybrid_build.py` (chunk bisection), `m6_long_ctx_attn.py`
(`kv8` cores, host simulation), `m6_attn_layer_int8.py` (a real attention layer, part by part),
`m6_softmax_probe.py` (softmax forms, one tile), `m6_attn_logit_stats.py` (real score ranges and simulated error),
`m6_long_ctx_eval.py`, `m6_kl512_eval.py`, `m6_chunk_ab.py`, `m6_compare_bench.py`.
