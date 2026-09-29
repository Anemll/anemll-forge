> Historical research notes imported on 2026-09-29. Paths were generalized; results were not rerun. Later sections may supersede earlier findings. See the current [release workflow](docs/WORKFLOW.md).

# Core AI port notes (Qwen3.8-27B target on the M6 ANE)

Why: Core ML multifunction models wire one weight copy per loaded function on the ANE
(`FEEDBACK_ANE_MULTIFUNCTION_MEMORY.md`), so prefill-64 + verify-8 + a context ladder cannot co-reside for a 27B
model. Core AI entry points share one weight copy (same ODIX + MPSGraph machinery as Apple's AFM package).

## Measured (M6, macOS 27, 2026-09-26)

- 537 MB FP16 toy, 4 entry points (8 rows at KV 2K / 8K / 16K, 64 rows at 2K) in one `.aimodel`: wired +0.50 GB
  total (Core ML multifunction: +1.75 GB), all `mps.fullyPlacedOnANE`, 4.8-5.0 ms per call; 64 rows cost the same as
  8. Compiled cache: one `resources.bin` (537 MB) + one ANE region per entry point.
  (`fp8-mlp-metal41-bench/coreai/coreai_entry_share.py [ladder]`, `ane-vector-lut/scripts/coreml_entry_share.py`)
- Weight formats: FP16 LUT values, scalar and vector (`cluster_dim` 2-16) palettization run on the ANE through
  coreai-opt. No FP8 / INT8 LUT values. INT8 / FP8 per-channel weights (non-LUT) do export.

### Real layers of the target (2026-09-27; `coreai/probes/coreai_chunk_port.py`, `fp8-mlp-metal41-bench/coreai/coreai_attn_entries.py`)

Port = torch mirror of the v4 chunk (lazy-commit DeltaNet with host-owned conv / rec / pend buffers, KV as read-only
inputs + mask (1, CTX), k/v rows out), exact exported weights: `coreai_chunk_ref.py dump` (vq27b venv) writes the
weights and the Core ML chunk's I/O for 3 calls (cold, commit 8, commit 3); palettization injects the exported LUT /
indices exactly (vector 2x16 MLP, scalar LUT4 mixers, Hadamard +-1/32 as UNIQUE); per-channel scales are a mul after
the conv. Palettize BEFORE `prog.optimize()` (optimize folds the scale mul into W and breaks the per-tensor LUT).

| piece | placement | parity | wired | call |
| --- | --- | --- | --- | --- |
| layer 0: DeltaNet + MLP (4-bit) | fully ANE, 1 region | vs torch fp32: conv 1.00000, rec 1.00000, pend 0.9998, y 0.9988 | +0.30 GB (198 MB pkg) | 2.3 ms |
| layer 3: attention + MLP, entries 2K / 8K / 16K | fully ANE, 3 regions, one 208 MB resources.bin | vs torch: y / k_new / v_new 0.99999-1.00000, norm 1.000, every entry | 2 extra entries: +0.09 GB total (a weight copy = 0.2 GB) | 1.9 / 2.4 / 3.3 ms |
| **full chunk L0-3, entries 2K + 8K** | fully ANE, 2 regions, 758 MB pkg (Core ML v4: 702 MB) | vs Core ML v4: y 0.993-0.997, conv / rec 1.0000; vs torch fp32: y 0.988-0.994 (the ANE fp16 floor; Core ML is as close) | cold 1st entry +1.14 GB, 2nd entry +0.04 GB | 12.1 / 12.4 ms vs Core ML 11.3 / 11.7 ms back to back under the same load (+6%); near-quiet 7.4 / 7.8 vs 5.75 / 6.3 ms (different times) |

Other measurements (M6, 2026-09-27):
- Per-call overhead (slope of 1 vs 8 chained 1x1 convs, 8 rows, both on the ANE; `overhead_slope.py`): Core ML
  0.13 ms, Core AI 0.22 ms (Core AI contended by an A/B). A bare mul / add graph runs on the CPU in Core ML.
- Big inputs: a 128 MB KV-like input adds nothing per call in either framework (mapped, not copied).
- Program size, one entry point, FP16 conv chain: 1 GB 93 GB/s, 2 GB 139 GB/s, 3 GB 158 GB/s, all fully on the
  ANE; 4 GB compiled (a 4 GB cached program) but `load_function` failed with the generic error below - unresolved.
- **Reliability:** 10-20 min after compiling, cached Core AI models failed `load_function` with
  `Foundation._GenericObjCError error 0` while the disk was 99% full (11-16 GB free); moving the model's
  `~/Library/Caches/coreai-cache/<ver>/python/<hash>` entry aside forces a recompile (19 s) and it works again.
  Likely the purgeable ANE program cache is evicted under disk pressure and Core AI does not recompile on its own.
  A runtime needs a retry-with-cache-purge path (and free disk).

Compiler differences found (MIL/E5RT graph -> Core AI MLIR/MPSGraph), each fixed in the port:
1. **Doubling inverse `(I + N)^-1 = (I - N)(I + N^2)(I + N^4)`** (chained matmuls of computed 8x8 matrices; on MIL
   written as `I - (I - N)(I + N)` for the -14 rule): MPSGraph puts NO ANE region in the whole graph (layer 0 ran at
   131 ms off-ANE). A single dynamic 8x8 batched matmul is placed fine. Fix: forward substitution over the 8 rows
   (elementwise + reduce, solves `u` and `wk` together): fully on the ANE (1.6 ms standalone).
2. **RMSNorm with a /64 pre-scale** (`rms_hidden`, against fp16 overflow of massive activations): at layer 0 the
   embeddings are tiny (rms 0.015), (x/64)^2 ~ 5e-8 is fp16-subnormal, and Core AI's ANE lowering mis-normalizes
   (qkv 0.70x, per-token scalar, cos 1.0 - invisible to cosine checks). Core ML's ANE path handles it. Fix: scale-free
   RMSNorm xs = x / max|x| (squares in [0, 1]); after it conv / rec match to 1.0000 in cos AND norm.
   Lesson: check norms / magnitudes, not only cosines.
3. `index_copy_` KV writes lower to `scatter_nd` -> the graph goes to the GPU (15 ms). Keep the v4 design (KV read-only
   inputs, host commits rows) - big inputs are not copied per call anyway.

**Found in the production Core ML model (not a Core AI issue):** the ANE's fp16 `softplus` returns 0 for inputs
above ~11 (exp overflow). The DeltaNet gate g = softplus(a + dt) * -exp(A_log) is therefore 0 (no decay) instead of
up to -7.5 per token for heads with large a + dt: in layer 0, 74-81 of 384 (head, token) values (~20%) have
a+dt >= 11.2 and ALL of them are g = 0 on the ANE, spread over 15-16 of the 48 heads (1, 6, 8, 11, 18, 19, 20, 30,
31, 32, 37, 38, 45, 46, 47, ...); 10 <= a+dt < 11 is partly affected, < 10 is exact. Those heads should nearly
reset their state every token and instead never decay. Fix: softplus(x) = relu(x) + log(1 + exp(-|x|)) in
qwen38_ane_chunk.py (all DeltaNet paths). The M3U KL numbers (PyTorch) do not include this error.

## Full-target port (2026-09-27, in progress)

Files:
- `coreai/qwen38_coreai_build.py`: build straight from the checkpoint + export (no npz dumps).
  Weights come from `qwen38_ane_model.Checkpoint` / `layer_quant`; `sklearn` is stubbed because the Core AI venv
  lacks it and loading needs no k-means. One `.aimodel` per chunk with entries `v8_<ctx>k` (verify, lazy commit)
  and `p64_<ctx>k` (64-row prefill: all rows committed, padding masked by `valid`, conv rows returned in the
  P-row layout, zero pending rows, like the MIL `gdn_lazy_prefill_block`). Also `head_T8.aimodel` and
  `manifest.json`. Output: `~/Models/vq27b/coreai/<export>/`.
  ```
  cd ane-vector-lut/coreai && unset USE_LOCAL_COREAI
  .venv/bin/python qwen38_coreai_build.py chunk 0-3 --ctx 2048,8192,16384,32768,65536 --pctx 2048
  .venv/bin/python qwen38_coreai_build.py all --ctx 2048,8192,16384,32768,65536 --pctx 2048   # 16 x 4 layers + head
  ```
- Layout (moved from fp8-mlp-metal41-bench/coreai on 2026-09-28): `coreai/` holds the builder, `qwen38_coreai_batch.py` (parallel chunk builds, e.g. on the M3U), `qwen38_coreai_greedy.py` (size-limited chunk plan + single-chunk ANE load test: `probe <package>`), `coreai_util.py`; `coreai/swift_bridge/` the Swift runtime bridge (libcoreai_bridge.dylib, coreai_bridge.py) and its validation / timing tools; `coreai/probes/` one-off Qwen Core AI probes. Build venv: `coreai/.venv` (not in git), made with `uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python -r requirements_build.txt`. The serving runtime stays in `scripts/` (`qwen38_coreai_model.py`, venv ~/venvs/vq27b-coreai).
- `qwen38_coreai_stage1.py entries|inplace|size8`: entry timing, prefill-64 vs 8 x verify-8 parity, in-place KV
  check on a real chunk.
- `qwen38_coreai_bisect.py "<layers>:<entries>" ...`: builds small packages and reports ANE placement, then
  deletes each package and its cache entry.
- `ane-vector-lut/scripts/qwen38_coreai_model.py`: runtime `CoreAIQwen` with the AneQwen3 API. All entry points are
  loaded up front; `resize` swaps KV buffers and entry names with no reload. DeltaNet buffers are passed straight from
  one call's outputs to the next call's inputs. KV caches are host NDArrays written in place. Loads precompiled
  `.aimodelc` when present, with a purge-cache-and-retry fallback.
- `ane-vector-lut/scripts/qwen38_coreai_verify.py dump coreml|coreai`, `compare`, `greedy <rt>`, `speed`: stage 4
  harness (teacher-forced logits vs the ane5 Core ML build, ppl, greedy, per-entry speed, switch time).

Findings:
- **Core AI's weight-key bug:** conv weights reach coreai-opt's `_blockwise_compress` as (Cout, Cin, 1, 1), so
  LUT-injection keys must ignore the shape. With shape-keyed lookup the vector 2x16 matrices fell back to UNIQUE
  mode (NUM_PALETTES 256 vs 4-bit indices), and `optimize()` failed.
- **ANE dimension limit:** attention concatenates [history | block], so 65536 + 8 = 65544 > 65536 fails ANEC for the
  whole package. KV history is capped at 65536 - T (the 64K entry holds 65528 rows).
- **32K / 64K attention fails ANEC** ("MLIR MPS to ANEC conversion failed"). One such entry takes every entry of the
  package off the ANE: all ran on the CPU at ~1040 ms per call, and load failures don't say which entry.
  16K compiles. Core ML has the matching cliff at 32K (23 ms vs 10.7 ms at 64K).
  **Fix: blocked attention above 16K.** History is sliced into 16K blocks, one global max is taken over all block
  scores, then exp(s - m) per block with summed numerators and denominators. That is exactly softmax over
  [history | block] (CPU fp32: cos 1.0, max rel 1e-6). Layer 3 at 32K, at 64K, and 16K + 32K + 64K in one package:
  all fully on the ANE (`qwen38_coreai_bisect.py`).
- Bisect of the other entries: chunk L00-03 `v8_2k`, layer 0 `p64_2k` and layer 3 `v8_16k` are each fully on the ANE.
- Prefill-64 vs 8 chained verify-8 calls (L00-03, cold start): y per block cos 1.00000, k rows 1.00000, and the next
  verify call's y / rec match 1.00000. **Measured on the CPU fallback of the failing package**, so it proves the
  graph logic, not ANE numerics.
- In-place KV writes: the NDArray constructor copies its source and `numpy()` returns fresh buffers, but the
  buffer-protocol pointer of an NDArray's storage is stable. A writable ctypes view over it lets the host write
  rows in place: 8 rows in 0.001 ms vs 3.1 ms to rebuild a 32 MB cache NDArray. The chunk reads the in-place rows
  (y bit-identical to a fresh NDArray), again measured on the CPU fallback.
  `coreai_inplace_probe.py` also compiled to the CPU / GPU (0 ANE regions), so the ANE case is still open.
- Writing KV inside the graph: `slice_scatter` lowers to `coreai.slice_update` (not scatter_nd) and writes the right
  rows of a state buffer, but the graph had no ANE region (`coreai_kv_slice_probe.py`). A separate writer program
  sharing the state failed to compile (`placement.region_call operand type mismatch`, `coreai_kv_writer_probe.py`).
  Host in-place writes stay the plan.

**Blocked (stage 2, full export): disk.** 10 GB free on the M6. A chunk costs its package (0.76 GB) plus its compile
cache (~0.84 GB), so 16 chunks + head need ~25 GB (~13 GB if compiled once to `.aimodelc` and the `.aimodel`
deleted). The space is held by 19 leftover Core ML temp compiles of the drafter in `$TMPDIR`
(`dflash2_lut4_rtn_*.mlmodelc`, 1.5 GB each, 28 GB, 09-26 18:34 to 09-27 00:52, none open). Loading the drafter
`.mlpackage` compiles a new copy every time and never deletes it. Fix: compile the drafter once to `.mlmodelc`,
load that, and delete the temp copies. Other large non-port items: `fp8-mlp-metal41-bench/coreai/artifacts_vector_lut`
(16 GB, 09-25), `artifacts_chunk` (3.3 GB incl. a 1.5 GB `.mlir`), and older coreai-cache entries (~20 GB, 09-22..26).

## Test plan (Core AI is a different compiler: MLIR -> MPSGraph -> ANE regions, not MIL -> E5RT)

Nothing learned on the MIL path carries over by default: the MLState -14 rules, the width-axis slice bug, the
duplicate-output bug, fp16 overflow workarounds (scale-free RMSNorm, LUT scale folding), matmul(N, N) failures.
Test in this order, each against the MIL / torch reference, before porting anything bigger:
1. Op-level parity on the ANE: our DeltaNet lazy-commit math, KV-input attention (concat softmax), RMSNorm,
   LUT + per-channel scale, 1-bit-LUT Hadamard conv. Output cos / max error vs torch fp32, placement per region.
2. I/O mechanics: host-owned ping-pong buffers vs `MutableBuffers` states; whether large inputs are copied every call
   (`coreai_call_overhead.py` KV cases); strided views of one max-length KV buffer across entry points (AFM style).
3. One real chunk (layers 0-3: 3 DeltaNet + 1 attention, exported weights) in Core AI vs the v4 MIL chunk: outputs,
   call time, wired memory, placement; then with 2+ entry points (verify-8, prefill-64, two context lengths).
4. Only then the full target, and a teacher-forced perplexity / KL check against the Core ML runtime.

## Open checks before porting

- [~] **Chunk / program size limit on the ANE.** Core ML: about 1 GB of weights per chunk on iOS, about 2 GB on
      macOS. Core AI: 3 GB in one entry point fully on the ANE (158 GB/s); 4 GB compiled but failed to load
      (inconclusive, see Reliability). Entry points sharing one `resources.bin`: 758 MB x 2 entries fine.
- [x] **Per-call overhead** vs Core ML: 0.22 vs 0.13 ms per call (slope method); the real 4-layer chunk is ~6%
      slower in Core AI under equal load.
- [x] Zero-copy I/O: `NDArray(data, StorageKind.IO_SURFACE | BYTES)`; `.numpy()` is a zero-copy view (fp16). Big
      inputs are NOT copied per call (a 128 MB KV-like input costs nothing extra). No output backings: outputs are
      allocated per call (a few MB per chunk). Inputs bind by name; undeclared extras are ignored; results deterministic.
- [x] States: a torch buffer mutated in place -> `MutableBuffers` handle `tensor<4x65536x256>`, passed per call as
      `state={"kv": NDArray}`; in-place updates land correctly and persist; several entry points take the same
      max-length handle and read windows of it (AFM style). BUT the dynamic write (`index_copy_` -> `scatter_nd`) puts
      the graph on the GPU. Prefer read-only KV inputs + host-committed rows (v4 design).
- [x] Per-channel scale after a LUT: a mul after the conv, placed on the ANE; palettize before `optimize()`.
- [x] Exact codebooks / indices through coreai-opt: `_blockwise_compress` injection (by weight bytes) works for
      vector 2x16, scalar LUT4 and the Hadamard convs in one pass (n_bits=4, cluster_dim=2, fast k-means off).
- [x] Graph pieces in torch: lazy-commit Gated DeltaNet (forward substitution instead of the doubling inverse),
      KV-input attention, online Hadamard as a grouped conv - all fully on the ANE, parity 0.9998-1.0000.


## Incident 2026-09-27 03:18: M6 watchdog reset during the full-target port

- The M6 reset (`ResetCounter` report: `Boot faults: wdog,reset_in_1`) while the port agent rebuilt chunk L00-03 with 6
  entry points (verify-8 at 2K / 8K / 16K / 32K / 64K with blocked attention above 16K, + prefill-64). Earlier the same
  night a Core AI compile aborted in `MPSGraphExecutable applyOptimizationPasses` (python3.13 crash 01:17).
- After the reboot a root `ANECompilerService` kept compiling for 4+ hours at ~500% CPU, stuck in
  `ZinMirGraphSplitLatencyCostModel` (graph-split cost model), with no client; it needs `sudo kill -9 <pid>`.
- Until understood: build the large-context entry points (32K / 64K) one at a time in separate packages, never
  several huge ANE compiles concurrently, and keep the M6 otherwise idle during those compiles.


## Stage A: full target in Core AI, export mix25_aw_cal_lr64mix (2026-09-27, M6) - NOT ready for the server

Build: 16 chunks x 4 layers + head, entries v8_2k / v8_8k / v8_16k / p64_2k, one package per chunk (0.73-0.78 GB),
9.9 GB total, ~3.3 min per chunk (two build processes in parallel, separate OUT dirs, merged with
`coreai_merge_builds.py`); first load compiles every entry (402 s), cached loads ~1-7 s. All entries of every chunk
fully on the ANE, one region each, no GPU.
```
cd ane-vector-lut/coreai && unset USE_LOCAL_COREAI
EXPORT_DIR=~/Models/vq27b/export/mix25_aw_cal_lr64mix OUT=~/Models/vq27b/coreai_ane6 .venv/bin/python qwen38_coreai_build.py all \
    --plan 0-3,...,60-63 --ctx 2048,8192,16384 --pctx 2048
COREAI_DIR=~/Models/vq27b/coreai_ane6/mix25_aw_cal_lr64mix .venv/bin/python ../scripts/qwen38_coreai_verify.py speed
```
`xcrun coreai-build compile` (--compile) produced packages this OS refuses ("MPSGraph Package Version ... up to 7.0.80"):
don't precompile; load the .aimodel (the cache compiles once).

| check | Core AI | Core ML ane6 16K |
| --- | --- | --- |
| logits parity (256 prompt tok + 32 verify blocks, 16K) | top-1 98.4%, ppl 5.332 vs 5.325, KL mean 0.0016 (p99 0.010), cos mean 0.994 | reference |
| verify-8 call, steady | 134-161 ms (16K); 10-14 ms per chunk | 108 ms; 6.0-7.4 ms per chunk |
| verify-8 over a long run | degrades (chunk: 8.7 -> 12.9 ms over 600 calls); full model p50 226 / p90 360 ms after ~50 calls | flat |
| prefill-64 | 188-192 tok/s (2K entry only) | 62-73 tok/s (8-row calls) |
| context switch 2K -> 8K -> 16K | 75-135 / 159-199 ms (KV rows moved, no reload) | 60-80 s reload |
| wired, target + 16K KV | 19.6 GB (only v8_16k called), 21.1-21.3 GB with all 4 entries warmed | 18.8 GB |

**Blocker: Core AI runtime output pool.** Every call allocates new output NDArrays (12 per chunk: conv / rec / pend
states, y, k / v rows, taps). Over a long run the per-call time rises and wired memory grows (+0.5 GB / 600 calls);
after ~850 function calls (~50 full verifies) the process died: `CoreAIRuntime/NDArray+Pool.swift:77: Fatal error:
Failed to allocate storage for NDArray with byteCount: 307200, sk: ioSurface`. Not Python: `gc.collect()` every call
with `gc.freeze()` does not change it (`coreai_leak_probe.py`). There is no output-backing API. Feeding outputs back
as inputs also costs ~3 ms per call over a single call (`coreai_output_cost.py`: 1.33 ms single, 4.5 ms chained;
copying into persistent IOSurface inputs 2.7 ms in the probe, no gain in the full runtime).
Candidate fix: DeltaNet states as Core AI state buffers (MutableBuffers updated in place, whole-buffer writes, not
scatter) so a chunk returns only y / k / v / taps; test on one chunk that it stays on the ANE. Otherwise report to Apple.

Memory per extra entry (full model): ~0.75 GB (ANE program + scratch per entry; weights shared). Budget at 16K:
Core AI 4 entries 21.2 GB + drafter 2.2 GB = 23.4 GB (target ~24 GB); Core ML 16K 18.8 + 2.2 = 21.0 GB.
Projected 64K (KV 4 GB instead of 1 GB, + 32K / 64K entries ~1.5 GB): Core AI ~27.9 GB (over), Core ML ~24 GB.
Runtime changes (qwen38_coreai_model.py): reset() shrinks to the smallest ladder context, restore(snap) shrinks to the
smallest context holding the snapshot and raises if its KV rows were dropped (restorable(snap) tells), persistent
DeltaNet / hidden-state input buffers, stats created before the first reset.

## ANE memory: what a Core AI chunk wires, and the compile-mode fix (2026-09-28)

aned logs every program load (`log stream --info --predicate 'process == "aned"'`): `[ANE Model Stats] : modelSize=...
: wiredMemory=...`; wired = the compiled program (hwx) + one scratch (intermediate) buffer per program.
4-layer chunk L00-03 at 24K (0.52 GB of source weights, v8_24k + p64_24k), MPSGRAPH_ANE_BONDED_COMPILE_MODE:

| mode | program (modelSize) | wired | verify-8 / prefill-64 per chunk |
| --- | ---: | ---: | --- |
| default (= 0) | 1.09 GB | 1.25 GB | 7.50 / 23.56 ms |
| 1 | 0.57 GB | 0.71 GB | 10.30 / 36.85 ms |
| **2** | **0.64 GB** | **0.79 GB** | **7.46 / 23.52 ms** |
| 3 | 0.57 GB | 0.71 GB | 10.46 / 36.55 ms |

Mode 2 outputs are bit-identical to the default (12 outputs x 3 calls, both entries). MPSGraph normally keeps a
"bonded" and a "nonbonded" variant of every procedure (Apple's AFM binary has 44 procedure variants for 22 entries);
mode 2 keeps the fast one. Set by qwen38_server.sh for RUNTIME=coreai. The AFM package ships its hwx precompiled
(`enableCompileResourcesForPackage`), and its KV caches are prewired IOSurface state inputs; neither is available through
the public coreai-torch / coreai-build today. Private-API survey: fp8-mlp-metal41-bench/coreai/private_ane_research/.

Full model with mode 2 (coreai_mixr/mix25in_mixr_lr64mix: 16 chunks x 4 entries v8/p64 at 16K and 24K + head, 9.9 GB
of packages on disk; bridge runtime, 2026-09-28). aned loads 17 programs, one per package; all 4 entries of a chunk share
its weights. System wired memory over the idle baseline:

| step | wired |
| --- | ---: |
| all packages loaded | +12.1 GB |
| first verify at 16K (KV, DeltaNet states and per-program scratch get wired) | +15.6 GB |
| first prefill at 16K | +15.7 GB |
| resize to 24K, then first verify / prefill | +17.1 / +16.2 GB |

Before mode 2 the same target took ~28.5 GB. Speed: verify-8 is 121.7 ms at 16K and 129.8 ms at 24K; prefill-64 is
353 ms (181 tok/s) at 16K and 378 ms (169 tok/s) at 24K. The 16K -> 24K resize takes 107 ms. Trace ppl 2.4148 over
40023 tokens (ane7i 2.426).

Compile mode is part of the cached specialization, not of the cache key (`~/Library/Caches/coreai-cache/<OS build>/
<executable name>/<main.hash>/...`; the mode is `aneBondedCompileMode` in the cached mpsgraphpackage manifest). Loading
a package cached in mode 0 with mode 2 aborts the process (`failed assertion 'Unable to use cached specializations
and original module not available'`); the other direction loads the cached mode-2 program. qwen38_coreai_model.py
therefore defaults the mode to 2 itself (every entry point: server, chat, trace ppl, validation) and, before each load,
purges this process's cached specializations of that package that were compiled in another mode, so it recompiles
instead of aborting. The DFlash2 drafter is a single-function Core ML package (1.5 GB on disk, 2.02 GB ANE program,
+1.9 GB wired) and is not affected by the MPSGraph setting.

## Context ladder 8K-64K: 12 entry points per chunk (2026-09-28)

One 4-layer chunk (L60-63, LUT4 mixers) with v8 + p64 entries at 8K, 16K, 24K, 32K, 48K and 64K (12 entries, 824 MB
package; built on the M3U with `qwen38_coreai_batch.py --ctx 8192,16384,24576,32768,49152,65536`), loaded alone on the
M6 in mode 2 next to the running server (`tests/op_limit_test.py`): all 12 entries run, first load (ANE compile) 56 s.
Compared with the deployed 4-entry chunk of the same layers (16K / 24K), measured back to back:

| chunk L60-63 | package | program (modelSize) | wired | scratch (wired - program) |
| --- | ---: | ---: | ---: | ---: |
| 4 entries (16K, 24K) | 785 MB | 1069.9 MB | 1227.7 MB | 158 MB |
| 12 entries (8K-64K) | 797 MB | 1085.3 MB | 1393.8 MB | 309 MB |

Entry points are nearly free (+15 MB program for 8 more); the scratch buffer (one per program) is sized by the largest
entry (p64 at 64K: 64 x 65472 attention scores), +151 MB per chunk. Same speed at 16K / 24K in both packages.

| per chunk, ms | 8K | 16K | 24K | 32K | 48K | 64K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| verify-8 | 7.7 | 8.5 | 9.3 | 10.2 | 11.7 | 13.4 |
| prefill-64 | 17.6 | 22.0 | 23.6 | 27.8 | 34.4 | 41.1 |

Projection for the full model (16 chunks, one attention layer each, so the context-dependent part scales x16 from the
measured full-model 16K numbers; 24K predicted 134 / 378 ms vs measured 129.8 / 378): verify-8 ~110 / 122 / 134 / 149 /
173 / 201 ms and prefill-64 ~281 / 353 / 378 / 446 / 551 / 658 ms (228 / 181 / 169 / 143 / 116 / 97 tok/s) at 8K / 16K
/ 24K / 32K / 48K / 64K. Memory: +2.4 GB of scratch over the deployed build (16 x 151 MB), KV 1.07 GB per 16K
(16 attention layers x K and V x 4 heads x 256 x fp16; 4.29 GB at 65472 rows). The server's usable context at 64K is
65472 positions (the 64K entry's KV rows); qwen38_server.py caps it and leaves 8 rows for the last verify.

Full model, measured (coreai_mixr12/mix25in_mixr_lr64mix: 16 chunks x 12 entries + head, 10 GB on disk; mode 2; first
load compiles 16 x ~60 s, then 1-3 s from the cache). `qwen38_coreai_verify.py speed` (target alone; wired over a
2.87 GB idle baseline):

| ctx | verify-8 median / mean / max | prefill-64 | switch in | wired |
| --- | --- | ---: | ---: | ---: |
| 8K | 107.7 / 107.7 / 110.0 ms | 225 tok/s | - | +17.5 GB |
| 16K | 126.2 / 126.1 / 127.8 ms | 177 tok/s | 108 ms | +18.1 GB |
| 24K | 134.9 / 135.8 / 140.6 ms | 161 tok/s | 139 ms | +18.6 GB |
| 32K | 150.7 / 151.2 / 154.5 ms | 140 tok/s | 237 ms | +19.0 GB |
| 48K | 178.6 / 178.1 / 181.4 ms | 109 tok/s | 287 ms | +20.1 GB |
| 64K | 204.7 / 205.3 / 209.5 ms | 80 tok/s | 411 ms | +21.1 GB |

Wired grows by the KV only (~0.55 GB per 8K); with the Core AI drafter +1.5 GB more (64K: +22.6 GB, ~25.4 GB of
32 GB). Decode (server Engine + Core AI drafter, target pinned to each entry, `tests/decode_ctx_test.py`, 512 tokens,
coding prompt, cold): sampled 32.3 / 30.0 / 25.5 / 23.8 / 20.6 / 18.3 tok/s at 8K / 16K / 24K / 32K / 48K / 64K;
greedy 35.2 tok/s at 8K, 20.8 at 64K (4.70 tok/cycle). Greedy text is identical on all six entries; trace ppl on 8
sequences (5896 tokens) 2.1986 at 8K vs 2.1983 at 64K.

Context transitions on real prompts (`tests/transition_test.py`: the ladder grows mid-prefill or mid-decode vs the same
request pinned to the larger entry): greedy text identical in all three cases (8K -> 16K during decode and during
prefill; a 32.6K-token prompt through 8K -> 16K -> 24K -> 32K, then 32K -> 48K during decode). KV rows moved in
64-513 ms per switch. Growing is faster than pinning: the 32.6K prompt prefills in 207 s (158 tok/s) vs 327 s
(100 tok/s) pinned at 48K.

Drafter stalls (fixed in qwen38_server.py, DRAFT_GAP_MS): the drafter's ANE call stalls 300-700 ms a few times per
100 cycles (the next verify sometimes too) when it is submitted within ~2 ms of the target verify's return. Greedy
leaves ~1.5 ms of host work there, sampling ~6 ms, so it showed as a slow drafter in greedy runs only (20-47 ms/cycle
vs 16). `tests/drafter_gap_test.py` at 16K, 4 interleaved runs each: no gap 10 drafter + 4 verify stalls, 26.4-29.5
tok/s; 3 ms gap none, 29.4-29.5 tok/s. The server now waits until 3 ms after the verify (counted in the draft phase);
with it, 4 runs had no drafter stalls (2 isolated verify stalls of 222 / 490 ms remain). Server after restart on this
build (CTX=64K, usable 65472): short greedy 47.8 tok/s (6.25 tok/call; the 16K/24K build: 43.5), short sampled 40.6.
