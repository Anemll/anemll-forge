# TODO

Experiments queued for later: what to measure and what would make it worth keeping. Measurements live in the research notes; the compute work is in [M6 compute acceleration, Follow-up, 4 October](docs/research/M6_COMPUTE_ACCELERATION_2026-10-03.md#follow-up-4-october-measured) and [5 October](docs/research/M6_COMPUTE_ACCELERATION_2026-10-03.md#follow-up-5-october-compiled-ane-programs-and-8-bit-attention-measured) (`R` below).

## Open

### 1. INT8 activations with our LUT weights (prefill weight matmuls)

INT8 x INT8 runs on the M6 ANE through Core AI at up to about 53 TOPS against 13 to 19 for FP16 (R, INT8 compute), but INT8 weights for the whole model (about 27 GB) do not fit on 32 GB. Test our 2/4-bit LUT weights with INT8 activations on one chunk's MLP at the 64-row prefill size, built the way that works: compile-time weights, shared activation scales in and out, output-channel splits. An earlier LUT x INT8-activation probe failed ANEC before these corrections. Our weights are not all LUT4: half the MLP layers and the large mixers of layers 0 to 23 use vector LUT 2x16 (2 bits per weight), the rest LUT4, K/V projections INT8 ([docs/QUANTIZATION.md](docs/QUANTIZATION.md)); every LUT holds FP16 values, so W8A8 needs a quantized LUT (INT8 codebook entries with a scale: `lut_to_dense` of an INT8 table, then a compile-time shift/scale, which `coreai_opt` supports) to give INT8 weight values at the same storage. Keep it if prefill per row drops clearly and KL-512 holds; quality is the open risk (activation quantization of a 27B model).

### 2. Output-channel splits (TP2) for FP16 and LUT weight matmuls (MLP, head)

Splitting a constant INT8 weight into two output-channel branches lifted 256-row W8A8 from 34.4 to 43.3 TOPS, and INT8 weights with FP16 activations from 23.0 to 32.2 (R, INT8 compute); a 4096 x 4096 FP16 matmul was unchanged at 256 rows. Our weights are LUT constants of other shapes (MLP 5120 to 17408 and back, head 5120 to 248,320), which the compiler may tile differently. Measure one chunk's MLP projections and the head as TP1 / TP2 / TP4 at 8 and 64 rows (chunk A/B for verify and prefill, head in a standalone package). Keep any split that shortens the verify or prefill call without hurting compile time.

### 3. 8-bit attention: INT8-only path (M5)

On V8 (FP16 keys) the M6 form C (INT8 scores, FP8 softmax sum, FP8 PV weights) matches V8 on KL-512 (R, 5 October); the INT8 / UINT8 form A does not (direct KL 0.011 to V8): folding the per-token value scales into UINT8 weights rounds 84% of codes to zero and drops about 7% of the softmax mass. Options for M5, host simulation first (`scripts/m6_attn_logit_stats.py`): take the softmax sum from the same UINT8 weights (4.7% against 5.4% attention error), per-tile value scales (3.7%, a cache-format change), or retrain the rank-64 corrections with the 8-bit attention simulated.

### 4. Contexts above 64K: production ladder

80K runs on 32 GB as an 80K-only package (25.7 GB wired); 80K and 100K together swap (R, Contexts above 64K). Next: one chunk of a full ladder plus 80K (8K to 64K + 80K) for its first-call wiring, then the full build, compile time and memory at the 80K entry; a one-chunk 100K-only package for wiring. Needs `qwen38_pi_config.py` to accept 80K windows if adopted.

### 5. Long prefill split between the ANE and the GPU (idea)

Prefill is sequential through tokens, so only a layer pipeline can use both: the ANE on chunks 0 to k, the GPU on the rest, overlapping consecutive 64-row blocks; only the hidden state crosses (about 640 KB per block). Splash's GPU prefill is about 1.1 to 1.2x ours, so balanced halves might roughly halve a long cold prefill (unmeasured). Needs GPU kernels for our LUT format (dequantize to tile, then `matmul2d`), KV and DeltaNet state written in our layouts, a KL check of the mixed model, and whole-machine power (about 31 W ANE plus about 46 W GPU when each runs alone). First measure one chunk's prefill on each device.

### 6. Drafter on the GPU

A decode cycle is 98% ANE work in sequence (8K: 114 ms = drafter 15.3 + verify 96.6 + sampling 1.9 + host 0.6). The drafter package already runs on the GPU (`COREAI_DRAFTER_COMPUTE=gpu`), but `forge.py serve` forces `ane`: allow the override and measure draft time, `DRAFT_GAP_MS` stalls, acceptance and power (at most 3 to 13% of decode). In prefill the drafter reads the prompt after each target block (about 10% of prefill, estimated); on the GPU it could overlap the next block.

### 7. Build only what is needed

One extra cache format compiles about 1.65x longer, and every context adds tiles. Keep the default build to one format; generalize `--kv-cache-dtype` to any list or `all` (selectable manifests with any subset); add a quick-test preset (8K, one format) so a new graph or format builds and compiles in minutes; have the compile guide name these options when the estimate is long.

### 8. Form C in production (M6)

Full 8K to 64K build of form C (`ATT_INT8MM=s8,s8b,sm8,pvf8`, `ATT_S8_UNIT=ATT_S8B_UNIT=0.25`, V8): server benchmark against V8 and the 64K long-context check are running (R, 5 October). If they hold: make the forms a builder option (FP8 enabled per chip, off on M5), add a host exactness test for the forms without quantization, record the setting in the release manifest, and update the model card numbers.

### 9. Transposed key cache

Every history tile of keys goes through an INT8 transpose pass before QK (the cache stores (token, 256), QK reads (256, token)): 19% of verify cycles and 6% of prefill, through DRAM in prefill (R, 5 October). Store keys as (head, 256, token): one `kv8` core first (the HWX should lose the transpose tasks), then the cache writer (`scripts/qwen38_kv_cache.py`), saved caches and the tests.

### 10. Smaller items

- **16-bit matched pair:** Splash `--kv-format bf16` against our FP16 cache (`fast_b2k` two-format build), same harness.
- **DFlash2 temperature sampling:** [docs/research/DFLASH2_SAMPLING_PLAN.md](docs/research/DFLASH2_SAMPLING_PLAN.md).
- **Compile mode and the Core AI cache:** the cache does not key on `MPSGRAPH_ANE_BONDED_COMPILE_MODE`, so a package first compiled in another mode keeps that program (2.2x the cycles for one attention core in mode 0). Have the loaders record the mode they compiled with and warn on a mismatch.
- **Dynamic quantization scales:** `coreai.symmetric_quantization_statistics` computes per-axis scales at run time but has no torch op; try it in hand-written MLIR (a runtime scale written in torch fails ANEC).

## Closed (4 and 5 October; details in R)

- **`kv8` (INT8 keys and values):** quality unchanged against V8 (KL-512, direct KL 6.6e-5), speed within 2%, a third less cache. In the faster graph's 2K tiles INT8 caches save memory, not time.
- **Softmax forms:** online and split are exact but within about 1% of two-pass on a full chunk (split -3.4% only for 64K prefill). Switches `ATT_SOFTMAX` / `ATT_SOFTMAX_PREFILL` stay, default two-pass.
- **128-row prefill calls:** 21 to 45% slower per row than 64-row calls.
- **INT8 matmuls in the history attention:** the matmuls are under a tenth of the core, so INT8 multiply-adds cannot pay there (see item 3 for the score tensors).
- **Contexts above 64K on the ANE:** the 65,472-row cap is gone for the tiled graph; 80K and 100K attention compiles fully onto the ANE.
- **INT8 history QK:** the HWX confirms INT8 x INT8 with a query pair, but QK is bound by data movement: a fixed step gains nothing, a per-row step costs a pass (+4 to 5%), a per-row scale inside the pair fails ANEC.
- **Offset codes for the attention weights:** zero point -128 or `minval` mode does not fuse; both operands get dequantized to FP16 in DRAM (17% slower than no pair).
- **Native softmax:** `coreai.softmax` compiles to the same program as the hand-written passes.
- **Per-tile dequantization of the INT8 cache:** the compiler already reads the INT8 tiles directly; the same program either way.
- **INT8 K/V projection weights:** compile-time INT8 constants are now the builder default (`QCONV_INT8=1`, 10 MB less per attention layer); needs KL-512 with the next full build.
