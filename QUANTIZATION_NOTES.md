> Historical research notes imported on 2026-09-29. Paths were generalized; results were not rerun. Later sections may supersede earlier findings. See the current [release workflow](docs/WORKFLOW.md).

> Release context — 2026-09-29: the target-quality and repetition conclusions below describe their original experiments, not a blanket attribution of all loops. Later work found serving-policy and wrong-build confounders; see [session lessons](docs/SESSION_LESSONS.md). The intended fast release includes the matching Core AI DFlash2 drafter. Algorithmic sampling correctness and CPU tests do not establish compiled-runtime parity or task quality. Use [the current speculative-decoding guide](docs/SPECULATIVE_DECODING.md) for the release contract; historical commands and results below are preserved.

# Qwen3.8-27B on the ANE: quantization approach, layout, experiments, findings, dead ends

Living document. Every experiment lists the command that reproduces it. Core AI / memory topics live in
`COREAI_PORT_NOTES.md` and `FEEDBACK_ANE_MULTIFUNCTION_MEMORY.md`.

Machines: **M6** (32 GB, macOS 27, ANE target, `~/venvs/vq27b`), **M3U** (M3 Ultra, 96 GB, `ssh user@quantization-host`,
bf16 checkpoint + calibration + KL reference on `/path/to/data`). Scripts: `ane-vector-lut/scripts/` (same tree on
both machines; copy changed scripts to the M3U with `scp`).

## 1. Model and weight layout

- Qwen3.8-27B (`qwen3_5` arch): 64 layers = 48 Gated DeltaNet (linear attention) + 16 gated full attention (every 4th
  layer, 3, 7, ..., 63), hidden 5120, MLP 17408, 24 q / 4 kv heads, head_dim 256, vocab 248320 (untied head).
- ~25.6B weights read per token: MLP 17.1B, DeltaNet projections ~5.5B (in_proj_qkv 10240x5120, in_proj_z
  6144x5120, out_proj 5120x6144 per layer), attention ~1.5B, head 1.27B.
- Deployed export `full_mix25_mixer4_head4` (9.06 GiB of quantized weights):
  | part | format | notes |
  | --- | --- | --- |
  | MLP gate/up/down | `vector 2x16 + pcs` (2 bits/w) in 47 layers, `LUT4 per-tensor + pcs` in 17 layers (0-6, 8, 9, 28, 34-40) | plan `plan_optiq_top48.json` from WikiText sensitivity sweeps; GPTQ; online block Hadamard (1024) on MLP input and down input |
  | DeltaNet in_proj_qkv / in_proj_z / out_proj | `LUT4 per-tensor + pcs` | GPTQ, plain basis |
  | attention q (+gate) / o | `LUT4 per-tensor + pcs` | k / v projections `INT8 per-channel` |
  | lm_head | `LUT4 per-tensor + pcs` | |
  | in_proj_a / in_proj_b, norms, conv, embeddings | bf16 / fp16 | |
- `pcs` = per-output-channel fp16 scale after a per-tensor LUT (free on the ANE). The LUT is fit on `w / row_RMS`.

## 2. ANE format constraints (measured, see README.md)

- LUT at most **256 values** (entries x vector size); vectors along Cout only; per-tensor vector LUTs only.
- Per-group **scalar** LUTs run on the ANE; per-group **vector** LUTs and any grouping along Cin fall to the CPU.
- Speed is weight-bandwidth bound down to ~2 bits/w (~125-165 GB/s). 3-bit and 6-bit indices are read like 4- and
  8-bit. MIL `constexpr_lut_to_dense` allows 1/2/3/4/6/8-bit indices (no 7-bit).
- 27B-shape MLP block, 1 token (qwen38_mlp_ane_probe.py): INT8 1.63 ms, LUT4 per-group-8 1.19, LUT4 per-tensor +
  pcs 0.83-0.85, 2-bit vector 2x16 / ternary + pcs 0.43. Each MLP layer 2-bit -> 4-bit costs ~+0.4 ms per call.
- Grouped scalar LUT4 at the 27B MLP shape (2026-09-27, same run, per-block slope S=8 vs S=16,
  `qwen38_mlp_ane_probe.py lut4_pcs lut4_g1024_pcs lut4_g512_pcs lut4_g128_pcs lut4_g8` + `time_coreml_pair.swift`
  with a (1, 5120, 1, 1) input): per-tensor + pcs 0.847 ms, 1024-row groups 1.025 (+21%), 512-row 1.026 (+21%),
  128-row 1.008 (+19%), anemll per-group-8 1.244 (+47%). Any grouping costs ~20%.
- Weight SNR on Gaussian weights (k-means, per-tensor): LUT4 20.1 dB, vector 2x64 (3 b) 15.2, scalar 3-bit 14.5,
  vector 2x16 (2 b) 9.7, scalar 2-bit 9.3.

## 3. Pipeline

| step | script (machine) | reproduce |
| --- | --- | --- |
| sensitivity sweeps (per layer, WikiText ppl) | `qwen38_gptq_27b.py` SWEEP=mlp / mixer (M3U) | `SWEEP=mlp python qwen38_gptq_27b.py` |
| bit plan for a size budget | `qwen38_plan.py` (M3U) | `python qwen38_plan.py sweep_mlp.json sweep_mixer.json 8.0 9.9` |
| GPTQ export | `qwen38_gptq_27b.py` (M3U, ~85 min) | see the env block below |
| KL vs bf16 (in-domain trace, 64 seq, 40K tok) | `qwen38_kl.py eval` (M3U, ~9 min) | `TRACE=/path/to/data/vq27b/kl EXPORT_DIR=<export> python qwen38_kl.py eval` |
| ANE build (v4: 16 chunks x 4 layers, T=8) | `qwen38_ane_model.py build_v3` (M6) | `ANE_OUT=~/Models/vq27b/ane4 EXPORT_DIR=<export> CTX=8192 python qwen38_ane_model.py build_v3` |
| ANE teacher-forced ppl | `qwen38_ane_ppl.py` (M6) | `CTX=8192 MODE=block N=512 python qwen38_ane_ppl.py` |

Deployed export command (M3U, in `scripts/`):
```
PLAN=/path/to/data/vq27b/plan_optiq_top48.json MIXER="LUT4 per-tensor + pcs" KV_FMT="INT8 per-channel" \
HEAD="LUT4 per-tensor + pcs" BASELINE=0 TAG=full_mix25_mixer4_head4 python -X faulthandler -u qwen38_gptq_27b.py
```
GPTQ options added 2026-09-26: `CAL_MIX="<rows.npy>:<n>,..."` (in-domain calibration rows replacing WikiText rows;
`qwen38_calib_gen.py` = bf16 self-generated chat with thinking + tool prompts, `qwen38_calib_pi.py` = rendered pi
agent sessions), `AW=1` (codebook k-means weighted by diag(H), the imatrix idea), `NCAL`, format
`vector 2x64 + pcs`. KL eval options: `PARTS=mlp,gdn,attn,head` + `QLAYERS=lo-hi` (ablation: quantize only these
parts), `LR_RANK=r` (add the rank-r SVD of the quantization error), per-position KL saved as `klpos_<tag>.npy`.

## 4. Experiments and results

KL = mean KL(bf16 || quant) over the 40K-token in-domain trace (bf16 ppl 2.136). Run on the M3U.

| tag | what | size GiB | mean KL | median | p99 | top-1 | ppl |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full_mix25_mixer4_head4 | deployed | 9.06 | 0.619 | 0.168 | 5.48 | 72.8% | 3.09 |
| optiq_top48_mix | same MLP plan, mixers + head bf16 | 20.83 | 0.496 | 0.084 | 7.77 | 78.0% | 2.80 |
| sweep_8.0GB | 2-bit MLP + 2-bit mixers plan | 7.24 | 0.778 | 0.361 | 4.85 | 68.9% | 3.64 |
| abl_mlp | deployed MLP only (rest bf16) | | 0.477 | 0.087 | 5.08 | 77.8% | 2.87 |
| abl_gdn | DeltaNet projections only | | 0.170 | 0.021 | 3.20 | 87.9% | 2.20 |
| abl_attn | attention projections only | | 0.044 | 0.004 | 0.67 | 93.2% | 2.07 |
| abl_head | lm_head only | | 0.028 | 0.003 | 0.35 | 93.9% | 2.23 |
| abl_mlp_0_23 | MLP of layers 0-23 only | | 0.066 | 0.006 | 0.92 | 92.5% | 2.28 |
| abl_mlp_24_63 | MLP of layers 24-63 only | | 0.453 | 0.074 | 6.07 | 79.4% | |
| abl_gdn_lr64 | DeltaNet projections only + rank-64 SVD of the quantization error (fp16 A @ B) | +0.23 | **0.024** | 0.003 | 0.43 | 95.0% | 2.137 |
| abl_mlp_lr64 | MLP only + rank-64 SVD of the error | +0.55 | 0.453 | 0.077 | 6.23 | 78.7% | 2.80 |
| mix25_aw_cal_lr64mix | mix25_aw_cal + rank-64 correction on DeltaNet + attention | +0.28 | pending | | | | |
| full_mix25_br | deployed export after block reconstruction (LUT values + scales, same size) | 0 | pending | | | | |
| full_mix25_br_lr64 | block reconstruction + trained rank-64 factors on DeltaNet + attention | +0.28 | pending | | | | |

Block reconstruction (M3U, streamed layers, ~115 s per DeltaNet layer, ~40 s per attention layer, 64 steps):
```
MODEL=/path/to/data/Qwen3.8-27B EXPORT_DIR=/path/to/data/vq27b/runs/export/full_mix25_mixer4_head4 \
OUT_DIR=/path/to/data/vq27b/runs/export/full_mix25_br CAL=/path/to/data/vq27b/calib_pi_ids.npy:16 WIKI_ROWS=16 \
HOLD=4 STEPS=64 BS=2 [LR_RANK=64 LR_PARTS=gdn,attn] python -u qwen38_blockrecon.py
```
Exported factors are `{key}.lr_a` (Cout x r) / `{key}.lr_b` (r x Cin) fp16 in the layer files (original basis; for the
MLP on the unrotated input); `qwen38_kl.py` applies them. Test on the M6 (CPU, parity check vs the export):
`DEVICE=cpu LAYERS=0-3 SEQ=256 CAL=~/Models/vq27b/calib_pi_ids.npy:4 WIKI_ROWS=0 HOLD=2 STEPS=8 CHECK=1 ...`.
| mix25_aw_cal | deployed plan, AW=1, calib 16 wiki + 16 chat + 16 pi (size-neutral) | 9.06 | pending | | | | |
| mlp2_aw_cal | every MLP vector 2x16, same calibration (smallest; 2-bit weights for the band sweep) | ~8.0 | pending | | | | |
| band2_<lo-hi> | only one 8-layer MLP band of mlp2_aw_cal at 2-bit, rest bf16 (x8 bands) | | pending | | | | |
| mlp4_aw_cal | all MLP LUT4 (quality-ceiling reference, not a deployment candidate) | ~12 | not queued | | | | |
| mlp2x64_aw_cal | all MLP vector 2x64 (reference) | ~10.5 | not queued | | | | |

Reproduce the ablations (M3U): `EXPORT_DIR=/path/to/data/vq27b/runs/export/full_mix25_mixer4_head4 TAG=abl_mlp
PARTS=mlp python qwen38_kl.py eval` (see `/path/to/data/vq27b/run_ablation.sh`). The size-neutral series
(low-rank evals, calibration data, mix25_aw_cal, mlp2_aw_cal, band sweep) is `/path/to/data/vq27b/run_quality2.sh`;
its env block for the exports:
```
MIXER="LUT4 per-tensor + pcs" KV_FMT="INT8 per-channel" HEAD="LUT4 per-tensor + pcs" BASELINE=0 AW=1 NCAL=48 \
CAL_MIX="/path/to/data/vq27b/calib_chat_ids.npy:16,/path/to/data/vq27b/calib_pi_ids.npy:16" \
PLAN=/path/to/data/vq27b/plan_optiq_top48.json TAG=mix25_aw_cal python -X faulthandler -u qwen38_gptq_27b.py
FORMAT="vector 2x16 + pcs" TAG=mlp2_aw_cal ...                       # (same env, no PLAN)
EXPORT_DIR=.../runs/export/mlp2_aw_cal TAG=band2_24-31 PARTS=mlp QLAYERS=24-31 python qwen38_kl.py eval
```
Calibration rows: `calib_pi_ids.npy` (48 x 1024 tokens, 25 pi sessions rendered with the Qwen template + pi tool
schemas; built on the M6 with `MAX_ROWS=48 python qwen38_calib_pi.py`, copied to the M3U), `calib_chat_ids.npy`
(M3U, `CAL_OUT=... python qwen38_calib_gen.py`: bf16 answers to 32 prompts that are not in the KL set).

Weight SNR of LUT splits (RTN, bf16 checkpoint; M6: `python qwen38_lut_split_snr.py`, ~5 min):

| matrix | one per-tensor LUT4 + pcs | split | per-group-8 (anemll) |
| --- | ---: | ---: | ---: |
| in_proj_qkv L0 / L12 / L30 / L48 / L62 | 19.76 / 19.64 / 19.28 / 19.34 / 18.54 dB | q/k/v LUTs: 19.76 / 19.63 / 19.27 / 19.31 / 18.62 | L30: 19.38 |
| in_proj_qkv L30, 5 / 20 / 80 row groups | 19.28 | 19.28 / 19.29 / 19.31 | 19.38 |
| mlp.down_proj L40, 10 / 40 row groups | 19.77 | 19.76 / 19.76 | 19.83 (640 groups) |

GPTQ log SNRs of the deployed export (`run_full_mix25.log`): 2-bit MLP gate/up ~7.9 dB, down 6.3-7.3 dB, MLP output
11-18 dB; 4-bit mixers 20-47 dB; head 28 dB (held-out logits). WikiText ppl: all-MLP 2-bit +23%, all-MLP LUT4 +1.3%.

## 5. Findings

- The MLP is the main error (0.48 of 0.62), concentrated in the later layers: layers 0-23 alone cost only 0.066.
  The deployed plan spent its 17 LUT4 layers mostly on layers 0-9 because it was allocated from **WikiText**
  perplexity sweeps; in-domain (chat / thinking / code / agentic) sensitivity differs.
- DeltaNet projections at 4-bit per-tensor cost KL 0.17 on their own (likely error accumulation in the recurrent
  state). Splitting the in_proj_qkv LUT into q / k / v LUTs does not change the weight SNR (+-0.08 dB): after the
  per-channel scale every row block has nearly the same value distribution.
- **The harmful part of the DeltaNet quantization error is low-rank**: adding the rank-64 SVD of (W_bf16 - W_q) to
  the DeltaNet projections (0.23 GB fp16 for all 48 layers, ~+1.7 ms per verify) takes their KL from 0.170 to 0.024
  (top-1 87.9% -> 95.0%, ppl 2.20 -> 2.137 vs bf16 2.136). Probably a few outlier input directions that a 16-level
  LUT cannot represent. By far the best KL per GB found so far.
- ANE-only error on top of the quantization (found in the Core AI port work, COREAI_PORT_NOTES.md): the ANE's fp16
  `softplus` returns 0 above ~11, so DeltaNet decay gates with a + dt >= ~11 become 0 (no decay) in the deployed
  Core ML chunks (layer 0: 81 of 384 head-token values). The M3U KL numbers (PyTorch) do not include it; the next
  ANE build must use the stable softplus (relu(x) + log(1 + exp(-|x|))) in qwen38_ane_chunk.py.
- Block reconstruction with low-rank factors (LR_RANK, SVD-initialized, trained with the LUT values and scales):
  M6 test, rank 16 on DeltaNet + attention, layers 0-3: held-out stream error starts at 0.012-0.034 instead of
  0.067-0.149 (table-only) and ends at 0.009-0.031; layer 0 out_proj weight error 0.38 -> 0.13.
- The MLP damage is in layers 24-63 (0.453 of 0.477): the deployed plan put 9 of its 17 LUT4 MLP layers in 0-9.
- Block-wise reconstruction (qwen38_blockrecon.py, LUT values + per-channel scales only, indices fixed) reduces the
  per-layer held-out stream error by 14-44% after only 8 steps (M6 CPU test, layers 0-3, 256-token rows); the
  functional quantized layer matches the export's dequantized weights to <1e-6.
- More LUTs per matrix buy almost nothing: 80 row groups +0.03 dB, anemll per-group-8 +0.06-0.10 dB over one
  per-tensor LUT4 + pcs. The per-channel scale already does the job a grouped LUT would.
- The error is heavy-tailed (median 0.17, p99 5.5): a few positions derail completely, consistent with the observed
  sudden loops ("grid color grid ...").
- Bytes alone do not fix it: optiq_top48_mix at 2.3x the size only reaches 0.496.
- GPTQ was calibrated on WikiText only (32 x 1024 tokens); the LUT codebooks and per-channel scales were fit on the
  raw weights (no activation importance); only GPTQ's rounding used H.
- Loops are a target-quality problem: speculative sampling in the server is exact (CPU statistical test,
  `qwen38_spec_unit_test.py`), and plain decoding loops too.
- Constraint: any increase in quantized size slows decode (bandwidth bound), so fixes are ranked by KL gained per ms.

## 6. Dead ends and non-options

- 2-D VQ at 3.5 bits (2x128): 7-bit indices are not expressible in MIL; 2x256 exceeds the 256-value LUT limit.
- VQ at ~4-bit cost: 2x64 carries 3 bits but reads like 8-bit indices per pair (~LUT4 bandwidth) at ~5 dB lower SNR
  than LUT4. Vector LUTs pay off only at <= 2 bits (2x16 beats scalar 2-bit by 0.4-1 dB).
- FP8 / INT8 weights for decode: 2x the bytes of LUT4, so slower per token; useful only for accuracy on small,
  sensitive matrices or for compute-bound prefill. Core AI has no FP8/INT8 LUT values.
- Per-group vector LUTs, Cin-grouped scales (e.g. g128 blockwise, Bonsai-style): fall off the ANE.
- Reduced drafter heads (32K-64K rows): lose more acceptance than they save (DFLASH2_ANE_PLAN.md).
- Core ML multifunction dedup does not reduce ANE memory (see FEEDBACK_ANE_MULTIFUNCTION_MEMORY.md).
- Separate q / k / v LUTs for in_proj_qkv, and grouped scalar LUTs (5-1280 groups): <= 0.1 dB weight SNR
  (`qwen38_lut_split_snr.py`) for ~+20% ANE time per matrix (section 2). Dead end.
- Closed-form rank-64 correction of the MLP error: KL 0.477 -> 0.453 (-5%) for +0.55 GB. The 2-bit MLP error is not
  low-rank; bits or better placement (in-domain plan, block reconstruction) are the levers there.
- `</think>` hypothesis (draft runs close the thinking less often): refuted by the fixed-build batch (seed 3 draft
  closes it; block rows are not sharper).

## 7. Next (size-neutral first)

Rank by KL gained per ms of verify; report size delta (GB), verify delta (ms), KL for each. Queue on the M3U:
`/path/to/data/vq27b/run_quality3.sh` (log `run_quality3.out`); block reconstruction runs separately
(`br_full_mix25.log`).
1. mix25_aw_cal: same plan and size, in-domain calibration + AW (0 GB, 0 ms). Then + rank-64 correction on DeltaNet
   + attention (`LR_RANK=64 LR_PARTS=gdn,attn`, +0.28 GB, ~+2 ms).
2. full_mix25_br: block reconstruction of the deployed export (0 GB, 0 ms), then of the best export. Next variant:
   train the low-rank factors jointly (SVD-initialized, LoRA-style) inside the block reconstruction.
3. mlp2_aw_cal + in-domain band sweep: KL of each 8-layer MLP band at 2-bit; then a new plan with the same number of
   LUT4 matrices (48 in plan_optiq_top48) placed by in-domain damage (0 GB, 0 ms):
   `python qwen38_plan_indomain.py --out /path/to/data/vq27b/plan_indomain.json` (M3U; band KLs split among a band's
   layers by the WikiText per-layer sweep), then GPTQ `PLAN=.../plan_indomain.json TAG=mix25_indomain_aw_cal` with
   the section-4 env, KL, block reconstruction (+ low-rank on DeltaNet / attention).
4. MLP low-rank (abl_mlp_lr64): measured, 0.477 -> 0.453 for +0.55 GB (~+4 ms): not worth it; the MLP error is
   full-rank (2-bit rounding noise), unlike the DeltaNet error. Dropped (section 6).
5. DeltaNet at 8-bit: superseded by the low-rank correction (0.23 GB instead of +1.26 GB).
6. Quality-ceiling references when the M3U is free: mlp4_aw_cal (all LUT4, +~3 GB, ~+19 ms), mlp2x64_aw_cal.
7. ANE: low-rank branch in the chunk builder - DONE (2026-09-27): `lut_linear` adds `conv(conv(x, b), a)` in fp16
   when the quant tuple carries factors; `as_quant` reads `{key}.lr_a / lr_b` and, for MLP matrices in the online
   basis, moves b to the rotated input basis (b @ M, exact to 1e-15). Still to verify: a one-chunk ANE build with
   factors vs PyTorch (placement + parity). Grouped-LUT speed: measured, +19-21% (section 2), dropped.

## 8. Speculative decoding and repetition

Draft vs plain on the M6 ANE target (16K v4 build + DFlash2 LUT4 drafter), same prompt / sampling / seeds, cold
state per run, DRY 0.8 (allowed 8), loop guard 6, then a teacher-forced check of 8-row verify blocks vs 1-row steps
(KL, top-1, entropy per row in the block):
```
cd ane-vector-lut/scripts && ANE_OUT=~/Models/vq27b/ane4 CTX=16384 SEEDS=1,2,3 python -u qwen38_spec_ab.py
# more seeds: SEEDS=4,5,6 TF=0; non-thinking: THINK=0 TEMP=0.7 MAX=2000 PROMPT="what is apple neural engine"
```
Batch 1 (2026-09-27 00:05, buggy-softplus build `~/Models/vq27b/ane4`, tetris, thinking on, temp 1.0, MAX 4000;
log `~/Models/vq27b/tests/qwen38_spec_ab2.log`, texts `~/Models/dflash2/spec_ab2/`):

| seed | plain: tokens, finish, rep4, </think>, </html> | draft: tokens, finish, rep4, tok/call, </think>, </html> |
| --- | --- | --- |
| 1 | 2868, stop, 0.310, yes, yes | 4000, length, 0.691, 2.77, no, no |
| 2 | 4000, length, 0.567, yes, no | 4000, length, 0.686, 2.34, no, yes |
| 3 | 4000, length, 0.524, no, yes | 2093, stop, 0.127, 2.83, yes, yes |

Each side finished once in three runs; mean rep4 plain 0.47, draft 0.50: no systematic difference at n=3 (the
direction flips between seeds). Speed on this build at 16K: plain 8.2-8.6 tok/s, draft 17.9-20.2 tok/s.
(The teacher-forced part of this run crashed on an off-by-one, fixed in the script.) Remaining batches moved to
the fixed build.

Batch 2 (2026-09-27 01:01, FIXED build `~/Models/vq27b/ane5`, same settings; note `MODEL_DIR` must point at the
build, `ANE_OUT` alone is overridden by the server engine; log `~/Models/vq27b/tests/qwen38_spec_ab5.log`, texts
`~/Models/dflash2/spec_ab5/`):
```
MODEL_DIR=~/Models/vq27b/ane5/full_mix25_mixer4_head4 ANE_OUT=~/Models/vq27b/ane5 CTX=16384 SEEDS=1,2,3 \
OUTD=~/Models/dflash2/spec_ab5 python -u qwen38_spec_ab.py
```

| seed | plain: tokens, finish, rep4, </think>, </html> | draft: tokens, finish, rep4, tok/call, </think>, </html> |
| --- | --- | --- |
| 1 | 931, stop (loop on "let me stop overthinking"), 0.363, no, no | 4000, length, 0.583, 3.39, no, yes |
| 2 | 4000, length, 0.713, no, no | 4000, length, 0.583, 2.74, no, no |
| 3 | 4000, length, 0.509, no, yes | 2737, stop, 0.429, 3.41, yes, yes |

plain: finished 1/3, mean rep4 0.529, 8.8 tok/s; draft: finished 1/3, mean rep4 0.531, 22.6 tok/s (2.6x).

Teacher-forced 8-row verify block vs 1-row step (931 positions of the seed-1 plain output, fixed build):

| row in block | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KL | 0.010 | 0.015 | 0.016 | 0.022 | 0.020 | 0.015 | 0.032 | 0.021 | 0.019 |
| top-1 match | 98.3% | 98.3% | 95.7% | 97.4% | 93.1% | 94.8% | 95.7% | 95.7% | 96.1% |
| H(block) - H(step) | +0.008 | +0.030 | +0.015 | -0.001 | -0.005 | +0.008 | +0.006 | +0.012 | +0.009 (se 0.004) |

**Conclusion: speculative decoding does not change repetition or looping.** Over 6 draft/plain pairs on two builds,
both finish 2/6 and have the same mean rep4 (0.50-0.53); the verify block is not sharper than the step (entropy
+0.009 nats, i.e. marginally flatter), and the sampling is exact (`qwen38_spec_unit_test.py`). Block vs step differ
by KL ~0.02 (fp16 lazy-commit of 8 DeltaNet rows at once vs 1). The loops come from the quantized target. The first attempt (2026-09-26 23:30) is
void: the server's prompt cache treated an empty cache as "continue", so runs 2-6 decoded on top of the previous
run's state (fixed in qwen38_server.py; the script now also resets the model and drafter per run).

## 9. Numerics on the ANE (not visible in the M3U KL)

- **DeltaNet ~50% wrong on the ANE in every build up to ane7 (found 2026-09-27): see ANE_DELTANET_NUMERICS.md.**
  The ANE's native silu has ~1e-3 absolute error near 0 (DeltaNet conv, MLP gate) and q . S is fp16-subnormal.
  Fixed (tanh-form silu, q / v scaling with a matching gated-norm eps, MLP down-input scaling); ane7f ANE trace ppl
  2.420 = the PyTorch sim (2.428). The "~0.1-nat ANE gap" below was this bug, not general fp16 numerics.
- **fp16 softplus overflow** (found 2026-09-27 in the Core AI port work; COREAI_PORT_NOTES.md): the ANE's
  `mb.softplus` returns 0 for every input >= ~11 (exp overflow). The DeltaNet decay g = softplus(a + dt) * -exp(A_log)
  is then 0 (no decay) instead of strongly negative: ~20% of (head, token) values in layer 0; heads that should nearly
  reset their state every token never decay. Fix in qwen38_ane_chunk.py (all 5 DeltaNet paths):
  softplus(x) = relu(x) + log(1 + exp(-|x|)) (max error 0.002 on the ANE). Fixed build:
  `~/Models/vq27b/ane5/full_mix25_mixer4_head4` (16K, v4 layout, 16 chunks of 4 layers).
- Before / after (M6 ANE, WikiText test, 4096 tokens at CTX 16384, 8-row blocks):
  `ANE_OUT=~/Models/vq27b/ane{4,5} CTX=16384 MODE=block N=4096 python qwen38_ane_ppl.py`
  | build | ppl | first / second half |
  | --- | ---: | ---: |
  | ane4 (buggy softplus) | 7.157 | 6.087 / 8.416 |
  | ane5 (fixed) | 7.116 | 6.042 / 8.382 |
  -0.6% perplexity on WikiText. The ANE model tracks the PyTorch quantized model (GPTQ log: quantized WikiText ppl
  7.19 over 16 x 1024 tokens), so apart from this bug the ANE numerics add little; the quantization is the gap.
- Draft vs plain on the fixed build: section 8, batch 2 (loops persist with the fix: the quantization is the cause).


## ANE validation of the in-domain calibration + low-rank export (2026-09-27, M6)

In-domain trace perplexity on the ANE (64 bf16 chat answers, 40K tokens, same definition as `qwen38_kl.py eval`;
bf16 2.136). Builds: v4 layout (16 x 4-layer chunks, T=8, 16K), softplus fix. Script `scripts/qwen38_ane_trace_ppl.py`
(`ANE_OUT=~/Models/vq27b/<build> EXPORT_DIR=~/Models/vq27b/export/<export> CTX=16384 python qwen38_ane_trace_ppl.py`).

| build | export | PyTorch ppl | ANE ppl | ANE gap (nats) |
| --- | --- | ---: | ---: | ---: |
| ane5 | full_mix25_mixer4_head4 (deployed quant) | 3.092 | 3.416 | +0.100 |
| ane6 | mix25_aw_cal_lr64mix (in-domain calibration + AW + rank-64 mixer factors, +0.29 GiB) | 2.533 | 2.833 | +0.112 |

- The low-rank factors work on the ANE (fp16 `a @ (b @ x)` convs): the gain carries over (-0.188 nats on the ANE vs
  -0.199 in PyTorch).
- The ANE adds ~0.10-0.11 nats for both exports: a general fp16-numerics gap (to localize with a per-layer ANE vs torch
  divergence analysis; candidates: DeltaNet fp16 state / chunked math, attention softmax, RMSNorm pre-scale, head).
- WikiText ppl on the ANE (4096 tokens): ane5 7.116, ane6 7.419 - the in-domain calibration trades WikiText for chat /
  agentic quality (the M3U GPTQ log shows the same: 7.19 -> 7.59); judge exports on the in-domain trace.
- Factor export: `scripts/qwen38_lowrank_export.py` (M3U, ~2 min). Build: `CHUNK_PLAN="0-3,...,60-63"
  EXPORT_DIR=~/Models/vq27b/export/mix25_aw_cal_lr64mix ANE_OUT=~/Models/vq27b/ane6 CTX=16384 python qwen38_ane_model.py
  build_v3` (the head builder needed `lut, idx, sc = qq[:3]` after as_quant gained the low-rank slot).


## In-domain MLP band sweep and re-plan (2026-09-27, M3U, run_quality5.sh)

All-MLP 2-bit export `mlp2_aw_cal` (vector 2x16 + pcs, in-domain calibration, AW=1; WikiText ppl 7.827), then one KL
eval per band with only that band's MLP quantized (rest bf16):
`EXPORT_DIR=/path/to/data/vq27b/runs/export/mlp2_aw_cal PARTS=mlp QLAYERS=<lo-hi> TAG=band2_<lo-hi> python qwen38_kl.py eval`.

| MLP band at 2-bit | mean KL | median | p99 | top-1 | ppl |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0-7 | 0.0077 | 0.0007 | 0.086 | 97.1% | 2.146 |
| 8-15 | 0.0079 | 0.0008 | 0.079 | 97.0% | 2.145 |
| 16-23 | 0.0188 | 0.0019 | 0.203 | 95.4% | 2.177 |
| 24-31 | 0.0448 | 0.0048 | 0.516 | 93.0% | 2.236 |
| 32-39 | 0.0541 | 0.0053 | 0.555 | 92.9% | 2.169 |
| 40-47 | 0.0577 | 0.0072 | 0.687 | 92.3% | 2.151 |
| 48-55 | 0.0519 | 0.0075 | 0.542 | 92.5% | 2.208 |
| 56-63 | 0.0693 | 0.0081 | 0.754 | 91.3% | 2.235 |
| sum | 0.312 | | | | |

(The quality4 band numbers from a partial export - ENOSPC - were invalid: bands 8-63 returned 0.0.)

`python qwen38_plan_indomain.py --kl-dir /path/to/data/vq27b/kl --plan plan_optiq_top48.json --out plan_indomain.json`:
same budget (48 LUT4 matrices), new LUT4 MLP layers 24, 25, 28, 35, 36, 40, 41, 42, 44, 45, 53, 54, 58, 60, 61, 63
(old WikiText plan: 0-6, 8, 9, 28, 34-40). Estimated MLP KL removed by the 4-bit layers: new 0.173 vs old 0.060.
Export `mix25in_aw_cal` (same size as the deployed export; GPTQ with the new plan, same calibration + AW):
KL 0.2256 (median 0.0448, p99 2.470, top-1 84.1%, ppl 2.411; WikiText ppl 7.107) vs mix25_aw_cal 0.3097 - beats even
mix25_aw_cal_lr64mix (0.269) without factors. + rank-64 mixer factors: pending.
Reproduce (M3U): `PLAN=/path/to/data/vq27b/plan_indomain.json FORMAT="vector 2x16 + pcs" MIXER="LUT4 per-tensor + pcs"
KV_FMT="INT8 per-channel" HEAD="LUT4 per-tensor + pcs" BASELINE=0 AW=1 NCAL=48 CAL_MIX="calib_chat_ids.npy:16,calib_pi_ids.npy:16"
TAG=mix25in_aw_cal python qwen38_gptq_27b.py` (see run_quality5.sh).
