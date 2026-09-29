# M3U historical pipeline archive

These 28 shell scripts were copied with the owner's permission from the M3U's `/Volumes/SN8100/vq27b/` on 2026-09-29. Script contents are retained byte-for-byte; [manifest.json](manifest.json) records source hashes verified over SSH. They were **not executed** during import.

This is the execution history, not the portable user interface. The files retain original mount points, repository paths, venv paths, process matching, hold-file behavior, output redirection and artifact renames. Some guards send STOP/CONT signals to other processes; queue helpers wait for named jobs. Read and adapt them before execution. Use the current [workflow](../../docs/WORKFLOW.md) for the initial port.

## Deployed mixer/MLP trade sequence

1. `run_mixr.sh` builds the mixer-two-bit calibration export, measures mixer LUT4 + rank-64 quality, measures eight mixer bands at two bits + rank-64, then checks the all-mixer interaction.
2. `run_mixr_final.sh` calls `qwen38_plan_mixr.py`, quantizes `mix25in_mixr`, adds plain rank-64 factors to DeltaNet/attention, and evaluates the resulting `mix25in_mixr_lr64mix` export. Its historical acceptance rule is candidate KL <= 0.97 × the stored baseline; failing artifacts are renamed `_rejected`. That rule is a workflow condition, not proof that a particular run passed.
3. `run_mixr_drafter.sh` recalibrates the optional DFlash2 drafter against this target, uses the matching target head, and compares two-fold replay before export. Reusing a drafter/head from a different target changes the acceptance experiment.

The planner trades less-sensitive mixer-band bytes for more four-bit MLP matrices at approximately constant total size. It uses empirical constants (`R4`, `REAL`) and measured KL inputs; it is an experiment-specific heuristic, not an architecture-independent optimizer.

## Other families

- `run_sweeps.sh`, `run_plans.sh`, `run_queue*.sh`: original sensitivity/format scheduling.
- `run_kl.sh`, `run_ablation.sh`, `run_quality*.sh`, `run_mlp4.sh`, `run_wikippl.sh`: reference, ablation, calibration and quality comparisons.
- `run_lr_aw.sh`, `run_lrdyn.sh`: activation-weighted and dynamic-rank correction experiments.
- `run_dflash_now.sh`, `run_ane7_drafter.sh`, `run_after_q5.sh`: earlier drafter/calibration sequences.
- `dflash_guard*.sh`, `trigger_mixr_final.sh`: machine/job coordination, not model algorithms.

## Still-needed inputs

The code and exact commands are now present. This archive does not include the original checkpoint, bit-plan outputs, KL JSON measurements/reference trace, calibration token arrays, generated exports, drafter assets, per-run logs, or auxiliary `.env`/hold files. Source availability alone is not complete artifact reproducibility. Retain or publish suitable hashes and data provenance before making historical results reproducibility claims.
