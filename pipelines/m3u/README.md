# Localized M3U research pipelines

These 28 shell scripts were imported from the M3U on 2026-09-29, then localized to Forge. [manifest.json](manifest.json) retains the original source paths and SHA-256 hashes verified over SSH and records separate hashes for the localized files. Original bytes remain in Git history. Neither import nor localization executed the research jobs.

All recipes source [common.zsh](../common.zsh), discover this checkout from the script location and use its `scripts/` and local DFlash reference source. They accept configurable data and Python paths; no sibling repository or M3U mount is required. Process matching, hold-file behavior, experimental thresholds and artifact renames remain historical. Guards send STOP/CONT signals and the trigger stops named jobs; review those coordination scripts before running them on a shared machine. Use the current [workflow](../../docs/WORKFLOW.md) for the initial port.

## Configure and run from Forge

Prepare the compatible environments described in [ENVIRONMENT.md](../../docs/ENVIRONMENT.md). From the Forge root:

```zsh
export FORGE_PYTHON="$PWD/.venv/bin/python"
export FORGE_WORK_DIR="$HOME/Models/anemll-forge/experiments"
export FORGE_MODEL_DIR="$HOME/Models/Qwen3.8-27B"
export FORGE_DRAFTER_DIR="$HOME/Models/DFlash2-27B"
export FORGE_DFLASH_WORK_DIR="$FORGE_WORK_DIR/dflash2"
zsh pipelines/m3u/run_mixr_final.sh
```

Choose **one** recipe after supplying its inputs; the example final phase needs the preceding mixer-band measurements, MLP sweep, starting plan and baseline KL record. It is not a download-and-smoke command. The scripts also work when invoked by absolute path from another directory. `FORGE_ROOT` is always discovered from this checkout, not taken from a former checkout's environment.

`FORGE_MODEL_DIR` and `FORGE_DRAFTER_DIR` refer to original BF16 checkpoints for quantization/reference work, not the prepared inference bundle. The default Python is Forge's `.venv/bin/python`; set `FORGE_PYTHON` to a compatible research environment when needed. Work defaults to `~/Models/anemll-forge/experiments`, keeping logs, traces and exports outside the source tree. `WIKI`, `TRACE`, `OUT`, `KL_TRACE`, `HEAD_EXPORT` and `REF_CODE` can override individual inputs. `WIKI` needs the WikiText files/token arrays described in the workflow. Planner calls pass all measured-input paths explicitly. Nested guards and the final-phase trigger resolve their scripts from `pipelines/m3u/`, not the data directory.

## Deployed mixer/MLP trade sequence

1. `run_mixr.sh` builds the mixer-two-bit calibration export, measures mixer LUT4 + rank-64 quality, measures eight mixer bands at two bits + rank-64, then checks the all-mixer interaction.
2. `run_mixr_final.sh` calls `qwen38_plan_mixr.py`, quantizes `mix25in_mixr`, adds plain rank-64 factors to DeltaNet/attention, and evaluates the resulting `mix25in_mixr_lr64mix` export. Its historical acceptance rule is candidate KL <= 0.97 × the stored baseline; failing artifacts are renamed `_rejected`. That rule is a workflow condition, not proof that a particular run passed.
3. `run_mixr_drafter.sh` recalibrates the required release DFlash2 drafter against this target, uses the matching target head, and compares two-fold replay before export. Reusing a drafter/head from a different target changes the acceptance experiment.

The planner trades less-sensitive mixer-band bytes for more four-bit MLP matrices at approximately constant total size. It uses empirical constants (`R4`, `REAL`) and measured KL inputs; it is an experiment-specific heuristic, not an architecture-independent optimizer.

## Other families

- `run_sweeps.sh`, `run_plans.sh`, `run_queue*.sh`: original sensitivity/format scheduling.
- `run_kl.sh`, `run_ablation.sh`, `run_quality*.sh`, `run_mlp4.sh`, `run_wikippl.sh`: reference, ablation, calibration and quality comparisons.
- `run_lr_aw.sh`, `run_lrdyn.sh`: activation-weighted and dynamic-rank correction experiments.
- `run_dflash_now.sh`, `run_ane7_drafter.sh`, `run_after_q5.sh`: earlier drafter/calibration sequences.
- `dflash_guard*.sh`, `trigger_mixr_final.sh`: machine/job coordination, not model algorithms.

## Still-needed inputs

The code and localized commands are present. Original checkpoints, measured KL inputs/reference trace, calibration token arrays, generated exports, drafter weights, per-run logs and auxiliary `.env`/hold files remain external data. The recovered final bit plan is in [configs/quantization](../../configs/quantization/mix25in_mixr.json). Calibration recipes that name `calib_pi_ids.npy` require your own consented data or a documented public replacement; no private session arrays are distributed. Source availability alone is not complete artifact reproducibility. Retain or publish suitable hashes and data provenance before making historical results reproducibility claims.
