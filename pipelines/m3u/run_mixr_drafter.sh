#!/bin/zsh
source "${0:A:h}/../common.zsh" || exit $?
# DFlash2 drafter GPTQ recalibration for the served mixr target (M6 session, 2026-09-28). Logs: $W/<step>.log;
# touch $W/.hold_mixr to stop before the next step. Resume: rerun (finished steps are skipped by their outputs).
set -u
W=$WORK
S="$FORGE_ROOT/scripts"
PY="$FORGE_PYTHON"
EXP=$OUT/export/mix25in_mixr_lr64mix
DEQ=$WORK/deq_mix25in_mixr_lr64mix
HEADX=$OUT/export/mix25in_mixr/lm_head.safetensors
cd "$S" || exit 1
step() { [[ -f $W/.hold_mixr ]] && { echo "$(date +%T) hold file: stopping before $1"; exit 0; }; echo "$(date +%T) START $1"; }
if [[ ! -f $DEQ/lm_head.safetensors ]]; then
  step dequant; DEQ_OUT=$DEQ EXPORT_DIR=$EXP $PY dflash2_target_ref.py dequant_export > $W/dequant_mixr.log 2>&1 || { echo "dequant FAILED"; exit 1; }
fi
if [[ ! -f $W/traces_qmixr_cal.npz ]]; then
  step "simulate qmixr_cal"; TAG=qmixr_cal EXPORT_DIR=$EXP DEQ_DIR=$DEQ PROMPT_SET=calib N_PROMPTS=32 MAX_NEW=256 THREADS=16 \
    $PY dflash2_target_ref.py simulate > $W/sim_qmixr_cal.log 2>&1 || { echo "simulate FAILED"; exit 1; }
fi
if ! grep -q "^{" $W/baseline_q7_on_qmixr.log 2>/dev/null; then
  step "baseline replay (deployed q7_cal GPTQ on qmixr traces)"; TRACES=$W/traces_qmixr_cal.npz DRAFT_EXPORT=$W/drafter_lut4_gptq_q7_cal \
    HEAD_EXPORT=$HEADX MASK_SCALE=0.7 OMP_NUM_THREADS=16 $PY dflash2_ref_replay.py > $W/baseline_q7_on_qmixr.log 2>&1 || echo "baseline FAILED (continuing)"
fi
if ! grep -q "lut4_gptq" $W/quant_eval_qmixr_cal.log 2>/dev/null || ! grep -q "mean accepted" $W/quant_eval_qmixr_cal.log; then
  step "quant eval 2-fold"; VARIANTS=lut4_gptq TRACE_TAG=qmixr_cal HEAD=lut4 HEAD_EXPORT=$HEADX MASK_SCALE=0.7 THREADS=16 \
    $PY dflash2_quant.py eval > $W/quant_eval_qmixr_cal.log 2>&1 || echo "quant eval FAILED (continuing)"
fi
if [[ ! -f $W/drafter_lut4_gptq_qmixr_m07/drafter_quant.safetensors ]]; then
  step "export lut4_gptq"; VARIANT=lut4_gptq TRACE_TAG=qmixr_cal HEAD=lut4 HEAD_EXPORT=$HEADX MASK_SCALE=0.7 THREADS=16 \
    $PY dflash2_quant.py export > $W/export_qmixr_m07.log 2>&1 && mv $W/drafter_lut4_gptq $W/drafter_lut4_gptq_qmixr_m07 || { echo "export FAILED"; exit 1; }
fi
echo "$(date +%T) ALL DONE: $W/drafter_lut4_gptq_qmixr_m07"
