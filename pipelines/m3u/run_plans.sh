#!/bin/zsh
source "${0:A:h}/../common.zsh" || exit $?
while pgrep -f "run_kl.sh" >/dev/null; do sleep 60; done
cd "$FORGE_ROOT/scripts" || exit 1
export PYTHONWARNINGS=ignore TRACE=$TRACE
PY="$FORGE_PYTHON"
L=$FORGE_WORK_DIR
for b in 8.0 9.9; do
  PLAN=$L/plan_sweep_${b}GB.json FORMAT="vector 2x16 + pcs" MIXER="vector 2x16 + pcs" HEAD="LUT4 per-tensor + pcs" BASELINE=0 \
    TAG=sweep_${b}GB $PY -X faulthandler -u qwen38_gptq_27b.py > $L/run_sweep_${b}GB.log 2>&1
  EXPORT_DIR=$OUT/export/sweep_${b}GB $PY -u qwen38_kl.py eval > $L/kl_eval_sweep_${b}GB.log 2>&1
done
