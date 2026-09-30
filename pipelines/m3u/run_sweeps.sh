#!/bin/zsh
source "${0:A:h}/../common.zsh" || exit $?
cd "$FORGE_ROOT/scripts" || exit 1
export PYTHONWARNINGS=ignore NEVAL=4
PY="$FORGE_PYTHON"
L=$FORGE_WORK_DIR
SWEEP=mlp SWEEP_FMT="vector 2x16 + pcs" BASIS=online $PY -X faulthandler -u qwen38_gptq_27b.py > $L/sweep_mlp.log 2>&1
SWEEP=mixer SWEEP_FMT="vector 2x16 + pcs" $PY -X faulthandler -u qwen38_gptq_27b.py > $L/sweep_mixer.log 2>&1
