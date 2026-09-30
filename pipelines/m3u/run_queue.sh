#!/bin/zsh
source "${0:A:h}/../common.zsh" || exit $?
cd "$FORGE_ROOT/scripts" || exit 1
export PYTHONWARNINGS=ignore
PY="$FORGE_PYTHON"
FORMAT="vector 2x16 + pcs" BASELINE=1 $PY -X faulthandler -u qwen38_gptq_27b.py > $FORGE_WORK_DIR/run_v2x16_pcs.log 2>&1
FORMAT="LUT4 per-tensor + pcs" BASELINE=0 $PY -X faulthandler -u qwen38_gptq_27b.py > $FORGE_WORK_DIR/run_lut4_pcs.log 2>&1
