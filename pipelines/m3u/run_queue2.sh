#!/bin/zsh
source "${0:A:h}/../common.zsh" || exit $?
while pgrep -f run_queue.sh >/dev/null; do sleep 60; done
cd "$FORGE_ROOT/scripts" || exit 1
PYTHONWARNINGS=ignore PLAN=$FORGE_WORK_DIR/plan_optiq_top48.json TAG=optiq_top48_mix BASELINE=0 "$FORGE_PYTHON" -X faulthandler -u qwen38_gptq_27b.py > $FORGE_WORK_DIR/run_optiq_top48.log 2>&1
