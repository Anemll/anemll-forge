#!/bin/zsh
source "${0:A:h}/../common.zsh" || exit $?
while pgrep -f "run_queue.sh|run_queue2.sh" >/dev/null; do sleep 60; done
cd "$FORGE_ROOT/scripts" || exit 1
PYTHONWARNINGS=ignore FORMAT="vector 4x64 + pcs" BASELINE=0 "$FORGE_PYTHON" -X faulthandler -u qwen38_gptq_27b.py > $FORGE_WORK_DIR/run_v4x64_pcs.log 2>&1
