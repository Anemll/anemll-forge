#!/bin/zsh
source "${0:A:h}/../common.zsh" || exit $?
while pgrep -f "run_queue.sh|run_queue2.sh|run_queue3.sh" >/dev/null; do sleep 60; done
cd "$FORGE_ROOT/scripts" || exit 1
PYTHONWARNINGS=ignore FORMAT="vector 2x16 + pcs" BASIS=plain BASELINE=0 "$FORGE_PYTHON" -X faulthandler -u qwen38_gptq_27b.py > $FORGE_WORK_DIR/run_v2x16_pcs_plain.log 2>&1
