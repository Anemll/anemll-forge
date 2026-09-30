#!/bin/zsh
source "${0:A:h}/../common.zsh" || exit $?
while pgrep -f "run_sweeps.sh" >/dev/null; do sleep 60; done
cd "$FORGE_ROOT/scripts" || exit 1
export PYTHONWARNINGS=ignore TRACE=$TRACE
PY="$FORGE_PYTHON"
L=$FORGE_WORK_DIR
$PY -u qwen38_kl.py generate > $L/kl_generate.log 2>&1 || exit 1
$PY -u qwen38_kl.py reference > $L/kl_reference.log 2>&1 || exit 1
$PY -u qwen38_kl.py eval > $L/kl_eval_bf16.log 2>&1
for e in full_mix25_mixer4_head4 optiq_top48_mix; do
  EXPORT_DIR=$OUT/export/$e $PY -u qwen38_kl.py eval > $L/kl_eval_$e.log 2>&1
done
