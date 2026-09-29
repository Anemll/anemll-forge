#!/bin/zsh
while pgrep -f "run_sweeps.sh" >/dev/null; do sleep 60; done
cd ~/SourceRelease/GITHUB/ML_playground/ane-vector-lut/scripts
export PYTHONWARNINGS=ignore TRACE=/Volumes/SN8100/vq27b/kl
PY=~/venvs/vq27b/bin/python
L=/Volumes/SN8100/vq27b
$PY -u qwen38_kl.py generate > $L/kl_generate.log 2>&1 || exit 1
$PY -u qwen38_kl.py reference > $L/kl_reference.log 2>&1 || exit 1
$PY -u qwen38_kl.py eval > $L/kl_eval_bf16.log 2>&1
for e in full_mix25_mixer4_head4 optiq_top48_mix; do
  EXPORT_DIR=$L/runs/export/$e $PY -u qwen38_kl.py eval > $L/kl_eval_$e.log 2>&1
done
