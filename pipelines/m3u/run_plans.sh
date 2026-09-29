#!/bin/zsh
while pgrep -f "run_kl.sh" >/dev/null; do sleep 60; done
cd ~/SourceRelease/GITHUB/ML_playground/ane-vector-lut/scripts
export PYTHONWARNINGS=ignore TRACE=/Volumes/SN8100/vq27b/kl
PY=~/venvs/vq27b/bin/python
L=/Volumes/SN8100/vq27b
for b in 8.0 9.9; do
  PLAN=$L/plan_sweep_${b}GB.json FORMAT="vector 2x16 + pcs" MIXER="vector 2x16 + pcs" HEAD="LUT4 per-tensor + pcs" BASELINE=0 \
    TAG=sweep_${b}GB $PY -X faulthandler -u qwen38_gptq_27b.py > $L/run_sweep_${b}GB.log 2>&1
  EXPORT_DIR=$L/runs/export/sweep_${b}GB $PY -u qwen38_kl.py eval > $L/kl_eval_sweep_${b}GB.log 2>&1
done
