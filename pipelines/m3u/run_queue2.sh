#!/bin/zsh
while pgrep -f run_queue.sh >/dev/null; do sleep 60; done
cd ~/SourceRelease/GITHUB/ML_playground/ane-vector-lut/scripts
PYTHONWARNINGS=ignore PLAN=/Volumes/SN8100/vq27b/plan_optiq_top48.json TAG=optiq_top48_mix BASELINE=0 ~/venvs/vq27b/bin/python -X faulthandler -u qwen38_gptq_27b.py > /Volumes/SN8100/vq27b/run_optiq_top48.log 2>&1
