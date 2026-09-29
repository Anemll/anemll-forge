#!/bin/zsh
while pgrep -f "run_queue.sh|run_queue2.sh" >/dev/null; do sleep 60; done
cd ~/SourceRelease/GITHUB/ML_playground/ane-vector-lut/scripts
PYTHONWARNINGS=ignore FORMAT="vector 4x64 + pcs" BASELINE=0 ~/venvs/vq27b/bin/python -X faulthandler -u qwen38_gptq_27b.py > /Volumes/SN8100/vq27b/run_v4x64_pcs.log 2>&1
