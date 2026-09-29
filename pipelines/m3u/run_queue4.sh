#!/bin/zsh
while pgrep -f "run_queue.sh|run_queue2.sh|run_queue3.sh" >/dev/null; do sleep 60; done
cd ~/SourceRelease/GITHUB/ML_playground/ane-vector-lut/scripts
PYTHONWARNINGS=ignore FORMAT="vector 2x16 + pcs" BASIS=plain BASELINE=0 ~/venvs/vq27b/bin/python -X faulthandler -u qwen38_gptq_27b.py > /Volumes/SN8100/vq27b/run_v2x16_pcs_plain.log 2>&1
