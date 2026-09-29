#!/bin/zsh
cd ~/SourceRelease/GITHUB/ML_playground/ane-vector-lut/scripts
export PYTHONWARNINGS=ignore
PY=~/venvs/vq27b/bin/python
FORMAT="vector 2x16 + pcs" BASELINE=1 $PY -X faulthandler -u qwen38_gptq_27b.py > /Volumes/SN8100/vq27b/run_v2x16_pcs.log 2>&1
FORMAT="LUT4 per-tensor + pcs" BASELINE=0 $PY -X faulthandler -u qwen38_gptq_27b.py > /Volumes/SN8100/vq27b/run_lut4_pcs.log 2>&1
