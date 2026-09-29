#!/bin/zsh
cd ~/SourceRelease/GITHUB/ML_playground/ane-vector-lut/scripts
export PYTHONWARNINGS=ignore NEVAL=4
PY=~/venvs/vq27b/bin/python
L=/Volumes/SN8100/vq27b
SWEEP=mlp SWEEP_FMT="vector 2x16 + pcs" BASIS=online $PY -X faulthandler -u qwen38_gptq_27b.py > $L/sweep_mlp.log 2>&1
SWEEP=mixer SWEEP_FMT="vector 2x16 + pcs" $PY -X faulthandler -u qwen38_gptq_27b.py > $L/sweep_mixer.log 2>&1
