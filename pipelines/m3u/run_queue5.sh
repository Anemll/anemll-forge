#!/bin/zsh
while pgrep -f "run_queue.sh|run_queue2.sh|run_queue3.sh|run_queue4.sh" >/dev/null; do sleep 60; done
cd ~/SourceRelease/GITHUB/ML_playground/ane-vector-lut/scripts
export PYTHONWARNINGS=ignore PY=~/venvs/vq27b/bin/python
export PLAN=/Volumes/SN8100/vq27b/plan_optiq_top48.json MIXER="LUT4 per-tensor + pcs" KV_FMT="INT8 per-channel" HEAD="LUT4 per-tensor + pcs" BASELINE=0
NLAYERS=2 NCAL=4 NEVAL=2 BATCH=2 TAG=smoke_mixer $PY -X faulthandler -u qwen38_gptq_27b.py > /Volumes/SN8100/vq27b/run_smoke_mixer.log 2>&1 || exit 1
TAG=full_mix25_mixer4_head4 $PY -X faulthandler -u qwen38_gptq_27b.py > /Volumes/SN8100/vq27b/run_full_mix25.log 2>&1
