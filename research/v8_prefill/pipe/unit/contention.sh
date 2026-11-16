#!/bin/zsh
# GPU<->ANE coupling: GPU attention@60K + DeltaNet per-call GPU time alone and under two ANE loads (compute-heavy /
# weight-light "hot", and weight-heavy ffn2_L0 = 2 FFN blocks streaming ~270 MB of weights per call); then the
# DeltaNet mixer compiled by Core AI for the GPU (mix2_L0) vs our Metal engine's pair call.
U=${0:a:h}; SH=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/v8/share; R=/Volumes/SSD4TB/anemll-forge-research
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
export MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 MODEL=/Volumes/SSD4TB/Qwen3.8-27B
HOT="$R/tiny_ane_probe/hot.aimodel t"; FFN="$R/v8_shared2/chunk_L00-03.aimodel ffn2_L0"
q() { grep -E "^(GPU|ANE) " ; }
$PY $U/gpu_steady.py 15 61440 "alone" 2>&1 | q
$PY $SH/ane_load.py ${=HOT} 10 ane "hot alone" 2>&1 | q
$PY $SH/ane_load.py ${=FFN} 10 ane "ffn2 alone" 2>&1 | q
for LD in HOT FFN; do
  sleep 5
  $PY $SH/ane_load.py ${=${(P)LD}} 35 ane "$LD under GPU" 2>&1 | q & A=$!
  sleep 12
  $PY $U/gpu_steady.py 15 61440 "with ANE $LD" 2>&1 | q
  wait $A
done
sleep 5
$PY $U/gpu_steady.py 10 61440 "alone again" 2>&1 | q
$PY $SH/ane_load.py $R/probe_mix_fresh/mix_L00-02.aimodel mix2_L0 15 gpu "Core AI GPU mixer" 2>&1 | grep -E "^GPU |Error|error" | head -3
