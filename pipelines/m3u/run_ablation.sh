#!/bin/zsh
# which quantized part of full_mix25_mixer4_head4 costs the KL: one part at a time, the rest bf16
cd ~/SourceRelease/GITHUB/ML_playground/ane-vector-lut/scripts
export PYTHONWARNINGS=ignore TRACE=/Volumes/SN8100/vq27b/kl
PY=~/venvs/vq27b/bin/python
L=/Volumes/SN8100/vq27b
E=$L/runs/export/full_mix25_mixer4_head4
for spec in "mlp:mlp:" "gdn:gdn:" "attn:attn:" "head:head:" "mlp_0_23:mlp:0-23" "mlp_24_63:mlp:24-63"; do
  tag=abl_${spec%%:*}; rest=${spec#*:}; parts=${rest%%:*}; ql=${rest#*:}
  echo "$(date +%H:%M:%S) $tag parts=$parts layers=${ql:-all}"
  EXPORT_DIR=$E TAG=$tag PARTS=$parts QLAYERS=$ql $PY -u qwen38_kl.py eval > $L/kl_eval_$tag.log 2>&1
  tail -1 $L/kl_eval_$tag.log
done
echo "$(date +%H:%M:%S) done"
