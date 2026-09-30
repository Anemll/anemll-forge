#!/bin/zsh
source "${0:A:h}/../common.zsh" || exit $?
# which quantized part of full_mix25_mixer4_head4 costs the KL: one part at a time, the rest bf16
cd "$FORGE_ROOT/scripts" || exit 1
export PYTHONWARNINGS=ignore TRACE=$TRACE
PY="$FORGE_PYTHON"
L=$FORGE_WORK_DIR
E=$OUT/export/full_mix25_mixer4_head4
for spec in "mlp:mlp:" "gdn:gdn:" "attn:attn:" "head:head:" "mlp_0_23:mlp:0-23" "mlp_24_63:mlp:24-63"; do
  tag=abl_${spec%%:*}; rest=${spec#*:}; parts=${rest%%:*}; ql=${rest#*:}
  echo "$(date +%H:%M:%S) $tag parts=$parts layers=${ql:-all}"
  EXPORT_DIR=$E TAG=$tag PARTS=$parts QLAYERS=$ql $PY -u qwen38_kl.py eval > $L/kl_eval_$tag.log 2>&1
  tail -1 $L/kl_eval_$tag.log
done
echo "$(date +%H:%M:%S) done"
