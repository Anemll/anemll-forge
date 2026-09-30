#!/bin/zsh
source "${0:A:h}/../common.zsh" || exit $?
while pgrep -f "run_queue.sh|run_queue2.sh" >/dev/null; do sleep 60; done
cd "$FORGE_ROOT/scripts" || exit 1
export PYTHONWARNINGS=ignore
PY="$FORGE_PYTHON"
L=$FORGE_WORK_DIR
# 6: everything quantized (MLP 2.5-bit OptiQ mix, mixers LUT4 + scale, K/V INT8, lm_head LUT4 + scale), smoke test first
if PLAN=$L/plan_optiq_top48.json MIXER="LUT4 per-tensor + pcs" HEAD="LUT4 per-tensor + pcs" BASELINE=0 NLAYERS=2 NCAL=4 NEVAL=2 BATCH=2 TAG=smoke_mixer \
     $PY -X faulthandler -u qwen38_gptq_27b.py > $L/run_smoke_mixer.log 2>&1; then
  PLAN=$L/plan_optiq_top48.json MIXER="LUT4 per-tensor + pcs" HEAD="LUT4 per-tensor + pcs" BASELINE=0 TAG=full_mix25_mixer4_head4 \
     $PY -X faulthandler -u qwen38_gptq_27b.py > $L/run_full_mix25.log 2>&1
fi
# 4: MLP vector 4x64 + scale; 5: MLP 2x16 + scale without Hadamard
FORMAT="vector 4x64 + pcs" BASELINE=0 $PY -X faulthandler -u qwen38_gptq_27b.py > $L/run_v4x64_pcs.log 2>&1
FORMAT="vector 2x16 + pcs" BASIS=plain BASELINE=0 $PY -X faulthandler -u qwen38_gptq_27b.py > $L/run_v2x16_pcs_plain.log 2>&1
