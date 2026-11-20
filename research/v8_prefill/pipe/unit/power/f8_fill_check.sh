#!/bin/zsh
# K8V16 fill path check (f8_fill_check.py), plain then F8=x, one fresh process each, the server's environment
# (v8_trace.sh's env), swap watchdog; forge must be stopped.   f8_fill_check.sh OUT OUT_TF
O=$1; TF=$2; PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1 && { echo "forge is up on 8765: stop it first"; exit 1; }
mkdir -p $O
for TAG in plain f8x; do
  if [[ $TAG == f8x ]]; then X=(F8=x EXPECT_F8=x); else X=(EXPECT_F8=off); fi
  env -u F8 V8_DYLIB=libgpu_v8_f8.dylib $X MODEL=/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model SILU=tanh MLP_SILU=tanh GDN_SQ=16 GDN_SV=64 MLP_DS=1 MLP_DS_DYN=0 V3_KV_IN=1 V3_PREFILL=0 \
      EMBED_NPY=/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model/embed_tokens_fp16.npy RUNTIME=coreai MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 \
      FUSE=all I8=1 I8T=1 I8TV=_k256 LUTC=1 I8A=1 I8KV=1 LRM=1 QK8=1 TKV=2048 \
    $PY -u /Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power/f8_fill_check.py $O $TAG $TF > $O/$TAG.log 2>&1 & G=$!
  /Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh $G 1536 $O/$TAG.watchdog.log >/dev/null 2>&1 & K=$!
  wait $G; rc=$?; kill $K 2>/dev/null; echo "$(date +%T) f8_fill_check $TAG rc=$rc"
  grep -E "^BANNER|^fill path|^cont|^fill|^\{|ABORT|Traceback" $O/$TAG.log | cut -c1-300
  sleep 15
done
