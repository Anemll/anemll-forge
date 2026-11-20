#!/bin/zsh
# Traced V8 prefill with the server's environment (v8_server_dec8.sh's env), forge must be stopped; swap watchdog.
#   v8_trace.sh OUT [N,N,...]
O=$1; shift
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
env MODEL=/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model SILU=tanh MLP_SILU=tanh GDN_SQ=16 GDN_SV=64 MLP_DS=1 MLP_DS_DYN=0 V3_KV_IN=1 V3_PREFILL=0 \
    EMBED_NPY=/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model/embed_tokens_fp16.npy RUNTIME=coreai MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 \
    FUSE=all I8=1 I8T=1 I8TV=${I8TV:-_k256} LUTC=1 I8A=1 I8KV=1 LRM=1 QK8=1 TKV=2048 \
  $PY -u /Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power/v8_trace.py $O "$@" > $O.log 2>&1 & G=$!
/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh $G 1536 $O.watchdog.log >/dev/null 2>&1 & K=$!
wait $G; rc=$?; kill $K 2>/dev/null; echo "v8_trace rc=$rc"
