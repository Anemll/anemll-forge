#!/bin/zsh
# Queue 11 (19 Nov 2026): short-context attention placement at 2K / 4K / 8K / 16K, fresh processes in two rounds:
#   v8   the served V8 prefill (attention on the GPU: FP8 x + QKE=tg, libgpu_v8_f8q.dylib) - v8_trace.py
#   ane  ANE-only (the bridge's p64 prefill entries: attention on the ANE, no GPU engine) - ane_place.py (+ the queue 13
#        cost model: 8-row verify and 64-row prefill call ms at each context)
#   apj  V8 with the attention projections (q / k / v / o) on the ANE (ffnq2 / offn2 entries, attproj_from 0), the
#        attention core on the GPU - v8_trace.py
# Last-token logits against v8 a. Forge stays down (the machine is free for benches).
P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe; PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
D=$P/unit/power; R=/Volumes/SSD4TB/anemll-forge-research/contention/place; mkdir -p $R; WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh
LENS=2048,4096,8192,16384
lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1 && { echo "$(date +%T) stopping forge on 8765"; PORT=8765 $P/srv_stop.sh; sleep 30; }
SRV=(MODEL=/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model SILU=tanh MLP_SILU=tanh GDN_SQ=16 GDN_SV=64 MLP_DS=1 MLP_DS_DYN=0 V3_KV_IN=1 V3_PREFILL=0
     EMBED_NPY=/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model/embed_tokens_fp16.npy RUNTIME=coreai MPSGRAPH_ANE_BONDED_COMPILE_MODE=2)
for r in a b; do
  for A in v8 ane apj; do
    T=${A}_$r; mkdir -p $R/tr_$T; echo "$(date +%T) $T (swap $(sysctl -n vm.swapusage | awk '{print $6}'))"
    case $A in
      v8)  env V8_DYLIB=libgpu_v8_f8q.dylib F8=x QKE=tg EXPECT_F8=x EXPECT_QK=attn_qk8isk_tg $D/v8_trace.sh $R/tr_$T $LENS ;;
      apj) env V8_DYLIB=libgpu_v8_f8q.dylib F8=x QKE=tg EXPECT_F8=x EXPECT_QK=attn_qk8isk_tg V8_ATTPROJ_FROM=0 $D/v8_trace.sh $R/tr_$T $LENS ;;
      ane) env $SRV $PY -u $D/ane_place.py $R/tr_$T $LENS > $R/tr_$T.log 2>&1 & G=$!; $WD $G 1536 $R/tr_$T.watchdog.log >/dev/null 2>&1 & K=$!; wait $G; kill $K 2>/dev/null ;;
    esac
    grep -E "^BANNER|ABORT|Traceback|attention projections|^run" $R/tr_$T.log | cut -c1-260
    [[ $A != ane ]] && $PY $D/trace_analyze.py $R/tr_$T > /dev/null 2>$R/tr_$T/analyze.err
    sleep 30
  done
done
$PY $D/logits_cmp.py $R/tr_v8_a $R/tr_v8_b $R/tr_ane_a $R/tr_ane_b $R/tr_apj_a $R/tr_apj_b
echo "$(date +%T) all done (forge left down)"
