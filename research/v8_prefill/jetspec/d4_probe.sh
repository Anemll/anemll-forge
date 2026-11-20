#!/bin/zsh
# D4 probe: build the per-form packages (one process), then compile + check each form's dir separately
set -u
O=/Volumes/SSD4TB/anemll-forge-research/tree20/d4probe
R=${RD:-/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/20nov/d4/probe}
W=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/v8; PW=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power
WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh
SPY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
BPY=/Users/anemll/SourceRelease/GITHUB/ML_playground/ane-vector-lut/coreai/.venv/bin/python
FORMS=${FORMS:-corr,corrc,corrd,corrx,corrx2}
mkdir -p $O $R
lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1 && { echo "forge listens on 8765: stop it first"; exit 1; }
echo "$(date +%T) build forms $FORMS" | tee -a $R/run.log
echo "STATE=working ITEM=D4-corrx-probe NOTE=building per-form probe packages ($FORMS)" > /tmp/m6_yolo_status.txt
( cd $W && env KV_CACHE_DTYPE=kv8 KV_KEYS_T=1 ATT_INT8MM=s8,s8b,sm8,pvf8 ATT_S8_UNIT=0.25 ATT_S8B_UNIT=0.25 ATT_PF8_UNIT=0.015625 \
  ATT_BLOCK=2048 ATT_BLOCK_PREFILL=4096 ATT_BLOCK_LONG=4096 ATT_LONG_FROM=131072 TPS=64 TPS_LONG=32 PREFILL_LONG_FROM=98304 \
  ATT_BLOCK_PREFILL_LONG=8192 GDN_FAST=1 QCONV_INT8=0 KV_SEG=8192,8192,16384,32768,32768 V8_SPLIT=8192,9216 V8_ATTPROJ=1 \
  EXPORT_DIR=/Users/anemll/Models/vq27b/export/release_vq3pA_mixh_s600_k1_mat MODEL=/Volumes/SSD4TB/Qwen3.8-27B \
  $BPY -u build_tree_corrx_probe.py $O $FORMS ) > $R/build.log 2>&1 & BP=$!
$WD $BP 1024 $R/build.watchdog.log >/dev/null 2>&1 &
wait $BP || { echo "build failed"; tail -20 $R/build.log; exit 1; }
grep "MB (" $R/build.log | tee -a $R/run.log
for f in ${(s:,:)FORMS}; do
  echo "STATE=working ITEM=D4-corrx-probe NOTE=compile + check form $f" > /tmp/m6_yolo_status.txt
  echo "$(date +%T) compile $f" | tee -a $R/run.log
  MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 $SPY $W/journal_compile.py $O/$f > $R/compile_$f.log 2>&1 & CP=$!
  $WD $CP 1024 $R/compile_$f.watchdog.log >/dev/null 2>&1 &
  if wait $CP; then
    grep "done in" $R/compile_$f.log | tail -1 | tee -a $R/run.log
    MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 $SPY $PW/d4_probe_check.py $O/$f $R/check_$f.json 2>&1 | grep -v -i warn | tee -a $R/run.log
  else
    echo "$(date +%T) $f: COMPILE FAILED ($(grep -o 'MLIR[^\"]*failed[^\"]*' $R/compile_$f.log | head -1))" | tee -a $R/run.log
  fi
done
echo ALLDONE | tee -a $R/run.log
