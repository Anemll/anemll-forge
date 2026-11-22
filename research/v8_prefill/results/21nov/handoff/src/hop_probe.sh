#!/bin/zsh
# 21 Nov 2026 (M6 ANE bot, item 1): standalone ANE <-> GPU handoff probe. Build the tiny scratch package
# (v8/build_hop_probe.py), compile it via forge.py (journal_compile.py), build libhop_gpu.dylib, wait for Pi idle
# (no 8765 connection for 60 s, last request >= 120 s ago), run hop_probe.py under the swap watchdog. Forge is never
# touched. Cleanup (Core AI entries via coreai_cache_delete.py, then the files) is the separate `clean` step.
#   zsh hop_probe.sh run | probe | clean   (probe: timing only, after a run built and compiled)
set -u
P=${0:a:h}; V=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/v8
O=/Volumes/SSD4TB/anemll-forge-research/hop_probe
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/21nov/handoff
SL=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log
WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh
SPY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
BPY=/Users/anemll/SourceRelease/GITHUB/ML_playground/ane-vector-lut/coreai/.venv/bin/python
FORGE=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge
mkdir -p $R
if [ "${1:-run}" = clean ]; then
  $SPY $FORGE/scripts/coreai_cache_delete.py --delete $O 2>&1 | tee -a $R/clean.log
  rm -rf $O && echo "$(date +%T) removed $O" | tee -a $R/clean.log
  df -h / /Volumes/SSD4TB | tail -2 | tee -a $R/clean.log
  exit 0
fi
pi_idle() {
  local i quiet=0 last now age
  for i in $(seq 1 720); do
    if [ "$(lsof -nP -iTCP:8765 -sTCP:ESTABLISHED 2>/dev/null | tail -n +2 | wc -l | tr -d ' ')" = 0 ]; then quiet=$((quiet + 5)); else quiet=0; fi
    last=$(grep -oE "^\[[0-9:]+\] request: prompt" $SL 2>/dev/null | tail -1 | cut -c2-9)
    now=$(date +%s); age=999
    [ -n "$last" ] && age=$(( now - $(date -j -f "%Y-%m-%d %H:%M:%S" "$(date +%Y-%m-%d) $last" +%s) ))
    [ $age -lt 0 ] && age=999
    [ $quiet -ge 60 ] && [ $age -ge 120 ] && return 0
    sleep 5
  done
  return 1
}
if [ "${1:-run}" = probe ]; then echo "$(date +%T) probe only (build + compile done)" | tee -a $R/run.log; else
echo "$(date +%T) build" | tee $R/run.log
( cd $V && env KV_CACHE_DTYPE=kv8 KV_KEYS_T=1 ATT_INT8MM=s8,s8b,sm8,pvf8 ATT_S8_UNIT=0.25 ATT_S8B_UNIT=0.25 ATT_PF8_UNIT=0.015625 \
  ATT_BLOCK=2048 ATT_BLOCK_PREFILL=4096 ATT_BLOCK_LONG=4096 ATT_LONG_FROM=131072 TPS=64 TPS_LONG=32 PREFILL_LONG_FROM=98304 \
  ATT_BLOCK_PREFILL_LONG=8192 GDN_FAST=1 QCONV_INT8=0 KV_SEG=8192,8192,16384,32768,32768 V8_SPLIT=8192,9216 \
  EXPORT_DIR=/Users/anemll/Models/vq27b/export/release_vq3pA_mixh_s600_k1_mat MODEL=/Volumes/SSD4TB/Qwen3.8-27B \
  $BPY -u build_hop_probe.py $O ) > $R/build.log 2>&1 & BP=$!
$WD $BP 1024 $R/build.watchdog.log > /dev/null 2>&1 &
wait $BP || { echo "build failed"; tail -20 $R/build.log; exit 1; }
tail -1 $R/build.log | tee -a $R/run.log
MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 $SPY $V/journal_compile.py $O > $R/compile.log 2>&1 || { echo "compile failed"; tail -20 $R/compile.log; exit 1; }
grep "done in" $R/compile.log | tail -1 | tee -a $R/run.log
( cd $P && swiftc -O -emit-library -o libhop_gpu.dylib hop_gpu.swift -framework Metal -framework QuartzCore ) 2>&1 | tail -3 | tee -a $R/run.log
fi
echo "$(date +%T) waiting for Pi idle" | tee -a $R/run.log
pi_idle || { echo "$(date +%T) Pi not idle for 60 min: not timed" | tee -a $R/run.log; exit 2; }
echo "$(date +%T) Pi idle: probe" | tee -a $R/run.log
MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 $SPY -u $P/hop_probe.py $O $R/hop_probe.json ${REPS:-300} > $R/hop_probe.log 2>&1 & HP=$!
$WD $HP 1024 $R/probe.watchdog.log > /dev/null 2>&1 &
wait $HP; rc=$?
grep -vi warn $R/hop_probe.log | tee -a $R/run.log
echo "$(date +%T) probe rc $rc" | tee -a $R/run.log
exit $rc
