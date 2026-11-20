#!/bin/zsh
# B8 (20 Nov 2026): the JetSpec++ adapter's 12 blocks on the ANE at LUT6 (build_jsp_adapter_ane.py), 3 packages of 4
# blocks, entries blk16_8k / blk16_32k; forge.py compile (journal_compile.py); then tree_decomp_time.py timing.
# Forge down (one model at a time); swap watchdog on every step.   zsh build_jsp_adapter_all.sh OUT_DIR RESULT_DIR
set -u
O=$1; R=$2; mkdir -p $O $R
W=${0:a:h}; PW=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power
WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh
SPY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
BPY=/Users/anemll/SourceRelease/GITHUB/ML_playground/ane-vector-lut/coreai/.venv/bin/python
LUT=${LUT:-/Volumes/SSD4TB/anemll-forge-research/jetspec_pp/adapter_lut6}
lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1 && { echo "forge listens on 8765: stop it first"; exit 1; }
for f in 0 4 8; do
  echo "$(date +%T) build blocks $f-$((f+3))" | tee -a $R/run.log
  ( cd $W && env KV_CACHE_DTYPE=kv8 KV_KEYS_T=1 ATT_INT8MM=s8,s8b,sm8,pvf8 ATT_S8_UNIT=0.25 ATT_S8B_UNIT=0.25 ATT_PF8_UNIT=0.015625 \
    ATT_BLOCK=2048 ATT_BLOCK_PREFILL=4096 ATT_BLOCK_LONG=4096 ATT_LONG_FROM=131072 TPS=64 TPS_LONG=32 PREFILL_LONG_FROM=98304 \
    ATT_BLOCK_PREFILL_LONG=8192 GDN_FAST=1 QCONV_INT8=0 KV_SEG=8192,8192,16384,32768,32768 V8_SPLIT=8192,9216 V8_ATTPROJ=1 \
    EXPORT_DIR=/Users/anemll/Models/vq27b/export/release_vq3pA_mixh_s600_k1_mat MODEL=/Volumes/SSD4TB/Qwen3.8-27B \
    $BPY -u build_jsp_adapter_ane.py $LUT $O $f 4 ${CTXS:-8192,32768} ) > $R/build_b$f.log 2>&1 & BP=$!
  $WD $BP 1024 $R/build_b$f.watchdog.log >/dev/null 2>&1 &
  wait $BP || { echo "build $f failed"; tail -20 $R/build_b$f.log; exit 1; }
  tail -1 $R/build_b$f.log | tee -a $R/run.log
done
echo "$(date +%T) compile (forge.py)" | tee -a $R/run.log
MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 $SPY $W/journal_compile.py $O > $R/compile.log 2>&1 & CP=$!
$WD $CP 1024 $R/compile.watchdog.log >/dev/null 2>&1 &
wait $CP || { echo "compile failed"; tail -20 $R/compile.log; exit 1; }
grep "done in" $R/compile.log | tail -1 | tee -a $R/run.log
echo "$(date +%T) timing" | tee -a $R/run.log
MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 $SPY $PW/tree_decomp_time.py $O $R/timing.json 2>&1 | grep -v -i warn | tee -a $R/run.log
