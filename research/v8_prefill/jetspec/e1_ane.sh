#!/bin/zsh
# E1 (20 Nov 2026): one row-sweep adapter package (v8/build_jsp_adapter_rows.py), forge.py compile (journaled), timing
# (pipe/unit/power/e1_ane_rows_time.py, banner with the HWX mode), HWX json (hwx_mode.py). Forge must be down.
#   TAG=lut4hg LUT=... JSP_ROT=1 [MLP=fp16|w4a8|w4a8t] [ROWS=..] [ROWS_MLP=..] [FIRST=0 COUNT=3] zsh e1_ane.sh
set -u
V8=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/v8; PW=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power
WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh
SPY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
BPY=/Users/anemll/SourceRelease/GITHUB/ML_playground/ane-vector-lut/coreai/.venv/bin/python
O=/Volumes/SSD4TB/anemll-forge-research/jetspec_pp/e1_rows_$TAG
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/20nov/${RDIR:-e1}/$TAG
mkdir -p $O $R
lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1 && { echo "forge listens on 8765: stop it first"; exit 1; }
echo "$(date +%T) build $TAG (LUT $LUT rot ${JSP_ROT:-0} mlp ${MLP:-fp16} rows ${ROWS:-1,4,8,16,32,64} mlp rows ${ROWS_MLP:-})" | tee -a $R/run.log
( cd $V8 && env KV_CACHE_DTYPE=kv8 KV_KEYS_T=1 ATT_INT8MM=s8,s8b,sm8,pvf8 ATT_S8_UNIT=0.25 ATT_S8B_UNIT=0.25 ATT_PF8_UNIT=0.015625 \
  ATT_BLOCK=2048 ATT_BLOCK_PREFILL=4096 ATT_BLOCK_LONG=4096 ATT_LONG_FROM=131072 TPS=64 TPS_LONG=32 PREFILL_LONG_FROM=98304 \
  ATT_BLOCK_PREFILL_LONG=8192 GDN_FAST=1 QCONV_INT8=0 KV_SEG=8192,8192,16384,32768,32768 V8_SPLIT=8192,9216 V8_ATTPROJ=1 \
  EXPORT_DIR=/Users/anemll/Models/vq27b/export/release_vq3pA_mixh_s600_k1_mat MODEL=/Volumes/SSD4TB/Qwen3.8-27B \
  JSP_ROT=${JSP_ROT:-0} SPLIT_OC=${SPLIT_OC:-} SPLIT_N=${SPLIT_N:-2} MLP=${MLP:-fp16} MLP_ASCALE=${MLP_ASCALE:-} ROWS_MLP=${ROWS_MLP:-} \
  PYTHONPATH=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/w8a8 \
  $BPY -u build_jsp_adapter_rows.py $LUT $O ${FIRST:-0} ${COUNT:-3} ${CTX:-8192} ${ROWS:-1,4,8,16,32,64} ) > $R/build.log 2>&1 & BP=$!
$WD $BP 1024 $R/build.watchdog.log >/dev/null 2>&1 &
wait $BP || { echo "build failed"; tail -20 $R/build.log; exit 1; }
tail -1 $R/build.log | tee -a $R/run.log
echo "$(date +%T) compile (forge.py, mode 2)" | tee -a $R/run.log
MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 $SPY $V8/journal_compile.py $O > $R/compile.log 2>&1 & CP=$!
$WD $CP 1024 $R/compile.watchdog.log >/dev/null 2>&1 &
wait $CP || { echo "compile failed"; tail -20 $R/compile.log; exit 1; }
grep -i "done in\|error\|fail" $R/compile.log | tail -3 | tee -a $R/run.log
echo "$(date +%T) timing" | tee -a $R/run.log
MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 $SPY $PW/e1_ane_rows_time.py $O $R/timing.json 2>&1 | grep -v -i warn | tee -a $R/run.log
$SPY $PW/hwx_mode.py $O/*.aimodel --json $R/hwx.json > $R/hwx.log 2>&1
echo "$(date +%T) done $TAG" | tee -a $R/run.log
