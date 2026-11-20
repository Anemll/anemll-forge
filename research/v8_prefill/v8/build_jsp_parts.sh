#!/bin/zsh
# C1 (20 Nov 2026): one JetSpec++ adapter block (streams.0, LUT6) on the ANE: full block at 8K / 32K / 64K / 96K of KV,
# split parts pa16 / pb16 / mlp16; forge.py compile; tree_decomp_time.py timing. Forge down.  zsh build_jsp_parts.sh OUT RES
set -u
O=$1; R=$2; mkdir -p $O $R
W=${0:a:h}; PW=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power
WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh
SPY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
BPY=/Users/anemll/SourceRelease/GITHUB/ML_playground/ane-vector-lut/coreai/.venv/bin/python
( cd $W && env JSP_PARTS=1 KV_CACHE_DTYPE=kv8 KV_KEYS_T=1 ATT_INT8MM=s8,s8b,sm8,pvf8 ATT_S8_UNIT=0.25 ATT_S8B_UNIT=0.25 ATT_PF8_UNIT=0.015625 \
  ATT_BLOCK=2048 ATT_BLOCK_PREFILL=4096 ATT_BLOCK_LONG=4096 ATT_LONG_FROM=131072 TPS=64 TPS_LONG=32 PREFILL_LONG_FROM=98304 \
  ATT_BLOCK_PREFILL_LONG=8192 GDN_FAST=1 QCONV_INT8=0 KV_SEG=8192,8192,16384,32768,32768 V8_SPLIT=8192,9216 V8_ATTPROJ=1 \
  EXPORT_DIR=/Users/anemll/Models/vq27b/export/release_vq3pA_mixh_s600_k1_mat MODEL=/Volumes/SSD4TB/Qwen3.8-27B \
  $BPY -u build_jsp_adapter_ane.py /Volumes/SSD4TB/anemll-forge-research/jetspec_pp/adapter_lut6 $O 0 1 8192,32768,65536,98304 ) > $R/build.log 2>&1 & BP=$!
$WD $BP 1024 $R/build.watchdog.log >/dev/null 2>&1 &
wait $BP || { echo "build failed"; tail -20 $R/build.log; exit 1; }
tail -1 $R/build.log | tee -a $R/run.log
MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 $SPY $W/journal_compile.py $O > $R/compile.log 2>&1 || { echo "compile failed"; tail -20 $R/compile.log; exit 1; }
grep "done in" $R/compile.log | tail -1 | tee -a $R/run.log
MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 $SPY $PW/tree_decomp_time.py $O $R/timing.json 2>&1 | grep -v -i warn | tee -a $R/run.log
