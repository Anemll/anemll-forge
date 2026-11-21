#!/bin/zsh
# U2: decomposition probe (v8/build_u2_decomp.py) P = 8 and 16, forge.py compile, HWX (mode + per-stream cycles),
# tree_decomp_time.py, cleanup. Forge must be down.
set -u
V8=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/v8; PW=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power
WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh
SPY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
BPY=/Users/anemll/SourceRelease/GITHUB/ML_playground/ane-vector-lut/coreai/.venv/bin/python
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/20nov/u/decomp; mkdir -p $R
lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1 && { echo "forge listens on 8765: stop it first"; exit 1; }
for p in 8 16; do
  O=/Volumes/SSD4TB/anemll-forge-research/u2decomp_p$p; mkdir -p $O
  echo "$(date +%T) build P=$p" | tee -a $R/run.log
  ( cd $V8 && env KV_CACHE_DTYPE=kv8 KV_KEYS_T=1 ATT_INT8MM=s8,s8b,sm8,pvf8 ATT_S8_UNIT=0.25 ATT_S8B_UNIT=0.25 ATT_PF8_UNIT=0.015625 \
    ATT_BLOCK=2048 ATT_BLOCK_PREFILL=4096 ATT_BLOCK_LONG=4096 ATT_LONG_FROM=131072 TPS=64 TPS_LONG=32 PREFILL_LONG_FROM=98304 \
    ATT_BLOCK_PREFILL_LONG=8192 GDN_FAST=1 QCONV_INT8=0 KV_SEG=8192,8192,16384,32768,32768 V8_SPLIT=8192,9216 V8_ATTPROJ=1 \
    EXPORT_DIR=/Users/anemll/Models/vq27b/export/release_vq3pA_mixh_s600_k1_mat MODEL=/Volumes/SSD4TB/Qwen3.8-27B TREE_P=$p \
    $BPY -u build_u2_decomp.py $O ) > $R/build_p$p.log 2>&1 & BP=$!
  $WD $BP 1024 $R/build_p$p.watchdog.log >/dev/null 2>&1 &
  wait $BP || { echo "build P=$p failed"; tail -20 $R/build_p$p.log; exit 1; }
  tail -1 $R/build_p$p.log | tee -a $R/run.log
  MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 $SPY $V8/journal_compile.py $O > $R/compile_p$p.log 2>&1 & CP=$!
  $WD $CP 1024 $R/compile_p$p.watchdog.log >/dev/null 2>&1 &
  wait $CP || { echo "compile P=$p failed"; tail -20 $R/compile_p$p.log; exit 1; }
  grep -i "done in" $R/compile_p$p.log | tail -1 | tee -a $R/run.log
  $SPY $PW/hwx_mode.py $O/*.aimodel --json $R/hwx_p$p.json > $R/hwx_p$p.log 2>&1; cut -c1-320 $R/hwx_p$p.log | tee -a $R/run.log
  $SPY $PW/u2_streams.py $O/*.aimodel > $R/streams_p$p.log 2>&1; cat $R/streams_p$p.log | tee -a $R/run.log
  MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 $SPY $PW/tree_decomp_time.py $O $R/timing_p$p.json 2>&1 | grep -v -i warn | tee -a $R/run.log
  DEL=$(cd /Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge && $SPY scripts/coreai_cache_delete.py --delete "$O" 2>&1 | tail -1)
  echo "$DEL" | tee -a $R/run.log
  if [[ "$DEL" == "deleted the entries of 1 of 1 packages"* ]]; then rm -rf "$O"; else echo "cache delete incomplete: KEPT $O" | tee -a $R/run.log; fi
done
