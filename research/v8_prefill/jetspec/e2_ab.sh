#!/bin/zsh
# E2 A/B (20 Nov 2026, Anemll: "is it actual int4 x int8? palettized?"): block 0's MLP alone at 16 / 64 / 256 rows (+ the
# 1-block 16-row entry), one package per weight form, so each compiles one kernel format:
#   fp16   LUT4 palette (4-bit indices -> FP16 values) x FP16 activations   (the served adapter form)
#   w4a8t  LUT4 palette (4-bit indices -> INT8 values) x INT8 activations   (E2)
#   w8a8t  the same INT8 values stored DENSE (1 byte per weight)  x INT8    (no palette)
# PHASE=build: build + forge.py compile (CPU / ANE compiler; may run beside the GPU replay)
# PHASE=time:  e1_ane_rows_time.py + hwx_mode.py on each (quiet machine only), then the Core AI entries deleted
set -u
V8=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/v8; PW=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power
WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh
SPY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
BPY=/Users/anemll/SourceRelease/GITHUB/ML_playground/ane-vector-lut/coreai/.venv/bin/python
LUT=/Volumes/SSD4TB/anemll-forge-research/jetspec_pp/adapter_lut4hg
RR=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/20nov
lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1 && { echo "forge listens on 8765: stop it first"; exit 1; }
for v in ${=VARIANTS:-fp16 w4a8t w8a8t w4a8q w4a16q}; do
  O=/Volumes/SSD4TB/anemll-forge-research/jetspec_pp/e2_ab_$v; R=$RR/e2/ab_$v; mkdir -p $O $R
  if [[ $PHASE == build ]]; then
    echo "$(date +%T) build $v" | tee -a $R/run.log
    ( cd $V8 && env KV_CACHE_DTYPE=kv8 KV_KEYS_T=1 ATT_INT8MM=s8,s8b,sm8,pvf8 ATT_S8_UNIT=0.25 ATT_S8B_UNIT=0.25 ATT_PF8_UNIT=0.015625 \
      ATT_BLOCK=2048 ATT_BLOCK_PREFILL=4096 ATT_BLOCK_LONG=4096 ATT_LONG_FROM=131072 TPS=64 TPS_LONG=32 PREFILL_LONG_FROM=98304 \
      ATT_BLOCK_PREFILL_LONG=8192 GDN_FAST=1 QCONV_INT8=0 KV_SEG=8192,8192,16384,32768,32768 V8_SPLIT=8192,9216 V8_ATTPROJ=1 \
      EXPORT_DIR=/Users/anemll/Models/vq27b/export/release_vq3pA_mixh_s600_k1_mat MODEL=/Volumes/SSD4TB/Qwen3.8-27B \
      JSP_ROT=1 SPLIT_OC= MLP=$v MLP_ASCALE=$RR/e2/scales/ane_ascale_timing.json ROWS_MLP=16,64,256 \
      PYTHONPATH=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/w8a8 \
      $BPY -u build_jsp_adapter_rows.py $LUT $O 0 1 8192 16 ) > $R/build.log 2>&1 & BP=$!
    $WD $BP 1024 $R/build.watchdog.log >/dev/null 2>&1 &
    wait $BP || { echo "build $v failed"; tail -20 $R/build.log; continue; }
    tail -1 $R/build.log | tee -a $R/run.log
    MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 $SPY $V8/journal_compile.py $O > $R/compile.log 2>&1 & CP=$!
    $WD $CP 1024 $R/compile.watchdog.log >/dev/null 2>&1 &
    wait $CP || { echo "compile $v failed"; tail -20 $R/compile.log; continue; }
    grep -i "done in" $R/compile.log | tail -1 | tee -a $R/run.log
  else
    echo "$(date +%T) timing $v" | tee -a $R/run.log
    MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 $SPY $PW/e1_ane_rows_time.py $O $R/timing.json 2>&1 | grep -v -i warn | tee -a $R/run.log
    $SPY $PW/hwx_mode.py $O/*.aimodel --json $R/hwx.json > $R/hwx.log 2>&1
    DEL=$(cd /Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge && $SPY scripts/coreai_cache_delete.py --delete "$O" 2>&1 | tail -1)
    echo "$DEL" | tee -a $R/run.log
    if [[ "$DEL" == "deleted the entries of 1 of 1 packages"* ]]; then rm -rf "$O"; else echo "cache delete incomplete: KEPT $O" | tee -a $R/run.log; fi
  fi
done
