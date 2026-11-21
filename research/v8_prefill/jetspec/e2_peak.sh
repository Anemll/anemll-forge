#!/bin/zsh
# E2 speed check: build (e2_int8_peak.py, coreai venv), forge.py compile (mode 2), time + HWX banner, then delete the
# Core AI entries and the scratch dir.   TAG=... [DIMS="N C H W"] [FORMS=fp16,lut4,w8a8,w4a8] zsh e2_peak.sh
set -u
V8=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/v8
WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh
SPY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
BPY=/Users/anemll/SourceRelease/GITHUB/ML_playground/ane-vector-lut/coreai/.venv/bin/python
O=/Volumes/SSD4TB/anemll-forge-research/jetspec_pp/e2_peak_$TAG
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/20nov/e2/peak_$TAG
mkdir -p $O $R
lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1 && { echo "forge listens on 8765: stop it first"; exit 1; }
echo "$(date +%T) build $TAG (${FORMS:-fp16,lut4,w8a8,w4a8}; ${DIMS:-64 512 64 64})" | tee -a $R/run.log
( cd $V8 && env KV_CACHE_DTYPE=kv8 KV_KEYS_T=1 ATT_INT8MM=s8,s8b,sm8,pvf8 GDN_FAST=1 QCONV_INT8=0 V8_SPLIT=8192,9216 \
  EXPORT_DIR=/Users/anemll/Models/vq27b/export/release_vq3pA_mixh_s600_k1_mat MODEL=/Volumes/SSD4TB/Qwen3.8-27B \
  $BPY -u e2_int8_peak.py build $O ${FORMS:-fp16,lut4,w8a8,w4a8} ${=DIMS:-64 512 64 64} ) > $R/build.log 2>&1 & BP=$!
$WD $BP 1024 $R/build.watchdog.log >/dev/null 2>&1 &
wait $BP || { echo "build failed"; tail -20 $R/build.log; exit 1; }
grep "MB in" $R/build.log | tee -a $R/run.log
MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 $SPY $V8/journal_compile.py $O > $R/compile.log 2>&1 & CP=$!
$WD $CP 1024 $R/compile.watchdog.log >/dev/null 2>&1 &
wait $CP || { echo "compile failed"; tail -20 $R/compile.log; exit 1; }
grep -i "done in" $R/compile.log | tail -1 | tee -a $R/run.log
MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 $SPY $V8/e2_int8_peak.py time $O $R/timing.json 2>&1 | grep -v -i warn | tee -a $R/run.log
NP=$(ls -d $O/*.aimodel | wc -l | tr -d ' ')
DEL=$(cd /Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge && $SPY scripts/coreai_cache_delete.py --delete "$O" 2>&1 | tail -1)
echo "$DEL" | tee -a $R/run.log
if [[ "$DEL" == "deleted the entries of $NP of $NP packages"* ]]; then rm -rf "$O"; else echo "cache delete incomplete: KEPT $O" | tee -a $R/run.log; fi
