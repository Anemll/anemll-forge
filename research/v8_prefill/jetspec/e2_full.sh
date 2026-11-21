#!/bin/zsh
# E2 (20 Nov 2026): the full 12-block adapter pass at 16 rows / 8K on the ANE, one variant: 4 packages of 3 blocks
# (build_jsp_adapter_rows.py ROWS=16), one forge.py compile, e1_ane_rows_time.py (banner with the HWX mode), hwx json,
# then the build's Core AI entries deleted (coreai_cache_delete.py on the build dir) and the scratch dir removed.
#   TAG=... [MLP=..] [MLP_ASCALE=..] [SPLIT_OC=..] zsh e2_full.sh
set -u
V8=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/v8; PW=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power
WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh
SPY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
BPY=/Users/anemll/SourceRelease/GITHUB/ML_playground/ane-vector-lut/coreai/.venv/bin/python
LUT=/Volumes/SSD4TB/anemll-forge-research/jetspec_pp/adapter_lut4hg
CTX=${CTX:-8192}; O=/Volumes/SSD4TB/anemll-forge-research/jetspec_pp/e2_full_$TAG
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/20nov/e2/full_$TAG
mkdir -p $O $R
lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1 && { echo "forge listens on 8765: stop it first"; exit 1; }
for f in 0 3 6 9; do
  echo "$(date +%T) build $TAG blocks $f-$((f+2)) (mlp ${MLP:-fp16} split ${SPLIT_OC:-none})" | tee -a $R/run.log
  ( cd $V8 && env KV_CACHE_DTYPE=kv8 KV_KEYS_T=1 ATT_INT8MM=s8,s8b,sm8,pvf8 ATT_S8_UNIT=0.25 ATT_S8B_UNIT=0.25 ATT_PF8_UNIT=0.015625 \
    ATT_BLOCK=2048 ATT_BLOCK_PREFILL=4096 ATT_BLOCK_LONG=4096 ATT_LONG_FROM=131072 TPS=64 TPS_LONG=32 PREFILL_LONG_FROM=98304 \
    ATT_BLOCK_PREFILL_LONG=8192 GDN_FAST=1 QCONV_INT8=0 KV_SEG=8192,8192,16384,32768,32768 V8_SPLIT=8192,9216 V8_ATTPROJ=1 \
    EXPORT_DIR=/Users/anemll/Models/vq27b/export/release_vq3pA_mixh_s600_k1_mat MODEL=/Volumes/SSD4TB/Qwen3.8-27B \
    JSP_ROT=1 SPLIT_OC=${SPLIT_OC:-} SPLIT_N=2 MLP=${MLP:-fp16} MLP_ASCALE=${MLP_ASCALE:-} ROWS_MLP= \
    PYTHONPATH=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/w8a8 \
    $BPY -u build_jsp_adapter_rows.py $LUT $O $f 3 $CTX 16 ) > $R/build_b$f.log 2>&1 & BP=$!
  $WD $BP 1024 $R/build_b$f.watchdog.log >/dev/null 2>&1 &
  wait $BP || { echo "build $f failed"; tail -20 $R/build_b$f.log; exit 1; }
  tail -1 $R/build_b$f.log | tee -a $R/run.log
done
echo "$(date +%T) compile (forge.py, mode 2)" | tee -a $R/run.log
MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 $SPY $V8/journal_compile.py $O > $R/compile.log 2>&1 & CP=$!
$WD $CP 1024 $R/compile.watchdog.log >/dev/null 2>&1 &
wait $CP || { echo "compile failed"; tail -20 $R/compile.log; exit 1; }
grep -i "done in" $R/compile.log | tail -1 | tee -a $R/run.log
echo "$(date +%T) timing" | tee -a $R/run.log
MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 $SPY $PW/e1_ane_rows_time.py $O $R/timing.json 2>&1 | grep -v -i warn | tee -a $R/run.log
$SPY $PW/hwx_mode.py $O/*.aimodel --json $R/hwx.json > $R/hwx.log 2>&1
$SPY -c "
import json; d=json.load(open('$R/timing.json')); t=sum(v['ms_median'] for v in d['entries'].values())
print('PASS $TAG: 12 blocks, 16 rows, ctx $CTX: %.2f ms (sum of %d package medians)' % (t, len(d['entries'])))" | tee -a $R/run.log
DEL=$(cd /Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge && $SPY scripts/coreai_cache_delete.py --delete $O 2>&1 | tail -1)
echo "$DEL" | tee -a $R/run.log
if [[ "$DEL" == "deleted the entries of 4 of 4 packages"* ]]; then rm -rf $O; else echo "cache delete incomplete: KEPT $O" | tee -a $R/run.log; exit 1; fi
echo "$(date +%T) done $TAG (scratch deleted)" | tee -a $R/run.log
