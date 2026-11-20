#!/bin/zsh
# Softmax-fold tests, stages B and C (18 Nov 2026; forge stopped by the caller): B = plain vs fold128 at 8K / 32K / 60K
# / 88K keys, alone and next to the ANE FFN (coupling_run.py fold_<K>k_128); C = traced V8 prefill with the fold
# (libgpu_v8_fold.dylib, FOLD=128, every call) at 2K-95K, and the plain engine traced at 95K; then forge back on 8765.
P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe; PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
D=$P/unit/power; WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh; R=/Volumes/SSD4TB/anemll-forge-research/contention/fold
cd $D
for S in fold_8k_128 fold_32k_128 fold_60k_128 fold_88k_128; do
  mkdir -p $R/$S; echo "$(date +%T) set $S"
  $PY -u coupling_run.py $S $R/$S > $R/$S/driver.log 2>&1 & G=$!
  $WD $G 1024 $R/$S/watchdog.log >/dev/null 2>&1 & K=$!
  wait $G; echo "$(date +%T) set $S rc=$?"; kill $K 2>/dev/null; sleep 2
  $PY coupling_analyze.py $R/$S > /dev/null 2>$R/$S/analyze.err
  sleep 30
done
sleep 60
echo "$(date +%T) trace fold128"; mkdir -p $R/trace_fold128
V8_DYLIB=libgpu_v8_fold.dylib FOLD=128 FOLD_MIN=0 $D/v8_trace.sh $R/trace_fold128 2048,8192,32768,65536,95232
$PY $D/trace_analyze.py $R/trace_fold128 > /dev/null 2>$R/trace_fold128/analyze.err
sleep 60
echo "$(date +%T) trace plain 95K"; mkdir -p $R/trace_plain95
$D/v8_trace.sh $R/trace_plain95 95232
$PY $D/trace_analyze.py $R/trace_plain95 > /dev/null 2>$R/trace_plain95/analyze.err
echo "$(date +%T) restarting forge on 8765"; cd $P; PORT=8765 nohup $P/v8_server_dec8.sh >/dev/null 2>&1 &
for i in $(seq 1 60); do curl -s -m 2 localhost:8765/v1/models | grep -q -i qwen && break; sleep 5; done
echo "$(date +%T) forge ready"
echo "$(date +%T) all done"
