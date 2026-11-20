#!/bin/zsh
# FP8 P.V tests, forge-down stage (18 Nov 2026): plain vs F8=x next to the ANE FFN at 32K / 60K / 88K keys
# (coupling_run.py f8_<K>k_x), then interleaved repeated V8 traces (plain, F8=x, plain, F8=x) at 2K / 8K / 32K / 64K with
# the last-token logits saved, then forge back on 8765. Swap watchdog on every process.
P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe; PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
D=$P/unit/power; WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh; R=/Volumes/SSD4TB/anemll-forge-research/contention/fold/f8
cd $D
echo "$(date +%T) stopping forge on 8765"; PORT=8765 $P/srv_stop.sh
sleep 30
for S in f8_32k_x f8_60k_x f8_88k_x; do
  mkdir -p $R/$S; echo "$(date +%T) set $S"
  $PY -u coupling_run.py $S $R/$S > $R/$S/driver.log 2>&1 & G=$!
  $WD $G 1024 $R/$S/watchdog.log >/dev/null 2>&1 & K=$!
  wait $G; echo "$(date +%T) set $S rc=$?"; kill $K 2>/dev/null; sleep 2
  $PY coupling_analyze.py $R/$S > /dev/null 2>$R/$S/analyze.err
  sleep 30
done
for T in plain_a f8x_a plain_b f8x_b; do
  mkdir -p $R/tr_$T; echo "$(date +%T) trace $T"
  if [[ $T == f8x* ]]; then V8_DYLIB=libgpu_v8_f8.dylib F8=x $D/v8_trace.sh $R/tr_$T 2048,8192,32768,65536
  else V8_DYLIB=libgpu_v8_f8.dylib $D/v8_trace.sh $R/tr_$T 2048,8192,32768,65536; fi
  $PY $D/trace_analyze.py $R/tr_$T > /dev/null 2>$R/tr_$T/analyze.err
  sleep 30
done
echo "$(date +%T) restarting forge on 8765"; cd $P; PORT=8765 nohup $P/v8_server_dec8.sh >/dev/null 2>&1 &
for i in $(seq 1 60); do curl -s -m 2 localhost:8765/v1/models | grep -q -i qwen && break; sleep 5; done
echo "$(date +%T) forge ready"
echo "$(date +%T) all done"
