#!/bin/zsh
# 18 Nov: counter calibration, the compute-type contention set (forge stopped), then the V8 benchmark under telemetry.
P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe; PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh; R=/Volumes/SSD4TB/anemll-forge-research/contention/runs
echo "$(date +%T) stopping forge on 8765"; PORT=8765 $P/srv_stop.sh
for i in $(seq 1 60); do lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1 || break; sleep 1; done
for S in cal wa; do
  mkdir -p $R/$S; echo "$(date +%T) set $S"
  cd /Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power && $PY -u coupling_run.py $S $R/$S > $R/$S/driver.log 2>&1 & G=$!
  $WD $G 1024 $R/$S/watchdog.log >/dev/null 2>&1 & K=$!
  wait $G; echo "$(date +%T) set $S rc=$?"; kill $K 2>/dev/null; sleep 3
  $PY /Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power/coupling_analyze.py $R/$S > /dev/null 2>$R/$S/analyze.err
  [ $S = cal ] && $PY /Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power/cal_analyze.py $R/cal > /dev/null 2>>$R/cal/analyze.err
  [ $S = cal ] && { echo "$(date +%T) idle 120 s"; sleep 120; }
done
echo "$(date +%T) restarting forge on 8765"; PORT=8765 nohup $P/v8_server_dec8.sh >/dev/null 2>&1 &
for i in $(seq 1 60); do curl -s -m 2 localhost:8765/v1/models | grep -q -i qwen && break; sleep 5; done
echo "$(date +%T) forge ready"; sleep 60
mkdir -p $R/v8bench
$PY -u /Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power/bench_tel.py $R/v8bench -- zsh -c 'for s in 2048:95000 2048:96500 8192:85000 16384:25000 32768:45000 65536:5000; do sleep 60; /Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python /Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/srv_bench.py --port 8765 --tokens ${s%%:*} --offset ${s##*:} --max-tokens 256; done' > $R/v8bench.log 2>&1 & G=$!
$WD $G 1024 $R/v8bench/watchdog.log >/dev/null 2>&1 & K=$!
wait $G; echo "$(date +%T) v8bench rc=$?"; kill $K 2>/dev/null
echo "$(date +%T) all done"
