#!/bin/zsh
# Time and inspect the q/dq INT8 FFN redo (qdq_probe.py, 17 Nov 2026): stops the forge server on 8765 (one model at a
# time), times the suites (each load compiles its probe package), restarts forge, then the HWX analysis (no ANE).
# Swap watchdog on every process. Logs in RES.
#   qdq_run.sh OUT RES [SUITES]
O=$1; R=$2; SU=${3:-probe,served,decode}
P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe; W=${0:a:h}
BP=/Users/anemll/SourceRelease/GITHUB/ML_playground/ane-vector-lut/coreai/.venv/bin/python
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh
echo "$(date +%T) stopping forge on 8765"; PORT=8765 $P/srv_stop.sh
for i in $(seq 1 60); do lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1 || break; sleep 1; done
cd $W && $BP -u qdq_probe.py time --out $O --suites $SU > $R/time.log 2>&1 & G=$!
$WD $G 1024 $R/time_watchdog.log >/dev/null 2>&1 & K=$!
wait $G; echo "$(date +%T) time rc=$?"; kill $K 2>/dev/null
echo "$(date +%T) restarting forge on 8765"; PORT=8765 nohup $P/v8_server_dec8.sh >/dev/null 2>&1 &
for i in $(seq 1 60); do curl -s -m 2 localhost:8765/v1/models | grep -q -i qwen && break; sleep 5; done
echo "$(date +%T) forge ready: $(curl -s -m 2 localhost:8765/v1/models | head -c 50)"
cd $W && $PY -u qdq_probe.py hwx --out $O --suites $SU > $R/hwx.log 2>&1; echo "$(date +%T) hwx rc=$?"
for s in ${(s:,:)SU}; do cp $O/$s/time.json $R/time_$s.json 2>/dev/null; cp $O/$s/hwx.json $R/hwx_$s.json 2>/dev/null; cp $O/$s/build_report.json $R/build_report_$s.json 2>/dev/null; done
