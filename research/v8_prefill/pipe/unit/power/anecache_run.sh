#!/bin/zsh
# ANE on-chip memory probe run (18 Nov 2026): stops forge on 8765 (one model at a time), runs ane_cache_probe.py run
# under the swap watchdog, restarts forge.   anecache_run.sh OUT
O=$1; D=${0:a:h}; P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh
echo "$(date +%T) stopping forge on 8765"; PORT=8765 $P/srv_stop.sh
for i in $(seq 1 60); do lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1 || break; sleep 1; done
cd $D && $PY -u ane_cache_probe.py run $O > $O/run.log 2>&1 & G=$!
$WD $G 1024 $O/watchdog.log >/dev/null 2>&1 & K=$!
wait $G; echo "$(date +%T) probe rc=$?"; kill $K 2>/dev/null
echo "$(date +%T) restarting forge on 8765"; PORT=8765 nohup $P/v8_server_dec8.sh >/dev/null 2>&1 &
for i in $(seq 1 60); do curl -s -m 2 localhost:8765/v1/models | grep -q -i qwen && break; sleep 5; done
echo "$(date +%T) forge ready: $(curl -s -m 2 localhost:8765/v1/models | head -c 60)"
