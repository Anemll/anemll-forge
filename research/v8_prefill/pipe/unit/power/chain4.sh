#!/bin/zsh
# 18 Nov: after chain3: long decode under telemetry (forge up), CPU references at 512 rows, ANE 512-row check (forge
# stopped briefly), forge back.
P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe; PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
BP=/Users/anemll/SourceRelease/GITHUB/ML_playground/ane-vector-lut/coreai/.venv/bin/python
WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh; R=/Volumes/SSD4TB/anemll-forge-research/contention/runs
until grep -q "all done" /Volumes/SSD4TB/anemll-forge-research/contention/runs/chain3.log 2>/dev/null; do sleep 15; done
echo "$(date +%T) chain3 done"; sleep 120
mkdir -p $R/longdec; echo "$(date +%T) long decode"
$PY -u /Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power/bench_tel.py $R/longdec -- $PY /Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power/long_decode.py --port 8765 --max-tokens 2500 > $R/longdec.log 2>&1 & G=$!
$WD $G 1024 $R/longdec/watchdog.log >/dev/null 2>&1 & K=$!
wait $G; echo "$(date +%T) long decode rc=$?"; kill $K 2>/dev/null
sleep 30
echo "$(date +%T) CPU references at 512 rows"
cd /Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/w8a8 && env $(cat /Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/w8a8/build_env_dec8.txt) $BP -u ref512.py > $R/ref512.log 2>&1 & G=$!
$WD $G 2048 $R/ref512.watchdog.log >/dev/null 2>&1 & K=$!
wait $G; echo "$(date +%T) ref512 rc=$?"; kill $K 2>/dev/null
echo "$(date +%T) stopping forge on 8765"; PORT=8765 $P/srv_stop.sh
for i in $(seq 1 60); do lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1 || break; sleep 1; done
cd /Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/w8a8 && $PY -u check512.py > $R/check512.log 2>&1 & G=$!
$WD $G 1024 $R/check512.watchdog.log >/dev/null 2>&1 & K=$!
wait $G; echo "$(date +%T) check512 rc=$?"; kill $K 2>/dev/null
echo "$(date +%T) restarting forge on 8765"; PORT=8765 nohup $P/v8_server_dec8.sh >/dev/null 2>&1 &
for i in $(seq 1 60); do curl -s -m 2 localhost:8765/v1/models | grep -q -i qwen && break; sleep 5; done
echo "$(date +%T) forge ready"
echo "$(date +%T) all done"
