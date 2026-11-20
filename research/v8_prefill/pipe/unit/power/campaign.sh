#!/bin/zsh
# GPU / ANE coupling campaign (18 Nov 2026): the sets of coupling_run.py in order, 5 min idle between sets (heat),
# swap watchdog on each, analysis after each; forge must already be stopped; restarts forge on 8765 at the end.
#   campaign.sh OUT_ROOT [SETS]     (pm_service.sh must be running as root)
O=$1; SETS=${2:-slc,real,types,gdn,cpu,types_rev}
D=${0:a:h}; P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh
mkdir -p $O; first=1
for S in ${(s:,:)SETS}; do
  if (( ! first )); then echo "$(date +%T) idle 300 s"; sleep 300; fi; first=0
  mkdir -p $O/$S; echo "$(date +%T) set $S"
  cd $D && $PY -u coupling_run.py $S $O/$S > $O/$S/driver.log 2>&1 & G=$!
  $WD $G 1024 $O/$S/watchdog.log >/dev/null 2>&1 & K=$!
  wait $G; rc=$?; kill $K 2>/dev/null; echo "$(date +%T) set $S rc=$rc"
  sleep 3; $PY $D/coupling_analyze.py $O/$S > /dev/null 2>$O/$S/analyze.err; echo "$(date +%T) analyzed $S"
done
echo "$(date +%T) restarting forge on 8765"; PORT=8765 nohup $P/v8_server_dec8.sh >/dev/null 2>&1 &
for i in $(seq 1 60); do curl -s -m 2 localhost:8765/v1/models | grep -q -i qwen && break; sleep 5; done
echo "$(date +%T) forge ready: $(curl -s -m 2 localhost:8765/v1/models | head -c 60)"
