#!/bin/zsh
# 18 Nov: after chain2: traced V8 prefill (forge stopped), attention-core variants vs the FFN, then forge back.
P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe; PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh; R=/Volumes/SSD4TB/anemll-forge-research/contention/runs
until grep -q "all done" /Volumes/SSD4TB/anemll-forge-research/contention/runs/chain2.log 2>/dev/null; do sleep 15; done
echo "$(date +%T) chain2 done; stopping forge on 8765"; PORT=8765 $P/srv_stop.sh
for i in $(seq 1 60); do lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1 || break; sleep 1; done
sleep 120
echo "$(date +%T) v8 trace"; mkdir -p $R/trace1
/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power/v8_trace.sh $R/trace1 2048,2048,8192,32768,65536
$PY /Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power/trace_analyze.py $R/trace1 > /dev/null 2>$R/trace1/analyze.err; echo "$(date +%T) trace analyzed"
sleep 120
cd /Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power
for V in unf fa2 fa2mm fa2s flash fa2_64; do
  echo "$(date +%T) attention $V alone check"
  env $($PY -c "import sys; sys.argv=['x']; import coupling_run as C; print(' '.join(f'{k}={v}' for k, v in C.LOADS['a_$V'][1].items()))") \
     timeout 120 $PY gpu_mix_win.py att 3 61440 $V > $R/attn_check_$V.log 2>&1; echo "  rc=$? $(tail -1 $R/attn_check_$V.log | cut -c1-120)"
done
for V in unf fa2 fa2mm fa2s flash fa2_64; do
  grep -q "attention@60K" $R/attn_check_$V.log || { echo "skip $V"; continue; }
  S=attn_$V; mkdir -p $R/$S; echo "$(date +%T) set $S"
  $PY -u coupling_run.py $S $R/$S > $R/$S/driver.log 2>&1 & G=$!
  $WD $G 1024 $R/$S/watchdog.log >/dev/null 2>&1 & K=$!
  wait $G; echo "$(date +%T) set $S rc=$?"; kill $K 2>/dev/null; sleep 2
  $PY coupling_analyze.py $R/$S > /dev/null 2>$R/$S/analyze.err
  sleep 60
done
echo "$(date +%T) restarting forge on 8765"; PORT=8765 nohup $P/v8_server_dec8.sh >/dev/null 2>&1 &
for i in $(seq 1 60); do curl -s -m 2 localhost:8765/v1/models | grep -q -i qwen && break; sleep 5; done
echo "$(date +%T) forge ready"
echo "$(date +%T) all done"
