#!/bin/zsh
# T1 (20 Nov 2026): Splash on the realistic 9-prompt rungs 2K / 8K / 32K (bench/tree/splash_bench.py), one arm at a time,
# forge stopped first (one model at a time) and NOT restored here. Arms: ARMS="s170_auto s170_all s161".
#   s170_auto  Splash 1.7.0 defaults: tree up to 16 rows + prompt lookup, rows by measured costs
#   s170_all   Splash 1.7.0 --verify-rows all: every cycle verifies the draft block a lane (tree), repeatable
#   s161       Splash 1.6.1 defaults: the 8-row draft block as a chain (no trees, no prompt lookup): the chain reference
set -u
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/20nov/t
T=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/bench/tree; P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe
L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs; PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
mkdir -p $R
if lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1; then echo "$(date +%T) stopping forge on 8765" | tee -a $R/t1.log; PORT=8765 $P/srv_stop.sh; sleep 10; fi
for arm in ${=ARMS:-s170_auto s170_all s161}; do
  case $arm in
    s170_auto) SV=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/splash170/serve.sh; LOG=$L/splash170_serve.log; EX=() ;;
    s170_all)  SV=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/splash170/serve.sh; LOG=$L/splash170_serve.log; EX=(--verify-rows all) ;;
    s161)      SV=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/splash161/serve.sh; LOG=$L/splash161_serve.log; EX=() ;;
  esac
  echo "$(date +%T) $arm: start $SV $EX" | tee -a $R/t1.log
  ( zsh $SV $EX > $R/$arm.serve_launcher.log 2>&1 & )
  for i in $(seq 1 360); do curl -s -m 2 localhost:8000/v1/models | grep -q -i qwen && break; sleep 5; done
  curl -s -m 2 localhost:8000/v1/models | grep -q -i qwen || { echo "$arm: splash did not come up" | tee -a $R/t1.log; tail -30 $LOG; pkill -INT -f "server.server .*--port 8000"; exit 1; }
  sleep 5
  $PY $T/splash_bench.py --prompts $T/realistic_v1.json --label $arm --out $R/$arm.json --rungs 2k,8k,32k --log $LOG 2>&1 | tee $R/$arm.log
  cp $LOG $R/$arm.serve.log
  echo "$(date +%T) $arm: stop splash" | tee -a $R/t1.log
  pkill -INT -f "server.server .*--port 8000"; sleep 15
  lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1 && { pkill -TERM -f "server.server .*--port 8000"; sleep 10; }
  sleep 60   # cool down between arms
done
echo "$(date +%T) T1 done (forge NOT restored here)" | tee -a $R/t1.log
