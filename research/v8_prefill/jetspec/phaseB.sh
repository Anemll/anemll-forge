#!/bin/zsh
# 20 Nov 2026 Phase B (JetSpec++ by replay): capture 32 gsm8k traces of the SERVED model (jsp_capture_server.sh,
# port 8771, forge on 8765 must be down), then the replay check (--rows all vs outcome, 1 trace x 64 tokens) and the
# J3-R3 replay over all traces. Swap watchdog on the replay. Logs and results in RES.
#   CAP=<capture dir> RES=<results dir> zsh phaseB.sh capture|check|replay
set -u
J=${0:a:h}; PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh
PR=${PR:-/Volumes/SSD4TB/anemll-forge-research/jetspec_pp/src/JetSpec/bench/jetspec_plus_plus/prompts/27b-prompts_gsm8k128.json}
COUNT=${COUNT:-32}; DATASET=${DATASET:-gsm8k}; TAG=${TAG:-j3r3}
mkdir -p $RES
case $1 in
capture)
  lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1 && { echo "forge listens on 8765: stop it first"; exit 1; }
  OUT=$CAP PROMPTS=$PR nohup zsh $J/jsp_capture_server.sh > $RES/capture_server.out 2>&1 &
  for i in $(seq 1 120); do grep -q "serving OpenAI API" $CAP/server.log 2>/dev/null && break; sleep 2; done
  grep -q "serving OpenAI API" $CAP/server.log || { echo "capture server did not come up"; tail -20 $CAP/server.log; exit 1; }
  $PY $J/jsp_capture_client.py --first 0 --count $COUNT --max-tokens 2112 --out $RES/capture_client_$TAG.jsonl
  P=$(lsof -tnP -iTCP:8771 -sTCP:LISTEN); [ -n "$P" ] && kill $P
  for i in $(seq 1 30); do lsof -nP -iTCP:8771 -sTCP:LISTEN >/dev/null 2>&1 || break; sleep 1; done
  grep -E "\| gen [0-9]+ in .*tok/verify" $CAP/server.log > $RES/capture_decode_lines_$TAG.txt
  echo "capture done: $(ls $CAP/traces | wc -l) traces";;
check)
  for r in all outcome; do
    $PY $J/jsp_replay.py --traces $CAP/traces --limit 1 --max-new 64 --rows $r --out $RES/check_$r.json > $RES/check_$r.log 2>&1 & BP=$!
    $WD $BP 3072 $RES/check_$r.watchdog.log >/dev/null 2>&1 &
    wait $BP || { echo "check $r failed"; tail -20 $RES/check_$r.log; exit 1; }
    tail -3 $RES/check_$r.log
  done;;
replay)
  $PY $J/jsp_replay.py --traces $CAP/traces --max-new 2048 --rows outcome --schedule J3-R3 --dataset $DATASET --out $RES/replay_$TAG.json > $RES/replay_$TAG.log 2>&1 & BP=$!
  $WD $BP ${WDMB:-3072} $RES/replay_$TAG.watchdog.log >/dev/null 2>&1 &
  wait $BP; echo "replay rc $?"; tail -5 $RES/replay_$TAG.log;;
esac
