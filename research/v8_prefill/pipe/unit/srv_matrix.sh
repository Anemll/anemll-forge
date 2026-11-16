#!/bin/zsh
# V8 server speed matrix (16 Nov 2026): per config a fresh test server on port 8767 (v8_server.sh + the config's
# options), then one request per length in ascending order (a different doc slice each, so only the system snapshot is
# reused), then the server is stopped. One JSON line per request in $OUT (config tag added).
#   [FOLLOW=300,600] [MAXTOK=128] srv_matrix.sh OUT "TAG|v8_server options|len1 len2 ..." ...
#   e.g. srv_matrix.sh m.jsonl "k8v16|--v8-k8v16 1|2000 6100 16200" "kv8||2000 6100" "fp16|--v8-kv8 0|2000|CTX=32768"
PIPE=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe; PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
LOG=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log
OUT=$1; shift
for spec in "$@"; do
  TAG=${spec%%|*}; rest=${spec#*|}; OPTS=${rest%%|*}; LENS=${rest#*|}; SENV="X_=1"
  [[ $LENS == *"|"* ]] && { SENV=${LENS#*|}; LENS=${LENS%%|*}; }
  $PIPE/srv_stop.sh >/dev/null
  echo "$(date +%T) config $TAG: $SENV v8_server.sh $OPTS"
  nohup env ${=SENV} ${SRV:-$PIPE/v8_server.sh} ${=OPTS} > ${LOG:r}_launch.log 2>&1 &
  sleep 3
  for i in $(seq 1 300); do lsof -nP -iTCP:8767 -sTCP:LISTEN >/dev/null 2>&1 && break; sleep 2; done
  grep -E "v8:.*(engine|handover)" $LOG | tail -2 | cut -c1-140
  off=0
  for n in ${=LENS}; do
    $PY $PIPE/unit/srv_bench.py --port 8767 --tokens $n --offset $off --max-tokens ${MAXTOK:-128} --follow ${FOLLOW:-0} 2>&1 \
      | $PY -c "
import json, sys
for l in sys.stdin:
    try: d = json.loads(l)
    except Exception: d = {'error': l.strip()[:300]}
    d['config'] = '$TAG'; d['opts'] = '$OPTS'; d['len'] = $n; print(json.dumps(d), flush=True)" | tee -a $OUT | $PY $PIPE/unit/srv_summ.py 2>/dev/null | cut -c1-150
    echo "   swap $(sysctl -n vm.swapusage | awk '{print $6}')"
    off=$((off + ${STEP:-7000}))
  done
  grep -E "v8: [0-9]+ tok" $LOG | tail -${#${=LENS}} | cut -c1-120
done
$PIPE/srv_stop.sh >/dev/null; echo "$(date +%T) matrix done"
