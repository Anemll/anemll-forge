#!/bin/zsh
# V8 vs splash 1.3.2 at the same max context (98304), the same requests (pipe/unit/cmp_bench.py), one server at a time:
# the running V8 server (port 8765) first, then it is stopped and splash is served with --max-context 98304 (port
# 8011, the earlier runs' model revision and DFlash2 drafter, swap watchdog), benched and stopped. V8 is restarted by
# hand afterwards (pipe/v8_server.sh). Results: coreai_seg_logs/cmp_{v8,splash}_96k.jsonl.
D=${0:a:h}; P=$D/../pipe; L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
ARGS=(--lengths ${LENGTHS:-2048,8192,16384,32768,65536,95000} --follow 300 --max-tokens 256 --idle ${IDLE:-120})
echo "$(date +%H:%M:%S) V8 bench (port 8765)"
$PY $P/unit/cmp_bench.py --port 8765 --label v8 $ARGS 2>&1 | grep -v Warning > $L/cmp_v8_96k.jsonl
echo "$(date +%H:%M:%S) V8 done: $(wc -l < $L/cmp_v8_96k.jsonl) lines"
PORT=8765 $P/srv_stop.sh
DM=/Users/anemll/.cache/huggingface/hub/models--incoai--Qwen3.8-27B-DFlash2/snapshots/015e795645c74b1a0eeef3b570031fb62e769bc5
$D/serve.sh --revision 4ca720788d1e01f1bff70c033e0d0028fd02e502 --draft-model $DM --max-context 96K > $L/splash132_serve_cmp96k.log 2>&1 &
G=$!; /Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh $G 1024 $L/watchdog_splash132_cmp96k.log >/dev/null 2>&1 & W=$!
for i in $(seq 1 120); do curl -s -m 2 localhost:8011/v1/models >/dev/null && break; kill -0 $G 2>/dev/null || break; sleep 5; done
echo "$(date +%H:%M:%S) splash ready after $((i*5)) s: $(grep -m1 Ready $L/splash132_serve_cmp96k.log | cut -c1-140)"
$PY $P/unit/cmp_bench.py --port 8011 --label splash $ARGS 2>&1 | grep -v Warning > $L/cmp_splash_96k.jsonl
echo "$(date +%H:%M:%S) splash done: $(wc -l < $L/cmp_splash_96k.jsonl) lines"
kill $G; sleep 5; kill $W 2>/dev/null; echo "$(date +%H:%M:%S) splash stopped"
