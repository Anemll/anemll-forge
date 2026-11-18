#!/bin/zsh
# splash 1.3.2 GPU-only (--disable-ane: the FFN stays on the GPU) with the same requests as cmp_96k.sh
# (pipe/unit/cmp_bench.py, max context 96K, same model revision and DFlash2 drafter, port 8011, swap watchdog), to set
# splash's own Neural Engine gain against our measurements instead of its published figure. Stops whatever serves on
# 8765 first (one model at a time); restart the V8 server by hand afterwards. Result: coreai_seg_logs/cmp_splash_gpu_96k.jsonl.
D=${0:a:h}; P=$D/../pipe; L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
ARGS=(--lengths ${LENGTHS:-2048,8192,16384,32768,65536,95000} --follow 300 --max-tokens 256 --idle ${IDLE:-120})
PORT=8765 $P/srv_stop.sh
for i in $(seq 1 30); do lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1 || break; sleep 1; done
DM=/Users/anemll/.cache/huggingface/hub/models--incoai--Qwen3.8-27B-DFlash2/snapshots/015e795645c74b1a0eeef3b570031fb62e769bc5
$D/serve.sh --revision 4ca720788d1e01f1bff70c033e0d0028fd02e502 --draft-model $DM --max-context 96K --disable-ane \
  > $L/splash132_serve_gpu96k.log 2>&1 &
G=$!; /Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh $G 1024 $L/watchdog_splash132_gpu96k.log >/dev/null 2>&1 & W=$!
for i in $(seq 1 120); do curl -s -m 2 localhost:8011/v1/models | grep -q Qwen && break; kill -0 $G 2>/dev/null || break; sleep 5; done
echo "$(date +%H:%M:%S) splash GPU-only ready after $((i*5)) s: $(grep -m1 Ready $L/splash132_serve_gpu96k.log | cut -c1-140)"
grep -i -E 'neural engine|ane' $L/splash132_serve_gpu96k.log | head -3
$PY $P/unit/cmp_bench.py --port 8011 --label splash_gpu $ARGS 2>&1 | grep -v Warning > $L/cmp_splash_gpu_96k.jsonl
echo "$(date +%H:%M:%S) splash GPU-only done: $(wc -l < $L/cmp_splash_gpu_96k.jsonl) lines"
kill $G; sleep 5; kill $W 2>/dev/null; echo "$(date +%H:%M:%S) splash stopped"
