#!/bin/zsh
# splash 1.3.2 cold prompts at 8K / 16K / 32K / 64K / 128K, 3 min idle before each; server started and stopped here
D=${0:a:h}; L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs; PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
DM=/Users/anemll/.cache/huggingface/hub/models--incoai--Qwen3.8-27B-DFlash2/snapshots/015e795645c74b1a0eeef3b570031fb62e769bc5
$D/serve.sh --revision 4ca720788d1e01f1bff70c033e0d0028fd02e502 --draft-model $DM > $L/splash132_serve_273_long.log 2>&1 &
G=$!; /Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh $G 1024 $L/watchdog_splash132_273_long.log >/dev/null 2>&1 & W=$!
for i in $(seq 1 120); do curl -s -m 2 localhost:8011/v1/models >/dev/null && break; kill -0 $G 2>/dev/null || break; sleep 5; done
echo "splash ready after $((i*5)) s"
for n in 8192 16384 32768 65536 131072; do sleep 180; cd $D && $PY bench.py 8011 $n 2>&1 | cut -c1-200; done | tee $L/splash132_273_long.jsonl
kill $G; sleep 3; kill $W 2>/dev/null; echo stopped
