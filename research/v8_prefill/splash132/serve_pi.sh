#!/bin/zsh
# splash 1.3.2 for Pi (provider splash-local, http://127.0.0.1:8000/v1): the same model revision, DFlash2 drafter and
# 96K max context as the V8 comparison (cmp_96k.sh), swap watchdog. Foreground; log in $L/splash132_serve_pi.log.
D=${0:a:h}; L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs
DM=/Users/anemll/.cache/huggingface/hub/models--incoai--Qwen3.8-27B-DFlash2/snapshots/015e795645c74b1a0eeef3b570031fb62e769bc5
lsof -nP -iTCP:${PORT:-8000} -sTCP:LISTEN >/dev/null 2>&1 && { echo "port ${PORT:-8000} in use"; exit 1; }
[ -s $L/splash132_serve_pi.log ] && mv $L/splash132_serve_pi.log $L/splash132_serve_pi.$(date -r $L/splash132_serve_pi.log +%Y%m%d-%H%M%S).log
$D/serve.sh --revision 4ca720788d1e01f1bff70c033e0d0028fd02e502 --draft-model $DM --max-context ${CTX:-96K} --port ${PORT:-8000} \
  > $L/splash132_serve_pi.log 2>&1 &
G=$!; echo "splash pid $G (port ${PORT:-8000}, log $L/splash132_serve_pi.log)"
/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh $G 1536 $L/watchdog_splash132_pi.log >/dev/null 2>&1 &
wait $G
