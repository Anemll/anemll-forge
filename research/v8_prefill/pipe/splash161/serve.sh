#!/bin/zsh
# Splash 1.6.1 from the checkout (~/SourceRelease/GITHUB/ML_playground/incoai_splash, built with make), defaults
# (ANE split on), the Oct 4 comparison's model: unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS at revision 4ca72078 (the only
# one cached) and the cached DFlash2 draft (snapshot 015e7956, a local directory: offline cannot resolve its branch), offline. Foreground; log in $L/splash161_serve.log; swap watchdog.
S=/Users/anemll/SourceRelease/GITHUB/ML_playground/incoai_splash; L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs
lsof -nP -iTCP:${PORT:-8000} -sTCP:LISTEN >/dev/null 2>&1 && { echo "port ${PORT:-8000} in use"; exit 1; }
[ -s $L/splash161_serve.log ] && mv $L/splash161_serve.log $L/splash161_serve.$(date -r $L/splash161_serve.log +%Y%m%d-%H%M%S).log
cd $S && ./splash serve --model unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS --revision 4ca720788d1e01f1bff70c033e0d0028fd02e502 \
  --draft-model /Users/anemll/.cache/huggingface/hub/models--incoai--Qwen3.8-27B-DFlash2/snapshots/015e795645c74b1a0eeef3b570031fb62e769bc5 \
  --offline --port ${PORT:-8000} "$@" > $L/splash161_serve.log 2>&1 &
G=$!; echo "splash pid $G (port ${PORT:-8000}, log $L/splash161_serve.log)"
/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh $G 1536 $L/watchdog_splash161.log >/dev/null 2>&1 &
wait $G
