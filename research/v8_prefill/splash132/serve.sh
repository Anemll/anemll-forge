#!/bin/zsh
# splash 1.3.2 release, isolated HOME (the brew 1.2.0 install keeps its own state), local HF cache, no SSD cache
X=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/splash132/x/splash-1.3.2-arm64-macos26
export HOME=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/splash132/home HF_HUB_CACHE=/Users/anemll/.cache/huggingface/hub PYTHONDONTWRITEBYTECODE=1
exec $X/python/bin/python3 -u $X/install/launcher.py serve --model unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS --offline \
  --port 8011 --language-only --max-cache-disk 0 --idle-release off "$@"
