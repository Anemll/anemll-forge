#!/bin/zsh
# Queue 16d (19 Nov 2026): Splash 1.7.0 at 96K (--ctx 98304), exactly the method of splash170/ (cmp_chain.sh's Splash
# arm): serve.sh (worktree incoai_splash_170 at 61f6d516, unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS rev 4ca72078, cached
# DFlash2 draft, defaults + --offline), scripts/m6_compare_bench.py with mactop whole-machine power. Banner (binary /
# model identity, sha256, resolved flags) before the timed bench. Forge on 8765 stays down.
F=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge; PY=$F/.venv/bin/python
S=/Users/anemll/SourceRelease/GITHUB/ML_playground/incoai_splash_170; D=${0:a:h}; O=/Volumes/SSD4TB/anemll-forge-research/compare/nov19_170
L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs; TOK=/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model
lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1 && { echo "forge is up on 8765: stopping (not expected)"; exit 1; }
lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1 && { echo "port 8000 in use"; exit 1; }
GG=$(ls /Users/anemll/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/*UD-IQ3_XXS*.gguf | head -1)
echo "$(date +%T) hashing (outside the timed region): $GG"
GSHA=$(shasum -a 256 "$(readlink -f "$GG")" | cut -d' ' -f1); BSHA=$(shasum -a 256 $S/splash | cut -d' ' -f1)
echo "$(date +%T) starting splash 1.7.0 on 8000 (swap $(sysctl -n vm.swapusage | awk '{print $6}'))"
nohup $D/serve.sh >/dev/null 2>&1 &
for i in $(seq 1 360); do curl -s -m 2 localhost:8000/v1/models | grep -q -i qwen && break; sleep 5; done
curl -s -m 2 localhost:8000/v1/models | grep -q -i qwen || { echo "splash did not come up"; tail -30 $L/splash170_serve.log; pkill -INT -f "server.server .*--port 8000"; exit 1; }
MID=$(curl -s -m 2 localhost:8000/v1/models | $PY -c "import sys, json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "BANNER splash $(cd $S && ./splash --version 2>/dev/null | head -1) | git $(cd $S && git rev-parse HEAD) | binary $S/splash sha256 $BSHA | model id $MID rev 4ca72078 file $(basename $GG) sha256 $GSHA | draft incoai/Qwen3.8-27B-DFlash2 snapshot 015e7956 | flags: serve defaults (--verify-rows auto) + --offline | resolved: $(grep -i -m6 -E 'Ready|neural engine|verify|split|share|context' $L/splash170_serve.log | cut -c10- | tr '\n' ' ' | cut -c1-700)"
cd $F/scripts && $PY m6_compare_bench.py --url http://127.0.0.1:8000/v1 --model "$MID" --tokenizer $TOK --ctx 98304 --label splash170_96k --out $O/splash170_ud_iq3xxs_96k.json 2>&1 | tee $O/splash170_ud_iq3xxs_96k.log
cp $L/splash170_serve.log $O/splash170_serve_96k.log
echo "$(date +%T) stopping splash"; pkill -INT -f "server.server .*--port 8000"
for i in $(seq 1 60); do lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1 || break; sleep 1; done
lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1 && { pkill -TERM -f "server.server .*--port 8000"; sleep 10; }
echo "$(date +%T) all done (forge left down)"
