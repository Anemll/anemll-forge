#!/bin/zsh
# Queue 14 (19 Nov 2026): Splash 1.7.0 (worktree incoai_splash_170, 61f6d516; tree verify up to 16 rows, --verify-rows
# auto) vs forge (v8_server_dec8_qk.sh: FP8 x + QKE=tg + keep-alive), scripts/m6_compare_bench.py at 2K / 8K / 16K /
# 32K / 64K (decode at 2K / 16K / 32K / 64K, prefill at 8K / 32K / 64K from the same runs), each server alone, whole-
# machine W from mactop. Forge is left down at the end.
F=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge; PY=$F/.venv/bin/python; P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe
S=/Users/anemll/SourceRelease/GITHUB/ML_playground/incoai_splash_170; D=${0:a:h}; O=/Volumes/SSD4TB/anemll-forge-research/compare/nov19_170
L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs; TOK=/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model; CTX=2048,8192,16384,32768,65536
mkdir -p $O
lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1 && { echo "$(date +%T) stopping forge on 8765"; PORT=8765 $P/srv_stop.sh; sleep 20; }
echo "$(date +%T) starting splash 1.7.0 on 8000 (swap $(sysctl -n vm.swapusage | awk '{print $6}'))"
nohup $D/serve.sh >/dev/null 2>&1 &
for i in $(seq 1 360); do curl -s -m 2 localhost:8000/v1/models | grep -q -i qwen && break; sleep 5; done
curl -s -m 2 localhost:8000/v1/models | grep -q -i qwen || { echo "splash did not come up"; tail -30 $L/splash170_serve.log; pkill -INT -f "server.server .*--port 8000"; exit 1; }
MID=$(curl -s -m 2 localhost:8000/v1/models | $PY -c "import sys, json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "BANNER splash $(cd $S && ./splash --version 2>/dev/null | head -1) | git $(cd $S && git rev-parse --short HEAD) $(cd $S && git log -1 --format=%s | cut -c1-70) | model id $MID | revision 4ca72078 | flags: defaults (--verify-rows auto) + --offline | log: $(grep -i -m4 -E 'neural engine|verify|tree|split' $L/splash170_serve.log | tr '\n' ' ' | cut -c1-400)"
cd $F/scripts && $PY m6_compare_bench.py --url http://127.0.0.1:8000/v1 --model "$MID" --tokenizer $TOK --ctx $CTX --label splash170 --out $O/splash170_ud_iq3xxs.json 2>&1 | tee $O/splash170_ud_iq3xxs.log
cp $L/splash170_serve.log $O/splash170_serve.log
echo "$(date +%T) stopping splash"; pkill -INT -f "server.server .*--port 8000"
for i in $(seq 1 60); do lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1 || break; sleep 1; done
lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1 && { pkill -INT -f "server.server .*--port 8000"; sleep 10; }
lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1 && { pkill -TERM -f "server.server .*--port 8000"; sleep 10; }
sleep 20
lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1 && { echo "splash still up on 8000"; exit 1; }
echo "$(date +%T) starting forge on 8765 (swap $(sysctl -n vm.swapusage | awk '{print $6}'))"; cd $P; PORT=8765 nohup $P/v8_server_dec8_qk.sh >/dev/null 2>&1 &
for i in $(seq 1 60); do curl -s -m 2 localhost:8765/v1/models | grep -q -i qwen && break; sleep 5; done
FID=$(curl -s -m 2 localhost:8765/v1/models | $PY -c "import sys, json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "BANNER forge model id $FID | $(grep -E 'v8: variant' $L/v8_server.log | tail -1 | cut -c12-420) | $(grep -E 'dec8: decode GPU engine' $L/v8_server.log | tail -1 | cut -c12-300)"
cd $F/scripts && $PY m6_compare_bench.py --url http://127.0.0.1:8765/v1 --model "$FID" --tokenizer $TOK --ctx $CTX --label forge_qk --out $O/forge_qk.json 2>&1 | tee $O/forge_qk.log
cp $L/v8_server.log $O/forge_server.log
echo "$(date +%T) stopping forge (left down)"; PORT=8765 $P/srv_stop.sh
echo "$(date +%T) all done"
