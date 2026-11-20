#!/bin/zsh
# Splash 1.6.1 vs forge, whole server, whole machine (18 Nov 2026): scripts/m6_compare_bench.py at 8K / 16K / 32K / 64K,
# each server alone. Forge stopped -> Splash served, banner, bench, stopped -> forge restarted (FP8 x serving, keep-alive),
# banner, bench; forge left up on 8765.
F=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge; PY=$F/.venv/bin/python; P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe
S=/Users/anemll/SourceRelease/GITHUB/ML_playground/incoai_splash; D=${0:a:h}; O=/Volumes/SSD4TB/anemll-forge-research/compare/nov18
L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs; TOK=/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model; CTX=8192,16384,32768,65536
mkdir -p $O
echo "$(date +%T) stopping forge on 8765"; PORT=8765 $P/srv_stop.sh; sleep 20
echo "$(date +%T) starting splash 1.6.1 on 8000 (swap $(sysctl -n vm.swapusage | awk '{print $6}'))"
nohup $D/serve.sh >/dev/null 2>&1 &
for i in $(seq 1 240); do curl -s -m 2 localhost:8000/v1/models | grep -q -i qwen && break; sleep 5; done
curl -s -m 2 localhost:8000/v1/models | grep -q -i qwen || { echo "splash did not come up"; tail -20 $L/splash161_serve.log; pkill -INT -f "server.server .*--port 8000"; sleep 10
  echo "$(date +%T) restoring forge on 8765"; cd $P; PORT=8765 nohup $P/v8_server_dec8.sh >/dev/null 2>&1 &; exit 1; }
MID=$(curl -s -m 2 localhost:8000/v1/models | $PY -c "import sys, json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "BANNER splash $(cd $S && ./splash --version 2>/dev/null | head -1) | git $(cd $S && git rev-parse --short HEAD) $(cd $S && git log -1 --format=%s | cut -c1-60) | model id $MID | revision 4ca72078 | flags: defaults + --offline | ANE: $(grep -i -m3 -E 'neural engine|ane' $L/splash161_serve.log | tr '\n' ' ' | cut -c1-300)"
cd $F/scripts && $PY m6_compare_bench.py --url http://127.0.0.1:8000/v1 --model "$MID" --tokenizer $TOK --ctx $CTX --label splash161 --out $O/splash161_ud_iq3xxs.json 2>&1 | tee $O/splash161_ud_iq3xxs.log
grep -i -E "ane|neural" $L/splash161_serve.log | tail -5 | cut -c1-300 > $O/splash161_ane_lines.txt
echo "$(date +%T) stopping splash (Ctrl+C to its server process)"; pkill -INT -f "server.server .*--port 8000"
for i in $(seq 1 60); do lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1 || break; sleep 1; done
lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1 && { pkill -INT -f "server.server .*--port 8000"; sleep 10; }
lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1 && { pkill -TERM -f "server.server .*--port 8000"; sleep 10; }
sleep 20
lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1 && { echo "splash still up on 8000"; exit 1; }
echo "$(date +%T) starting forge on 8765 (swap $(sysctl -n vm.swapusage | awk '{print $6}'))"; cd $P; PORT=8765 nohup $P/v8_server_dec8.sh >/dev/null 2>&1 &
for i in $(seq 1 60); do curl -s -m 2 localhost:8765/v1/models | grep -q -i qwen && break; sleep 5; done
FID=$(curl -s -m 2 localhost:8765/v1/models | $PY -c "import sys, json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "BANNER forge model id $FID | $(grep -E 'v8: variant' $L/v8_server.log | tail -1 | cut -c12-400) | $(grep -E 'dec8: decode GPU engine' $L/v8_server.log | tail -1 | cut -c12-300)"
cd $F/scripts && $PY m6_compare_bench.py --url http://127.0.0.1:8765/v1 --model "$FID" --tokenizer $TOK --ctx $CTX --label forge_f8x_ka --out $O/forge_f8x_ka.json 2>&1 | tee $O/forge_f8x_ka.log
echo "$(date +%T) all done (forge up on 8765)"
