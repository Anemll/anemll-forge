#!/bin/zsh
# The forge half of cmp_chain.sh (18 Nov 2026: the chain's Splash stop step missed the server process; Splash was
# stopped by hand): forge started on 8765 (FP8 x serving, keep-alive), banner, scripts/m6_compare_bench.py at the
# same contexts; forge left up.
F=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge; PY=$F/.venv/bin/python; P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe
O=/Volumes/SSD4TB/anemll-forge-research/compare/nov18; L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs
TOK=/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model; CTX=8192,16384,32768,65536
lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1 && { echo "splash still up on 8000"; exit 1; }
echo "$(date +%T) starting forge on 8765 (swap $(sysctl -n vm.swapusage | awk '{print $6}'))"; cd $P; PORT=8765 nohup $P/v8_server_dec8.sh >/dev/null 2>&1 &
for i in $(seq 1 60); do curl -s -m 2 localhost:8765/v1/models | grep -q -i qwen && break; sleep 5; done
FID=$(curl -s -m 2 localhost:8765/v1/models | $PY -c "import sys, json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "BANNER forge model id $FID | $(grep -E 'v8: variant' $L/v8_server.log | tail -1 | cut -c12-400) | $(grep -E 'dec8: decode GPU engine' $L/v8_server.log | tail -1 | cut -c12-300)"
cd $F/scripts && $PY m6_compare_bench.py --url http://127.0.0.1:8765/v1 --model "$FID" --tokenizer $TOK --ctx $CTX --label forge_f8x_ka --out $O/forge_f8x_ka.json 2>&1 | tee $O/forge_f8x_ka.log
echo "$(date +%T) all done (forge up on 8765)"
