#!/bin/zsh
# Clean, matched, cool Splash 1.7.0 vs forge grid (19 Nov 2026, external fan on): contexts 2K / 8K / 16K / 32K / 64K /
# 96K, each server alone, Splash first, then forge (left up on 8765). scripts/m6_compare_bench.py once per context (its
# own 30 s idle baseline, cold prompt = entry - 512, 256 greedy tokens, 3 repeats), mactop whole-machine power.
# Thermal hygiene: before each server >= 180 s and until SoC <= 45 C (max 600 s); before each context until SoC <= 48 C
# (max 240 s); SoC / GPU temperature, fan 0 RPM and thermal state recorded before each server and context (temps.jsonl).
# Banners (paths + sha256, resolved flags) outside the timed region. Output: compare/nov19_170_cool/ (fresh).
F=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge; PY=$F/.venv/bin/python; P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe
S=/Users/anemll/SourceRelease/GITHUB/ML_playground/incoai_splash_170; D=${0:a:h}; O=/Volumes/SSD4TB/anemll-forge-research/compare/nov19_170_cool
L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs; TOK=/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model; MT=/Users/anemll/homebrew/bin/mactop
CTXS=(${=CTXS:-2048 8192 16384 32768 65536 98304})
[ -e $O ] && { echo "$O exists: refusing to overwrite"; exit 1; }; mkdir -p $O
temp() {   # temp LABEL -> one line in temps.jsonl and stdout
  timeout 10 $MT --headless --format json -i 1000 --count 1 2>/dev/null | $PY -c "
import sys, json, time
d = json.loads(sys.stdin.read()); d = d[0] if isinstance(d, list) else d
r = {'t': time.strftime('%H:%M:%S'), 'label': '$1', 'soc_c': round(d['soc_metrics']['soc_temp'], 1), 'gpu_c': round(d['soc_metrics']['gpu_temp'], 1),
     'fan0_rpm': (d.get('fans') or [{}])[0].get('rpm'), 'thermal': d.get('thermal_state'), 'total_w': round(d['soc_metrics'].get('total_power', 0) or 0, 1)}
print(json.dumps(r)); open('$O/temps.jsonl', 'a').write(json.dumps(r) + '\n')"
}
cool() {   # cool MIN_S MAX_S LIMIT_C LABEL
  local t0=$(date +%s); sleep $1
  while true; do
    local c=$(timeout 10 $MT --headless --format json -i 1000 --count 1 2>/dev/null | $PY -c "import sys, json; d = json.loads(sys.stdin.read()); d = d[0] if isinstance(d, list) else d; print(d['soc_metrics']['soc_temp'])")
    (( $(date +%s) - t0 >= $2 )) && { echo "$(date +%T) cooldown $4: max wait reached at SoC $c C"; break; }
    (( ${c%.*} <= $3 )) && { echo "$(date +%T) cooldown $4: SoC $c C <= $3 after $(( $(date +%s) - t0 )) s"; break; }
    sleep 15
  done
}
bench() {   # bench URL MODEL LABEL
  for C in $CTXS; do
    cool 0 240 48 "$3 before ${C}"; temp "$3 ctx $C start"
    (cd $F/scripts && $PY m6_compare_bench.py --url $1 --model "$2" --tokenizer $TOK --ctx $C --label ${3}_$C --out $O/${3}_$C.json > $O/${3}_$C.log 2>&1)
    echo "$(date +%T) $3 $C rc=$? | $(grep -E 'cold:|decode:' $O/${3}_$C.log | tr '\n' ' ' | cut -c1-260)"
  done
  temp "$3 end"
}
echo "$(date +%T) start (JetSpec++ download left running)"; temp "start"
lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1 && { echo "$(date +%T) stopping forge on 8765"; PORT=8765 $P/srv_stop.sh; sleep 10; }
lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1 && { echo "port 8000 in use"; exit 1; }
# ---- Splash 1.7.0 ----
cool 180 600 45 "before splash"; temp "splash before load"
GG=$(readlink -f /Users/anemll/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-IQ3_XXS.gguf)
GSHA=$(shasum -a 256 "$GG" | cut -d' ' -f1); BSHA=$(shasum -a 256 $S/splash | cut -d' ' -f1)
nohup $D/serve.sh >/dev/null 2>&1 &
for i in $(seq 1 360); do curl -s -m 2 localhost:8000/v1/models | grep -q -i qwen && break; sleep 5; done
curl -s -m 2 localhost:8000/v1/models | grep -q -i qwen || { echo "splash did not come up"; tail -30 $L/splash170_serve.log; pkill -INT -f "server.server .*--port 8000"; exit 1; }
MID=$(curl -s -m 2 localhost:8000/v1/models | $PY -c "import sys, json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "BANNER splash $(cd $S && ./splash --version 2>/dev/null | head -1) | git $(cd $S && git rev-parse HEAD) | binary $S/splash sha256 $BSHA | model $MID rev 4ca72078 file $GG sha256 $GSHA | draft incoai/Qwen3.8-27B-DFlash2 snapshot 015e7956 | flags: serve defaults (--verify-rows auto) + --offline | resolved: $(grep -i -m6 -E 'Ready|verify|share|context' $L/splash170_serve.log | cut -c10- | tr '\n' ' ' | cut -c1-600)" | tee $O/banner_splash.txt
bench http://127.0.0.1:8000/v1 "$MID" splash170
cp $L/splash170_serve.log $O/splash170_serve.log
echo "$(date +%T) stopping splash"; pkill -INT -f "server.server .*--port 8000"
for i in $(seq 1 60); do lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1 || break; sleep 1; done
lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1 && { pkill -TERM -f "server.server .*--port 8000"; sleep 10; }
lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1 && { echo "splash still up on 8000"; exit 1; }
# ---- forge (served config) ----
cool 180 600 45 "before forge"; temp "forge before load"
cat /Volumes/SSD4TB/anemll-forge-research/contention/ctl/ka > $O/ka_switch.txt 2>/dev/null
(cd $P && env -u V8_DEC8 -u DEC_DYLIB -u DEC_METAL -u DEC_ATTN -u DEC_F8V -u DEC_LOGITS_DIR -u DRAFT_PROF -u DRAFT_TREE_LOG -u PREFILL_LOGITS_DIR -u DEC_KA_HOLD_MS PORT=8765 nohup $P/v8_server_dec8_qk.sh >/dev/null 2>&1 &)
for i in $(seq 1 120); do curl -s -m 2 localhost:8765/v1/models | grep -q -i qwen && break; sleep 5; done
curl -s -m 2 localhost:8765/v1/models | grep -q -i qwen || { echo "forge did not come up"; tail -30 $L/v8_server.log; exit 1; }
FID=$(curl -s -m 2 localhost:8765/v1/models | $PY -c "import sys, json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "BANNER forge model $FID | launcher $P/v8_server_dec8_qk.sh sha256 $(shasum -a 256 $P/v8_server_dec8_qk.sh | cut -d' ' -f1) | build /Volumes/SSD4TB/anemll-forge-research/v8_dec8 manifest sha256 $(shasum -a 256 /Volumes/SSD4TB/anemll-forge-research/v8_dec8/manifest.json | cut -d' ' -f1) | prefill engine $P/engine/libgpu_v8_f8q.dylib sha256 $(shasum -a 256 $P/engine/libgpu_v8_f8q.dylib | cut -d' ' -f1) | $(grep -E 'v8: variant' $L/v8_server.log | tail -1 | cut -c12- | cut -d'|' -f1-3) | $(grep -E 'dec8: BANNER' $L/v8_server.log | tail -1 | cut -c12-) | $(grep -E 'keep-alive hold' $L/v8_server.log | tail -1 | grep -o 'keep-alive[^,]*') | ka switch $(cat $O/ka_switch.txt)" | tee $O/banner_forge.txt
bench http://127.0.0.1:8765/v1 "$FID" forge
cp $L/v8_server.log $O/forge_server.log
echo "$(date +%T) all done; forge left UP on 8765 (served config)"
