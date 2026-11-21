#!/bin/zsh
# 21 Nov 2026: tree8 served default (Anemll 02:30). Restart forge on 8765 with the standard serving launcher
# (v8_server_dec8_qk.sh, now tree8; launcher defaults only, no tree/lookup env), banner check, smoke twice on the same
# server (run-to-run), vs u1_tree8 (cross-launch), then the realistic 2K rung + the code / agent sets (U1 chain
# baselines 42.74 / 39.46 / 54.48 tok/s). Leaves the server up.
set -u
T=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/bench/tree
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/21nov
R20=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/20nov
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
mkdir -p $R/bench
EX="v8=qk=attn_qk8isk_tg v8=F8=x v8=KV8=true v8=libgpu_v8_f8q.dylib dec8=libdec_gpu_tree.dylib tree=mode=tree8 tree=policy=branched tree=v8_dec8tx lookup=off"
EXA=(); for e in ${=EX}; do EXA+=(--expect $e); done
echo "$(date +%T) serve tree8: restart with v8_server_dec8_qk.sh (launcher defaults)" | tee -a $R/serve_tree8.log
( cd $T && env -u BUILD -u GM -u NGRAM_DRAFT -u TREE -u SERVED_TREE -u QKE -u V8_SRV_DYLIB RDIR=$R LAUNCHER=v8_server_dec8_qk.sh ARM=served_tree8_smoke1 RUNGS=2k TASKS=1,8 EXPECT="$EX" timeout 900 zsh $T/run_arm.sh ) > $R/served_tree8_smoke1.driver.log 2>&1
grep -E "^BANNER|SUMMARY|ABORT" $R/bench/served_tree8_smoke1.log | tee -a $R/serve_tree8.log
SL=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log
grep -E "^\[qwen38-27b-ane\] (started|model dir|speculative)|keep-alive|serving OpenAI" $SL | tee -a $R/serve_tree8.log
grep -q "keep-alive hold 60 ms" $SL || { echo "ABORT keep-alive not on"; exit 1; }
grep "tree: BANNER" $SL | grep -q "lookup off$" || { echo "ABORT tree banner: lookup not off"; exit 1; }
grep -q "SUMMARY" $R/bench/served_tree8_smoke1.log || { echo "smoke1 failed"; tail -30 $R/served_tree8_smoke1.driver.log; exit 1; }
$PY $T/tree_bench.py --prompts $T/realistic_v1.json --label served_tree8_smoke2 --out $R/bench/served_tree8_smoke2.json --rungs 2k --tasks 1,8 $EXA > $R/bench/served_tree8_smoke2.log 2>&1
grep -E "SUMMARY|ABORT" $R/bench/served_tree8_smoke2.log | tee -a $R/serve_tree8.log
$PY - <<PYEOF | tee -a $R/serve_tree8.log
import json
a = json.load(open("$R/bench/served_tree8_smoke1.json")); b = json.load(open("$R/bench/served_tree8_smoke2.json"))
u = json.load(open("$R20/bench/u1_tree8.json"))
# keyed by (cat, index): index alone repeats across categories (the 21 Nov first run keyed by index and misreported)
ua = {(r["cat"], r["index"]): r["sha"] for r in u["rungs"]["2k"]}
for ra, rb in zip(a["rungs"]["2k"], b["rungs"]["2k"]):
    i = (ra["cat"], ra["index"])
    print(f"smoke 2k {i}: run1 {ra['sha']} run2 {rb['sha']} u1_tree8 {ua.get(i)} | run-to-run {'IDENTICAL' if ra['sha'] == rb['sha'] else 'DIFFER'}, vs u1_tree8 {'IDENTICAL' if ra['sha'] == ua.get(i) else 'DIFFER'}")
PYEOF
sleep 20
$PY $T/tree_bench.py --prompts $T/realistic_v1.json --label served_tree8_2k --out $R/bench/served_tree8_2k.json --rungs 2k $EXA > $R/bench/served_tree8_2k.log 2>&1
grep -E "SUMMARY|ABORT" $R/bench/served_tree8_2k.log | tee -a $R/serve_tree8.log
$PY $T/tree_bench.py --prompts $T/u1_code_agent.json --label served_tree8_ca --out $R/bench/served_tree8_ca.json --rungs code,agent $EXA > $R/bench/served_tree8_ca.log 2>&1
grep -E "SUMMARY|ABORT" $R/bench/served_tree8_ca.log | tee -a $R/serve_tree8.log
echo "$(date +%T) serve tree8 done, forge pid $(lsof -tnP -iTCP:8765 -sTCP:LISTEN)" | tee -a $R/serve_tree8.log
