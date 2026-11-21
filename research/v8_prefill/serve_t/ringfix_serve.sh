#!/bin/zsh
# 21 Nov 2026 (Anemll 07:27): DRAFT_RING_SNAP=fix served. Restart forge on 8765 with the standard serving launcher
# (launcher defaults only: no tree / lookup / ring env), banner + ring line check, fresh smoke (realistic 2K positions
# 1, 8; expect 29aef2ec9db18a3f), then on the same (used) server the 9-prompt 2K rung vs U1 tree8 (a fresh launch,
# keyed by (cat, index)) and the smoke again. Leaves the server up.
set -u
T=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/bench/tree
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/21nov
R20=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/20nov
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
SL=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log
G=$R/ringfix_serve.log
mkdir -p $R/bench
EX="v8=qk=attn_qk8isk_tg v8=F8=x v8=KV8=true v8=libgpu_v8_f8q.dylib dec8=libdec_gpu_tree.dylib tree=mode=tree8 tree=policy=branched tree=v8_dec8tx lookup=off"
EXA=(); for e in ${=EX}; do EXA+=(--expect $e); done
echo "$(date +%T) ringfix serve: restart with v8_server_dec8_qk.sh (launcher defaults)" | tee -a $G
( cd $T && env -u BUILD -u GM -u NGRAM_DRAFT -u TREE -u SERVED_TREE -u QKE -u V8_SRV_DYLIB -u DRAFT_RING_SNAP RDIR=$R LAUNCHER=v8_server_dec8_qk.sh ARM=served_ringfix_smoke1 RUNGS=2k TASKS=1,8 EXPECT="$EX" timeout 900 zsh $T/run_arm.sh ) > $R/served_ringfix_smoke1.driver.log 2>&1
grep -E "^BANNER|SUMMARY|ABORT" $R/bench/served_ringfix_smoke1.log | tee -a $G
grep -E "^\[qwen38-27b-ane\] (started|model dir|speculative)|v8: variant|dec8: BANNER|tree: BANNER|drafter ring snapshot:|keep-alive|serving OpenAI" $SL | tee -a $G
grep -q "keep-alive hold 60 ms" $SL || { echo "ABORT keep-alive not on" | tee -a $G; exit 1; }
grep "tree: BANNER" $SL | grep -q "lookup off$" || { echo "ABORT tree banner: lookup not off" | tee -a $G; exit 1; }
grep -q "drafter ring snapshot: fix" $SL || { echo "ABORT ring snapshot not fix" | tee -a $G; exit 1; }
grep -q "SUMMARY" $R/bench/served_ringfix_smoke1.log || { echo "smoke1 failed" | tee -a $G; tail -30 $R/served_ringfix_smoke1.driver.log; exit 1; }
$PY $T/tree_bench.py --prompts $T/realistic_v1.json --label served_ringfix_2k --out $R/bench/served_ringfix_2k.json --rungs 2k $EXA > $R/bench/served_ringfix_2k.log 2>&1
grep -E "SUMMARY|ABORT" $R/bench/served_ringfix_2k.log | tee -a $G
$PY $T/tree_bench.py --prompts $T/realistic_v1.json --label served_ringfix_smoke2 --out $R/bench/served_ringfix_smoke2.json --rungs 2k --tasks 1,8 $EXA > $R/bench/served_ringfix_smoke2.log 2>&1
grep -E "SUMMARY|ABORT" $R/bench/served_ringfix_smoke2.log | tee -a $G
$PY - <<PYEOF | tee -a $G
import json
L = lambda n: json.load(open(f"$R/bench/{n}.json"))["rungs"]["2k"]
u = {(r["cat"], r["index"]): r for r in json.load(open("$R20/bench/u1_tree8.json"))["rungs"]["2k"]}
rows = L("served_ringfix_2k")
same = 0
for r in rows:
    i = (r["cat"], r["index"]); f = u.get(i)
    ok = f is not None and r["sha"] == f["sha"]; same += ok
    tv = r["server"]["tok_verify"] if r.get("server") else None
    print(f"used 2k {i}: {r['sha']} vs u1_tree8 (fresh) {f and f['sha']} {'IDENTICAL' if ok else 'DIFFER'}"
          f" | tok/verify {tv} vs {f and f['server'] and f['server']['tok_verify']}")
print(f"used-vs-fresh 2K rung: {same}/{len(rows)} identical")
for n in ("served_ringfix_smoke1", "served_ringfix_smoke2"):
    print(n, [(r["cat"], r["index"], r["sha"]) for r in L(n)])
PYEOF
grep -E "drafter ring \(|restore '|cache: cold" $SL | sed -E 's/ \| prefilling.*//' | sed "s/^/server: /" >> $G
grep -c "all written back" $SL | sed "s/^/ring write-backs: /" | tee -a $G
echo "$(date +%T) ringfix serve done, forge pid $(lsof -tnP -iTCP:8765 -sTCP:LISTEN)" | tee -a $G
