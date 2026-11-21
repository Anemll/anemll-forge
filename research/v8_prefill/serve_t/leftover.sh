#!/bin/zsh
# 21 Nov 2026 (Anemll 10:45, Pi idle): the 9-prompt used-vs-fresh check on the served config, then the chat:2 leftover.
#   1 used     the running server (used by the sampled A/B: 8K / 32K / code / agent) -> realistic 2K rung, all 9
#   2 fresh    restart (served config) -> 2K rung (the fresh reference; smoke = positions 1, 8); then one long sampled
#              request (realistic 8k task 0, thinking on, temperature 1.0, seed 777, 1500 tokens: ~9K positions, past the
#              8K entry, as this morning's Pi request) -> 2K rung again (control: the leftover reproduced?)
#   3 scrub    restart with COLD_SCRUB=kv (a cold start puts every KV row back in its fresh state) -> 2K rung -> the
#              same long request -> 2K rung (fixed?)
#   4 served   restart (served config, no research env) -> smoke; stays up
set -u
T=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/bench/tree
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/21nov
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
SL=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log
O=$R/leftover; G=$O/leftover.log
mkdir -p $O $R/bench
EX="v8=qk=attn_qk8isk_tg v8=F8=x v8=KV8=true v8=libgpu_v8_f8q.dylib dec8=libdec_gpu_tree.dylib tree=mode=tree8 tree=policy=branched tree=v8_dec8tx tree=sampled=on lookup=off"
EXA=(); for e in ${=EX}; do EXA+=(--expect $e); done
out() { grep -E "sha [0-9a-f]{16}$|SUMMARY|ABORT" $R/bench/$1.log | sed -E "s/^\[[^]]*\] //; s/ prompt .* sha / sha /" | sed "s/^/$1: /" | tee -a $G; }
rung() { $PY $T/tree_bench.py --prompts $T/realistic_v1.json --label $1 --out $R/bench/$1.json --rungs 2k --idle-s 5 $EXA > $R/bench/$1.log 2>&1; out $1; }
pollute() { $PY $T/tree_bench.py --prompts $T/realistic_v1.json --label $1 --out $R/bench/$1.json --rungs 8k --tasks 0 --temp 1.0 --think 1 \
              --seed 777 --max-tokens 1500 --idle-s 5 $EXA > $R/bench/$1.log 2>&1; out $1; grep -E "\[ctx\]" $SL | tail -2 | sed "s/^/$1 server: /" | tee -a $G; }
fresh() {   # label env...
  local lab=$1; shift
  ( cd $T && env -u BUILD -u GM -u NGRAM_DRAFT -u TREE -u SERVED_TREE -u QKE -u V8_SRV_DYLIB -u DRAFT_RING_SNAP -u TREE_SAMPLED -u COLD_SCRUB \
      "$@" RDIR=$R LAUNCHER=v8_server_dec8_qk.sh ARM=$lab RUNGS=2k EXPECT="$EX" timeout 900 zsh $T/run_arm.sh ) > $O/$lab.driver.log 2>&1
  grep -E "drafter ring snapshot:|v8: variant|dec8: decode GPU engine|tree: BANNER|serving OpenAI" $SL | cut -c1-240 | sed "s/^/$lab banner: /" | tee -a $G
  out $lab
}
savelog() { grep -vE "^   (state|prompt)( )?:" $SL > $O/$1.server.log; }
echo "$(date +%T) leftover: start (forge pid $(lsof -tnP -iTCP:8765 -sTCP:LISTEN), used by the sampled A/B)" | tee -a $G
grep -E "drafter ring snapshot:|tree: BANNER" $SL | cut -c1-240 | sed "s/^/lo_used_ab banner: /" | tee -a $G
rung lo_used_ab
savelog s1_used
fresh lo_fresh_2k
pollute lo_pollute_a
rung lo_used_ctl
savelog s2_fresh_ctl
fresh lo_scrub_fresh_2k COLD_SCRUB=kv
pollute lo_pollute_b
rung lo_scrub_used_2k
grep -E "cold scrub" $SL | head -3 | sed "s/^/scrub server: /" | tee -a $G
savelog s3_scrub
( cd $T && env -u BUILD -u GM -u NGRAM_DRAFT -u TREE -u SERVED_TREE -u QKE -u V8_SRV_DYLIB -u DRAFT_RING_SNAP -u TREE_SAMPLED -u COLD_SCRUB \
    RDIR=$R LAUNCHER=v8_server_dec8_qk.sh ARM=lo_served_smoke RUNGS=2k TASKS=1,8 EXPECT="$EX" timeout 900 zsh $T/run_arm.sh ) > $O/lo_served_smoke.driver.log 2>&1
grep -E "drafter ring snapshot:|v8: variant|dec8: BANNER|dec8: decode GPU engine|tree: BANNER|keep-alive|serving OpenAI" $SL | cut -c1-260 | sed "s/^/served banner: /" | tee -a $G
out lo_served_smoke
$PY - <<PYEOF | tee -a $G
import json
L = lambda n: {(r["cat"], r["set"], r["index"]): r for r in json.load(open(f"$R/bench/{n}.json"))["rungs"]["2k"]}
F = L("lo_fresh_2k")
U1 = {(r["cat"], r["set"], r["index"]): r for r in json.load(open("$R/../20nov/bench/u1_tree8.json"))["rungs"]["2k"]}
for n, ref, rn in (("lo_fresh_2k", U1, "u1_tree8 (20 Nov, fresh)"), ("lo_used_ab", F, "fresh"), ("lo_used_ctl", F, "fresh"),
                   ("lo_scrub_fresh_2k", F, "fresh"), ("lo_scrub_used_2k", F, "fresh")):
    A = L(n)
    diff = [f"{k[0]}:{k[1]}:{k[2]}" for k in A if k in ref and A[k]["sha"] != ref[k]["sha"]]
    print(f"{n} vs {rn}: {sum(A[k]['sha'] == ref[k]['sha'] for k in A if k in ref)}/{len(A)} identical" + (f"; differ: {', '.join(diff)}" if diff else ""))
pa, pb = (json.load(open(f"$R/bench/{n}.json"))["rungs"]["8k"][0]["sha"] for n in ("lo_pollute_a", "lo_pollute_b"))
print(f"long request (seed 777) identical in both arms: {pa == pb} ({pa} / {pb})")
PYEOF
savelog s4_served
echo "$(date +%T) leftover: done, forge pid $(lsof -tnP -iTCP:8765 -sTCP:LISTEN)" | tee -a $G
