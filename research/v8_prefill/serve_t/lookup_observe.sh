#!/bin/zsh
# 21 Nov 2026 (M6 ANE bot batch: lookup gate + combined tree, data). Runs only while Pi is idle. One fresh launch of
# the served config + NGRAM_DRAFT=6 LOOKUP_OBSERVE=1 TREE_LOG TREE_LOG_DIST=1: the tree serves every cycle (no lookup
# cycle is taken), and each cycle records the lookup proposal (drafts, match length), the drafter inputs, the tree, the
# accepted path and every verified row's sampling distribution. Workloads (the seeds of the earlier lookup A/Bs):
#   lgo_g_ca  u1 code + agent, greedy (320 tok)          lgo_s_ca  the same, sampled seed 4100 (384 tok)
#   lgo_g_rm  realistic math + chat (tasks 3-8) at 2k / 8k / 32k, greedy    lgo_s_rm  the same, sampled seed 5100
# Then the served config is restored (launcher defaults) and smoke-tested. Arrays for every argument list (zsh).
set -u
T=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/bench/tree
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/21nov
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
SL=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log
TL=/Volumes/SSD4TB/anemll-forge-research/treelog_lgo21
O=$R/lookup_gate; G=$O/run.log
mkdir -p $O $R/bench
BASE=(v8=qk=attn_qk8isk_tg v8=F8=x v8=KV8=true v8=libgpu_v8_f8q.dylib dec8=libdec_gpu_tree.dylib tree=mode=tree8 tree=policy=branched tree=v8_dec8tx tree=sampled=on)
EXOBS=($BASE tree=NGRAM_DRAFT=); EXOFF=($BASE lookup=off)
out() { grep -E "SUMMARY|ABORT" $R/bench/$1.log | sed "s/^/$1: /" | tee -a $G; }
pi_idle() {
  local i quiet=0 last now age
  for i in $(seq 1 540); do
    if [ "$(lsof -nP -iTCP:8765 -sTCP:ESTABLISHED 2>/dev/null | tail -n +2 | wc -l | tr -d ' ')" = 0 ]; then quiet=$((quiet + 5)); else quiet=0; fi
    last=$(grep -oE "^\[[0-9:]+\] request: prompt" $SL 2>/dev/null | tail -1 | cut -c2-9)
    now=$(date +%s); age=999
    [ -n "$last" ] && age=$(( now - $(date -j -f "%Y-%m-%d %H:%M:%S" "$(date +%Y-%m-%d) $last" +%s) ))
    [ $quiet -ge 60 ] && [ $age -ge 120 ] && return 0
    sleep 5
  done
  return 1
}
bench() {   # label prompts rungs tasks(or -) temp(0 = greedy) seed
  local args=(--prompts $T/$2 --label $1 --out $R/bench/$1.json --rungs $3 --idle-s 5)
  [ "$4" != "-" ] && args+=(--tasks $4)
  [ "$5" != "0" ] && args+=(--temp $5 --think 1 --seed $6 --max-tokens 384)
  for e in "${EXOBS[@]}"; do args+=(--expect "$e"); done
  $PY $T/tree_bench.py "${args[@]}" > $R/bench/$1.log 2>&1
  out $1
}
echo "$(date +%T) lookup observe: waiting for Pi idle" | tee -a $G
pi_idle || { echo "$(date +%T) Pi not idle for 45 min: nothing run" | tee -a $G; exit 2; }
echo "$(date +%T) Pi idle: launch the observe server" | tee -a $G
rm -rf $TL; mkdir -p $TL
( cd $T && env -u BUILD -u GM -u NGRAM_DRAFT -u TREE -u SERVED_TREE -u QKE -u V8_SRV_DYLIB -u DRAFT_RING_SNAP -u TREE_SAMPLED -u COLD_SCRUB \
    -u TREE_LOG -u TREE_LOG_DIST -u LOOKUP_OBSERVE NGRAM_DRAFT=6 LOOKUP_OBSERVE=1 TREE_LOG=$TL TREE_LOG_DIST=1 \
    RDIR=$R LAUNCHER=v8_server_dec8_qk.sh ARM=lgo_b RUNGS=2k TASKS=1 EXPECT="${EXOBS[*]}" timeout 900 zsh $T/run_arm.sh ) > $O/lgo.driver.log 2>&1
grep -E "drafter ring snapshot:|tree: BANNER|prompt-lookup drafts:" $SL | cut -c1-230 | sed "s/^/lgo banner: /" | tee -a $G
if grep -q "SUMMARY" $R/bench/lgo_b.log; then
  bench lgo_g_ca u1_code_agent.json code,agent - 0 0
  bench lgo_s_ca u1_code_agent.json code,agent - 1.0 4100
  bench lgo_g_rm realistic_v1.json 2k,8k,32k 3,4,5,6,7,8 0 0
  bench lgo_s_rm realistic_v1.json 2k,8k,32k 3,4,5,6,7,8 1.0 5100
  echo "lgo: Pi requests: $(grep -c 'max_tokens 32768' $SL), errors: $(grep -cE 'Traceback|StopIteration' $SL), lookup cycles taken: $(grep -oE 'lookup [0-9]+/' $SL | grep -vc 'lookup 0/'), tree log files $(ls $TL | wc -l | tr -d ' ') ($(du -sh $TL | cut -f1))" | tee -a $G
  grep -vE "^   (state|prompt)( )?:" $SL > $O/lgo.server.log
else
  echo "lgo: launch failed" | tee -a $G; tail -5 $O/lgo.driver.log | tee -a $G
fi
echo "$(date +%T) restore the served config" | tee -a $G
( cd $T && env -u BUILD -u GM -u NGRAM_DRAFT -u TREE -u SERVED_TREE -u QKE -u V8_SRV_DYLIB -u DRAFT_RING_SNAP -u TREE_SAMPLED -u COLD_SCRUB \
    -u TREE_LOG -u TREE_LOG_DIST -u LOOKUP_OBSERVE RDIR=$R LAUNCHER=v8_server_dec8_qk.sh ARM=served_after_lgo RUNGS=2k TASKS=1,8 EXPECT="${EXOFF[*]}" timeout 900 zsh $T/run_arm.sh ) > $O/restore.driver.log 2>&1
grep -E "drafter ring snapshot:|tree: BANNER|serving OpenAI" $SL | cut -c1-230 | sed "s/^/served banner: /" | tee -a $G
out served_after_lgo
echo "$(date +%T) done, forge pid $(lsof -tnP -iTCP:8765 -sTCP:LISTEN)" | tee -a $G
