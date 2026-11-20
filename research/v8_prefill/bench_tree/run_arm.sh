#!/bin/zsh
# 20 Nov 2026 tree study: one bench arm = (re)start forge on 8765 with the given launcher + env, wait until it serves,
# run tree_bench.py on the realistic set with banner expectations, leave the server up.
#   LAUNCHER=v8_server_tree.sh|v8_server_dec8_qk.sh ARM=<label> [RUNGS=2k,8k,...] [TASKS=0,1] [EXPECT="k=v k=v"] [env...] run_arm.sh
set -u
P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe; T=${0:a:h}
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/20nov
L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
cd $P && PORT=8765 zsh srv_stop.sh
echo "$(date '+%H:%M') arm $ARM: start $LAUNCHER ($(env | grep -E '^(TREE|DEC_|BUILD|GM|V8_)' | tr '\n' ' '))" >> $R/ops.log
[ -s $L ] && mv $L ${L:r}.$(date -r $L +%Y%m%d-%H%M%S).log   # so the wait below reads the new server's log
PORT=8765 nohup zsh $P/$LAUNCHER > /dev/null 2>&1 &
for i in $(seq 1 120); do grep -q "serving OpenAI API" $L 2>/dev/null && break; sleep 2; done
grep -q "serving OpenAI API" $L || { echo "server did not come up"; tail -30 $L; exit 1; }
EX=(); for e in ${=EXPECT:-}; do EX+=(--expect $e); done
$PY $T/tree_bench.py --prompts $T/realistic_v1.json --label $ARM --out $R/bench/$ARM.json --rungs ${RUNGS:-2k,8k,32k,64k,96k} \
  ${=TASKS:+--tasks $TASKS} $EX 2>&1 | tee $R/bench/$ARM.log
