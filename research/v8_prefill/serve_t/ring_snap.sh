#!/bin/zsh
# 21 Nov 2026: fresh vs used server, the drafter ring in the 'system' snapshot. Fresh launches of the served
# launcher (tree8 unless SERVED_TREE=0) with DRAFT_RING_SNAP = log (control: behavior unchanged, restores logged) /
# fix (the snapshot's ring slots written back); smoke = realistic_v1 2K positions 1, 8 (tree_bench: its 'Say OK.'
# warm-up is a cold request), same-server repeats, the full 2K rung. Ends with the served default (no env) + smoke.
set -u
T=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/bench/tree
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/21nov
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
SL=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log
mkdir -p $R/ring
EXT="v8=qk=attn_qk8isk_tg v8=F8=x v8=KV8=true v8=libgpu_v8_f8q.dylib dec8=libdec_gpu_tree.dylib tree=mode=tree8 tree=policy=branched tree=v8_dec8tx lookup=off"
EXC="v8=qk=attn_qk8isk_tg v8=F8=x v8=KV8=true v8=libgpu_v8_f8q.dylib dec8=libdec_gpu_clk.dylib tree=mode=chain lookup=off"
out() { grep -E "sha [0-9a-f]{16}$|SUMMARY|ABORT" $R/bench/$1.log | sed -E "s/^\[[^]]*\] //; s/ prompt .* sha / sha /" | sed "s/^/$1: /" | tee -a $R/ring/ring.log; }
same() {   # label expect [rungs tasks]
  local EXA=(); for e in ${=2}; do EXA+=(--expect $e); done
  $PY $T/tree_bench.py --prompts $T/realistic_v1.json --label $1 --out $R/bench/$1.json --rungs 2k ${=3:-} $EXA > $R/bench/$1.log 2>&1
  out $1
}
fresh() {   # label expect env...
  local lab=$1 ex=$2; shift 2
  ( cd $T && env -u BUILD -u GM -u NGRAM_DRAFT -u TREE -u SERVED_TREE -u QKE -u V8_SRV_DYLIB -u DRAFT_RING_SNAP "$@" RDIR=$R LAUNCHER=v8_server_dec8_qk.sh ARM=$lab RUNGS=2k TASKS=1,8 EXPECT="$ex" timeout 900 zsh $T/run_arm.sh ) > $R/ring/$lab.driver.log 2>&1
  grep -E "drafter ring snapshot:|tree: BANNER|dec8: BANNER" $SL | cut -c1-200 | sed "s/^/$lab banner: /" | tee -a $R/ring/ring.log
  out $lab
}
ringlog() { grep -E "drafter ring \(|restore '|cache: cold" $SL | sed -E 's/ \| prefilling.*//' | sed "s/^/$1 server: /" >> $R/ring/ring.log; grep -c "hold another prompt" $SL | sed "s/^/$1 ring restore lines: /" | tee -a $R/ring/ring.log; }
echo "$(date +%T) ring: start" | tee -a $R/ring/ring.log
# (1) log: control, the off behavior (smoke1 / smoke2 / served_tree8_2k sequence) with the restores logged
fresh ring_log_run1 "$EXT" DRAFT_RING_SNAP=log
same ring_log_run2 "$EXT" "--tasks 1,8"
same ring_log_2k "$EXT"
ringlog ring_log
# (2) fix, tree8: the same sequence + a 3rd run
fresh ring_fix_run1 "$EXT" DRAFT_RING_SNAP=fix
same ring_fix_run2 "$EXT" "--tasks 1,8"
same ring_fix_2k "$EXT"
same ring_fix_run3 "$EXT" "--tasks 1,8"
ringlog ring_fix
# (3) fix, chain: fresh + same server
fresh ring_fixc_run1 "$EXC" DRAFT_RING_SNAP=fix SERVED_TREE=0
same ring_fixc_run2 "$EXC" "--tasks 1,8"
ringlog ring_fixc
# (4) the served default again (no DRAFT_RING_SNAP): fresh smoke
fresh ring_served_final "$EXT"
echo "$(date +%T) ring: done, forge pid $(lsof -tnP -iTCP:8765 -sTCP:LISTEN)" | tee -a $R/ring/ring.log
