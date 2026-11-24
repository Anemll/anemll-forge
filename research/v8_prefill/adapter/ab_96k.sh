#!/bin/zsh
# 24 Nov 2026 (yolo run): 96K A/B of the super-adapter tree build (v8_dec8ts_sa + v8_dec8ts_sa_w) against the served
# tree build (v8_dec8ts + w_dec8ts), on test port 8767 with the served launcher (v8_server_dec8_qk.sh: tree8 greedy +
# sampled, sliced 24/40/48/56K, the _uw leak fix) and libgpu_v8_f8q_uwlr.dylib (rank-generic low-rank; at rank 64 the
# served math) in BOTH arms. Base first: its smoke must give the served checksum 29aef2ec9db18a3f (the new dylib changes
# nothing at rank 64), else stop. Per arm: smoke, realistic_v1 2K / 32K (9 tasks) + 96K (3 tasks), the U1 agent set
# (tools), the long agent session (tools, 8K -> 96K). The server is stopped after each arm; nothing is left serving.
#   ARMS="base sa" ab_96k.sh
set -u
R=/Volumes/SSD4TB/anemll-forge-research; P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe
VP=${0:a:h:h}; T=$VP/bench_tree; RES=$VP/results/24nov/adapter/ab96; mkdir -p $RES
L=$R/coreai_seg_logs/v8_server.log; PORT=8767; SMOKE=29aef2ec9db18a3f
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
LOCK=$VP/results/22nov_review/windows/runner.lock
log() { echo "$(date '+%H:%M:%S') $*" | tee -a $RES/ops.log; }
reviewer_busy() { pgrep -f "window[.]zsh" > /dev/null && return 0; [ -e $LOCK ] && [ -n "$(lsof -t $LOCK 2>/dev/null)" ] && return 0; return 1; }
stop() { cd $P && PORT=$PORT zsh srv_stop.sh >> $RES/ops.log 2>&1; }
lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1 && { log "a server is on 8765 (one model at a time): stop"; exit 4; }
bench() {  # arm name prompts rungs [tasks] [extra]
  local arm=$1 nm=$2 pr=$3 rg=$4 tk=$5; shift 5
  local ta=(); [ -n "$tk" ] && ta=(--tasks $tk)
  $PY $T/tree_bench.py --prompts $T/$pr --label ${arm}_$nm --out $RES/${arm}_$nm.json --rungs $rg --port $PORT --log $L \
    $ta "$@" > $RES/${arm}_$nm.log 2>&1 || { log "$arm $nm: bench failed: $(tail -2 $RES/${arm}_$nm.log)"; return 1; }
  log "$arm $nm: $(grep SUMMARY $RES/${arm}_$nm.log | tail -1 | cut -c1-300)"
}
for arm in ${=ARMS:-base sa}; do
  while reviewer_busy; do log "reviewer window running: waiting"; sleep 60; done
  if [ $arm = sa ]; then B=$R/v8_dec8ts_sa; G=$R/v8_dec8ts_sa_w/manifest_best.json; else B=$R/v8_dec8ts; G=$P/w_dec8ts/manifest_best.json; fi
  [ -e $B/manifest.json ] && [ -e $G ] || { log "$arm: missing $B or $G"; exit 1; }
  stop
  [ -s $L ] && mv $L ${L:r}.$(date -r $L +%Y%m%d-%H%M%S).log
  log "$arm: start (BUILD $B, GM $G)"
  env PORT=$PORT BUILD=$B GM=$G V8_SRV_DYLIB=libgpu_v8_f8q_uwlr.dylib DEC_DYLIB=libdec_gpu_tree_uw.dylib \
    nohup zsh $P/v8_server_dec8_qk.sh > $RES/${arm}_launch.out 2>&1 &
  for i in $(seq 1 300); do grep -q "serving OpenAI API" $L 2>/dev/null && break; sleep 2; done
  grep -q "serving OpenAI API" $L || { log "$arm: server did not come up: $(tail -5 $L | tr '\n' ' ')"; stop; exit 1; }
  for want in "libgpu_v8_f8q_uwlr.dylib" "libdec_gpu_tree_uw.dylib" "tree: BANNER mode=tree8 policy=branched" "sampled=on" "sliced decode entries [24, 40, 48, 56]K" "model dir     $B"; do
    grep -qF "$want" $L || { log "$arm: banner lacks '$want'"; stop; exit 1; }
  done
  grep -E "tree: mode|slice|wraps|GPU engine|speculative" $L | head -8 > $RES/${arm}_banner.txt
  bench $arm smoke smoke_2k_v1.json 2k 1,8 --idle-s 5 --no-power || { stop; exit 1; }
  cs=$(grep SUMMARY $RES/${arm}_smoke.log | tail -1)
  if [ $arm = base ] && [[ $cs != *"output checksum $SMOKE"* ]]; then log "base smoke differs from $SMOKE: the uwlr dylib changed numerics, stop"; stop; exit 5; fi
  bench $arm r2k32k realistic_v1.json 2k,32k ""
  bench $arm r96k realistic_v1.json 96k 0,3,6
  bench $arm agent u1_code_agent.json agent ""
  bench $arm long long_agent_v1.json long ""
  cp $L $RES/${arm}_server.log
  grep -E "footprint" $R/coreai_seg_logs/watchdog_v8_server.log | tail -2 >> $RES/${arm}_banner.txt
  stop; log "$arm: done, server stopped"
done
log "A/B done"
