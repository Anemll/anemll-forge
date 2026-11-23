#!/bin/zsh
# 22 Nov 2026 KV-wrap leak A/B (Anemll: option 2): fresh launch of the served config (2K smoke 29aef2ec9db18a3f), then
# memleak_alt.py (cold 2K / 56K alternation, footprint after each request), for arm old (served dylibs) and arm new
# (libgpu_v8_f8q_uw.dylib + libdec_gpu_tree_uw.dylib: v8_unwrap / v8_kv8_clear / dec_unwrap). Stops the server at the
# end (not left serving; served defaults unchanged). Aborts on foreign (Pi) traffic.
T=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/bench/tree; P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe
RS=${0:a:h}; R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/22nov; O=$R/memleak; mkdir -p $O $R/bench
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
SL=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log; SMOKE=29aef2ec9db18a3f; G=$O/ab.log
UNSET=(-u BUILD -u GM -u NGRAM_DRAFT -u TREE -u SERVED_TREE -u SERVED_SLICE -u QKE -u V8_SRV_DYLIB -u DEC_DYLIB -u DRAFT_RING_SNAP -u TREE_SAMPLED
       -u COLD_SCRUB -u TREE_LOG -u TREE_LOG_DIST -u LOOKUP_OBSERVE -u LOOKUP_GATE -u DRAFT_PROF -u TREE_POLICY -u RUNG_SLICE
       -u RUNG_SLICE_CHECK -u TREE_SLICED -u STREAM_PACE -u FORGE_WD -u FORGE_FP_MAX_MB)
for arm in ${=ARMS:-old new}; do
  lab=memleak_$arm$X
  if [ $arm = new ]; then EV=(V8_SRV_DYLIB=libgpu_v8_f8q_uw.dylib DEC_DYLIB=libdec_gpu_tree_uw.dylib); VD=libgpu_v8_f8q_uw.dylib; DD=libdec_gpu_tree_uw.dylib
  else EV=(); VD=libgpu_v8_f8q.dylib; DD=libdec_gpu_tree.dylib; fi
  EX="v8=qk=attn_qk8isk_tg v8=F8=x v8=KV8=true v8=$VD dec8=$DD tree=mode=tree8 tree=policy=branched tree=sampled=on lookup=off tree=v8_dec8ts"
  echo "$(date +%T) launch $lab ($EV)" | tee -a $G
  ( cd $T && env $UNSET $EV RDIR=$R LAUNCHER=v8_server_dec8_qk.sh ARM=$lab RUNGS=2k TASKS=1,8 EXPECT="$EX" timeout 900 zsh $T/run_arm.sh ) > $O/$lab.driver.log 2>&1
  rc=$?
  grep -E "tree: BANNER|slice: BANNER|v8: GPU engine|v8: variant|dec8: " $SL | head -8 | cut -c1-230 | sed "s/^/$lab banner: /" | tee -a $G
  grep -E "SUMMARY|ABORT|output checksum" $R/bench/$lab.log | sed "s/^/$lab: /" | tee -a $G
  if [ $rc != 0 ] || ! grep -q "output checksum $SMOKE" $R/bench/$lab.log; then echo "$(date +%T) $lab: launch or smoke FAILED (rc $rc)" | tee -a $G; break; fi
  echo "$(date +%T) $lab: smoke MATCH" | tee -a $G
  $PY $RS/memleak_alt.py $O/$lab.json --cycles ${CYCLES:-3} --big ${BIG:-56000} > $O/$lab.alt.log 2>&1 & BP=$!
  while kill -0 $BP 2>/dev/null; do
    if grep -q "max_tokens 32768" $SL; then kill $BP; echo "$(date +%T) Pi traffic during $lab: abort" | tee -a $G; break; fi
    sleep 2
  done
  wait $BP; cat $O/$lab.alt.log | cut -c1-200 | sed "s/^/$lab alt: /" | tee -a $G
  grep -E "KV segments registered|\[ctx\]|old wraps" $SL | cut -c1-120 > $O/$lab.switch.txt
  cp $SL $O/$lab.server.log
  (cd $P && PORT=8765 zsh srv_stop.sh) | tee -a $G
  grep -q "Pi traffic" $G && break
done
echo "$(date +%T) DONE (server stopped)" | tee -a $G
