#!/bin/zsh
# After fold_chain_e.sh (forge already down): (1) the K8V16 fill path under FP8 (f8_fill_check.sh: plain, then F8=x);
# (2) the missing prefill points 4K / 16K / 48K, interleaved plain / F8=x / plain / F8=x, one fresh process each, as
# fold_chain_c.sh's 2K-64K table (lengths in one process, v8_trace's 45 s idle before each run, 30 s between
# processes), banner from the loaded dylib, abort on a mismatch. Forge NOT restarted here (decided after).
P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe; PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
D=$P/unit/power; F=/Volumes/SSD4TB/anemll-forge-research/contention/fold/f8
while pgrep -f fold_chain_e.sh >/dev/null; do sleep 10; done
lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1 && { echo "forge is up on 8765: stop it first"; exit 1; }
echo "$(date +%T) fill check (swap $(sysctl -n vm.swapusage | awk '{print $6}'))"
$D/f8_fill_check.sh $F/fill $F/tf_kl
sleep 30
R=$F/pts; mkdir -p $R
for T in plain_a f8x_a plain_b f8x_b; do
  mkdir -p $R/tr_$T; echo "$(date +%T) trace $T (swap $(sysctl -n vm.swapusage | awk '{print $6}'))"
  if [[ $T == f8x* ]]; then env V8_DYLIB=libgpu_v8_f8.dylib F8=x EXPECT_F8=x $D/v8_trace.sh $R/tr_$T 4096,16384,49152
  else env -u F8 V8_DYLIB=libgpu_v8_f8.dylib EXPECT_F8=off $D/v8_trace.sh $R/tr_$T 4096,16384,49152; fi
  grep -E "^BANNER|ABORT" $R/tr_$T.log | cut -c1-300
  grep -q "^ABORT" $R/tr_$T.log && { echo "variant mismatch: stopping"; exit 1; }
  $PY $D/trace_analyze.py $R/tr_$T > /dev/null 2>$R/tr_$T/analyze.err
  grep "^run" $R/tr_$T.log
  sleep 30
done
$PY $D/logits_cmp.py $R/tr_plain_a $R/tr_plain_b $R/tr_f8x_a $R/tr_f8x_b
echo "$(date +%T) all done (forge left down)"
