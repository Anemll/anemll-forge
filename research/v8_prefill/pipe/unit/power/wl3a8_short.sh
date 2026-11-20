#!/bin/zsh
# Queue 4 addendum: WL3A8 vs FP16 FFN at 2K / 4K / 8K, interleaved base / wl / base / wl, FFN-only harness.
P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe; PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
D=$P/unit/power; R=/Volumes/SSD4TB/anemll-forge-research/contention/wl3a8; BASE=/Volumes/SSD4TB/anemll-forge-research/v8_ffn2c; WL=/Volumes/SSD4TB/anemll-forge-research/scratch/v8_ffn8x
for T in sbase_a swl_a sbase_b swl_b; do
  [[ $T == sbase* ]] && F=$BASE || F=$WL
  mkdir -p $R/tr_$T; echo "$(date +%T) trace $T"
  env FFN_ONLY_DIR=$F V8_DYLIB=libgpu_v8_f8q.dylib F8=x QKE=tg EXPECT_F8=x EXPECT_QK=attn_qk8isk_tg $D/v8_trace.sh $R/tr_$T 2048,2048,4096,8192
  grep -E "ABORT|Traceback" $R/tr_$T.log; grep "^run" $R/tr_$T.log; sleep 20
done
$PY $D/logits_cmp.py $R/tr_sbase_a $R/tr_swl_a | grep -v '"tokens": 2048, .*"kl_a_b": 0.0' | cut -c1-200
