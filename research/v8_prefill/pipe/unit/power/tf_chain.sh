#!/bin/zsh
# Teacher-forced KL, FP8 x vs plain, at 32K and 95K (18 Nov 2026): plain capture, then F8=x capture + head + KL, one
# fresh process each, the same binary (libgpu_v8_f8.dylib with v8_variant). Forge must already be stopped.
D=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power; R=/Volumes/SSD4TB/anemll-forge-research/contention/fold/f8/tf_kl
lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1 && { echo "forge is up on 8765: stop it first"; exit 1; }
mkdir -p $R
echo "$(date +%T) plain capture (swap $(sysctl -n vm.swapusage | awk '{print $6}'))"
env -u F8 V8_DYLIB=libgpu_v8_f8.dylib EXPECT_F8=off $D/tf_kl.sh $R plain 32768,95232
grep -E "^BANNER|^captured|^prefix|ABORT" $R/plain.log
grep -q "^prefix check" $R/plain.log || { echo "plain capture failed"; exit 1; }
sleep 20
echo "$(date +%T) F8=x capture + KL (swap $(sysctl -n vm.swapusage | awk '{print $6}'))"
env V8_DYLIB=libgpu_v8_f8.dylib F8=x EXPECT_F8=x $D/tf_kl.sh $R f8x 32768,95232 --kl-against $R/plain
grep -E "^BANNER|^captured|^prefix|ABORT|^\{" $R/f8x.log
echo "$(date +%T) done (swap $(sysctl -n vm.swapusage | awk '{print $6}'))"
