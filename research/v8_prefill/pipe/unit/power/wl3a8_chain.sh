#!/bin/zsh
# Queue 4 (19 Nov 2026): WL3A8 FFN on the ANE vs the served FP16 FFN, prefill only, through the FFN-only harness
# (ffn_only.py via FFN_ONLY_DIR: no served build packages loaded, one copy of the FFN weights per process):
#   base  v8_ffn2c (FP16 FFN-only build = the served FFN)      wl  the TEMPORARY WL3A8 build (scratch/v8_ffn8x)
# GPU engine as served (libgpu_v8_f8q.dylib, F8=x, QKE=tg). 1) smoke 4K: base against the served path's trace
# (bit-exact required), wl against base; 2) interleaved base / wl / base / wl at 95K, 64K, 32K; 3) teacher-forced KL of
# wl at 32K and 95K against the served F8=x captures (tf_kl/f8x; QKE=tg is bit-exact to it). Forge stays down.
P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe; PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
D=$P/unit/power; R=/Volumes/SSD4TB/anemll-forge-research/contention/wl3a8; mkdir -p $R
BASE=/Volumes/SSD4TB/anemll-forge-research/v8_ffn2c; WL=/Volumes/SSD4TB/anemll-forge-research/scratch/v8_ffn8x
SERVED4K=/Volumes/SSD4TB/anemll-forge-research/contention/gdn/g256_e2e/tr_s_smoke
lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1 && { echo "$(date +%T) stopping forge on 8765"; PORT=8765 $P/srv_stop.sh; sleep 30; }
run() {   # run NAME FFN_DIR LENGTHS
  mkdir -p $R/tr_$1; echo "$(date +%T) trace $1 $3 (swap $(sysctl -n vm.swapusage | awk '{print $6}'))"
  env FFN_ONLY_DIR=$2 V8_DYLIB=libgpu_v8_f8q.dylib F8=x QKE=tg EXPECT_F8=x EXPECT_QK=attn_qk8isk_tg $D/v8_trace.sh $R/tr_$1 $3
  grep -E "^BANNER|ABORT|Traceback|Error" $R/tr_$1.log | cut -c1-400
  grep -q -E "^ABORT|Traceback" $R/tr_$1.log && { echo "$1 failed: stopping"; exit 1; }
  $PY $D/trace_analyze.py $R/tr_$1 > /dev/null 2>$R/tr_$1/analyze.err
  grep "^run" $R/tr_$1.log; sleep 30
}
run base_smoke $BASE 4096; run wl_smoke $WL 4096
$PY $D/logits_cmp.py $SERVED4K $R/tr_base_smoke $R/tr_wl_smoke | tee $R/smoke_cmp.jsonl
$PY -c "
import json, sys
d = [json.loads(l) for l in open('$R/smoke_cmp.jsonl')]
b = [x for x in d if x['b'] == 'tr_base_smoke']; w = [x for x in d if x['b'] == 'tr_wl_smoke']
ok = b and all(x['bitexact'] for x in b) and w and all(x['argmax_same'] for x in w)
print('smoke', 'OK' if ok else 'FAILED', 'base bit-exact vs served:', [x['bitexact'] for x in b], 'wl KL:', [round(x['kl_a_b'], 5) for x in w]); sys.exit(0 if ok else 1)" || { echo "smoke failed: stopping"; exit 1; }
for T in base_a wl_a base_b wl_b; do
  [[ $T == base* ]] && F=$BASE || F=$WL
  run $T $F 95232,65536,32768
done
$PY $D/logits_cmp.py $R/tr_base_a $R/tr_base_b $R/tr_wl_a $R/tr_wl_b | tee $R/ab_cmp.jsonl
echo "$(date +%T) teacher-forced capture + KL (wl at 32K, 95K vs tf_kl/f8x)"
TF=/Volumes/SSD4TB/anemll-forge-research/contention/fold/f8/tf_kl
env FFN_ONLY_DIR=$WL V8_DYLIB=libgpu_v8_f8q.dylib F8=x QKE=tg EXPECT_F8=x $D/tf_kl.sh $TF wl3a8 32768,95232 --kl-against $TF/f8x
grep -E "^BANNER|^captured|^prefix|ABORT|^\{" $TF/wl3a8.log | cut -c1-600
echo "$(date +%T) all done (forge left down)"
