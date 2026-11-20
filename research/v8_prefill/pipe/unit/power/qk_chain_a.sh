#!/bin/zsh
# Q.K staged epilogue (19 Nov 2026), kernel-level A/B on the one-layer manifest (f8_check.py, layer 3, random INT8 KV):
# served F8=x vs F8=x + QKE=tg, the same dylib (libgpu_v8_f8q.dylib), interleaved x / tg / x / tg, one process each,
# then a PROFILE run of each (per-kernel GPU ms). Forge stays up (idle); no model load. Swap watchdog on every process.
P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe; PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
D=$P/unit/power; O=/Volumes/SSD4TB/anemll-forge-research/contention/qkf8/alone; WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh
mkdir -p $O; export MANIFEST=$P/w_dec8/manifest_L3.json DYLIB=libgpu_v8_f8q.dylib F8=x EXPECT_PV=attn_pv_f8v
POS=32768,61440,95232
run() { local name=$1; shift; "$@" > $O/$name.log 2>&1 & local G=$!; $WD $G 1536 $O/$name.watchdog.log >/dev/null 2>&1 & local K=$!; wait $G; local rc=$?; kill $K 2>/dev/null; echo "$(date +%T) $name rc=$rc"; grep -E "^BANNER|ABORT" $O/$name.log | cut -c1-330; }
cd $D
for r in a b; do
  QKE= EXPECT_QK=attn_qk8isk run x_$r $PY f8_check.py $O/x_$r $POS 16 98304; sleep 10
  QKE=tg EXPECT_QK=attn_qk8isk_tg run tg_$r $PY f8_check.py $O/tg_$r $POS 16 98304; sleep 10
done
QKE= EXPECT_QK=attn_qk8isk PROFILE=1 run px $PY f8_check.py $O/px 95232 8 98304
QKE=tg EXPECT_QK=attn_qk8isk_tg PROFILE=1 run ptg $PY f8_check.py $O/ptg 95232 8 98304
for f in x_a tg_a x_b tg_b; do echo "== $f $(grep '^{' $O/$f.log | $PY -c "import sys,json; print(' '.join(f\"{json.loads(l)['pos']}:{json.loads(l)['ms_median']}\" for l in sys.stdin))")"; done
$PY f8_check.py --compare $O/x_a.npz $O/x_b.npz $O/tg_a.npz $O/tg_b.npz | $PY -c "
import sys, json
for l in sys.stdin:
    d = json.loads(l); print(d['file'], d['pos'], 'rel %.2e' % d['rel_rms_vs_base'], 'maxabs', d['max_abs'], 'finite', d['finite'])"
echo "$(date +%T) done"
