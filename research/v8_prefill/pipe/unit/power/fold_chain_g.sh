#!/bin/zsh
# F8=x + folded softmax (18 Nov 2026, evening), stage 1: alone sweep on the one-layer manifest, every form in its own
# process (libgpu_v8_f8f.dylib; F8F at load): x (no fold) and x + F8F = w64 / w64d / w128 / w128d, outputs compared with
# x. Forge stopped first and left down (the e2e stage decides). Swap watchdog on every process.
P=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe; PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
D=$P/unit/power; O=/Volumes/SSD4TB/anemll-forge-research/contention/fold/f8f; WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh
mkdir -p $O; export MANIFEST=$P/w_dec8/manifest_L3.json DYLIB=libgpu_v8_f8f.dylib F8=x
run() { local name=$1; shift; "$@" > $O/$name.log 2>&1 & local G=$!; $WD $G 1536 $O/$name.watchdog.log >/dev/null 2>&1 & local K=$!; wait $G; local rc=$?; kill $K 2>/dev/null; echo "$(date +%T) $name rc=$rc"; grep -E "^BANNER|ABORT" $O/$name.log | cut -c1-260; }
echo "$(date +%T) stopping forge on 8765"; PORT=8765 $P/srv_stop.sh; sleep 30
cd $D
EXPECT_PV=attn_pv_f8v run a_x $PY f8_check.py $O/a_x 2048,8192,16384,32768,61440,95232 12 98304
for f in w64 w64d w128 w128d; do
  F8F=$f EXPECT_PV=attn_pv_f8m_$f run a_$f $PY f8_check.py $O/a_$f 2048,8192,16384,32768,61440,95232 12 98304
done
for f in x w64 w64d w128 w128d; do echo "== $f"; grep '^{' $O/a_$f.log | $PY -c "import sys,json; print(' '.join(f\"{json.loads(l)['pos']}:{json.loads(l)['ms_median']}\" for l in sys.stdin))"; done
$PY f8_check.py --compare $O/a_x.npz $O/a_w64.npz $O/a_w64d.npz $O/a_w128.npz $O/a_w128d.npz | $PY -c "
import sys, json
for l in sys.stdin:
    d = json.loads(l); print(d['file'], d['pos'], 'rel %.2e' % d['rel_rms_vs_base'], 'finite', d['finite'])"
echo "$(date +%T) stage 1 done (forge left down)"
