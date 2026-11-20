#!/bin/zsh
# Softmax-fold tests, stage A (18 Nov 2026; forge stopped): alone sweep over positions (every form on the same data in
# one process) and a per-kernel profile at 60K. Swap watchdog on every process.
O=/Volumes/SSD4TB/anemll-forge-research/contention/fold; PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
D=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power; WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh
run() { local name=$1; shift; "$@" > $O/$name.log 2>&1 & local G=$!; $WD $G 1536 $O/$name.watchdog.log >/dev/null 2>&1 & local K=$!; wait $G; local rc=$?; kill $K 2>/dev/null; echo "$name rc=$rc"; }
run sweep $PY $D/fold_check.py $O/sweep.json 2048,4096,8192,16384,32768,49152,61440,81920,95232 12 98304
PROFILE=1 run prof60k $PY $D/fold_check.py $O/prof60k.json 61440 6 65536
