#!/bin/zsh
# V8 driver run with the PID swap watchdog only (no export_launch guard: it counts reclaimable file cache as used).
# usage: run_wd.sh TOKENS REF TAG [driver args...]
L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs; PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
F=/Users/anemll/Documents/Codex/ios-prefill-20261030/claude_full96_prep/freeze_kv_slice_v1/v8_prefill.py
W=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work; N=$1; REF=$2; TAG=$3; shift 3
REFARG=$W/v8/$REF; [ "$REF" = run-a ] && REFARG=run-a
MAN=${MAN:-$W/pipe/manifest_all64.json}; BUILD=${BUILD:-/Volumes/SSD4TB/anemll-forge-research/coreai_seg2/release_vq3pA_mixh_s600_k1_mat_kv8}
echo "$(date +%T) start $TAG pressure $(sysctl -n kern.memorystatus_vm_pressure_level) swap $(sysctl -n vm.swapusage | awk '{print $6}')"
env MODEL=/Volumes/SSD4TB/Qwen3.8-27B MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 $PY $F --build $BUILD --ffn /Volumes/SSD4TB/anemll-forge-research/v8_ffn2c --gpu-manifest $MAN --tokens $N --ref $REFARG --mode v8 --pair --streams 4 --out $L/${TAG}.json "$@" > $L/${TAG}.log 2>&1 & G=$!
/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh $G 1024 $L/watchdog_${TAG}.log >/dev/null 2>&1 &
wait $G; rc=$?
echo "$(date +%T) done $TAG rc=$rc pressure $(sysctl -n kern.memorystatus_vm_pressure_level) swap $(sysctl -n vm.swapusage | awk '{print $6}')"; tail -2 $L/watchdog_${TAG}.log | tail -1
