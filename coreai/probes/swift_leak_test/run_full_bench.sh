#!/bin/zsh
# Full Qwen3.8-27B Core AI target timed from Swift (IOSurface I/O, .userInitiated). One process per phase.
#
#   ./run_full_bench.sh dry         # parse manifest, show which packages would load (no Core AI load)
#   ./run_full_bench.sh specialize  # one-time: OS-specialize the 16 chunks + head from .aimodel, one at a time
#                                   #   (~40 s and ~1.5 GB transient each, ~11 min total, ~10.5 GB disk cache under
#                                   #    ~/Library/Caches/coreai-cache/<os build>/full-model-bench/)
#   ./run_full_bench.sh time        # the timing run: verify-8 (60 calls after 5 warm-up), prefill-64 (30 after 3),
#                                   #   per-chunk breakdown, wired memory per chunk / per entry, head
#
# Why .aimodel: the .aimodelc from Xcode-beta coreai-build hold MPSGraph package 7.1.2; macOS 27.0 (26A428) reads
# <= 7.0.80 and the load segfaults (Python too). The binary picks the source automatically (PREFER=auto).
#
# Memory: with the Qwen server resident (~22-23 GB wired) only a few chunks fit, so `time` loads chunks in rotating
# groups (GROUP, default 4: load 4, time them as a sequence, unload, next 4; full-model numbers = per-call sums).
# If the server is stopped, run  GROUP=16 ./run_full_bench.sh time  for an all-resident run and the total wired
# memory of 16 chunks x 2 entries. Guards: in-process (free+purgeable+file cache >= AVAIL_MIN_GB after each load,
# swap growth <= SWAP_MAX_GB) and this script's watchdog (kills the process if swap grows > WATCHDOG_MB).
cd "${0:A:h}"
MODEL_DIR=${MODEL_DIR:-$HOME/Models/vq27b/coreai_ane7i/mix25in_aw_cal_lr64mix}
PHASE=${1:-time}
export HEAD=${HEAD:-1} GROUP=${GROUP:-4} AVAIL_MIN_GB=${AVAIL_MIN_GB:-1.5} SWAP_MAX_GB=${SWAP_MAX_GB:-0.5}
WATCHDOG_MB=${WATCHDOG_MB:-768}
LOG=log_full_${PHASE}_$(date +%m%d_%H%M).txt

if [[ ! -x full_model_bench || full_model_bench.swift -nt full_model_bench ]]; then
  DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer xcrun swiftc -O -swift-version 5 \
    full_model_bench.swift -o full_model_bench 2>&1 | grep -E "error:" && exit 1
fi
case $PHASE in
  dry) export DRY_RUN=1 ;;
  specialize) export SPECIALIZE_ONLY=1 ;;
  time) ;;
  *) echo "usage: $0 dry|specialize|time"; exit 2 ;;
esac

swap_mb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6}' }
wired() { vm_stat | awk '/page size of/{ps=$8} /wired down/{gsub("\\.","",$4); printf "%.2f GB", $4*ps/2^30}' }
S0=$(swap_mb)
echo "phase $PHASE, GROUP=$GROUP HEAD=$HEAD AVAIL_MIN_GB=$AVAIL_MIN_GB SWAP_MAX_GB=$SWAP_MAX_GB, swap used ${S0} MB, wired $(wired)" | tee $LOG
./full_model_bench $MODEL_DIR >> $LOG 2>&1 &
PID=$!
while kill -0 $PID 2>/dev/null; do
  S=$(swap_mb)
  if (( S - S0 > WATCHDOG_MB )); then
    echo "WATCHDOG: swap grew ${S0} -> ${S} MB, killing $PID" | tee -a $LOG
    kill $PID; break
  fi
  sleep 2
done
wait $PID; echo "exit code $?" >> $LOG
sleep 3; echo "3 s after exit: wired $(wired), swap used $(swap_mb) MB" >> $LOG
cat $LOG
