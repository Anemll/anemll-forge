#!/usr/bin/env bash
# swap_watchdog.sh <pid> [WATCHDOG_MB=1024] [log]: while <pid> runs, kill it (and its children) if swap grows more
# than WATCHDOG_MB over the swap used at start (the guard of coreai/swift_bridge/validate_full_model.sh).
PID=$1; MB=${2:-1024}; LOG=${3:-/dev/stdout}
swap_mb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); printf "%d", $6}'; }
S0=$(swap_mb); PEAK=$S0
echo "$(date +%T) watchdog armed: pid $PID ($(ps -o comm= -p $PID)), swap ${S0} MB, limit +${MB} MB" >> "$LOG"
while kill -0 "$PID" 2>/dev/null; do
  S=$(swap_mb); (( S > PEAK )) && PEAK=$S
  if (( S - S0 > MB )); then
    echo "$(date +%T) WATCHDOG: swap grew ${S0} -> ${S} MB; killing pid $PID and children" >> "$LOG"
    pkill -TERM -P "$PID"; kill -TERM "$PID"; exit 1
  fi
  sleep 2
done
echo "$(date +%T) watchdog done: pid $PID exited, swap ${S0} -> $(swap_mb) MB (peak ${PEAK})" >> "$LOG"
