#!/usr/bin/env bash
# forge_watchdog.sh <pid> [FP_MAX_MB=12288] [log] (22 Nov 2026, Anemll): the served forge's own memory guard. Kills <pid>
# (and its children) only for forge's own growth:
#   - forge's phys_footprint (footprint(1): its malloc, IOSurface KV, GPU buffers) above FP_MAX_MB, or
#   - system swap up more than SWAP_MB (6144) since arming AND forge's footprint up >= FP_SHARE_MB (1024) since its
#     first loaded sample (forge part of a real memory emergency).
# Swap growth from other processes is logged (every +512 MB step), never acted on. swap_watchdog.sh (system swap only)
# stays for build / bench jobs. Samples every 2 s; logs footprint peaks per 1 GB step.
PID=$1; FPMAX=${2:-12288}; LOG=${3:-/dev/stdout}; SWAP_MB=${SWAP_MB:-6144}; FP_SHARE_MB=${FP_SHARE_MB:-1024}
swap_mb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); printf "%d", $6}'; }
fp_mb() { footprint "$PID" 2>/dev/null | awk '/Footprint:/ {for (i=1;i<=NF;i++) if ($i=="Footprint:") {v=$(i+1); u=$(i+2)}} END {if (u=="GB") v*=1024; else if (u=="KB") v/=1024; printf "%d", v}'; }
S0=$(swap_mb); SPEAK=$S0; SSTEP=$S0; FP0=""; FPEAK=0; FSTEP=0
echo "$(date +%T) forge watchdog armed: pid $PID ($(ps -o comm= -p $PID)), footprint max ${FPMAX} MB, swap ${S0} MB (kill only with forge's own growth: swap +${SWAP_MB} and footprint +${FP_SHARE_MB})" >> "$LOG"
while kill -0 "$PID" 2>/dev/null; do
  S=$(swap_mb); F=$(fp_mb); F=${F:-0}
  (( S > SPEAK )) && SPEAK=$S
  (( F > FPEAK )) && FPEAK=$F
  [ -z "$FP0" ] && (( F > 2048 )) && FP0=$F && echo "$(date +%T) forge loaded: footprint ${F} MB" >> "$LOG"
  if (( F / 1024 > FSTEP )); then FSTEP=$(( F / 1024 )); echo "$(date +%T) footprint ${F} MB (swap ${S} MB)" >> "$LOG"; fi
  if (( S - SSTEP >= 512 )); then SSTEP=$S; echo "$(date +%T) swap ${S0} -> ${S} MB (forge footprint ${F} MB): logged only" >> "$LOG"; fi
  if (( F > FPMAX )); then
    echo "$(date +%T) FORGE WATCHDOG: footprint ${F} MB > ${FPMAX} MB; killing pid $PID and children" >> "$LOG"
    pkill -TERM -P "$PID"; kill -TERM "$PID"; exit 1
  fi
  if [ -n "$FP0" ] && (( S - S0 > SWAP_MB )) && (( F - FP0 >= FP_SHARE_MB )); then
    echo "$(date +%T) FORGE WATCHDOG: swap ${S0} -> ${S} MB with forge footprint ${FP0} -> ${F} MB; killing pid $PID and children" >> "$LOG"
    pkill -TERM -P "$PID"; kill -TERM "$PID"; exit 1
  fi
  sleep 2
done
echo "$(date +%T) forge watchdog done: pid $PID exited, footprint peak ${FPEAK} MB, swap ${S0} -> $(swap_mb) MB (peak ${SPEAK})" >> "$LOG"
