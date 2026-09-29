#!/bin/zsh
# M3U helper: memory guard for run_ane7_drafter.sh (copy of dflash_guard.sh, 12:58 2026-09-27).
# Pauses (SIGSTOP) the running dflash/lowrank python while a KL eval runs (52 GB model on the GPU + ~30 GB sim do not
# fit in 96 GB: swap 4 -> 18 GB in minutes, 09:47) or on critical pressure (swap growth alone is not a trigger since 11:17:
# with pressure normal it was compression of paused state, not thrashing); resumes it (SIGCONT)
# when no KL eval runs and pressure has been normal (1) for 60 s (was <= warn until 12:25: critical at 12:24).
# Exits when run_ane7_drafter.sh is gone and never leaves a process stopped.
L=/Volumes/SN8100/vq27b
OUT=$L/dflash_guard.log
log() { echo "$(date +%H:%M:%S) $*" >> $OUT; }
swap_mb() { sysctl -n vm.swapusage | awk '{print int($6)}'; }
base=$(swap_mb); ok=0; pauses=()
resume_all() { for p in $(pgrep -f "dflash2_target_ref.py|dflash2_quant.py|qwen38_lowrank_export.py"); do
  [[ $(ps -o state= -p $p) == T* ]] && kill -CONT $p && log "GUARD exit: resumed pid $p"; done }
trap 'resume_all; exit' INT TERM HUP  # not an EXIT trap: zsh runs those when $(...) subshells exit
log "GUARD v2 started (swap base ${base} MB)"
while pgrep -f run_ane7_drafter.sh > /dev/null; do
  p=$(sysctl -n kern.memorystatus_vm_pressure_level); s=$(swap_mb); now=$(date +%s)
  pid=$(pgrep -f "dflash2_target_ref.py|dflash2_quant.py|qwen38_lowrank_export.py" | head -1)
  if [[ -n $pid ]]; then
    if [[ $(ps -o state= -p $pid) != T* ]]; then
      if pgrep -f qwen38_kl.py > /dev/null || (( p >= 4 )); then
        kill -STOP $pid && pauses+=($now) && ok=0 && log "GUARD pause pid $pid (pressure $p, swap ${s} MB, base ${base} MB)"
      fi
    else
      recent=0; for t in $pauses; do (( t > now - 600 )) && (( recent++ )); done
      if (( p == 1 )); then (( ok++ )); else ok=0; fi
      if (( ok >= 3 )) && ! pgrep -f qwen38_kl.py > /dev/null; then
        kill -CONT $pid && log "GUARD resume pid $pid (pressure $p, swap ${s} MB, pauses in 10 min: $recent)"
        base=$(swap_mb); ok=0
      fi
    fi
  fi
  sleep 20
done
resume_all
log "GUARD v2 exit (run_ane7_drafter.sh finished)"
