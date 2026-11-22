#!/bin/zsh
# Run on m6.local only during an explicitly approved, exclusive 8765 bench window.
# zsh window.zsh observe  |  zsh window.zsh confirm CANDIDATE
set -eu
setopt pipefail
umask 077
C=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/22nov_review
PY=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python
SSD=/Volumes/SSD4TB/anemll-forge-research
L=$SSD/coreai_seg_logs/v8_server.log
P=$C/pipe
H=${0:a:h}
[[ $H == $R/lookup_adaptation && -r $R/window_bench.py ]] || exit 2
MODE=${1:-}
case "$MODE" in
  observe) arms=(observe); fixture=$C/bench/tree/long_agent_v1.json; seed=7100 ;;
  confirm) [[ $# == 2 && ($2 == feedback_1 || $2 == feedback_3 || $2 == shadow_3 || $2 == recent_floor || $2 == yield || $2 == ctx) && -r $H/holdout.json ]] || exit 2
           candidate=$2; arms=(off yield $candidate $candidate yield off); [[ $candidate != yield ]] || arms=(off yield yield off); fixture=$H/holdout.json; seed=8700 ;;
  *) exit 2 ;;
esac
fail() { print -u2 -- "$1"; return 1; }
mkdir -p "$R/windows"
mkdir -p "$H/runs"
D=$H/runs/$(date -u +%Y%m%dT%H%M%SZ)_${MODE}_$$
mkdir "$D"
exec > >(tee "$D/driver.log") 2>&1
# The kernel releases the lock on exit. Keep the lock file as part of the window records.
lock=$R/windows/runner.lock
: >> "$lock"
zmodload zsh/system
zsystem flock -t 0 -f lease "$lock" || { fail 'Another bench window owns the lock.'; exit 1; }
probe_pid=0
owned=0
lp=0
changed=0
listener() { /usr/sbin/lsof -tnP -iTCP:8765 -sTCP:LISTEN 2>/dev/null || true; }
identity() {
  [[ $1 == <1-> ]] || return 1
  local cmdline
  cmdline=$(/bin/ps -o command= -p "$1") || return 1
  [[ $cmdline == *"$P/v8_server.py "* && $cmdline == *' --port 8765 '* && $cmdline == *' --host 127.0.0.1 '* ]]
}
wait_gone() {
  local pid=$1 j
  for j in {1..60}; do
    /bin/kill -0 "$pid" 2>/dev/null || return 0
    sleep 1
  done
  fail 'Server still exiting; a second model cannot be loaded.'
}
stop_owned() {
  local pid
  pid=$(listener)
  if [[ -n $pid ]]; then
    [[ $pid == $owned ]] && identity "$pid" || { fail 'Port owner changed. No signal sent.'; return 1; }
  fi
  if (( owned > 0 )) && /bin/kill -0 "$owned" 2>/dev/null; then
    identity "$owned" || { fail 'Owned process identity changed. No signal sent.'; return 1; }
    while [[ -n $(/usr/sbin/lsof -tnP -iTCP:8765 -sTCP:ESTABLISHED 2>/dev/null || true) ]]; do
      sleep 3
      [[ $(listener) == $owned ]] && identity "$owned" || { fail 'Service owner changed during request.'; return 1; }
    done
    /bin/kill -TERM "$owned" || return 1
    wait_gone "$owned" || return 1
  fi
  owned=0
  [[ -z $(listener) ]] || return 1
}
startup_pid() {
  /usr/bin/sed -nE 's/^server pid ([0-9]+) \(port 8765,.*/\1/p' "$1" | head -1
}
banners() {
  local ss=$1 lookup=$2 tau=$3 mid=$4 outfile=$5
  /usr/bin/grep -E 'drafter ring snapshot|v8: variant|dec8: BANNER|tree: BANNER' "$L" | /usr/bin/tee "$outfile" || return 1
  local b
  for b in 'drafter ring snapshot: fix' 'v8: variant F8=x' 'KV8=true' 'qk=attn_qk8isk_tg' 'libgpu_v8_f8q.dylib' 'dec8: BANNER' 'libdec_gpu_tree.dylib' 'tree: BANNER mode=tree8 policy=branched topc=4' "tau=$tau sampled=$ss " 'split tree on' '/v8_dec8tx'; do
    /usr/bin/grep -Fq -- "$b" "$outfile" || { fail 'Unexpected served banner.'; return 1; }
  done
  if [[ $lookup == 0 ]]; then
    /usr/bin/grep -Fq '| lookup off' "$outfile" || return 1
  else
    /usr/bin/grep -Fq '| lookup on (NGRAM_DRAFT=6:' "$outfile" || return 1
  fi
  /usr/bin/grep -Fq "64K entry up to $mid keys" "$L" && /usr/bin/grep -Fq 'keep-alive hold 60 ms' "$L" || return 1
}
launch() {
  local label=$1 ts=$2 lookup=$3 tau=$4 mid=$5 default=$6
  local -a opts
  opts=()
  [[ $default == 1 ]] || opts=(TREE_SAMPLED=$ts NGRAM_DRAFT=$lookup TREE_TAU=$tau V8_DEC8_MID=$mid)
  if [[ $default != 1 ]]; then
    case "$arm" in
      observe) mkdir "$D/observe_counts" || return 1
               opts+=(LOOKUP_OBSERVE=1 FORGE_GATE_ARM=observe FORGE_GATE_COUNTS=$D/observe_counts PYTHONPATH=$H PYTHONDONTWRITEBYTECODE=1) ;;
      feedback_*|recent_floor|shadow_3) opts+=(LOOKUP_GATE=yield:1.25 FORGE_GATE_ARM=$arm PYTHONPATH=$H PYTHONDONTWRITEBYTECODE=1) ;;
      yield) opts+=(LOOKUP_GATE=yield:1.25) ;;
      ctx) opts+=(LOOKUP_GATE=ctx:32768:48:1.25) ;;
    esac
  fi
  local launchfile=$D/$label.launch
  : > "$launchfile" || return 1
  # Retain only OS identity, executables, Python/driver temp-path defaults. Restoring uses the launcher's own defaults.
  nohup /usr/bin/env -i HOME="$HOME" USER="$USER" TMPDIR="${TMPDIR:-/tmp}" PATH='/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin' \
    PORT=8765 "${opts[@]}" /bin/zsh "$P/v8_server_dec8_qk.sh" > "$launchfile" 2>&1 < /dev/null &!
  lp=$!
  local j reported listen
  for j in {1..30}; do
    reported=$(startup_pid "$launchfile") || { fail 'Could not read launcher status.'; return 1; }
    if [[ $reported == <1-> ]]; then owned=$reported; break; fi
    /bin/kill -0 "$lp" 2>/dev/null || { fail 'Launcher exited before reporting a PID.'; return 1; }
    sleep 1
  done
  (( owned > 0 )) || { fail 'Launcher did not report the server PID.'; return 1; }
  # Existing launcher has its own watchdog; this one enforces the requested 1024 MB arm limit during startup and decode.
  if [[ $default != 1 ]]; then
    nohup "$SSD/tools/swap_watchdog.sh" "$owned" 1024 "$D/$label.watchdog.log" > /dev/null 2>&1 < /dev/null &!
  fi
  for j in {1..240}; do
    /bin/kill -0 "$owned" 2>/dev/null || { fail 'Server exited during startup.'; return 1; }
    listen=$(listener)
    if [[ $listen == $owned ]] && /usr/bin/grep -Fq 'serving OpenAI API on http://127.0.0.1:8765/v1' "$L"; then
      identity "$owned" || return 1
      local expected=off; [[ $ts == 1 ]] && expected=on
      banners "$expected" "$lookup" "$tau" "$mid" "$D/$label.banners" || return 1
      return 0
    fi
    sleep 1
  done
  fail 'Timed out waiting for a newly launched server.'
}
finish() {
  local rc=$1
  trap - EXIT
  trap '' INT HUP TERM
  set +e
  # A signal can arrive while readback of a just-started launcher PID is pending.
  if (( lp > 0 && owned == 0 )); then
    for j in {1..30}; do
      local x=$(startup_pid "$current_launchfile" 2>/dev/null)
      if [[ $x == <1-> ]] && identity "$x"; then owned=$x; break; fi
      /bin/kill -0 "$lp" 2>/dev/null || break
      sleep 1
    done
    if (( owned == 0 )) && /bin/kill -0 "$lp" 2>/dev/null; then
      print -u2 'Launcher PID uncertain. Refusing to load another model.'; rc=1; changed=0
    fi
  fi
  if (( probe_pid > 0 )); then
    if /bin/kill -0 "$probe_pid" 2>/dev/null; then
      /bin/kill -TERM "$probe_pid" 2>/dev/null
    fi
    wait "$probe_pid" 2>/dev/null
  fi
  if (( changed )); then
    if stop_owned && launch restored 1 0 1.5 0 1; then
      print -r -- 'Served default restored; checked all banners.' | tee "$D/restoration.txt"
    else
      print -u2 -- 'RESTORATION FAILED. Inspect 8765 owner and last launch before proceeding.'; rc=1
    fi
  fi
  zsystem flock -u "$lease"
  exit "$rc"
}
trap 'finish $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
other_driver() {
  /bin/ps -ax -o command= | /usr/bin/awk '{for(i=1;i<=NF;i++) if ($i ~ /(^|\/)(ctx_gate_ab|gate_reverse_long|lookup_gate_ab|run_arm)\.sh$/) found=1} END {exit !found}'
}
other_driver && { fail 'Another M6 bench driver is running; no service action'; exit 2; }
owned=$(listener)
identity "$owned" || { fail 'No verified original served forge on 8765.'; exit 1; }
banners on 0 1.5 0 "$D/initial.banners" || { fail 'Initial service is not the served default.'; exit 2; }
# Do not terminate a live request. The caller has already reserved this exclusive window with Anemll.
"$PY" - "$L" <<'PY'
import re,sys
s=open(sys.argv[1],'rb').read()
markers=re.findall(rb'^\[\d\d:\d\d:\d\d\] (thinking (?:on|off)|prompt \d+ \(cached )',s,re.M)
if markers and markers[-1].startswith(b'thinking'): sys.exit('A model request is active; no process stopped.')
PY
# Wait for one full idle minute and no recent request before owning the initial launch.
quiet=0
for n in {1..540}; do
  connected=$(/usr/sbin/lsof -tnP -iTCP:8765 -sTCP:ESTABLISHED 2>/dev/null || true)
  if [[ -z $connected ]]; then ((quiet+=5)); else quiet=0; fi
  if ((quiet>=60)); then
    age=$(( $(date +%s) - $(stat -f %m "$L") ))
    ((age>=120)) && break
  fi
  sleep 5
done
((quiet>=60 && age>=120)) || { fail 'Pi did not become idle; no action'; exit 2; }
[[ $(listener) == $owned ]] && identity "$owned" || exit 2
[[ -z $(/usr/sbin/lsof -tnP -iTCP:8765 -sTCP:ESTABLISHED 2>/dev/null || true) ]] || exit 2
other_driver && { fail 'Another M6 bench driver appeared; no service action'; exit 2; }
changed=1
idx=0
for arm in "${arms[@]}"; do
  (( ++idx )); label=${idx}_${arm}; pairseed=$seed
  (( idx <= ${#arms} / 2 || ${#arms} == 1 )) || pairseed=$((seed+10000))
  stop_owned || exit 1
  lp=0; current_launchfile=$D/$label.launch
  lk=6; [[ $arm != off ]] || lk=0
  launch "$label" 1 "$lk" 1.5 0 0 || exit 1
  case "$arm" in
    observe|feedback_*|recent_floor|shadow_3) grep -Fq "review gate: $arm;" "$L" || { fail 'Research hook absent'; exit 1; } ;;
  esac
  "$PY" -B "$H/long_client.py" "$fixture" "$D/$label.json" "$pairseed" || exit 1
done
