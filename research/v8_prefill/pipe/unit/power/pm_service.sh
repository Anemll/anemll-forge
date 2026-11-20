#!/bin/bash
# powermetrics recording service for the GPU / ANE coupling tests (18 Nov 2026). The user starts it once as root:
#   sudo nohup /Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit/power/pm_service.sh >/dev/null 2>&1 &
# (or in the foreground of its own Terminal tab). It ignores Ctrl-C and hangups (closing the tab); it ends only on
# the quit file below or `sudo kill PID`. Clients (no root) control it with files in $CTL:
#   echo "NAME INTERVAL_MS" > $CTL/start   start powermetrics into $OUT/NAME.plist (plist samples, NUL separated);
#                                          a running recording is stopped first
#   touch $CTL/stop                        stop the current recording
#   touch $CTL/quit                        stop and exit the service
#   cat $CTL/status                        idle | recording NAME INTERVAL PID START_EPOCH
# It only ever runs powermetrics with the fixed sampler list below. NAME must match [A-Za-z0-9._-]{1,64} and the
# interval must be 20..2000 ms; anything else is refused and logged.
set -u
D=/Volumes/SSD4TB/anemll-forge-research/contention
CTL=$D/ctl OUT=$D/pm LOG=$D/pm_service.log
SAMPLERS=cpu_power,gpu_power,ane_power,thermal
umask 022
mkdir -p "$CTL" "$OUT"
[ -n "${SUDO_USER:-}" ] && chown "$SUDO_USER" "$CTL" 2>/dev/null   # (exFAT volume: no owners, already writable)
rm -f "$CTL/start" "$CTL/stop" "$CTL/quit"
PID="" NAME=""
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
status() { echo "$*" > "$CTL/status.tmp" && mv -f "$CTL/status.tmp" "$CTL/status"; }
stop_rec() {
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        kill -TERM "$PID"; for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$PID" 2>/dev/null || break; sleep 0.2; done
        kill -KILL "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; log "stopped $NAME (pid $PID)"
    fi
    PID="" NAME=""; status idle
}
trap '' INT HUP
trap 'stop_rec; log "service exit (TERM)"; exit 0' TERM
log "service start (pid $$, user ${SUDO_USER:-root})"
status idle
while true; do
    if [ -f "$CTL/start" ]; then
        read -r N IV _ < "$CTL/start"; rm -f "$CTL/start"
        if [[ ! "${N:-}" =~ ^[A-Za-z0-9._-]{1,64}$ ]] || [[ ! "${IV:-}" =~ ^[0-9]{2,4}$ ]] || [ "$IV" -lt 20 ] || [ "$IV" -gt 2000 ]; then
            log "refused start request: '${N:-}' '${IV:-}'"
        else
            stop_rec
            NAME=$N
            powermetrics -i "$IV" -f plist --samplers "$SAMPLERS" -o "$OUT/$NAME.plist" >/dev/null 2>>"$LOG" &
            PID=$!
            log "recording $NAME every $IV ms (pid $PID)"
            status "recording $NAME $IV $PID $(date +%s)"
        fi
    fi
    [ -f "$CTL/stop" ] && { rm -f "$CTL/stop"; stop_rec; }
    [ -f "$CTL/quit" ] && { rm -f "$CTL/quit"; stop_rec; log "service quit"; exit 0; }
    if [ -n "$PID" ] && ! kill -0 "$PID" 2>/dev/null; then log "powermetrics for $NAME exited on its own"; PID="" NAME=""; status idle; fi
    sleep 0.2
done
