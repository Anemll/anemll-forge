#!/usr/bin/env bash
# Start / stop the OpenAI-compatible Qwen3.8-27B ANE server from a Forge checkout.
#
#   scripts/qwen38_server.sh start|stop|restart|status|check|log
#
# The wrapper calls `forge.py serve`, so the prepared Core AI target, matching DFlash2 drafter and metrics are
# validated by the launcher before either large model is allocated. `--ctx` is a growth cap, not a pin: the runtime
# starts at the smallest advertised entry and grows through the ladder as positions are committed. See
# docs/SPECULATIVE_DECODING.md.
#
# Overrides (env):
#   FORGE_BUNDLE  bundle with model/ + coreai/ + drafter/ (default: ~/Models/anemll-forge-qwen3.8-27B)
#   MODEL, BUILD  override the checkpoint / Core AI build separately
#   CTX           context cap; a number or 8K / 16K / 24K / 32K / 48K / 64K (default: 16384)
#   KV_CACHE_DTYPE auto (manifest), fp16, v8 or kv8; requires the matching BUILD export
#   PORT          listen port (default: 8765)
#   BIND_HOST     listen address (default: 127.0.0.1; set 0.0.0.0 to expose)
#   DRAFT         on (default, sibling drafter/), off/--plain, or a path to a *.aimodel package
#   DRAFTER       directory holding the drafter config.json + selector.safetensors (default: package parent)
#   PY            python with the inference environment (default: .venv/bin/python if present, else python3)
#   MPSGRAPH_ANE_BONDED_COMPILE_MODE  ANE compile mode override (M5 family -> 1, M6+ -> 2; see
#                 docs/ANE_COMPILE_MODE_POLICY.md). Normally chosen by SoC policy automatically.
#   LOG, PIDFILE  runtime state paths (default: $ANEMLL_FORGE_STATE/server.log and .pid)
#   PI_SYNC       sync the Pi profile after start (default: 1); PI_DIR (default: ~/.pi/agent)
#   PI_BUILD      build name recorded in Pi's model display name (default: basename of BUILD)
#
#   START_WAIT_S  how long start / restart keep watching the startup (default: 7200)
#
# `restart` validates the requested build before stopping a running server. Startup shows completed target chunks
# and elapsed time; the chunk percentage excludes head/drafter loading. The first start of a build on a macOS build
# compiles its packages for the ANE once: the `[ANE compile]` lines show the plan, per-package progress, time left
# and options that compile faster. Ctrl-C only stops watching (the server keeps going; compiled packages stay cached
# and a later start resumes). `python forge.py compile --build <dir>` compiles without serving. `log` follows the
# server log.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$ROOT" && pwd)"

FORGE_BUNDLE="${FORGE_BUNDLE:-$HOME/Models/anemll-forge-qwen3.8-27B}"
MODEL="${MODEL:-$FORGE_BUNDLE/model}"
BUILD="${BUILD:-$FORGE_BUNDLE/coreai}"
PORT="${PORT:-8765}"
BIND_HOST="${BIND_HOST:-127.0.0.1}"
CTX="${CTX:-16384}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
PY="${PY:-}"
if [[ -z "$PY" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then PY="$ROOT/.venv/bin/python"; else PY="python3"; fi
fi
DRAFT="${DRAFT:-on}"
ANEMLL_FORGE_STATE="${ANEMLL_FORGE_STATE:-$HOME/.anemll-forge}"
LOG="${LOG:-$ANEMLL_FORGE_STATE/server.log}"
PIDFILE="${PIDFILE:-$ANEMLL_FORGE_STATE/server.pid}"
PI_SYNC="${PI_SYNC:-1}"
START_WAIT_S="${START_WAIT_S:-7200}"   # how long start / restart watches a (cold-compiling) startup
PI_DIR="${PI_DIR:-${PI_CODING_AGENT_DIR:-$HOME/.pi/agent}}"
PI_BUILD="${PI_BUILD:-$(basename "$BUILD")}"

CTX_UPPER=$(printf '%s' "$CTX" | tr '[:lower:]' '[:upper:]')
[[ "$CTX_UPPER" =~ ^[0-9]+K?$ ]] || { echo "CTX must be a number or 8K..64K, got '$CTX'"; exit 1; }
case "$CTX_UPPER" in *K) CTX=$(( 10#${CTX_UPPER%K} * 1024 )) ;; esac

managed_pids() {
  "$PY" "$ROOT/scripts/qwen38_server_process.py" pids --root "$ROOT" --port "$PORT" --pidfile "$PIDFILE"
}

running() { managed_pids >/dev/null; }

launch_args() {
  if [[ -n "${EXTRA_ARGS:-}" ]]; then
    echo "EXTRA_ARGS is not supported; use MODEL/BUILD/CTX/PORT/BIND_HOST/DRAFT/DRAFTER overrides" >&2
    return 1
  fi
  args=("$ROOT/forge.py" serve --runtime coreai --model "$MODEL" --build "$BUILD"
        --ctx "$CTX" --host "$BIND_HOST" --port "$PORT" --kv-cache-dtype "$KV_CACHE_DTYPE")
  case "$DRAFT" in
    0|off|no|false) args+=(--plain) ;;
    on|1|yes|true) ;;
    *) args+=(--draft "$DRAFT") ;;
  esac
  [[ -z "${DRAFTER:-}" ]] || args+=(--drafter "$DRAFTER")
}

check() {
  [[ -f "$MODEL/config.json" ]] || { echo "missing checkpoint config: $MODEL/config.json"; return 1; }
  "$PY" -c "import numpy" 2>/dev/null || {
    echo "PY=$PY cannot import numpy; set PY to the inference environment (README step 1)"; return 1; }
  [[ -f "$BUILD/manifest.json" ]] || { echo "no Core AI manifest.json in $BUILD"; return 1; }
  local ctxs
  ctxs=$("$PY" -c "import json,sys; print(' '.join(map(str, json.load(open(sys.argv[1]))['ctxs'])))" "$BUILD/manifest.json")
  if [[ " $ctxs " != *" $CTX "* ]]; then
    echo "build $BUILD has contexts: $ctxs (CTX=$CTX is not one of them)"; return 1
  fi
  local args
  launch_args || return 1
  "$PY" "${args[@]}" --dry-run >/dev/null || return 1
  "$PY" - "$ROOT" <<'PY' || return 1
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]) / "scripts"))
from hf_release import coreai_bridge_environment
try:
    os.environ.update(coreai_bridge_environment())
    sys.path.insert(0, os.environ["COREAI_BRIDGE_DIR"])
    import coreai_bridge
    coreai_bridge.lib()
except (ValueError, OSError, ImportError, RuntimeError) as error:
    print(f"Swift bridge check failed: {error}", file=sys.stderr)
    print("From this checkout, run: bash coreai/swift_bridge/build.sh", file=sys.stderr)
    sys.exit(1)
PY
}

pi_sync() {
  [[ "$PI_SYNC" == 0 ]] && return 0
  "$PY" "$ROOT/scripts/qwen38_pi_config.py" --ctx "$CTX" --pi-dir "$PI_DIR" --build "$PI_BUILD" \
      --draft "$([[ "$DRAFT" == off || "$DRAFT" == 0 || "$DRAFT" == no || "$DRAFT" == false ]] && echo off || echo on)" \
    || echo "pi config sync failed; set PI_SYNC=0 to skip"
}

start() {
  if running; then echo "already running (pids $(managed_pids)) on port $PORT"; return 0; fi
  if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "port $PORT is in use:"; lsof -nP -iTCP:"$PORT" -sTCP:LISTEN; return 1
  fi
  check || return 1
  mkdir -p "$ANEMLL_FORGE_STATE"
  [[ -s "$LOG" ]] && mv -f "$LOG" "$LOG.prev"

  local args
  launch_args || return 1

  local total_chunks loaded_chunks last_loaded=-1 startup_elapsed shown=0 n
  total_chunks=$("$PY" -c "import json,sys; print(len(json.load(open(sys.argv[1]))['chunks']))" "$BUILD/manifest.json")
  echo "starting (log $LOG)"
  nohup "$PY" -u "${args[@]}" > "$LOG" 2>&1 &
  echo $! > "$PIDFILE"
  # Ctrl-C only stops watching: the server keeps loading / compiling, and finished packages stay cached.
  trap 'echo; echo "stopped watching; the server keeps starting in the background (compiled packages are cached)."; \
        echo "follow: $0 log   status: $0 status   stop: $0 stop (a later start resumes the compile)"; \
        trap - INT; return 0' INT
  for (( startup_elapsed = 0; startup_elapsed < START_WAIT_S; startup_elapsed++ )); do
    n=$(grep -c "\[ANE compile\]" "$LOG" 2>/dev/null || true)
    if (( n > shown )); then grep "\[ANE compile\]" "$LOG" | tail -n +"$((shown + 1))"; shown=$n; fi
    if grep -q "serving OpenAI API" "$LOG" 2>/dev/null; then
      trap - INT
      { grep -E "target graph:|KV cache:" "$LOG" || true; } | tail -2
      grep -E "loaded|serving" "$LOG" | tail -3; pi_sync; return 0
    fi
    running || { trap - INT; echo "failed to start:"; tail -20 "$LOG"; return 1; }
    loaded_chunks=$(awk '/loaded chunk_[^ ]+\.aimodel \([0-9]+ entries,/ { n++ } END { print n+0 }' "$LOG")
    if (( loaded_chunks != last_loaded || startup_elapsed % 30 == 0 )); then
      if (( total_chunks > 0 && loaded_chunks < total_chunks )); then
        printf 'loading target chunks: %d/%d (%d%%), %ds elapsed\n' \
          "$loaded_chunks" "$total_chunks" "$((loaded_chunks * 100 / total_chunks))" "$startup_elapsed"
      else
        printf 'target chunks loaded: %d/%d; loading head/drafter and initializing, %ds elapsed\n' \
          "$loaded_chunks" "$total_chunks" "$startup_elapsed"
      fi
      last_loaded=$loaded_chunks
    fi
    sleep 1
  done
  trap - INT
  echo; echo "NOT SERVING YET after ${START_WAIT_S}s: still starting in the background."
  { grep "\[ANE compile\]" "$LOG" || true; } | tail -1
  echo "Follow: $0 log"
}

stop() {
  "$PY" "$ROOT/scripts/qwen38_server_process.py" stop --root "$ROOT" --port "$PORT" --pidfile "$PIDFILE"
}

status() {
  if running; then echo "running (pids $(managed_pids))"; else echo "no verified server from $PIDFILE"; fi
  curl -s --max-time 3 "http://127.0.0.1:$PORT/health" && echo || echo "no response on port $PORT"
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  restart) check || exit 1; stop || exit 1; sleep 1; start ;;
  status) status ;;
  check) check && echo "ok: $BUILD has a ctx $CTX build" ;;
  log) tail -F "$LOG" ;;
  *) echo "usage: $0 start|stop|restart|status|check|log"; exit 2 ;;
esac
