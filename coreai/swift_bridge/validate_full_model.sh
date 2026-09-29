#!/bin/zsh
# Full-model validation window for the Core AI bridge (see validate_full_model.py). Needs the Qwen server and chat
# stopped: each phase is one process and exits before the next (Core ML ~20 GB wired, Core AI ~23-25 GB wired).
#   ./validate_full_model.sh            # specialize -> coreml -> coreai -> compare
#   ./validate_full_model.sh coreai     # one phase (specialize | coreml | coreai | compare)
# env: FORCE=1 (skip the "server / chat running" check), WATCHDOG_MB (swap growth that kills a phase, 1024),
#      plus validate_full_model.py's (COREML_DIR, COREAI_DIR, CTX, PROMPT, G, NV, NP, STEPS, COREAI_BRIDGE)
cd "${0:A:h}"
PY_COREML=${PY_COREML:-$HOME/venvs/vq27b/bin/python}
PY_COREAI=${PY_COREAI:-$HOME/venvs/vq27b-coreai/bin/python}
WATCHDOG_MB=${WATCHDOG_MB:-1024}
LOG=val_full_$(date +%m%d_%H%M).log
PHASES=(${@:-specialize coreml coreai compare})
(( $# )) || PHASES=(specialize coreml coreai compare)  # zsh keeps the :- default as one word

if [[ -z $FORCE ]] && pgrep -f "qwen38_server.py|qwen38_chat.py" >/dev/null; then
  echo "the Qwen server / chat is running (it holds the Core ML model); stop it first or set FORCE=1"; exit 1
fi
[[ -f libcoreai_bridge.dylib && ! CoreAIBridge.swift -nt libcoreai_bridge.dylib ]] || ./build.sh || exit 1

swap_mb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); printf "%d", $6}' }
wired() { vm_stat | awk '/page size of/{ps=$8} /wired down/{gsub("\\.","",$4); printf "%.2f GB", $4*ps/2^30}' }

for ph in $PHASES; do
  [[ $ph == coreml ]] && PY=$PY_COREML || PY=$PY_COREAI
  S0=$(swap_mb)
  echo "=== $ph ($(date +%T), wired $(wired), swap used ${S0} MB)" | tee -a $LOG
  $PY -u validate_full_model.py $ph 2>&1 | grep --line-buffered -v "Redirects are currently\|has not been tested with coremltools" | tee -a $LOG &
  while pgrep -f "validate_full_model.py $ph" >/dev/null; do
    S=$(swap_mb)
    if (( S - S0 > WATCHDOG_MB )); then
      echo "WATCHDOG: swap grew ${S0} -> ${S} MB during $ph; killing it" | tee -a $LOG
      pkill -f "validate_full_model.py $ph"; break
    fi
    sleep 2
  done
  wait
  sleep 3
  echo "--- $ph done ($(date +%T), wired $(wired))" | tee -a $LOG
done
echo "log: $PWD/$LOG"
