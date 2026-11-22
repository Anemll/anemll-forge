#!/bin/zsh
# 21 Nov 2026 (item 4): ablation timing of gdn_chunk variants (CHUNK_VAR), alone, same wait / lock protocol as
# gdn_chunk_time.sh. NOSOLVE is a timing diagnostic (wrong output).   zsh gdn_chunk_abl.sh TAG "VAR1" "VAR2" ...
set -u
P=${0:a:h}; TAG=$1; shift
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/21nov/gdn_chunk
S=/private/tmp/claude-501/-Users-anemll-SourceRelease-GITHUB-ML-playground-anemll-forge/e4e29b3c-8360-4ddd-8f48-0762d3ef8250/scratchpad
SL=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log
WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh
G=$R/abl_$TAG.log
conns() { lsof -nP -iTCP:8765 -sTCP:ESTABLISHED 2>/dev/null | tail -n +2 | wc -l | tr -d ' '; }
age() {
  local last; last=$(grep -oE "^\[[0-9:]+\] request: prompt" $SL 2>/dev/null | tail -1 | cut -c2-9)
  [ -z "$last" ] && { echo 99999; return; }
  local a=$(( $(date +%s) - $(date -j -f "%Y-%m-%d %H:%M:%S" "$(date +%Y-%m-%d) $last" +%s) )); [ $a -lt 0 ] && a=99999; echo $a
}
echo "$(date +%T) waiting for timing_main.lock to clear and Pi idle" | tee -a $G
quiet=0
for i in $(seq 1 2160); do
  if [ -e $S/timing_main.lock ] || [ "$(conns)" != 0 ]; then quiet=0; else quiet=$((quiet + 5)); fi
  [ $quiet -ge 60 ] && [ $(age) -ge 120 ] && break
  sleep 5
done
[ $quiet -ge 60 ] && [ $(age) -ge 120 ] || { echo "$(date +%T) not clear within 3 h: not timed" | tee -a $G; exit 2; }
echo "gdn ablation $(date +%T)" > $S/timing_gdn.lock
trap 'rm -f $S/timing_gdn.lock' EXIT
PS="Pi idle (no 8765 connection 60 s, last request $(age) s ago), forge LISTENING not touched, parent timing lock clear | GPU otherwise idle"
echo "$(date +%T) timing ($PS)" | tee -a $G
cd $P
for gc in ${=GCS:-64 32}; do for var in "$@"; do
  CHUNK_VAR=$var PI_STATE="$PS" ./gdn_chunk_proto time $gc > $R/abl_${TAG}_gc${gc}_${var//,/+}.log 2>&1 & p=$!
  $WD $p 1024 /dev/null > /dev/null 2>&1 &
  wait $p; echo "GC=$gc var [$var] rc $?" | tee -a $G; grep -E "^gdn_" $R/abl_${TAG}_gc${gc}_${var//,/+}.log | tee -a $G
done; done
echo "$(date +%T) done; Pi connections now $(conns), last request $(age) s ago" | tee -a $G
