#!/bin/zsh
# 21 Nov 2026 (M6 ANE bot, item 4): timing of gdn_rec vs gdn_chunk (gdn_chunk_proto time GC). Waits until the parent
# session's timing lock is gone and Pi is idle (no 8765 connection for 60 s, last request >= 120 s ago), holds
# timing_gdn.lock while timing. Sets: GC 64 and 32 alone (GPU otherwise idle), then GC 64 and 32 under `dram_hog 2`
# (CPU memory load, the stand-in for the ANE weight stream). Swap watchdog on each run. Never touches forge.
set -u
P=${0:a:h}
R=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill/results/21nov/gdn_chunk
S=/private/tmp/claude-501/-Users-anemll-SourceRelease-GITHUB-ML-playground-anemll-forge/e4e29b3c-8360-4ddd-8f48-0762d3ef8250/scratchpad
SL=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log
WD=/Volumes/SSD4TB/anemll-forge-research/tools/swap_watchdog.sh
G=$R/time.log
conns() { lsof -nP -iTCP:8765 -sTCP:ESTABLISHED 2>/dev/null | tail -n +2 | wc -l | tr -d ' '; }
age() {
  local last; last=$(grep -oE "^\[[0-9:]+\] request: prompt" $SL 2>/dev/null | tail -1 | cut -c2-9)
  [ -z "$last" ] && { echo 99999; return; }
  local a=$(( $(date +%s) - $(date -j -f "%Y-%m-%d %H:%M:%S" "$(date +%Y-%m-%d) $last" +%s) )); [ $a -lt 0 ] && a=99999; echo $a
}
echo "$(date +%T) waiting for timing_main.lock to clear and Pi idle" | tee -a $G
quiet=0
for i in $(seq 1 2160); do   # up to 3 h
  if [ -e $S/timing_main.lock ] || [ "$(conns)" != 0 ]; then quiet=0; else quiet=$((quiet + 5)); fi
  [ $quiet -ge 60 ] && [ $(age) -ge 120 ] && break
  sleep 5
done
[ $quiet -ge 60 ] && [ $(age) -ge 120 ] || { echo "$(date +%T) not clear within 3 h: not timed" | tee -a $G; exit 2; }
echo "gdn timing $(date +%T)" > $S/timing_gdn.lock
trap 'rm -f $S/timing_gdn.lock' EXIT
PS="Pi idle (no 8765 connection 60 s, last request $(age) s ago), forge LISTENING not touched, parent timing lock clear"
echo "$(date +%T) timing ($PS)" | tee -a $G
cd $P
for gc in 64 32; do
  PI_STATE="$PS | GPU otherwise idle" ./gdn_chunk_proto time $gc > $R/time_alone_gc$gc.log 2>&1 & p=$!
  $WD $p 1024 $R/time_alone_gc$gc.watchdog.log > /dev/null 2>&1 &
  wait $p; echo "alone GC=$gc rc $?" | tee -a $G; cat $R/time_alone_gc$gc.log | tee -a $G
done
for gc in 64 32; do
  ./dram_hog 2 60 > $R/time_hog_gc$gc.hog.txt 2>&1 & hp=$!
  sleep 2
  PI_STATE="$PS | under dram_hog 2" ./gdn_chunk_proto time $gc > $R/time_hog_gc$gc.log 2>&1 & p=$!
  $WD $p 1024 $R/time_hog_gc$gc.watchdog.log > /dev/null 2>&1 &
  wait $p; echo "dram_hog 2, GC=$gc rc $?" | tee -a $G; cat $R/time_hog_gc$gc.log | tee -a $G
  kill $hp 2>/dev/null; wait $hp 2>/dev/null; cat $R/time_hog_gc$gc.hog.txt | tee -a $G
  [ "$(conns)" != 0 ] && echo "WARNING: 8765 connection seen at the end of the hog set" | tee -a $G
done
echo "$(date +%T) done; Pi connections now $(conns), last request $(age) s ago" | tee -a $G
