#!/usr/bin/env bash
# mem_sampler.sh OUT.csv: every 2 s, the 8765 forge's footprint (total, IOSurface, IOAccelerator graphics), system swap,
# wired and free pages (16 KB). Runs until killed.
OUT=$1; echo "t,pid,fp_mb,iosurface_mb,ioaccel_mb,swap_mb,wired_mb,free_mb" > $OUT
while true; do
  pid=$(lsof -tnP -iTCP:8765 -sTCP:LISTEN 2>/dev/null | head -1)
  if [ -n "$pid" ]; then
    footprint $pid 2>/dev/null | awk -v t=$(date +%s) -v pid=$pid '
      function mb(v,u){ return u=="GB"?v*1024:(u=="KB"?v/1024:(u=="B"?v/1048576:v)) }
      /Footprint:/ {for(i=1;i<=NF;i++) if($i=="Footprint:") fp=mb($(i+1),$(i+2))}
      /IOSurface$/ {ios=mb($1,$2)} /IOAccelerator \(graphics\)/ {ioa=mb($1,$2)}
      END {printf "%s,%s,%d,%d,%d", t, pid, fp, ios, ioa}' >> $OUT
    s=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); printf "%d", $6}')
    vm_stat | awk -v s=$s '/Pages wired down/ {gsub("\\.","",$4); w=$4} /Pages free/ {gsub("\\.","",$3); f=$3} END {printf ",%d,%d,%d\n", s, w*16/1024, f*16/1024}' >> $OUT
  fi
  sleep 2
done
