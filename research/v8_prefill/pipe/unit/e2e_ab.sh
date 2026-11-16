#!/bin/zsh
# Interleaved A/B of GPU engine variants on full prefills (v8_shared2 + w_shared2r, real KL vs the reference):
# each variant is its own process (engine env vars are read at load), rounds A B A B ... with cool-downs.
#   e2e_ab.sh TOKENS REF STREAMS ROUNDS "tagA|dylibA|ENV=.. ENV=..|--per-variant args" "tagB|dylibB|..." [-- extra driver args]
# MAN / BUILD can be overridden from the environment.
S=${0:a:h}; L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs
export BUILD=${BUILD:-/Volumes/SSD4TB/anemll-forge-research/v8_shared2} MAN=${MAN:-/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/w_shared2r/manifest.json}
N=$1; R=$2; ST=$3; RN=$4; shift 4
VS=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do VS+=("$1"); shift; done; [ "$1" = "--" ] && shift
for i in $(seq 1 $RN); do
  for v in "${VS[@]}"; do
    tag=${v%%|*}; rest=${v#*|}; dl=${rest%%|*}; ev=${rest#*|}; [ "$ev" = "$rest" ] && ev=""
    va=""; [ "${ev#*|}" != "$ev" ] && { va=${ev#*|}; ev=${ev%%|*}; }
    sleep 20
    env ${=ev} $S/run_wd.sh $N $R ab${N}_${tag}_$i --reps 3 --shared --streams $ST --dylib $dl --cool-s 30 ${=va} "$@"
  done
done
/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python - $L $N "${VS[@]}" <<'P'
import json, sys, glob
L, N, vs = sys.argv[1], sys.argv[2], sys.argv[3:]
for v in vs:
    tag = v.split("|")[0]; rs = []
    for f in sorted(glob.glob(f"{L}/ab{N}_{tag}_*.json")):
        rs += json.load(open(f))["runs"]
    if not rs: print(tag, "no runs"); continue
    tk = [r["v8_tok_s"] for r in rs]
    g = sum(r["timeline"]["gpu_busy_frac"] for r in rs) / len(rs); a = sum(r["timeline"]["ane_busy_frac"] for r in rs) / len(rs)
    print(f"{tag:10s} mean {sum(tk) / len(tk):6.1f} tok/s  best {max(tk):6.1f}  [{' '.join(f'{x:.0f}' for x in tk)}]  G{g * 100:3.0f} A{a * 100:3.0f}  KL {rs[0]['kl_mean']:.4f}")
P
