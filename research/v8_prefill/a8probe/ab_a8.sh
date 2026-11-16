#!/bin/zsh
# ab_a8.sh TOKENS ROUNDS "tag|ENV..|args" ...   interleaved A8 driver runs (prefill + KL), summary at the end
L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs; W=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work
N=$1; RN=$2; shift 2; r=ref_$((N / 1024))k.npz
BASE="MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 FUSE=all I8=1 I8T=1 I8TV=_k256 LUTC=1 I8A=1 I8KV=1 LRM=1"
[ $N -ge 4096 ] && BASE="$BASE QK8=1 TKV=2048"
for i in $(seq 1 $RN); do for v in "$@"; do
  tag=${v%%|*}; rest=${v#*|}; ev=${rest%%|*}; va=${rest#*|}
  sleep 15
  env ${=BASE} ${=ev} /Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python $W/pipe/v8_a8.py --build /Volumes/SSD4TB/anemll-forge-research/v8_shared2 \
    --gpu-manifest $W/pipe/w_a8/manifest_a8.json --a8-dir $W/pipe/w_a8 --a8-pkg /Volumes/SSD4TB/anemll-forge-research/a8probe/real_n1024.aimodel \
    --tokens $N --ref $W/v8/$r --out $L/a8ab${N}_${tag}_$i.json --reps 2 --cool-s 15 ${=va} > $L/a8ab${N}_${tag}_$i.log 2>&1 || echo "$tag $i FAILED"
done; done
/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python - $L $N "$@" <<'P'
import json, sys, glob
L, N, vs = sys.argv[1], sys.argv[2], sys.argv[3:]
for v in vs:
    tag = v.split("|")[0]; rs = []
    for f in sorted(glob.glob(f"{L}/a8ab{N}_{tag}_*.json")):
        rs += json.load(open(f))["runs"]
    if not rs: print(tag, "no runs"); continue
    tk = [r["tok_s"] for r in rs]
    print(f"{tag:10s} mean {sum(tk)/len(tk):6.1f}  best {max(tk):6.1f}  [{' '.join(f'{x:.0f}' for x in tk)}]  G{100*sum(r['gpu_busy_frac'] for r in rs)/len(rs):3.0f} A{100*sum(r['ane_busy_frac'] for r in rs)/len(rs):3.0f}  gpu {rs[0]['gpu_call_ms_median']:.2f} ms ane {rs[0]['ane_call_ms_median']:.1f} ms exp {rs[0]['expand_ms_median']:.2f} ms  KL {rs[0]['kl_mean']:.4f}")
P
