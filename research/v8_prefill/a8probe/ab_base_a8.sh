#!/bin/zsh
# interleaved: baseline V8 (fuse engine, prefill only, s8, QK8) vs A8 (--att all --out-layers gdn), ROUNDS each, one length
L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs; U=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/unit
N=$1; RN=$2; r=ref_$((N / 1024))k.npz
export BUILD=/Volumes/SSD4TB/anemll-forge-research/v8_shared2 MAN=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/w_shared2r/manifest_best.json
for i in $(seq 1 $RN); do
  sleep 20
  env FUSE=all I8=1 I8T=1 I8TV=_k256 LUTC=1 I8A=1 I8KV=1 LRM=1 QK8=1 TKV=2048 $U/run_wd.sh $N $r abx${N}_base_$i --reps 2 --shared --streams 8 --dylib libgpu_v8_fuse.dylib --cool-s 20 > /dev/null
  /Volumes/SSD4TB/anemll-forge-research/a8probe/ab_a8.sh $N 1 "x${i}_a8|A8EXP=t|--att all --out-layers gdn" > /dev/null
done
/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python - $L $N <<'P'
import json, sys, glob
L, N = sys.argv[1], sys.argv[2]
b = [r for f in sorted(glob.glob(f"{L}/abx{N}_base_*.json")) for r in json.load(open(f))["runs"]]
a = [r for f in sorted(glob.glob(f"{L}/a8ab{N}_x*_a8_1.json")) for r in json.load(open(f))["runs"]]
bt, at = [r["v8_tok_s"] for r in b], [r["tok_s"] for r in a]
print(f"{N}: baseline mean {sum(bt)/len(bt):.1f} best {max(bt):.1f} [{' '.join(f'{x:.0f}' for x in bt)}] KL {b[0]['kl_mean']:.4f}")
print(f"{N}: A8       mean {sum(at)/len(at):.1f} best {max(at):.1f} [{' '.join(f'{x:.0f}' for x in at)}] KL {a[0]['kl_mean']:.4f}")
P
