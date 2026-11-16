#!/bin/zsh
# Pi first prompt (~6K tokens): 6144-token prefill, attproj switched on from position P per rep
# (P = never / 4096 / 2048 / 0, twice each), 4 and 8 streams. ref_6k made first if missing.
S=${0:a:h}; L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs; V=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/v8
export BUILD=/Volumes/SSD4TB/anemll-forge-research/v8_shared2 MAN=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/w_shared2r/manifest.json
while pgrep -f short_sweep.sh >/dev/null; do sleep 10; done
[ -f $V/ref_6k.npz ] || $S/run_wd.sh 6144 ref_6k.npz mkref_6k --mode ref --reps 1
for ST in 4 8; do
  sleep 20
  $S/run_wd.sh 6144 ref_6k.npz pi6k_s$ST --reps 8 --shared --streams $ST --attproj --attproj-from-alt 1000000,4096,2048,0 --dylib libgpu_v8_attcore.dylib --cool-s 30
done
/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python - <<'P'
import json
L = "/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs"
for st in (4, 8):
    try:
        rs = json.load(open(f"{L}/pi6k_s{st}.json"))["runs"]
    except Exception as e:
        print(st, "missing", e); continue
    for P in (1000000, 4096, 2048, 0):
        x = [r for r in rs if r.get("attproj_from") == P]
        if x:
            v = sum(r["v8_tok_s"] for r in x) / len(x)
            g = sum(r["timeline"]["gpu_busy_frac"] for r in x) / len(x); a = sum(r["timeline"]["ane_busy_frac"] for r in x) / len(x)
            lab = "attproj never" if P == 1000000 else f"attproj from {P // 1024}K"
            print(f"6K s{st} {lab:16s} {v:6.1f} tok/s ({' '.join(f'{r['v8_tok_s']:.0f}' for r in x)})  G{g*100:3.0f} A{a*100:3.0f}  KL {x[0]['kl_mean']:.4f}")
P
