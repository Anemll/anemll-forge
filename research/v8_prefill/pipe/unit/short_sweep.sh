#!/bin/zsh
# Short prompts (full prefill from position 0, real KL vs the ANE-only reference): 2K / 4K / 8K, each with 4 and 8
# streams, attproj alternating off / on per rep. Build v8_shared2 + tables w_shared2r. ref_4k made first if missing.
S=${0:a:h}; L=/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs; V=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/v8
export BUILD=/Volumes/SSD4TB/anemll-forge-research/v8_shared2 MAN=/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/w_shared2r/manifest.json
[ -f $V/ref_4k.npz ] || { $S/run_wd.sh 4096 ref_4k.npz mkref_4k --mode ref --reps 1; ls -la $V/ref_4k.npz; }
for NR in 2048:ref_2k.npz 4096:ref_4k.npz 8192:ref_8k.npz; do
  N=${NR%%:*}; R=${NR#*:}
  for ST in 4 8; do
    sleep 20
    $S/run_wd.sh $N $R short${ST}_$N --reps 4 --shared --streams $ST --attproj --attproj-alt --dylib libgpu_v8_attcore.dylib --cool-s 30
  done
done
/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge/.venv/bin/python - <<'P'
import json
L = "/Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs"
print("tokens streams  attproj-off tok/s     attproj-on tok/s    (G%/A% busy, KL mean)")
for N in (2048, 4096, 8192):
    for st in (4, 8):
        try:
            rs = json.load(open(f"{L}/short{st}_{N}.json"))["runs"]
        except Exception as e:
            print(N, st, "missing", e); continue
        out = []
        for on in (False, True):
            x = [r for r in rs if r["attproj"] == on]
            if x:
                v = sum(r["v8_tok_s"] for r in x) / len(x)
                g = sum(r["timeline"]["gpu_busy_frac"] for r in x) / len(x); a = sum(r["timeline"]["ane_busy_frac"] for r in x) / len(x)
                out.append(f"{v:6.1f} ({' '.join(f'{r['v8_tok_s']:.0f}' for r in x)}) G{g*100:3.0f} A{a*100:3.0f} KL{x[0]['kl_mean']:.4f}")
        print(f"{N:6d}  s{st}   " + "   ".join(out))
P
