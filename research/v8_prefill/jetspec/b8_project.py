"""B8 projection (20 Nov 2026): decode tok/s of a serial ANE JetSpec++ (LUT6 drafter, 16-row chain verify) against the
served DFlash2 chain at the 8K entry (2K-8K contexts), from measured pieces only:
  adapter pass   27.2 ms   ANE, LUT6, 12 blocks x 16 rows, 8K KV (b8/ane/timing_8k.json: 9.08 + 9.07 + 9.09)
  B16 head pass  14.8 ms   PROJECTED: its LUT6 bytes (1.73B x 0.75 = 1.30 GB) at the adapter's measured 123 GB/s
                           (10.5 ms) + the target head on 16 rows (chain16/: 4.30 ms, measured)
  decoder pass    7.1 ms   PROJECTED: target layers 62 / 63 at 16 rows (chain16/: 5.50 ms per 4-layer group, measured)
                           / 2 + the target head on 16 rows (4.30 ms)
  verify, 16 rows 102.5 ms the served 8-row verify (94.5 ms) x 1.085 (chain16/, measured ratio)
  DFlash2 chain   107.7 ms per cycle (served: draft 13.2 + verify 94.5), its tok/verify on the same prompts
passes per round and tok/round from the replays (b8/passes.json; LUT6 runs where present).
    PY b8_project.py RESULTS_20NOV_DIR"""
import json, sys
from pathlib import Path

R = Path(sys.argv[1])
P = json.loads((R / "b8/passes.json").read_text())
ADAPTER, HEAD, DEC, VERIFY16, HOST = 27.2, 14.8, 7.1, 102.5, 1.0
DFL = {"gsm8k": 6.08, "humaneval": 7.30, "mbpp": 5.89}
CYC = 107.7
lut6 = {}
for f in R.glob("b8/b8_lut6t_*_*.json"):
    s, ds = f.stem.replace("b8_lut6t_", "").rsplit("_", 1)
    d = json.loads(f.read_text())
    if d.get("summary"):
        lut6[(s, ds)] = d["summary"]["tok_per_round"]
print("| schedule | set | drafter ms / round at LUT6 on the ANE (adapter + head + decoder) | round ms | acceptance, tok/round: bf16 replay (LUT6 replay) | JetSpec++ tok/s (LUT6 timing; LUT6 acceptance where measured) | DFlash2 tok/s | vs DFlash2 |")
print("|---|---|---|---|---|---|---|---|")
for s in ("J0-R0", "J1-R0", "J1-R1", "J3-R3"):
    for ds in ("gsm8k", "humaneval", "mbpp"):
        p = P[s][ds]
        draft = p["adapter"] * ADAPTER + p["head"] * HEAD + p["decoder"] * DEC
        rnd = draft + VERIFY16 + HOST
        tpr = lut6.get((s, ds), p["tok_round"])
        js = 1e3 * tpr / rnd
        df = 1e3 * DFL[ds] / CYC
        lab = f"{p['tok_round']:.2f}" + (f" ({lut6[(s, ds)]:.2f})" if (s, ds) in lut6 else "")
        print(f"| {s} | {ds} | {draft:.0f} ({p['adapter'] * ADAPTER:.0f} + {p['head'] * HEAD:.0f} + {p['decoder'] * DEC:.0f}) | {rnd:.0f} | {lab} | {js:.1f} | {df:.1f} | {js / df - 1:+.0%} |")
