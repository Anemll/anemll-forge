"""Per-round drafter passes by schedule (20 Nov 2026, B8), from the replays' RoundDriver counters: J (B16 head passes),
R (refreshes: one adapter predict + one anchor-decoder pass each) and runs (rounds + admissions: one final adapter
predict each). Prints tok/round and passes per round per set and schedule.
    PY b8_passes.py RESULTS_20NOV_DIR"""
import glob, json, sys
from pathlib import Path

R = Path(sys.argv[1])
files = {"J3-R3": {"gsm8k": R / "jetspec/replay_j3r3.json", "humaneval": R / "jetspec/replay_humaneval.json", "mbpp": R / "jetspec/replay_mbpp.json"}}
for f in sorted(glob.glob(str(R / "b8/b8_sched_*_*.json"))):
    s, ds = Path(f).stem.replace("b8_sched_", "").rsplit("_", 1)
    files.setdefault(s, {})[ds] = Path(f)
print("| schedule | set | tok/round | head passes / round | adapter passes / round | decoder passes / round |")
print("|---|---|---|---|---|---|")
out = {}
for s, by in files.items():
    for ds, f in sorted(by.items()):
        res = json.loads(f.read_text())["results"]
        res = sorted(res, key=lambda r: r["trace"])[:8]
        g = sum(r["generated"] for r in res); rounds = sum(r["rounds"] for r in res)
        J = sum(r["driver_counts"].get("J", 0) for r in res); Rr = sum(r["driver_counts"].get("R", 0) for r in res)
        runs = sum(r["driver_counts"].get("rounds", 0) + r["driver_counts"].get("admission", 0) for r in res)
        row = dict(tok_round=g / rounds, head=J / rounds, adapter=(Rr + runs) / rounds, decoder=Rr / rounds)
        out.setdefault(s, {})[ds] = row
        print(f"| {s} | {ds} | {row['tok_round']:.2f} | {row['head']:.2f} | {row['adapter']:.2f} | {row['decoder']:.2f} |")
(R / "b8/passes.json").write_text(json.dumps(out, indent=1))
