"""B7 table (20 Nov 2026): tok/round of each quantized JetSpec++ replay against the bf16 replay of the same traces
(gsm8k 0-7 from replay_j3r3.json; humaneval / mbpp 0-7 from replay_<ds>.json), and our DFlash2 chain on the same prompts.
    PY b7_table.py RESULTS_20NOV_DIR"""
import json, re, sys
from pathlib import Path

R = Path(sys.argv[1])
J = R / "jetspec"
SETS = ["gsm8k", "humaneval", "mbpp"]
BASE = {"gsm8k": J / "replay_j3r3.json", "humaneval": J / "replay_humaneval.json", "mbpp": J / "replay_mbpp.json"}
DFL = {"gsm8k": J / "capture_decode_lines_j3r3.txt", "humaneval": J / "capture_decode_lines_humaneval.txt",
       "mbpp": J / "capture_decode_lines_mbpp.txt"}
ORDER = [("bf16", None), ("int8 per channel, adapter + head", "b7_adapter_8c_head_8c"),
         ("6-bit g128, adapter + head", "b7_adapter_6_head_6"), ("LUT4 per row, adapter (head bf16)", "b7_adapter_lut4"),
         ("LUT3 per row, adapter (head bf16)", "b7_adapter_lut3")]


def tpr(results, names):
    rs = [r for r in results if r["trace"] in names]
    return sum(r["generated"] for r in rs) / max(1, sum(r["rounds"] for r in rs)), all(r["lossless_vs_trace"] for r in rs), len(rs)


def dflash(ds, n=8):
    g = c = 0.0
    for l in list(open(DFL[ds]))[:n]:
        m = re.search(r"gen (\d+) in .*?, ([\d.]+) tok/verify", l)
        g += int(m.group(1)); c += int(m.group(1)) / float(m.group(2))
    return g / c


names = {ds: sorted(r["trace"] for r in json.loads(BASE[ds].read_text())["results"])[:8] for ds in SETS}
base = {ds: tpr(json.loads(BASE[ds].read_text())["results"], names[ds])[0] for ds in SETS}
print("| drafter weights | " + " | ".join(f"{ds} tok/round (vs bf16)" for ds in SETS) + " | lossless |")
print("|---|" + "---|" * len(SETS) + "---|")
for label, tag in ORDER:
    cells, ok = [], True
    for ds in SETS:
        if tag is None:
            v, l, n = tpr(json.loads(BASE[ds].read_text())["results"], names[ds])
            cells.append(f"{v:.2f}")
        else:
            f = R / "b7" / f"{tag}_{ds}.json"
            if not f.exists():
                cells.append("not run"); continue
            d = json.loads(f.read_text())
            got = {r["trace"] for r in d["results"]}
            v, l, n = tpr(d["results"], names[ds])
            if n < 8:   # a stopped run: compare with bf16 on the same traces
                vb = tpr(json.loads(BASE[ds].read_text())["results"], got)[0]
                cells.append(f"{v:.2f} (bf16 on the same {n}: {vb:.2f}, {v / vb - 1:+.0%})"); ok = ok and l; continue
            cells.append(f"{v:.2f} ({v / base[ds] - 1:+.1%})" + ("" if n == 8 else f" [{n} traces]"))
        ok = ok and l
    print(f"| {label} | " + " | ".join(cells) + f" | {'yes' if ok else 'NO'} |")
print("| our DFlash2 chain (tok/verify, same prompts) | " + " | ".join(f"{dflash(ds):.2f}" for ds in SETS) + " | |")
