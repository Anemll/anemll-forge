"""U2 (20 Nov 2026): per-stream static ExeCycles of a package's HWX (hwx_mode._tasks), streams in pairs (one per ANE
instance per function), with each pair's balance (the busier stream / the pair's mean) and the NE / PE split.
    PY u2_streams.py PKG"""
import collections, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hwx_mode as HM  # noqa: E402

for pkg in sys.argv[1:]:
    for h in HM.HI.hwx_paths(Path(pkg)):
        ts = HM._tasks(h)
        cyc, ne = collections.Counter(), collections.Counter()
        for t in ts:
            cyc[t["stream"]] += t["cycles"]
            if t.get("ne"):
                ne[t["stream"]] += t["cycles"]
        ss = sorted(cyc)
        print(f"{Path(pkg).name}: {len(ss)} streams")
        for i in range(0, len(ss) - 1, 2):
            a, b = ss[i], ss[i + 1]
            m = (cyc[a] + cyc[b]) / 2
            print(f"  streams {a}/{b}: {cyc[a]} / {cyc[b]} static cycles (NE {ne[a]} / {ne[b]}), busier / mean {max(cyc[a], cyc[b]) / max(m, 1):.2f}")
