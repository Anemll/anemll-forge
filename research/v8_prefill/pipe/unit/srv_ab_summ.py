"""Decode A/B summary from a V8 server log (V8_AB=g1l,g3l,g3e; 16 Nov 2026): every request's '[ab] arm' line and its
summary line ('prompt ... ms/cycle N (draft d verify v sample s ...) ... (NK entry)'), grouped by arm and context
entry, weighted by verify cycles (generated tokens / tokens per verify). Requests with fewer than --min-gen tokens
are skipped (their cycle means are noise).
    python3 srv_ab_summ.py /Volumes/SSD4TB/anemll-forge-research/coreai_seg_logs/v8_server.log [--min-gen 100]"""
import argparse, re
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--min-gen", type=int, default=100)
    a = ap.parse_args()
    arm, rows = None, []
    for ln in open(a.log, encoding="utf-8", errors="replace"):
        m = re.search(r"\[ab\] arm (\S+):", ln)
        if m:
            arm = m[1]
            continue
        m = re.search(r"\] prompt \d+ \(cached.*\| gen (\d+) in [\d.]+s \(([\d.]+) tok/s, ([\d.]+) tok/verify.*ms/cycle ([\d.]+) "
                      r"\(draft ([\d.]+) verify ([\d.]+) sample ([\d.]+) ctx ([\d.]+) host ([\d.]+)\).*\((\d+)K entry\)", ln)
        if m and arm:
            g, tps, tpv = int(m[1]), float(m[2]), float(m[3])
            if g >= a.min_gen:
                rows.append(dict(arm=arm, entry=int(m[10]), cyc=g / tpv, tps=tps, tpv=tpv, cycle=float(m[4]),
                                 draft=float(m[5]), verify=float(m[6]), sample=float(m[7]), ctx=float(m[8])))
            arm = None
    agg = defaultdict(lambda: defaultdict(float))
    for r in rows:
        d = agg[(r["entry"], r["arm"])]
        d["n"] += 1; d["w"] += r["cyc"]
        for k in ("cycle", "draft", "verify", "sample", "ctx", "tpv"):
            d[k] += r[k] * r["cyc"]
    print(f"{'entry':>6} {'arm':>5} {'req':>4} {'cycles':>7} {'ms/cycle':>9} {'draft':>6} {'verify':>7} {'sample':>7} "
          f"{'ctx':>5} {'tok/verify':>10}")
    for (entry, arm_), d in sorted(agg.items()):
        w = d["w"]
        print(f"{entry:>5}K {arm_:>5} {int(d['n']):>4} {w:>7.0f} {d['cycle'] / w:>9.1f} {d['draft'] / w:>6.1f} "
              f"{d['verify'] / w:>7.1f} {d['sample'] / w:>7.2f} {d['ctx'] / w:>5.2f} {d['tpv'] / w:>10.2f}")


if __name__ == "__main__":
    main()
