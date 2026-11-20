"""Compare tree_bench.py records (20 Nov 2026): per rung token-weighted decode tok/s, tok/verify (server), decode W
(mactop total, mean over requests weighted by decode time), and output identity against the first record (equal reply
sha per request; the first differing character and token count where not).
    PY compare.py BASE.json OTHER.json [OTHER2.json ...]"""
import json, sys
from pathlib import Path


def rung_stats(rows):
    ok = [r for r in rows if r.get("server") and r.get("t_first") and r.get("t_last") and r["completion_tokens"] > 1]
    if not ok:
        return None
    dt = sum(r["t_last"] - r["t_first"] for r in ok)
    tps = sum(r["completion_tokens"] - 1 for r in ok) / dt
    tv = sum(r["server"]["gen"] for r in ok) / sum(r["server"]["gen"] / r["server"]["tok_verify"] for r in ok)
    pw = [(r["decode_power"]["mean_w"]["total_power"], r["t_last"] - r["t_first"]) for r in ok if r.get("decode_power")]
    w = sum(a * b for a, b in pw) / max(1e-9, sum(b for _, b in pw)) if pw else float("nan")
    cyc = sum(r["server"]["ms_cycle"] * r["server"]["gen"] / r["server"]["tok_verify"] for r in ok) / \
        sum(r["server"]["gen"] / r["server"]["tok_verify"] for r in ok)
    return {"tok_s": tps, "tok_verify": tv, "W": w, "ms_cycle": cyc, "n": len(ok)}


def main():
    recs = [json.loads(Path(p).read_text()) for p in sys.argv[1:]]
    base = recs[0]
    print("| rung | " + " | ".join(f"{r['label']} tok/s, tok/verify, ms/cycle, W" for r in recs) + " | identical to " + base["label"] + " |")
    print("|---|" + "---|" * len(recs) + "---|")
    for rung in base["rungs"]:
        cells, ident = [], []
        for r in recs:
            s = rung_stats(r["rungs"].get(rung, []))
            cells.append("-" if s is None else f"{s['tok_s']:.2f}, {s['tok_verify']:.3f}, {s['ms_cycle']:.0f}, {s['W']:.1f}")
        for r in recs[1:]:
            a, b = base["rungs"].get(rung, []), r["rungs"].get(rung, [])
            same = sum(1 for x, y in zip(a, b) if x["sha"] == y["sha"])
            ident.append(f"{same}/{min(len(a), len(b))}")
        print(f"| {rung} | " + " | ".join(cells) + " | " + ", ".join(ident) + " |")
    for r in recs[1:]:
        for rung in base["rungs"]:
            for x, y in zip(base["rungs"].get(rung, []), r["rungs"].get(rung, [])):
                if x["sha"] != y["sha"]:
                    a, b = x["content"], y["content"]
                    j = next((i for i, (c, d) in enumerate(zip(a, b)) if c != d), min(len(a), len(b)))
                    print(f"  {r['label']} {rung} {x['set']}:{x['index']} differs at char {j} of {len(a)} / {len(b)}: "
                          f"{a[max(0, j - 30):j + 30]!r} vs {b[max(0, j - 30):j + 30]!r}")


if __name__ == "__main__":
    main()
