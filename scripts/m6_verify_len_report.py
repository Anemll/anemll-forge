"""Verifier block length (T rows: 1 committed token + T - 1 drafts): measured full-target verify time per T and context
(scripts/m6_entry_sweep.py --chain records) combined with draft-acceptance histograms into expected tokens per cycle
and modeled decode tok/s. Prints Markdown tables.

    python scripts/m6_verify_len_report.py --sweep 8=t8.json 4=t4.json 3=t3.json \
        --accept session=session_hist.json --accept fixture=fixture_hist.json --other-ms 22.7
A histogram file is JSON {"hist": [p0, ..., p7]}: the fraction of T=8 cycles accepting exactly k of 7 drafts.
Model: with greedy prefix acceptance a T-row verifier accepts min(k, T - 1) drafts, so tokens/cycle = 1 + E[min(k, T - 1)];
tok/s = tokens/cycle / (verify ms + other ms) with the drafter, sampling and host time per cycle held fixed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def tokens_per_cycle(hist: list[float], T: int) -> float:
    s = sum(hist)
    return 1 + sum(p / s * min(k, T - 1) for k, p in enumerate(hist))


def verify_ms(sweep: dict) -> dict[int, float]:
    out = {}
    for name, r in sweep["entries"].items():
        if name.startswith("v"):
            out[int(name.split("_")[1].rstrip("k")) * 1024] = r["median_ms"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", nargs="+", required=True, help="T=path to an m6_entry_sweep --chain JSON")
    ap.add_argument("--accept", action="append", default=[], help="label=path to {'hist': [...]} (T=8 cycles)")
    ap.add_argument("--other-ms", type=float, required=True, help="drafter + sampling + host time per cycle")
    a = ap.parse_args()
    times = {int(t): verify_ms(json.loads(Path(p).read_text())) for t, p in (s.split("=") for s in a.sweep)}
    Ts = sorted(times, reverse=True)
    ctxs = sorted(set.intersection(*(set(v) for v in times.values())))
    print("| Context | " + " | ".join(f"T={T} verify" for T in Ts) + " |")
    print("| --- | " + " | ".join("---:" for _ in Ts) + " |")
    for c in ctxs:
        base = times[Ts[0]][c]
        cells = [f"{times[T][c]:.1f} ms" + ("" if T == Ts[0] else f" ({100 * (times[T][c] / base - 1):+.1f}%)") for T in Ts]
        print(f"| {c // 1024}K | " + " | ".join(cells) + " |")
    for spec in a.accept:
        label, path = spec.split("=", 1)
        hist = json.loads(Path(path).read_text())["hist"]
        tpc = {T: tokens_per_cycle(hist, T) for T in Ts}
        print(f"\n**{label}** acceptance: tokens per cycle " + ", ".join(f"T={T} {tpc[T]:.2f}" for T in Ts) + "\n")
        print("| Context | " + " | ".join(f"T={T} tok/s" for T in Ts) + " |")
        print("| --- | " + " | ".join("---:" for _ in Ts) + " |")
        for c in ctxs:
            rate = {T: tpc[T] / ((times[T][c] + a.other_ms) / 1e3) for T in Ts}
            base = rate[Ts[0]]
            print(f"| {c // 1024}K | " + " | ".join(
                f"{rate[T]:.1f}" + ("" if T == Ts[0] else f" ({100 * (rate[T] / base - 1):+.1f}%)") for T in Ts) + " |")


if __name__ == "__main__":
    main()
