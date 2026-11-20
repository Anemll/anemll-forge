"""D2 / D3 projection (20 Nov 2026): Jet Flash (JetSpec++ drafting + a 16-row chain verify on the ANE, serial) at the 8K and
32K entries vs the served DFlash2 chain, with the LUT4 (Hadamard + GPTQ) adapter's measured pass times.
  adapter pass   MEASURED: ANE (d2/b4 timing 8K; d2/b3 timing 8K / 32K), or GPU (d3/time_*.json + C1's GPU attention)
  B16 head pass  14.8 ms PROJECTED (b8_project.py: LUT6 bytes at 123 GB/s + the measured 16-row target head), same at 32K
  decoder pass    7.1 ms PROJECTED (b8_project.py), same at 32K
  verify, 16 rows MEASURED ratio x the served 8-row verify: 8K 94.1 ms x 1.085, 32K 106.5 ms x 1.17 (chain_v1, chain16/)
  DFlash2 chain  MEASURED cycle: 8K 107.7 ms (B8), 32K 121 ms (chain_v1 median), its tok/verify on the same prompts
passes / round from b8/passes.json; tok / round: the LUT4 GPTQ replays where measured (c4: J3-R3; d2: J0-R0), else bf16.
Acceptance comes from short-context traces at both entries (tok/round at 32K of context: UNMEASURED, assumed equal).
    PY d2_project.py RESULTS_20NOV_DIR ADAPTER_8K_MS ADAPTER_32K_MS LABEL"""
import json, sys
from pathlib import Path

R = Path(sys.argv[1])
AD = {"8K": float(sys.argv[2]), "32K": float(sys.argv[3])}
label = sys.argv[4] if len(sys.argv) > 4 else "ANE"
P = json.loads((R / "b8/passes.json").read_text())
HEAD, DEC, HOST = 14.8, 7.1, 1.0
VER = {"8K": 94.1 * 1.085, "32K": 106.5 * 1.17}
CYC = {"8K": 107.7, "32K": 121.0}
DFL = {"gsm8k": 6.08, "humaneval": 7.30, "mbpp": 5.89}
TPR = {("J3-R3", "gsm8k"): 10.17, ("J3-R3", "humaneval"): 13.55, ("J3-R3", "mbpp"): 10.45}
for f in R.glob("d2/d2_lut4hg_J0-R0_*.json"):
    d = json.loads(f.read_text())
    if d.get("summary"):
        TPR[("J0-R0", f.stem.rsplit("_", 1)[1])] = d["summary"]["tok_per_round"]
rows = []
print(f"| entry | schedule | set | adapter ({label}) + head + decoder ms / round | round ms | tok/round | Jet Flash tok/s | DFlash2 tok/s | vs DFlash2 |")
print("|---|---|---|---|---|---|---|---|---|")
for e in ("8K", "32K"):
    for s in ("J0-R0", "J3-R3"):
        for ds in ("gsm8k", "humaneval", "mbpp"):
            p = P[s][ds]
            a, h, dc = p["adapter"] * AD[e], p["head"] * HEAD, p["decoder"] * DEC
            rnd = a + h + dc + VER[e] + HOST
            tpr = TPR.get((s, ds)); src = "LUT4" if tpr else "bf16"
            tpr = tpr or p["tok_round"]
            js, df = 1e3 * tpr / rnd, 1e3 * DFL[ds] / CYC[e]
            rows.append({"entry": e, "schedule": s, "set": ds, "draft_ms": a + h + dc, "round_ms": rnd, "tok_round": tpr,
                         "tok_round_src": src, "jetflash_tps": js, "dflash2_tps": df})
            print(f"| {e} | {s} | {ds} | {a:.0f} + {h:.0f} + {dc:.0f} = {a + h + dc:.0f} | {rnd:.0f} | {tpr:.2f} ({src}) | {js:.1f} | "
                  f"{df:.1f} | {js / df - 1:+.0%} |")
(R / f"d2/projection_{label}.json").write_text(json.dumps({"adapter_ms": AD, "label": label, "rows": rows}, indent=1))
