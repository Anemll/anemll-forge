"""Time model of the V8 prefill from a traced run (18 Nov 2026, trace_analyze.py output). Per 128 prompt rows the GPU
runs 48 DeltaNet pair calls (cost independent of the position) and 16 attention pair calls (fixed + slope x position,
fitted on deciles 1-9 of the 32K+ runs); the Neural Engine runs 64 FFN pairs at FFN_MS (2.88 ms, the 2K trace, where it
sets the pace). The slower unit sets the rate. Prints today's rate per length and the rate with a cheaper attention:
    PY prefill_model.py TRACE_DIR [N,N,...]      -> TRACE_DIR/prefill_model.txt"""
import json, sys
from pathlib import Path
import numpy as np

FFN_MS = 2.88
PROJ_MS = 2.4               # an attention call at the start of a prompt (projections, gate, norms): the fixed part kept
FLOOR60_PEAK_MS = 7.5       # Q.K INT8 at 39.5 TOPS + P.V FP16 at 19.2 TOPS, 193 GFLOP per 128-row call at 60K keys
FLOOR60_I8PV_MS = 4.9       # P.V in INT8 too (8-bit probabilities, INT8 V)


def main():
    d = Path(sys.argv[1])
    ns = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "8192,32768,65536,95232").split(",")]
    S = json.loads((d / "trace_summary.json").read_text())
    P, A = [], []
    for s in S:
        if s["tokens"] >= 32768:
            for k in range(1, 10):
                P.append((k + 0.5) / 10 * s["tokens"] / 1000); A.append(s["gpu_ms_by_decile"][f"att{k}"])
    slope, fixed = np.polyfit(P, A, 1)
    resid = float(np.max(np.abs(np.array(A) - (fixed + slope * np.array(P)))))
    gdn = float(np.mean([np.median([v for k, v in s["gpu_ms_by_decile"].items() if k.startswith("gdn")]) for s in S]))
    ane = 64 * FFN_MS

    def rate(n, fx, sl):
        att = 16 * (fx + sl * n / 2000); gpu = 48 * gdn + att; t = max(gpu, ane)
        return 128e3 / t, att / gpu, ane / t

    forms = [("8-bit S/P + softmax fold (score part -25%)", fixed, 0.75 * slope),
             ("new kernel, 75% of peak", PROJ_MS, FLOOR60_PEAK_MS / 0.75 / 60),
             ("new kernel at peak", PROJ_MS, FLOOR60_PEAK_MS / 60),
             ("new kernel + INT8 P.V, 75% of peak", PROJ_MS, FLOOR60_I8PV_MS / 0.75 / 60)]
    L = [f"attention per 128-row call = {fixed:.2f} ms + {slope:.3f} ms per 1K keys of position (largest residual {resid:.2f} ms); "
         f"DeltaNet {gdn:.3f} ms per call; Neural Engine {ane:.0f} ms per 128 rows"]
    for n in ns:
        r0, sh, ab = rate(n, fixed, slope)
        row = [f"{n} tok: today {r0:.0f} tok/s (attention {100 * sh:.0f}% of GPU time, ANE busy {100 * ab:.0f}%)"]
        for lab, fx, sl in forms:
            r = rate(n, fx, sl)[0]; row.append(f"{lab}: {r:.0f} (+{100 * (r / r0 - 1):.0f}%)")
        L.append(" | ".join(row))
    (d / "prefill_model.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
