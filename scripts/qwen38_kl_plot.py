"""Plot mean KL divergence vs quantized weight size for Qwen3.8-27B: our ANE VQ variants (kl_*.json from qwen38_kl.py)
over the published reference points (transcribed from the Qwen 3.8 27B EXL3 / GGUF / NVFP4 / FP8 comparison chart,
which used a self-generated in-domain trace of 19,016 input + 45,930 output tokens; our trace is our own, so the
comparison is indicative).

    python qwen38_kl_plot.py /path/to/data/vq27b/kl  out.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REF = {  # series: [(label, GiB, mean KL)]
    "EXL3": [("3.00 bpw", 9.46, 0.0332), ("4.00 bpw", 12.28, 0.0082), ("5.00 bpw", 15.10, 0.0023),
             ("6.00 bpw", 17.93, 0.0009)],
    "EXL3-SC": [("1.40 bpw H3", 4.48, 0.5429), ("1.60 bpw H3", 5.03, 0.3988), ("1.80 bpw H3", 5.61, 0.2709),
                ("2.00 bpw H3", 6.17, 0.1168), ("2.20 bpw H3", 6.74, 0.0876), ("3.00 bpw H4", 9.16, 0.0257),
                ("4.00 bpw H5", 12.13, 0.0062), ("5.00 bpw H6", 15.10, 0.0016), ("6.00 bpw H6", 17.93, 0.0006)],
    "GGUF": [("UD-Q2_K_XL", 9.55, 0.0587), ("UD-Q3_K_XL", 12.01, 0.0209), ("UD-Q4_K_XL", 16.02, 0.0051),
             ("UD-Q5_K_XL", 18.00, 0.0028), ("UD-Q6_K_XL", 23.17, 0.0009)],
    "GGUF-IQ": [("IQ4_XS", 13.95, 0.0101)],
    "GGUF-UD3": [("UD3-IQ1_S", 5.37, 0.4724), ("UD3-IQ1_M", 5.88, 0.3148), ("UD3-IQ2_XXS", 6.37, 0.2219),
                 ("UD3-IQ2_S", 7.40, 0.1329)],
    "NVFP4": [("NVFP4 (Unsloth)", 17.81, 0.0092)],
    "FP8": [("FP8 (Qwen)", 25.08, 0.0022)],
}
REF_NOISE = 0.00045
COLORS = {"EXL3": "#f5c518", "EXL3-SC": "#e0306e", "GGUF": "#c8e39a", "GGUF-IQ": "#4a90d9", "GGUF-UD3": "#ff5a2a",
          "NVFP4": "#b0452f", "FP8": "#1f2fe0"}
LABELS = {"full_mix25_mixer4_head4": "OptiQ 2.5b MLP / 4b mixers (ANE build 1)",
          "sweep_8.0GB": "sweep plan 8.0 GB", "sweep_9.9GB": "sweep plan 9.9 GB",
          "optiq_top48_mix": "OptiQ 2.5b MLP only (mixers bf16)"}


def main():
    kl_dir, out = Path(sys.argv[1]), sys.argv[2]
    ours = [json.loads(p.read_text()) for p in sorted(kl_dir.glob("kl_*.json"))]
    noise = next((r["mean_kl"] for r in ours if r["tag"] == "bf16"), None)
    ours = [r for r in ours if r["size_gib"]]
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#444", "text.color": "#ddd", "axes.labelcolor": "#ddd",
                         "xtick.color": "#aaa", "ytick.color": "#aaa"})
    fig, ax = plt.subplots(figsize=(14, 9), dpi=140)
    fig.patch.set_facecolor("#16181d")
    ax.set_facecolor("#1f2128")
    for series, pts in REF.items():
        xs, ys = [p[1] for p in pts], [p[2] for p in pts]
        ax.plot(xs, ys, ":", color=COLORS[series], alpha=0.45, lw=1.5)
        ax.scatter(xs, ys, color=COLORS[series], alpha=0.55, s=45, edgecolor="white", lw=0.5, label=series, zorder=3)
        for lab, x, y in pts:
            ax.annotate(f"{lab}\n{y:.4f}", (x, y), xytext=(6, 6), textcoords="offset points", fontsize=7,
                        color=COLORS[series], alpha=0.6)
    if ours:
        pts = sorted(ours, key=lambda r: r["size_gib"])
        ax.plot([r["size_gib"] for r in pts], [r["mean_kl"] for r in pts], "-", color="#00e5a8", lw=2, alpha=0.8)
        ax.scatter([r["size_gib"] for r in pts], [r["mean_kl"] for r in pts], marker="*", s=320, color="#00e5a8",
                   edgecolor="white", lw=0.8, zorder=5, label="VQ on ANE (ours)")
        for r in pts:
            ax.annotate(f"{LABELS.get(r['tag'], r['tag'])}\n{r['mean_kl']:.4f}  (top-1 {100 * r['top1_agree']:.1f}%)",
                        (r["size_gib"], r["mean_kl"]), xytext=(10, -22), textcoords="offset points", fontsize=9,
                        color="#00e5a8", fontweight="bold")
    ax.axhline(REF_NOISE, ls=":", color="#aaa", lw=1.5)
    ax.text(3.2, REF_NOISE * 1.1, f"reference noise floor {REF_NOISE}", color="#aaa", fontsize=9)
    if noise:
        ax.axhline(max(noise, 1e-5), ls=":", color="#00e5a8", lw=1, alpha=0.6)
        ax.text(3.2, max(noise, 1e-5) * 0.75, f"our noise floor {noise:.5f}", color="#00e5a8", fontsize=9, alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlim(3, 27)
    ax.set_ylim(2.5e-4, 1.3)
    ax.grid(True, color="#333", lw=0.8)
    ax.set_xlabel("quantized weight size |W_q| / GiB (excl. embeddings, incl. output head)")
    ax.set_ylabel("mean KL divergence  D_KL(p_FP || p_quant)")
    tok = ours[0]["tokens"] if ours else 0
    ax.set_title("Qwen 3.8 27B: VQ on the Apple Neural Engine vs published quants\n"
                 f"ours: self-generated trace, {tok:,} scored tokens; reference points transcribed "
                 "(19,016 input + 45,930 output tokens)", color="#eee", fontsize=13)
    ax.legend(loc="upper right", facecolor="#1f2128", edgecolor="#444", labelcolor="#ddd")
    fig.tight_layout()
    fig.savefig(out, facecolor=fig.get_facecolor())
    print("wrote", out)


if __name__ == "__main__":
    main()
