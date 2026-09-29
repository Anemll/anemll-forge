"""Mixer -> MLP byte trade at equal size (M3U helper, 2026-09-27).

Downgrade whole mixer bands (all token-mixer matrices of 8 layers; k/v stay INT8) from LUT4 + pcs to vector 2x16 + pcs
and spend the freed bytes on promoting the remaining 2-bit MLP matrices to LUT4, while every trade gains more than it
costs. Inputs on the in-domain trace (qwen38_kl.py eval, rest bf16):
    kl_mband2lr_<lo>-<hi>.json  one band's mixers 2-bit + rank-64 plain factors re-fitted on the fly
    kl_mixers4lr_only.json      all mixers LUT4 + rank-64 factors (its KL is split over the bands in proportion to the
                                2-bit band KLs, the LUT4 share each band already costs)
    kl_band2_<lo>-<hi>.json + runs/sweep_mlp_...json + plan_indomain.json   the MLP side, as in qwen38_plan_indomain.py:
                                upgrading a matrix removes dmg / 3 * (1 - R4) * REAL (REAL = realized / predicted KL of
                                the in-domain re-plan, 0.084 / 0.113)
Bytes: 2 bits per parameter either way (LUT / pcs overheads are the same per matrix and ignored); the upgrade count is
rounded so the total stays within +-1% of today. Output: a PLAN for qwen38_gptq_27b.py (MLP formats + "<layer>.mixer").

    python qwen38_plan_mixr.py [--out /path/to/data/vq27b/plan_mixr.json]"""
import argparse
import json
from pathlib import Path

LOW, HIGH = "vector 2x16 + pcs", "LUT4 per-tensor + pcs"
R4, REAL = 0.06, 0.084 / 0.113
HID, INTER = 5120, 17408
MIXER_PARAMS = {"full": 12288 * HID + HID * 6144, "line": 10240 * HID + 6144 * HID + HID * 6144}  # q + o / qkv + z + out
MLP_MB = INTER * HID * 2 / 8 / 1e6


def main():
    L = Path("/path/to/data/vq27b")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kl-dir", default=str(L / "kl"))
    ap.add_argument("--sweep", default=str(L / "runs/sweep_mlp_vector_2x16_+_pcs_online.json"))
    ap.add_argument("--plan", default=str(L / "plan_indomain.json"))
    ap.add_argument("--out", default=str(L / "plan_mixr.json"))
    ap.add_argument("--total-gib", type=float, default=9.0617, help="today's quantized size (for the +-1% check)")
    a = ap.parse_args()
    kd = Path(a.kl_dir)
    kl = lambda tag: json.loads((kd / f"kl_{tag}.json").read_text())["mean_kl"]  # noqa: E731
    types = ["full" if i % 4 == 3 else "line" for i in range(64)]
    mb = {}
    for p in sorted(kd.glob("kl_mband2lr_*.json")):
        lo, hi = (int(x) for x in p.stem.split("_")[-1].split("-"))
        mb[(lo, hi)] = json.loads(p.read_text())["mean_kl"]
    lut4 = kl("mixers4lr_only")
    s2 = sum(mb.values())
    band = {}
    for (lo, hi), k2 in sorted(mb.items()):
        freed = sum(MIXER_PARAMS[types[i]] for i in range(lo, hi + 1)) * 2 / 8 / 1e6
        net = k2 - lut4 * k2 / s2
        band[(lo, hi)] = {"kl2": k2, "net": net, "freed_mb": freed, "per_mb": net / freed}
    # MLP candidates (qwen38_plan_indomain.py estimate)
    bands = {}
    for p in sorted(kd.glob("kl_band2_*.json")):
        lo, hi = (int(x) for x in p.stem.split("_")[-1].split("-"))
        bands[(lo, hi)] = json.loads(p.read_text())["mean_kl"]
    delta = json.loads(Path(a.sweep).read_text())["delta"]
    dmg = {}
    for (lo, hi), k in bands.items():
        w = [max(delta[i], 1e-4) for i in range(lo, hi + 1)]
        for i, wi in zip(range(lo, hi + 1), w):
            dmg[i] = k * wi / sum(w)
    plan = json.loads(Path(a.plan).read_text())
    cand = sorted(((dmg[i] / 3 * (1 - R4) * REAL, i, m) for i in range(64) for m in ("gate", "up", "down")
                   if plan[f"{i}.{m}"] != HIGH), key=lambda x: (-x[0], x[1], x[2]))
    # greedy: cheapest mixer band per MB first, keep it while the MLP upgrades it pays for gain more than it costs
    chosen, used, freed, gain_mlp, dmg_mix = [], 0, 0.0, 0.0, 0.0
    for b, v in sorted(band.items(), key=lambda x: x[1]["per_mb"]):
        n_new = round((freed + v["freed_mb"]) / MLP_MB) - used
        g = sum(c[0] for c in cand[used:used + n_new])
        print(f"band {b[0]:2d}-{b[1]:2d}: 2-bit KL {v['kl2']:.4f}, net {v['net']:.4f}, frees {v['freed_mb']:.1f} MB "
              f"-> {n_new} MLP upgrades gain {g:.4f}: {'TAKE' if g > v['net'] else 'skip'}")
        if g <= v["net"]:
            continue
        chosen.append(b)
        freed += v["freed_mb"]
        used += n_new
        gain_mlp += g
        dmg_mix += v["net"]
    ups = cand[:used]
    new = dict(plan)
    for _, i, m in ups:
        new[f"{i}.{m}"] = HIGH
    for lo, hi in chosen:
        for i in range(lo, hi + 1):
            new[f"{i}.mixer"] = LOW
    Path(a.out).write_text(json.dumps(new, indent=1))
    d_mb = used * MLP_MB - freed
    print(f"mixer bands -> 2-bit: {sorted(chosen)} (freed {freed:.1f} MB)")
    print(f"MLP -> LUT4: {used} matrices, layers {sorted({i for _, i, _ in ups})} (+{used * MLP_MB:.1f} MB)")
    print(f"net size {d_mb:+.1f} MB = {100 * d_mb / (a.total_gib * 2**30 / 1e6):+.2f}% of {a.total_gib} GiB")
    print(f"estimated: MLP gain {gain_mlp:.4f}, mixer damage {dmg_mix:.4f}, net {gain_mlp - dmg_mix:+.4f} KL")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
