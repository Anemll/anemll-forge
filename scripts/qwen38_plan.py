"""Bit allocation for Qwen3.8-27B from the per-layer sensitivity sweeps (qwen38_gptq_27b.py SWEEP=mlp / mixer):
every layer's MLP and token mixer starts at 2 bits (vector 2x16 + per-channel scale); the upgrades to 4 bits
(LUT4 per-tensor + per-channel scale) with the largest measured loss per extra byte are taken until the size
budget. K/V stay INT8, lm_head LUT4, everything with GPTQ + online Hadamard (MLP).

    python qwen38_plan.py sweep_mlp.json sweep_mixer.json 8.0 9.9      # budgets in GB (1e9 bytes)
"""
import json
import sys

import numpy as np

LOW, HIGH = "vector 2x16 + pcs", "LUT4 per-tensor + pcs"
R4 = 0.06  # 4-bit loss / 2-bit loss (all-MLP: +1.3% vs +23% perplexity)
MLP_W = 3 * 5120 * 17408
GDN_W = (2 * 2048 + 6144) * 5120 + 6144 * 5120 + 5120 * 6144   # in_proj_qkv + in_proj_z + out_proj
ATT_W = 12288 * 5120 + 5120 * 6144                             # q_proj (with gate) + o_proj
KV_W, HEAD_W = 2 * 1024 * 5120, 248320 * 5120
ROT = 64 * (5120 + 17408) * 1024 / 8                            # 1-bit Hadamard weights
SMALL = 48 * 2 * 48 * 5120 * 2 + 48 * 10240 * 4 * 2 + 30e6       # in_proj_a / b, conv taps, norms, scales


def main():
    mlp, mix = (np.clip(np.array(json.load(open(f))["delta"]), 0, None) for f in sys.argv[1:3])
    kinds = ["full" if i % 4 == 3 else "gdn" for i in range(64)]
    base = MLP_W * 64 * 0.25 + sum((GDN_W if k == "gdn" else ATT_W) * 0.25 for k in kinds) + KV_W * 16 + \
        HEAD_W * 0.5 + ROT + SMALL
    items = [(mlp[i] * (1 - R4), MLP_W * 0.25, i, "mlp") for i in range(64)] + \
            [(mix[i] * (1 - R4), (GDN_W if kinds[i] == "gdn" else ATT_W) * 0.25, i, "mixer") for i in range(64)]
    items.sort(key=lambda t: -t[0] / t[1])
    for budget in (float(b) * 1e9 for b in sys.argv[3:]):
        size, up = base, set()
        for gain, cost, i, part in items:
            if gain > 0 and size + cost <= budget:
                up.add((i, part))
                size += cost
        plan = {}
        for i in range(64):
            f = HIGH if (i, "mlp") in up else LOW
            plan.update({f"{i}.gate": f, f"{i}.up": f, f"{i}.down": f})
            plan[f"{i}.mixer"] = HIGH if (i, "mixer") in up else LOW
        loss = sum(d * (R4 if (i, "mlp") in up else 1) for i, d in enumerate(mlp)) + \
            sum(d * (R4 if (i, "mixer") in up else 1) for i, d in enumerate(mix))
        name = f"plan_sweep_{budget / 1e9:.1f}GB.json"
        json.dump(plan, open(name, "w"), indent=0)
        mlp_up = sorted(i for i, p in up if p == "mlp")
        mix_up = sorted(i for i, p in up if p == "mixer")
        bpw = size * 8 / (MLP_W * 64 + sum(GDN_W if k == "gdn" else ATT_W for k in kinds) + KV_W * 16 + HEAD_W)
        print(f"{name}: {size / 1e9:.2f} GB ({size / 2**30:.2f} GiB, {bpw:.2f} bpw); est. perplexity x{np.exp(loss):.3f}; "
              f"4-bit MLP layers {mlp_up}; 4-bit mixer layers {mix_up}")
    print(f"all 2-bit: {base / 1e9:.2f} GB, est. x{np.exp(mlp.sum() + mix.sum()):.3f}; "
          f"all 4-bit est. x{np.exp(R4 * (mlp.sum() + mix.sum())):.3f}")


if __name__ == "__main__":
    main()
