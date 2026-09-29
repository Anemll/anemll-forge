"""Re-allocate the MLP 4-bit budget from in-domain KL instead of WikiText perplexity, at the SAME size.

Inputs (M3U): the band sweep kl_band2_<lo>-<hi>.json (KL vs bf16 with only that MLP band at 2-bit, rest bf16, from
the all-2-bit export mlp2_aw_cal), the per-layer WikiText sweep (only to split a band's damage among its layers),
and the current plan (its number of LUT4 matrices is the budget). A layer's in-domain 2-bit damage is estimated as
its band's KL split by max(WikiText delta, floor); upgrading a matrix to LUT4 removes ~(1 - R4) of its share
(R4 = 4-bit loss / 2-bit loss = 0.06, qwen38_plan.py). The budget goes to the largest estimated damages, per matrix
(gate / up / down share a layer's damage equally).
    python qwen38_plan_indomain.py [--kl-dir /path/to/data/vq27b/kl] [--sweep runs/sweep_mlp_...json] \
        [--plan plan_optiq_top48.json] [--out plan_indomain.json]"""
import argparse
import json
from pathlib import Path

LOW, HIGH = "vector 2x16 + pcs", "LUT4 per-tensor + pcs"
R4 = 0.06


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    L = Path("/path/to/data/vq27b")
    ap.add_argument("--kl-dir", default=str(L / "kl"))
    ap.add_argument("--sweep", default=str(L / "runs/sweep_mlp_vector_2x16_+_pcs_online.json"))
    ap.add_argument("--plan", default=str(L / "plan_optiq_top48.json"))
    ap.add_argument("--out", default=str(L / "plan_indomain.json"))
    ap.add_argument("--budget", type=int, default=0, help="LUT4 matrices (0 = as many as --plan)")
    a = ap.parse_args()
    bands = {}
    for p in sorted(Path(a.kl_dir).glob("kl_band2_*.json")):
        lo, hi = (int(x) for x in p.stem.split("_")[-1].split("-"))
        bands[(lo, hi)] = json.loads(p.read_text())["mean_kl"]
    assert bands, f"no kl_band2_*.json in {a.kl_dir}"
    delta = json.loads(Path(a.sweep).read_text())["delta"]
    old = json.loads(Path(a.plan).read_text())
    budget = a.budget or sum(v == HIGH for v in old.values())
    dmg = {}
    for (lo, hi), kl in bands.items():
        w = [max(delta[i], 1e-4) for i in range(lo, hi + 1)]
        for i, wi in zip(range(lo, hi + 1), w):
            dmg[i] = kl * wi / sum(w)
    items = sorted(((dmg[i] / 3, i, m) for i in dmg for m in ("gate", "up", "down")), reverse=True)
    up = {(i, m) for _, i, m in items[:budget]}
    plan = {f"{i}.{m}": HIGH if (i, m) in up else LOW for i in range(64) for m in ("gate", "up", "down")}
    Path(a.out).write_text(json.dumps(plan, indent=1))
    old_up = {tuple(k.split(".")) for k, v in old.items() if v == HIGH}
    gain = lambda s: sum(dmg.get(int(i), 0) / 3 for i, _ in s) * (1 - R4)  # noqa: E731
    print("band KL (2-bit band, rest bf16):", {f"{lo}-{hi}": round(v, 4) for (lo, hi), v in sorted(bands.items())})
    print(f"budget {budget} LUT4 matrices; sum of band KLs {sum(bands.values()):.3f}")
    print(f"new LUT4 layers (matrices): {sorted({i for i, _ in up})}")
    print(f"estimated KL removed by the budget: new plan {gain({(str(i), m) for i, m in up}):.3f}, "
          f"old plan {gain(old_up):.3f} (same estimate, in-domain)")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
