"""Verbosity probe analysis: tokenizes each sample's thinking and answer with the model's own tokenizer (the same
count for both engines), then compares the engines on the (id, rep) samples both completed without error.
    FORGE_VENV_PYTHON analyze.py SPLASH.jsonl FORGE.jsonl [OUT.json]
"""
import json, math, statistics as st, sys
from pathlib import Path

from tokenizers import Tokenizer

TOK = Tokenizer.from_file("/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/model/tokenizer.json")
BUDGET = 6144


def load(p):
    rows = {}
    for line in Path(p).read_text().splitlines():
        r = json.loads(line)
        if r["error"]:
            continue
        r["think_tok"] = len(TOK.encode(r["reasoning"]).ids)
        r["ans_tok"] = len(TOK.encode(r["content"]).ids)
        u = r.get("usage") or {}
        r["out_tok"] = u.get("completion_tokens") or (r["think_tok"] + r["ans_tok"])
        rows[(r["id"], r["rep"])] = r
    return rows


s, f = load(sys.argv[1]), load(sys.argv[2])
keys = sorted(set(s) & set(f))
if not keys:
    sys.exit("no common samples")


def summ(rows):
    th = [rows[k]["think_tok"] for k in keys]
    an = [rows[k]["ans_tok"] for k in keys]
    wall = sum(rows[k]["wall_s"] for k in keys)
    out = sum(rows[k]["out_tok"] for k in keys)
    return {"samples": len(keys), "think_mean": round(st.mean(th)), "think_median": round(st.median(th)),
            "answer_mean": round(st.mean(an)), "answer_median": round(st.median(an)),
            "budget_hits": sum(t >= BUDGET for t in th), "wall_s": round(wall), "tok_per_s": round(out / wall, 1),
            "finish_length": sum(rows[k]["finish"] == "length" for k in keys)}


def sign_p(wins, n):   # two-sided exact sign test
    tail = sum(math.comb(n, i) for i in range(0, min(wins, n - wins) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


per = []
for k in keys:
    a, b = s[k], f[k]
    per.append({"id": k[0], "rep": k[1], "splash_think": a["think_tok"], "forge_think": b["think_tok"],
                "splash_answer": a["ans_tok"], "forge_answer": b["ans_tok"]})
think_wins = sum(p["splash_think"] > p["forge_think"] for p in per)
ties = sum(p["splash_think"] == p["forge_think"] for p in per)
ans_wins = sum(p["splash_answer"] > p["forge_answer"] for p in per)
lr = [math.log((p["splash_think"] + 1) / (p["forge_think"] + 1)) for p in per]
la = [math.log((p["splash_answer"] + 1) / (p["forge_answer"] + 1)) for p in per]
res = {"splash": summ(s), "forge": summ(f),
       "paired": {"samples": len(per),
                  "splash_thinks_longer": f"{think_wins}/{len(per) - ties}", "sign_p_think": round(sign_p(think_wins, len(per) - ties), 4),
                  "think_ratio_geomean": round(math.exp(st.mean(lr)), 2),
                  "splash_answers_longer": f"{ans_wins}/{len(per)}", "sign_p_answer": round(sign_p(ans_wins, len(per)), 4),
                  "answer_ratio_geomean": round(math.exp(st.mean(la)), 2)},
       "per_sample": per}
print(json.dumps({k: v for k, v in res.items() if k != "per_sample"}, indent=1))
print(f"\n{'prompt':16s} rep  think S / F     answer S / F")
for p in per:
    print(f"{p['id']:16s} {p['rep']:3d}  {p['splash_think']:6d} / {p['forge_think']:<6d}  {p['splash_answer']:6d} / {p['forge_answer']:<6d}")
if len(sys.argv) > 3:
    Path(sys.argv[3]).write_text(json.dumps(res, indent=1))
