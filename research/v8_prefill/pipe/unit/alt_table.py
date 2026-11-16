"""Per-config means of driver run JSONs. usage: alt_table.py KEY FILE... (KEY: attproj | attproj_from)"""
import json, sys
key = sys.argv[1]
for p in sys.argv[2:]:
    try:
        rs = json.load(open(p))["runs"]
    except Exception as e:
        print(p, "missing", e); continue
    vals = []
    for r in rs:
        if r.get(key) not in vals:
            vals.append(r.get(key))
    for v in vals:
        x = [r for r in rs if r.get(key) == v]
        tok = [r["v8_tok_s"] for r in x]
        g = sum(r["timeline"]["gpu_busy_frac"] for r in x) / len(x); a = sum(r["timeline"]["ane_busy_frac"] for r in x) / len(x)
        reps = " ".join("%.0f" % t for t in tok)
        print("%-16s %s=%-8s %6.1f tok/s (%s)  G%3.0f A%3.0f  KL %.4f" % (p.rsplit("/", 1)[-1][:-5], key, v, sum(tok) / len(tok), reps, g * 100, a * 100, x[0]["kl_mean"]))
