import json, sys, collections
r = json.load(open(sys.argv[1]))
by = collections.defaultdict(list)
print("rep  gpu_ffn       e2e tok/s  ane_ms  gpu_ms  ANE   GPU  decodeKL agree")
for x in r["runs"]:
    t, e = x["timeline"], x.get("e2e", {})
    k = ("apj+" if x.get("attproj") else "") + "mix%d" % len(x.get("ane_mix", [])) + ("+gffn" + ",".join(map(str, x["gpu_ffn"])) if x["gpu_ffn"] else "")
    by[k].append(e.get("tok_s", x["v8_tok_s"]))
    print("%3d  %-12s %8.1f   %5.2f   %5.2f  %.2f  %.2f  %.4f  %.3f" % (x["rep"], k, e.get("tok_s", x["v8_tok_s"]), t["ane_call_ms"]["median"],
          t["gpu_ms"]["median"], t["ane_busy_frac"], t["gpu_busy_frac"], e.get("decode_kl_mean", float("nan")), e.get("decode_argmax_agree", float("nan"))))
for k, v in by.items():
    print("%-12s mean %.1f  (%s)" % (k, sum(v) / len(v), " ".join("%.0f" % y for y in v)))
