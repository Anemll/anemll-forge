import sys, json
for l in sys.stdin:
    r = json.loads(l); t = r["timeline"]
    print("  rep %d: %.1f tok/s (heads run %.1f)  KL %.4f top1 %.3f  GPU %.2f ANE %.2f  gpu_ms %s ane_ms %s" % (r["rep"], r["v8_tok_s"], r["v8_heads_tok_s"], r["kl_mean"], r["top1_agree"], t["gpu_busy_frac"], t["ane_busy_frac"], t["gpu_ms"]["median"], t["ane_call_ms"]["median"]))
