# D3 sweep: per-shape timing of each v6 configuration (one block's 7 matrices, 50 calls each), then correctness of the best
import os, sys, time, json
sys.path.insert(0, "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/jetspec")
import torch, gpu_lut4_pass as G
T = {p: G.load(f"streams.0.{p}") for p in G.PROJ}
res = {}
for cfg in [f"v7_{n}_{k}" for n, k in G.V7] + ["v6_4_64"]:
    G.MMK = cfg
    tot = 0.0; per = {}
    for p, t in T.items():
        x = torch.randn(16, t["K"], dtype=torch.float16, device="mps") * 0.1; y = torch.empty(16, t["O"], dtype=torch.float16, device="mps")
        for _ in range(5): G.mm(t, x, y)
        torch.mps.synchronize(); t0 = time.perf_counter()
        for _ in range(50): G.mm(t, x, y)
        torch.mps.synchronize(); ms = 1e3 * (time.perf_counter() - t0) / 50
        per[p] = ms; tot += ms
    res[cfg] = {"block_ms": tot, "per": per}
    print(f"{cfg}: one block's 7 matrices {tot:.3f} ms -> 12 blocks {12 * tot:.1f} ms ({0.186 / (tot / 1e3):.0f} GB/s)", flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
