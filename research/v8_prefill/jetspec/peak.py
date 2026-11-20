import time, torch
for M in (16, 64, 256, 2048):
    w = torch.randn(17408, 5120, dtype=torch.float16, device="mps"); x = torch.randn(M, 5120, dtype=torch.float16, device="mps")
    for _ in range(3): y = x @ w.T
    torch.mps.synchronize(); t = time.perf_counter(); n = 20
    for _ in range(n): y = x @ w.T
    torch.mps.synchronize(); ms = (time.perf_counter() - t) / n * 1e3
    print(f"fp16 matmul M={M}: {ms:.3f} ms, {M*17408*5120/ms/1e9:.2f} TFMA/s, {w.numel()*2/1e9/(ms/1e3):.0f} GB/s weights", flush=True)
