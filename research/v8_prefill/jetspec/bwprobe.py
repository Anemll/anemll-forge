import time, torch
lib = torch.mps.compile_shader("""
kernel void rd(device const uint4 *a [[buffer(0)]], device uint *o [[buffer(1)]], constant uint &n [[buffer(2)]],
               uint i [[thread_position_in_grid]], uint G [[threads_per_grid]]) {
    uint4 s = 0u;
    for (uint j = i; j < n; j += G) s ^= a[j];
    if ((s.x ^ s.y ^ s.z ^ s.w) == 0xdeadbeefu) o[0] = 1u;
}
""")
n_bytes = 2_233_000_000 // 64 * 64
a = torch.randint(0, 255, (n_bytes,), dtype=torch.uint8, device="mps"); o = torch.zeros(1, dtype=torch.int32, device="mps")
n = n_bytes // 16
for G in (65536, 262144, 1048576):
    for _ in range(3): lib.rd(a, o, n, threads=G, group_size=256)
    torch.mps.synchronize(); t = time.perf_counter()
    for _ in range(5): lib.rd(a, o, n, threads=G, group_size=256)
    torch.mps.synchronize(); ms = (time.perf_counter() - t) / 5 * 1e3
    print(f"read kernel, {G} threads: {ms:.1f} ms for {n_bytes/1e9:.2f} GB -> {n_bytes/1e9/(ms/1e3):.0f} GB/s", flush=True)
w = torch.randn(17408, 5120, dtype=torch.float16, device="mps"); x = torch.randn(16, 5120, dtype=torch.float16, device="mps")
for _ in range(5): y = x @ w.T
torch.mps.synchronize(); t = time.perf_counter()
for _ in range(50): y = x @ w.T
torch.mps.synchronize(); ms = (time.perf_counter() - t) / 50 * 1e3
print(f"torch fp16 matmul 16 x 5120 @ 5120 x 17408: {ms:.3f} ms -> {w.numel()*2/1e9/(ms/1e3):.0f} GB/s of weights", flush=True)
