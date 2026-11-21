"""E1 (20 Nov 2026): row sweep of ONE LUT4 matmul on the GPU (the adapter's largest shape, streams.0 mlp.gate_proj,
17408 x 5120, Hadamard + GPTQ tensor), with D3's best kernel form (v6: NSG simdgroups x 8 output rows per threadgroup,
K chunks of KC columns, 4-bit indices decoded through a threadgroup palette) generalized to MT 8-row input tiles
(rows 1 / 4 / 8 padded to one tile; 16 = 2 tiles = D3's kernel; 32 = 4; 64 = 8). Controls on the same rows: torch fp16
dense matmul (the same matrix dequantized), the D3 read-bandwidth kernel (2.23 GB) and torch's fp16 matmul peak.
Every variant checked against float32 torch (rel error); per-call ms = batches of 20 back-to-back calls between MPS
synchronizes (median of 50 batches after 5 warm-up calls), as D3 timed per shape.
    PY e1_gpu_rows.py OUT.json [NSG KC]"""
import hashlib, json, statistics, sys, time
from pathlib import Path

import numpy as np
import torch

LUT = Path("/Volumes/SSD4TB/anemll-forge-research/jetspec_pp/adapter_lut4hg")
NAME = "streams.0.mlp.gate_proj"
outp = Path(sys.argv[1])
NSG, KC = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (4, 64)
ROWS = [1, 4, 8, 16, 32, 64]


def kern(nsg, kc, mt):
    R, TH, M = 8 * nsg, 32 * nsg, 8 * mt
    assert kc == 64 and nsg == 4, "v6 decode form: 4 threads x 8 bytes per row (KC 64), 4 simdgroups x 8 rows = TH / 4 rows"
    acc = ", ".join(f"c{j} = c0" for j in range(1, mt))
    mma = "\n".join(f"            simdgroup_load(a, xs + {8 * j} * {kc} + kk, {kc}); simdgroup_multiply_accumulate(c{j}, a, bt, c{j});" for j in range(mt))
    st = "\n".join(f"    simdgroup_store(c{j}, cs + sg * {64 * mt} + {64 * j}, 8);" for j in range(mt))
    return f"""
kernel void mm_{nsg}_{kc}_{mt}(device const uchar *idx [[buffer(0)]], device const half *pal [[buffer(1)]],
        device const half *scale [[buffer(2)]], device const half *x [[buffer(3)]], device half *y [[buffer(4)]],
        constant uint &O [[buffer(5)]], constant uint &K [[buffer(6)]], constant uint &MR [[buffer(7)]],
        uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {{
    threadgroup half ws[{max(R * kc, nsg * 32 * mt)}];
    threadgroup half xs[{M} * {kc}];
    threadgroup half ps[16];
    if (tid < 16) ps[tid] = pal[tid];
    const uint o0 = tg * {R}, kb = K / 2, wr = tid / 4, wq = tid % 4;
    simdgroup_float8x8 c0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f){', ' + acc if acc else ''};
    for (uint k0 = 0; k0 < K; k0 += {kc}) {{
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {{   // v6's decode: 4 threads per weight row, 8 index bytes (16 weights) each
            uint o = o0 + wr;
            uint2 bv = (o < O) ? *((device const uint2 *)(idx + (ulong)o * kb + k0 / 2 + wq * 8)) : uint2(0u);
            threadgroup half *dst = ws + wr * {kc} + wq * 16;
            _Pragma("unroll") for (uint j = 0; j < 2; j++) {{
                uint v = bv[j];
                *((threadgroup half4 *)(dst + j * 8)) = half4(ps[v & 0xFu], ps[(v >> 4) & 0xFu], ps[(v >> 8) & 0xFu], ps[(v >> 12) & 0xFu]);
                *((threadgroup half4 *)(dst + j * 8 + 4)) = half4(ps[(v >> 16) & 0xFu], ps[(v >> 20) & 0xFu], ps[(v >> 24) & 0xFu], ps[v >> 28]);
            }}
        }}
        for (uint i = tid; i < {M} * {kc} / 4; i += {TH}) {{ uint m = (i * 4) / {kc}, c = (i * 4) % {kc};   // x padded to {M} rows on the host
            *((threadgroup half4 *)(xs + m * {kc} + c)) = *((device const half4 *)(x + m * K + k0 + c)); }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        _Pragma("unroll") for (uint kk = 0; kk < {kc}; kk += 8) {{
            simdgroup_half8x8 a, bt;
            simdgroup_load(bt, ws + (sg * 8) * {kc} + kk, {kc}, ulong2(0, 0), true);
{mma}
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *cs = (threadgroup float *)ws;
{st}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < {nsg} * {64 * mt}; i += {TH}) {{
        uint g = i / {64 * mt}, e = i % {64 * mt}, m = e / 8, oc = e % 8, o = o0 + g * 8 + oc;
        if (o < O && m < MR) y[m * O + o] = half(cs[i] * float(scale[o]));
    }}
}}
"""


MTS = sorted({max(1, (r + 7) // 8) for r in ROWS})
SRC = "#include <metal_stdlib>\nusing namespace metal;\n" + "".join(kern(NSG, KC, mt) for mt in MTS)
lib = torch.mps.compile_shader(SRC)
sha = hashlib.sha256(SRC.encode()).hexdigest()[:16]
z = np.load(LUT / f"{NAME}.weight.npz")
idx = z["idx"].astype(np.uint8)
O, K = idx.shape
packed = (idx[:, 0::2] | (idx[:, 1::2] << 4)).astype(np.uint8)
dev, f = "mps", torch.float16
t_idx = torch.from_numpy(packed).to(dev)
t_pal = torch.from_numpy(z["lut"].astype(np.float16).reshape(-1)).to(dev)
t_sc = torch.from_numpy(z["scale"].astype(np.float16).reshape(-1)).to(dev)
w32 = (torch.from_numpy(z["lut"].astype(np.float32)).reshape(-1)[torch.from_numpy(idx.astype(np.int64))]
       * torch.from_numpy(z["scale"].astype(np.float32))[:, None])
w16 = w32.to(f).to(dev)
wbytes = packed.nbytes + 2 * O + 32
print(f"BANNER e1_gpu_rows | {NAME} {O} x {K} LUT4 ({wbytes / 1e6:.1f} MB) | kernel mm_{NSG}_{KC}_<MT> sha {sha} | rows {ROWS} | "
      f"torch {torch.__version__} mps", flush=True)


def timed(fn, n=50, batch=20):
    """per-call ms from batches of `batch` back-to-back calls between synchronizes (as in a real pass, D3's per-shape
    figures); median and min over n batches"""
    for _ in range(5):
        fn()
    torch.mps.synchronize()
    ms = []
    for _ in range(n):
        t0 = time.perf_counter()
        for _ in range(batch):
            fn()
        torch.mps.synchronize(); ms.append(1e3 * (time.perf_counter() - t0) / batch)
    return statistics.median(ms), min(ms)


res = {"banner": {"tensor": NAME, "O": O, "K": K, "weight_bytes": wbytes, "kernel_sha": sha, "nsg": NSG, "kc": KC}, "rows": {}}
for r in ROWS:
    mt = max(1, (r + 7) // 8)
    x = (torch.randn(r, K, dtype=torch.float32) * 0.1)
    xd = torch.zeros(8 * mt, K, dtype=f, device=dev); xd[:r] = x.to(f).to(dev)   # padded rows (zeros: the GPU has no zero skip)
    y = torch.empty(r, O, dtype=f, device=dev)
    groups = (O + 8 * NSG - 1) // (8 * NSG)
    k = getattr(lib, f"mm_{NSG}_{KC}_{mt}")
    run = lambda: k(t_idx, t_pal, t_sc, xd, y, O, K, r, threads=groups * 32 * NSG, group_size=32 * NSG)  # noqa: E731
    run(); torch.mps.synchronize()
    ref = x @ w32.T
    rel = float((y.float().cpu() - ref).norm() / ref.norm())
    med, mn = timed(run)
    xr_ = xd[:r].contiguous()
    dmed, _ = timed(lambda: xr_ @ w16.T)
    fl = 2 * r * O * K
    rec = {"lut4_ms": med, "lut4_ms_min": mn, "lut4_gbps": wbytes / 1e9 / (med / 1e3), "lut4_tflops": fl / 1e12 / (med / 1e3),
           "rel_err": rel, "mt": mt, "dense_fp16_ms": dmed, "dense_fp16_gbps": 2 * O * K / 1e9 / (dmed / 1e3),
           "dense_fp16_tflops": fl / 1e12 / (dmed / 1e3)}
    res["rows"][r] = rec
    print(f"rows {r:3d} (MT {mt}): LUT4 {med:6.3f} ms (min {mn:.3f}) {rec['lut4_gbps']:5.1f} GB/s {rec['lut4_tflops']:5.2f} TFLOP/s rel {rel:.1e} | "
          f"dense fp16 {dmed:6.3f} ms {rec['dense_fp16_gbps']:5.1f} GB/s {rec['dense_fp16_tflops']:5.2f} TFLOP/s", flush=True)
# peaks on this GPU, same harness
rd = torch.mps.compile_shader("""
kernel void rd(device const uint4 *a [[buffer(0)]], device uint *o [[buffer(1)]], constant uint &n [[buffer(2)]],
               uint i [[thread_position_in_grid]], uint G [[threads_per_grid]]) {
    uint4 s = 0u;
    for (uint j = i; j < n; j += G) s ^= a[j];
    if ((s.x ^ s.y ^ s.z ^ s.w) == 0xdeadbeefu) o[0] = 1u;
}""")
nb = 2_233_000_000 // 64 * 64
a = torch.randint(0, 255, (nb,), dtype=torch.uint8, device=dev); o = torch.zeros(1, dtype=torch.int32, device=dev)
best = 0
for G in (262144, 1048576):
    med, _ = timed(lambda: rd.rd(a, o, nb // 16, threads=G, group_size=256), 5, 1)
    best = max(best, nb / 1e9 / (med / 1e3))
del a
res["peak_read_gbps"] = best
pk = 0
for M in (256, 1024, 2048):
    x = torch.randn(M, 5120, dtype=f, device=dev)
    med, _ = timed(lambda: x @ w16.T, 10, 5)
    pk = max(pk, 2 * M * O * K / 1e12 / (med / 1e3))
res["peak_fp16_tflops"] = pk
print(f"GPU peaks, same harness: read {best:.1f} GB/s (2.23 GB kernel), fp16 matmul {pk:.2f} TFLOP/s (M 256-2048, 17408 x 5120)", flush=True)
outp.parent.mkdir(parents=True, exist_ok=True)
outp.write_text(json.dumps(res, indent=1))
