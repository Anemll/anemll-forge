"""D3 (20 Nov 2026): the JetSpec++ adapter at LUT4 (Hadamard + GPTQ tensors, quant_adapter_gptq.py) on the GPU, 16 rows.
Metal kernels through torch.mps.compile_shader:
  lut4_mm16  y (16, O) = x (16, K) @ (pal[idx] * scale_row)^T; idx packed 2 per byte (low nibble = even column), one
             16-entry palette per tensor (threadgroup memory), per-row scale; x tiled through threadgroup memory
             (TK columns), each simdgroup accumulates NR output rows x 16 inputs over its lanes' 8-column slices
  had1024    x' = blockdiag(H_1024) (sign * x) per row (the quantizer's rotation, applied to activations)
  rms16      RMSNorm of 16 rows;  swiglu16  silu(g) * u
One adapter pass = 12 blocks x (rms, had, q / k / v, had, o, rms, had, gate / up, swiglu, had, down), everything but the
attention core (q / k norm, RoPE, the 16 rows against the ctx KV), which C1 measured on the GPU (dec_attn_r4).
Modes:  check  one real matrix per shape vs a float32 torch reference (and the rotation round trip)
        time   the full 12-block pass, all 84 real tensors resident on the GPU; wall and GPU-event ms, N passes
    PY gpu_lut4_pass.py check|time OUT.json [N]"""
import hashlib, json, sys, time
from pathlib import Path

import numpy as np
import torch

LUT = Path("/Volumes/SSD4TB/anemll-forge-research/jetspec_pp/adapter_lut4hg")
BLOCKS = ["streams.0", "streams.1", "streams.2", "streams.3", "extra_stream_layers.3.0", "extra_stream_layers.3.1",
          "extra_stream_layers.3.2", "streams.4", "extra_stream_layers.4.0", "extra_stream_layers.4.1",
          "extra_stream_layers.4.2", "extra_stream_layers.4.3"]
PROJ = ["q_proj", "k_proj", "v_proj", "o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
NR, NSG, TK = 4, 8, 256
SRC = r"""
#include <metal_stdlib>
using namespace metal;
constant uint NR = %(NR)d, NSG = %(NSG)d, TK = %(TK)d;

// y[m][o] = scale[o] * sum_k pal[idx[o][k]] * x[m][k],  m < 16
kernel void lut4_mm16(device const uchar *idx [[buffer(0)]], device const half *pal [[buffer(1)]],
                      device const half *scale [[buffer(2)]], device const half *x [[buffer(3)]],
                      device half *y [[buffer(4)]], constant uint &O [[buffer(5)]], constant uint &K [[buffer(6)]],
                      uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint ln [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half xs[16 * TK];
    threadgroup half ps[16];
    if (tid < 16) ps[tid] = pal[tid];
    const uint r0 = (tg * NSG + sg) * NR;
    float acc[NR][16];
    for (uint r = 0; r < NR; r++) for (uint m = 0; m < 16; m++) acc[r][m] = 0.0f;
    const uint kb = K / 2;                                     // bytes per weight row
    for (uint k0 = 0; k0 < K; k0 += TK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 16 * TK / 4; i += NSG * 32) {   // x tile, half4 at a time
            uint m = (i * 4) / TK, c = (i * 4) %% TK;
            *((threadgroup half4 *)(xs + m * TK + c)) = *((device const half4 *)(x + m * K + k0 + c));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint c8 = ln * 8;                                // this lane's 8 columns of the tile
        half w[NR][8];
        for (uint r = 0; r < NR; r++) {
            uint o = r0 + r;
            uint b = (o < O) ? *((device const uint *)(idx + (ulong)o * kb + (k0 + c8) / 2)) : 0u;
            for (uint j = 0; j < 4; j++) {
                uint byte = (b >> (8 * j)) & 0xFFu;
                w[r][2 * j] = ps[byte & 0xFu];
                w[r][2 * j + 1] = ps[byte >> 4];
            }
        }
        for (uint m = 0; m < 16; m++) {
            half4 xa = *((threadgroup half4 *)(xs + m * TK + c8));
            half4 xb = *((threadgroup half4 *)(xs + m * TK + c8 + 4));
            for (uint r = 0; r < NR; r++) {
                acc[r][m] += float(dot(half4(w[r][0], w[r][1], w[r][2], w[r][3]), xa))
                           + float(dot(half4(w[r][4], w[r][5], w[r][6], w[r][7]), xb));
            }
        }
    }
    for (uint r = 0; r < NR; r++) {
        uint o = r0 + r;
        for (uint m = 0; m < 16; m++) {
            float s = simd_sum(acc[r][m]);
            if (ln == m && o < O) y[m * O + o] = half(s * float(scale[o]));
        }
    }
}

// v2: simdgroup 8x8 matrices. A threadgroup (4 simdgroups) owns 32 output rows; per K chunk of 256 it decodes the
// 32 x 256 weight tile (128 contiguous index bytes per row) and the 16 x 256 x tile into threadgroup memory; each
// simdgroup accumulates C (16 x 8) = x (16 x 256) . W^T (256 x 8) for its 8 rows; row scales applied at the end.
kernel void lut4_mm16s(device const uchar *idx [[buffer(0)]], device const half *pal [[buffer(1)]],
                       device const half *scale [[buffer(2)]], device const half *x [[buffer(3)]],
                       device half *y [[buffer(4)]], constant uint &O [[buffer(5)]], constant uint &K [[buffer(6)]],
                       uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half ws[32 * 256];
    threadgroup half xs[16 * 256];
    threadgroup half ps[16];
    threadgroup float cs[4][16 * 8];
    if (tid < 16) ps[tid] = pal[tid];
    const uint o0 = tg * 32, kb = K / 2;
    simdgroup_float8x8 c0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), c1 = c0;
    const uint wr = tid / 4, wq = tid %% 4;                     // weight tile: row wr, 64 columns at wq * 64
    for (uint k0 = 0; k0 < K; k0 += 256) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            uint o = o0 + wr;
            uint4 b0 = 0u, b1 = 0u;
            if (o < O) {
                device const uint4 *src = (device const uint4 *)(idx + (ulong)o * kb + k0 / 2 + wq * 32);
                b0 = src[0]; b1 = src[1];
            }
            threadgroup half *dst = ws + wr * 256 + wq * 64;
            uint wds[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
            for (uint j = 0; j < 8; j++) {
                uint v = wds[j];
                half4 lo = half4(ps[v & 0xFu], ps[(v >> 4) & 0xFu], ps[(v >> 8) & 0xFu], ps[(v >> 12) & 0xFu]);
                half4 hi = half4(ps[(v >> 16) & 0xFu], ps[(v >> 20) & 0xFu], ps[(v >> 24) & 0xFu], ps[v >> 28]);
                *((threadgroup half4 *)(dst + j * 8)) = lo;
                *((threadgroup half4 *)(dst + j * 8 + 4)) = hi;
            }
        }
        for (uint i = tid; i < 16 * 256 / 4; i += 128) {
            uint m = (i * 4) / 256, c = (i * 4) %% 256;
            *((threadgroup half4 *)(xs + m * 256 + c)) = *((device const half4 *)(x + m * K + k0 + c));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 256; kk += 8) {
            simdgroup_half8x8 a0, a1, bt;
            simdgroup_load(a0, xs + 0 * 256 + kk, 256);
            simdgroup_load(a1, xs + 8 * 256 + kk, 256);
            simdgroup_load(bt, ws + (sg * 8) * 256 + kk, 256, ulong2(0, 0), true);   // W rows -> W^T (k x o)
            simdgroup_multiply_accumulate(c0, a0, bt, c0);
            simdgroup_multiply_accumulate(c1, a1, bt, c1);
        }
    }
    simdgroup_store(c0, cs[sg], 8);
    simdgroup_store(c1, cs[sg] + 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 4 * 128; i += 128) {
        uint g = i / 128, e = i %% 128, m = e / 8, oc = e %% 8, o = o0 + g * 8 + oc;
        if (o < O) y[m * O + o] = half(cs[g][e] * float(scale[o]));
    }
}

// v3: v2 with the palette decoded by simd_shuffle (lane l holds entry l %% 16) instead of threadgroup lookups.
// v2: simdgroup 8x8 matrices. A threadgroup (4 simdgroups) owns 32 output rows; per K chunk of 256 it decodes the
// 32 x 256 weight tile (128 contiguous index bytes per row) and the 16 x 256 x tile into threadgroup memory; each
// simdgroup accumulates C (16 x 8) = x (16 x 256) . W^T (256 x 8) for its 8 rows; row scales applied at the end.
kernel void lut4_mm16q(device const uchar *idx [[buffer(0)]], device const half *pal [[buffer(1)]],
                       device const half *scale [[buffer(2)]], device const half *x [[buffer(3)]],
                       device half *y [[buffer(4)]], constant uint &O [[buffer(5)]], constant uint &K [[buffer(6)]],
                       uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint tid [[thread_index_in_threadgroup]], uint ln [[thread_index_in_simdgroup]]) {
    threadgroup half ws[32 * 256];
    threadgroup half xs[16 * 256];
    threadgroup float cs[4][16 * 8];
    const half pl = pal[ln & 15u];
    const uint o0 = tg * 32, kb = K / 2;
    simdgroup_float8x8 c0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), c1 = c0;
    const uint wr = tid / 4, wq = tid %% 4;                     // weight tile: row wr, 64 columns at wq * 64
    for (uint k0 = 0; k0 < K; k0 += 256) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            uint o = o0 + wr;
            uint4 b0 = 0u, b1 = 0u;
            if (o < O) {
                device const uint4 *src = (device const uint4 *)(idx + (ulong)o * kb + k0 / 2 + wq * 32);
                b0 = src[0]; b1 = src[1];
            }
            threadgroup half *dst = ws + wr * 256 + wq * 64;
            uint wds[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
            for (uint j = 0; j < 8; j++) {
                uint v = wds[j];
                half4 lo = half4(simd_shuffle(pl, ushort(v & 0xFu)), simd_shuffle(pl, ushort((v >> 4) & 0xFu)), simd_shuffle(pl, ushort((v >> 8) & 0xFu)), simd_shuffle(pl, ushort((v >> 12) & 0xFu)));
                half4 hi = half4(simd_shuffle(pl, ushort((v >> 16) & 0xFu)), simd_shuffle(pl, ushort((v >> 20) & 0xFu)), simd_shuffle(pl, ushort((v >> 24) & 0xFu)), simd_shuffle(pl, ushort(v >> 28)));
                *((threadgroup half4 *)(dst + j * 8)) = lo;
                *((threadgroup half4 *)(dst + j * 8 + 4)) = hi;
            }
        }
        for (uint i = tid; i < 16 * 256 / 4; i += 128) {
            uint m = (i * 4) / 256, c = (i * 4) %% 256;
            *((threadgroup half4 *)(xs + m * 256 + c)) = *((device const half4 *)(x + m * K + k0 + c));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 256; kk += 8) {
            simdgroup_half8x8 a0, a1, bt;
            simdgroup_load(a0, xs + 0 * 256 + kk, 256);
            simdgroup_load(a1, xs + 8 * 256 + kk, 256);
            simdgroup_load(bt, ws + (sg * 8) * 256 + kk, 256, ulong2(0, 0), true);   // W rows -> W^T (k x o)
            simdgroup_multiply_accumulate(c0, a0, bt, c0);
            simdgroup_multiply_accumulate(c1, a1, bt, c1);
        }
    }
    simdgroup_store(c0, cs[sg], 8);
    simdgroup_store(c1, cs[sg] + 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 4 * 128; i += 128) {
        uint g = i / 128, e = i %% 128, m = e / 8, oc = e %% 8, o = o0 + g * 8 + oc;
        if (o < O) y[m * O + o] = half(cs[g][e] * float(scale[o]));
    }
}

// v4: no weight tile in threadgroup memory. C^T (o x m) = W (o x k) . X^T (k x m) on 8x8 simdgroup matrices; lane l
// owns A-fragment elements (row fm, cols fn, fn + 1) (MLX's 8x8 fragment layout), so per 128-column block it loads 16
// index bytes of its row (4 lanes per row: 64 contiguous bytes) and fills 16 fragments from them: K is permuted within
// the block (fragment t, column c <-> k = (c / 2) * 32 + 2t + c %% 2), and x is staged into threadgroup memory with the
// same permutation. Each simdgroup keeps the block's 32 x fragments in registers and runs RT row tiles (8 rows each).
constant uint RT = %(RT)d;
kernel void lut4_mm16r(device const uchar *idx [[buffer(0)]], device const half *pal [[buffer(1)]],
                       device const half *scale [[buffer(2)]], device const half *x [[buffer(3)]],
                       device half *y [[buffer(4)]], constant uint &O [[buffer(5)]], constant uint &K [[buffer(6)]],
                       uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint tid [[thread_index_in_threadgroup]], uint ln [[thread_index_in_simdgroup]]) {
    threadgroup half xsp[16 * 8 * 16];
    threadgroup half ps[16];
    if (tid < 16) ps[tid] = pal[tid];
    const uint qid = ln / 4, fm = (qid & 4) + ((ln / 2) %% 4), fn = (qid & 2) * 2 + (ln %% 2) * 2, q = fn / 2;
    const uint obase = (tg * 4 + sg) * 8 * RT, kb = K / 2;
    simdgroup_float8x8 c[RT][2];
    _Pragma("unroll") for (uint r = 0; r < RT; r++) { c[r][0] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); c[r][1] = c[r][0]; }
    for (uint k0 = 0; k0 < K; k0 += 128) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 16 * 128; i += 128) {                 // x block (16 x 128) -> xsp[t][c][m]
            uint m = i / 128, kk = i %% 128, qq = kk / 32, rr = kk %% 32, t = rr / 2, cc = 2 * qq + (rr %% 2);
            xsp[(t * 8 + cc) * 16 + m] = x[m * K + k0 + kk];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_half8x8 b[16][2];
        _Pragma("unroll") for (uint t = 0; t < 16; t++) {
            simdgroup_load(b[t][0], xsp + t * 128, 16);
            simdgroup_load(b[t][1], xsp + t * 128 + 8, 16);
        }
        _Pragma("unroll") for (uint r = 0; r < RT; r++) {
            uint o = obase + r * 8 + fm;
            uint4 v = (o < O) ? *((device const uint4 *)(idx + (ulong)o * kb + k0 / 2 + q * 16)) : uint4(0u);
            _Pragma("unroll") for (uint j = 0; j < 4; j++) {
                uint w = v[j];
                _Pragma("unroll") for (uint bb = 0; bb < 4; bb++) {
                    uint t = j * 4 + bb, by = (w >> (8 * bb)) & 0xFFu;
                    simdgroup_half8x8 a;
                    a.thread_elements()[0] = ps[by & 0xFu];
                    a.thread_elements()[1] = ps[by >> 4];
                    simdgroup_multiply_accumulate(c[r][0], a, b[t][0], c[r][0]);
                    simdgroup_multiply_accumulate(c[r][1], a, b[t][1], c[r][1]);
                }
            }
        }
    }
    _Pragma("unroll") for (uint r = 0; r < RT; r++) {
        uint o = obase + r * 8 + fm;
        if (o < O) {
            float sc = float(scale[o]);
            for (uint h = 0; h < 2; h++) {
                y[(h * 8 + fn) * O + o] = half(c[r][h].thread_elements()[0] * sc);
                y[(h * 8 + fn + 1) * O + o] = half(c[r][h].thread_elements()[1] * sc);
            }
        }
    }
}

// v5: v2 at higher occupancy: 8 simdgroups x 8 rows = 64 output rows per threadgroup, K chunks of 64 (10 KB of
// threadgroup memory: ~3 threadgroups per core instead of 1); 4 threads decode a row's 32 index bytes per chunk.
kernel void lut4_mm16o(device const uchar *idx [[buffer(0)]], device const half *pal [[buffer(1)]],
                       device const half *scale [[buffer(2)]], device const half *x [[buffer(3)]],
                       device half *y [[buffer(4)]], constant uint &O [[buffer(5)]], constant uint &K [[buffer(6)]],
                       uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half ws[64 * 64];
    threadgroup half xs[16 * 64];
    threadgroup half ps[16];
    if (tid < 16) ps[tid] = pal[tid];
    const uint o0 = tg * 64, kb = K / 2, wr = tid / 4, wq = tid %% 4;
    simdgroup_float8x8 c0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), c1 = c0;
    for (uint k0 = 0; k0 < K; k0 += 64) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            uint o = o0 + wr;
            uint2 bv = (o < O) ? *((device const uint2 *)(idx + (ulong)o * kb + k0 / 2 + wq * 8)) : uint2(0u);
            threadgroup half *dst = ws + wr * 64 + wq * 16;
            _Pragma("unroll") for (uint j = 0; j < 2; j++) {
                uint v = bv[j];
                *((threadgroup half4 *)(dst + j * 8)) = half4(ps[v & 0xFu], ps[(v >> 4) & 0xFu], ps[(v >> 8) & 0xFu], ps[(v >> 12) & 0xFu]);
                *((threadgroup half4 *)(dst + j * 8 + 4)) = half4(ps[(v >> 16) & 0xFu], ps[(v >> 20) & 0xFu], ps[(v >> 24) & 0xFu], ps[v >> 28]);
            }
        }
        {   // x chunk: 16 x 64 halves, one half4 per thread
            uint m = tid / 16, c = (tid %% 16) * 4;
            *((threadgroup half4 *)(xs + m * 64 + c)) = *((device const half4 *)(x + m * K + k0 + c));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        _Pragma("unroll") for (uint kk = 0; kk < 64; kk += 8) {
            simdgroup_half8x8 a0, a1, bt;
            simdgroup_load(a0, xs + kk, 64);
            simdgroup_load(a1, xs + 8 * 64 + kk, 64);
            simdgroup_load(bt, ws + (sg * 8) * 64 + kk, 64, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(c0, a0, bt, c0);
            simdgroup_multiply_accumulate(c1, a1, bt, c1);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *cs = (threadgroup float *)ws;            // reuse: 8 simdgroups x 128 floats = 4 KB
    simdgroup_store(c0, cs + sg * 128, 8);
    simdgroup_store(c1, cs + sg * 128 + 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 8 * 128; i += 256) {
        uint g = i / 128, e = i %% 128, m = e / 8, oc = e %% 8, o = o0 + g * 8 + oc;
        if (o < O) y[m * O + o] = half(cs[i] * float(scale[o]));
    }
}

// x' = blockdiag(H_1024 / 32) (sign * x), per row; one threadgroup (256 threads) per (row, 1024-block)
kernel void had1024(device const half *x [[buffer(0)]], device const half *sign [[buffer(1)]], device half *y [[buffer(2)]],
                    constant uint &K [[buffer(3)]], uint tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
    threadgroup float v[1024];
    const uint nb = K / 1024, row = tg / nb, blk = tg %% nb;
    device const half *xr = x + row * K + blk * 1024;
    device const half *sr = sign + blk * 1024;
    for (uint i = t; i < 1024; i += 256) v[i] = float(xr[i]) * float(sr[i]);
    for (uint h = 1; h < 1024; h <<= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint p = t; p < 512; p += 256) {
            uint i = (p / h) * 2 * h + (p %% h), j = i + h;
            float a = v[i], b = v[j];
            v[i] = a + b; v[j] = a - b;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t; i < 1024; i += 256) y[row * K + blk * 1024 + i] = half(v[i] * 0.03125f);
}

// RMSNorm of one row per threadgroup (256 threads)
kernel void rms16(device const half *x [[buffer(0)]], device const half *g [[buffer(1)]], device half *y [[buffer(2)]],
                  constant uint &K [[buffer(3)]], uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]], uint ln [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    float s = 0.0f;
    for (uint i = t; i < K; i += 256) { float a = float(x[row * K + i]); s += a * a; }
    s = simd_sum(s);
    if (ln == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint i = 0; i < 8; i++) tot += part[i];
    float inv = rsqrt(tot / float(K) + 1e-6f);
    for (uint i = t; i < K; i += 256) y[row * K + i] = half(float(x[row * K + i]) * inv * float(g[i]));
}

kernel void swiglu16(device const half *g [[buffer(0)]], device const half *u [[buffer(1)]], device half *y [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    float a = float(g[i]);
    y[i] = half(a / (1.0f + exp(-a)) * float(u[i]));
}
""" % {"NR": NR, "NSG": NSG, "TK": TK, "RT": int(__import__("os").environ.get("RT", "4"))}

def _k6(nsg, kc):
    """v6 (generated): v5 with NSG6 simdgroups (8 rows each) per threadgroup and K chunks of KC6 columns"""
    R, TH = 8 * nsg, 32 * nsg
    bpt = kc // 8                     # index bytes per decoding thread (4 threads per row)
    xpt = 16 * kc // TH               # x halves per thread
    load = {8: "uint2 bv = (o < O) ? *((device const uint2 *)(idx + (ulong)o * kb + k0 / 2 + wq * 8)) : uint2(0u);",
            4: "uint2 bv = uint2((o < O) ? *((device const uint *)(idx + (ulong)o * kb + k0 / 2 + wq * 4)) : 0u, 0u);",
            16: "uint4 bv4 = (o < O) ? *((device const uint4 *)(idx + (ulong)o * kb + k0 / 2 + wq * 16)) : uint4(0u);"}[bpt]
    nw = bpt // 4
    dec = "\n".join(
        f"            {{ uint v = {'bv4' if bpt == 16 else 'bv'}[{j}]; "
        f"*((threadgroup half4 *)(dst + {j * 8})) = half4(ps[v & 0xFu], ps[(v >> 4) & 0xFu], ps[(v >> 8) & 0xFu], ps[(v >> 12) & 0xFu]); "
        f"*((threadgroup half4 *)(dst + {j * 8 + 4})) = half4(ps[(v >> 16) & 0xFu], ps[(v >> 20) & 0xFu], ps[(v >> 24) & 0xFu], ps[v >> 28]); }}"
        for j in range(nw))
    if xpt >= 4:
        xl = f"""for (uint i = tid; i < 16 * {kc} / 4; i += {TH}) {{ uint m = (i * 4) / {kc}, c = (i * 4) % {kc};
            *((threadgroup half4 *)(xs + m * {kc} + c)) = *((device const half4 *)(x + m * K + k0 + c)); }}"""
    else:
        xl = f"""for (uint i = tid; i < 16 * {kc}; i += {TH}) {{ uint m = i / {kc}, c = i % {kc}; xs[m * {kc} + c] = x[m * K + k0 + c]; }}"""
    return f"""
kernel void lut4_mm16_{nsg}_{kc}(device const uchar *idx [[buffer(0)]], device const half *pal [[buffer(1)]],
                       device const half *scale [[buffer(2)]], device const half *x [[buffer(3)]],
                       device half *y [[buffer(4)]], constant uint &O [[buffer(5)]], constant uint &K [[buffer(6)]],
                       uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint tid [[thread_index_in_threadgroup]]) {{
    threadgroup half ws[{max(R * kc, 2 * nsg * 128)}];
    threadgroup half xs[16 * {kc}];
    threadgroup half ps[16];
    if (tid < 16) ps[tid] = pal[tid];
    const uint o0 = tg * {R}, kb = K / 2, wr = tid / 4, wq = tid % 4;
    simdgroup_float8x8 c0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), c1 = c0;
    for (uint k0 = 0; k0 < K; k0 += {kc}) {{
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {{
            uint o = o0 + wr;
            {load}
            threadgroup half *dst = ws + wr * {kc} + wq * {2 * bpt};
{dec}
        }}
        {xl}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        _Pragma("unroll") for (uint kk = 0; kk < {kc}; kk += 8) {{
            simdgroup_half8x8 a0, a1, bt;
            simdgroup_load(a0, xs + kk, {kc});
            simdgroup_load(a1, xs + 8 * {kc} + kk, {kc});
            simdgroup_load(bt, ws + (sg * 8) * {kc} + kk, {kc}, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(c0, a0, bt, c0);
            simdgroup_multiply_accumulate(c1, a1, bt, c1);
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *cs = (threadgroup float *)ws;
    simdgroup_store(c0, cs + sg * 128, 8);
    simdgroup_store(c1, cs + sg * 128 + 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < {nsg} * 128; i += {TH}) {{
        uint g = i / 128, e = i % 128, m = e / 8, oc = e % 8, o = o0 + g * 8 + oc;
        if (o < O) y[m * O + o] = half(cs[i] * float(scale[o]));
    }}
}}
"""


V6 = [(4, 64), (8, 32), (8, 64), (8, 128), (16, 32), (16, 64)]
SRC += "".join(_k6(n_, k_) for n_, k_ in V6)


def _k7(nsg, kc):
    """v7 (generated): C^T (o x m) = W (o x k) . X^T (k x m); the weight tile is the untransposed A operand and x is staged
    transposed (xst[k][m]) once per chunk, so no transposed fragment loads"""
    src = _k6(nsg, kc).replace(f"kernel void lut4_mm16_{nsg}_{kc}(", f"kernel void lut4_mm16t_{nsg}_{kc}(")
    TH = 32 * nsg
    i0 = src.index("        for (uint i = tid; i < 16 *"); i1 = src.index("        threadgroup_barrier(mem_flags::mem_threadgroup);\n        _Pragma")
    src = src[:i0] + f"""        for (uint i = tid; i < 16 * {kc}; i += {TH}) {{ uint m = i / {kc}, c = i % {kc}; xs[c * 16 + m] = x[m * K + k0 + c]; }}
""" + src[i1:]
    src = src.replace(f"""            simdgroup_half8x8 a0, a1, bt;
            simdgroup_load(a0, xs + kk, {kc});
            simdgroup_load(a1, xs + 8 * {kc} + kk, {kc});
            simdgroup_load(bt, ws + (sg * 8) * {kc} + kk, {kc}, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(c0, a0, bt, c0);
            simdgroup_multiply_accumulate(c1, a1, bt, c1);""", f"""            simdgroup_half8x8 a, b0, b1;
            simdgroup_load(a, ws + (sg * 8) * {kc} + kk, {kc});
            simdgroup_load(b0, xs + kk * 16, 16);
            simdgroup_load(b1, xs + kk * 16 + 8, 16);
            simdgroup_multiply_accumulate(c0, a, b0, c0);
            simdgroup_multiply_accumulate(c1, a, b1, c1);""")
    src = src.replace("""        uint g = i / 128, e = i % 128, m = e / 8, oc = e % 8, o = o0 + g * 8 + oc;""",
                      """        uint g = i / 128, e = i % 128, h = e / 64, oc = (e % 64) / 8, m = h * 8 + e % 8, o = o0 + g * 8 + oc;""")
    assert "simdgroup_load(b0" in src and "h * 8 + e % 8" in src
    return src


V7 = [(4, 64), (8, 64), (4, 32), (8, 32)]
SRC += "".join(_k7(n_, k_) for n_, k_ in V7)

LIB = None


def lib():
    global LIB
    if LIB is None:
        LIB = torch.mps.compile_shader(SRC)
    return LIB


def load(name, dev="mps"):
    z = np.load(LUT / f"{name}.weight.npz")
    idx = z["idx"].astype(np.uint8)
    o, k = idx.shape
    packed = (idx[:, 0::2] | (idx[:, 1::2] << 4)).astype(np.uint8)
    return {"idx": torch.from_numpy(np.ascontiguousarray(packed)).to(dev), "pal": torch.from_numpy(z["lut"].astype(np.float16).reshape(-1)).to(dev),
            "scale": torch.from_numpy(z["scale"].astype(np.float16).reshape(-1)).to(dev),
            "sign": torch.from_numpy(z["sign"].astype(np.float16)).to(dev), "O": o, "K": k, "raw": z}


MMK = __import__("os").environ.get("MMK", "r")   # r: v4 (weights straight into fragments), q: v3, s: v2, v1: scalar


def mm(t, x, y):
    O, K = t["O"], t["K"]
    if MMK.startswith("v7_"):
        nsg, kc = (int(v) for v in MMK[3:].split("_"))
        groups = (O + 8 * nsg - 1) // (8 * nsg)
        getattr(lib(), f"lut4_mm16t_{nsg}_{kc}")(t["idx"], t["pal"], t["scale"], x, y, O, K, threads=groups * 32 * nsg, group_size=32 * nsg)
        return
    if MMK.startswith("v6_"):
        nsg, kc = (int(v) for v in MMK[3:].split("_"))
        groups = (O + 8 * nsg - 1) // (8 * nsg)
        getattr(lib(), f"lut4_mm16_{nsg}_{kc}")(t["idx"], t["pal"], t["scale"], x, y, O, K, threads=groups * 32 * nsg, group_size=32 * nsg)
        return
    if MMK == "o":
        groups = (O + 63) // 64
        lib().lut4_mm16o(t["idx"], t["pal"], t["scale"], x, y, O, K, threads=groups * 256, group_size=256)
        return
    if MMK == "r":
        rt = int(__import__("os").environ.get("RT", "4")); groups = (O + 4 * 8 * rt - 1) // (4 * 8 * rt)
        lib().lut4_mm16r(t["idx"], t["pal"], t["scale"], x, y, O, K, threads=groups * 128, group_size=128)
        return
    if MMK in ("s", "q"):
        groups = (O + 31) // 32
        getattr(lib(), "lut4_mm16s" if MMK == "s" else "lut4_mm16q")(t["idx"], t["pal"], t["scale"], x, y, O, K, threads=groups * 128, group_size=128)
        return
    groups = (O + NR * NSG - 1) // (NR * NSG)
    lib().lut4_mm16(t["idx"], t["pal"], t["scale"], x, y, O, K, threads=groups * NSG * 32, group_size=NSG * 32)


def had(x, sign, y):
    K = x.shape[1]
    lib().had1024(x, sign, y, K, threads=16 * (K // 1024) * 256, group_size=256)


def rms(x, g, y):
    lib().rms16(x, g, y, x.shape[1], threads=16 * 256, group_size=256)


def check(outp):
    res, torch_dev = [], "mps"
    torch.manual_seed(0)
    for p in PROJ:
        t = load(f"streams.0.{p}")
        z = t["raw"]
        w = torch.from_numpy(z["lut"].astype(np.float32).reshape(-1))[torch.from_numpy(z["idx"].astype(np.int64))] \
            * torch.from_numpy(z["scale"].astype(np.float32)).reshape(-1, 1)
        x = (torch.randn(16, t["K"]) * 0.5).half()
        ref = (x.float() @ w.T)
        y = torch.empty(16, t["O"], dtype=torch.float16, device=torch_dev)
        mm(t, x.to(torch_dev), y); torch.mps.synchronize()
        err = float((y.cpu().float() - ref).norm() / ref.norm())
        # rotation: had1024 vs the quantizer's rot_rows
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from quant_adapter_gptq import rot_rows
        xr = torch.empty(16, t["K"], dtype=torch.float16, device=torch_dev)
        had(x.to(torch_dev), t["sign"], xr); torch.mps.synchronize()
        rref = rot_rows(x.float(), torch.from_numpy(z["sign"].astype(np.float32)))
        rerr = float((xr.cpu().float() - rref).norm() / rref.norm())
        res.append({"proj": p, "O": t["O"], "K": t["K"], "rel_err_vs_fp32": err, "had_rel_err": rerr})
        print(f"check {p} ({t['O']} x {t['K']}): lut4_mm16 rel err {err:.2e}, had1024 rel err {rerr:.2e}", flush=True)
    outp.write_text(json.dumps({"mode": "check", "NR": NR, "NSG": NSG, "TK": TK, "rows": res}, indent=1))


def time_pass(outp, n):
    dev = "mps"
    T = {}
    t0 = time.time()
    for b in BLOCKS:
        for p in PROJ:
            t = load(f"{b}.{p}"); t.pop("raw"); T[(b, p)] = t
    nbytes = sum(t["idx"].numel() for t in T.values())
    print(f"loaded 84 LUT4 tensors, {nbytes / 1e9:.3f} GB of packed indices on the GPU, in {time.time() - t0:.0f}s", flush=True)
    H, I, A = 5120, 17408, 6144
    f = torch.float16
    g = torch.ones(H, dtype=f, device=dev)
    buf = {k: torch.randn(16, n_, dtype=f, device=dev) * 0.1 for k, n_ in
           (("x", H), ("h", H), ("hr", H), ("q", 12288), ("k", 1024), ("v", 1024), ("a", A), ("ar", A), ("o", H),
            ("gt", I), ("up", I), ("act", I), ("actr", I), ("dn", H))}

    def one_pass(parts):
        for b in BLOCKS:
            Tb = lambda p: T[(b, p)]  # noqa: E731
            if "norm" in parts:
                rms(buf["x"], g, buf["h"])
            if "had" in parts:
                had(buf["h"], Tb("q_proj")["sign"], buf["hr"])
            mm(Tb("q_proj"), buf["hr"], buf["q"]); mm(Tb("k_proj"), buf["hr"], buf["k"]); mm(Tb("v_proj"), buf["hr"], buf["v"])
            if "had" in parts:
                had(buf["a"], Tb("o_proj")["sign"], buf["ar"])
            mm(Tb("o_proj"), buf["ar"], buf["o"])
            if "norm" in parts:
                rms(buf["o"], g, buf["h"])
            if "had" in parts:
                had(buf["h"], Tb("mlp.gate_proj")["sign"], buf["hr"])
            mm(Tb("mlp.gate_proj"), buf["hr"], buf["gt"]); mm(Tb("mlp.up_proj"), buf["hr"], buf["up"])
            if "act" in parts:
                lib().swiglu16(buf["gt"], buf["up"], buf["act"], threads=16 * I, group_size=256)
            if "had" in parts:
                had(buf["act"], Tb("mlp.down_proj")["sign"], buf["actr"])
            mm(Tb("mlp.down_proj"), buf["actr"], buf["dn"])

    sha = hashlib.sha256(SRC.encode()).hexdigest()[:16]
    print(f"BANNER gpu_lut4_pass | kernel {MMK} | kernels sha {sha} NR {NR} NSG {NSG} TK {TK} | tensors {LUT} | 16 rows | torch {torch.__version__} mps", flush=True)
    out = {"mode": "time", "kernel": MMK, "kernel_sha": sha, "NR": NR, "NSG": NSG, "TK": TK, "packed_gb": nbytes / 1e9, "variants": {}}
    for label, parts in (("gemm_only", set()), ("gemm+had", {"had"}), ("full_no_attention", {"had", "norm", "act"})):
        for _ in range(3):
            one_pass(parts)
        torch.mps.synchronize()
        wall, gpu = [], []
        for _ in range(n):
            e0, e1 = torch.mps.event.Event(enable_timing=True), torch.mps.event.Event(enable_timing=True)
            p0 = time.perf_counter(); e0.record()
            one_pass(parts)
            e1.record(); torch.mps.synchronize()
            wall.append(1e3 * (time.perf_counter() - p0)); gpu.append(e0.elapsed_time(e1))
        wall.sort(); gpu.sort()
        rec = {"wall_ms_median": wall[n // 2], "wall_ms_min": wall[0], "event_ms_median": gpu[n // 2],
               "eff_gbps_weights": nbytes / 1e9 / (wall[n // 2] / 1e3)}
        out["variants"][label] = rec
        print(f"{label}: wall median {rec['wall_ms_median']:.2f} ms (min {rec['wall_ms_min']:.2f}), event {rec['event_ms_median']:.2f} ms, "
              f"weights {rec['eff_gbps_weights']:.0f} GB/s", flush=True)
    # per shape, one block's matrices 20x back to back (bandwidth per shape)
    shp = {}
    for p in PROJ:
        t = T[(BLOCKS[0], p)]
        xin = torch.randn(16, t["K"], dtype=f, device=dev) * 0.1; y = torch.empty(16, t["O"], dtype=f, device=dev)
        for _ in range(5):
            mm(t, xin, y)
        torch.mps.synchronize(); p0 = time.perf_counter()
        for _ in range(50):
            mm(t, xin, y)
        torch.mps.synchronize(); ms = 1e3 * (time.perf_counter() - p0) / 50
        shp[p] = {"O": t["O"], "K": t["K"], "ms": ms, "gbps": t["idx"].numel() / 1e9 / (ms / 1e3)}
        print(f"  {p} {t['O']} x {t['K']}: {ms:.3f} ms, {shp[p]['gbps']:.0f} GB/s", flush=True)
    out["per_shape"] = shp
    outp.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    mode, outp = sys.argv[1], Path(sys.argv[2])
    if mode == "check":
        check(outp)
    else:
        time_pass(outp, int(sys.argv[3]) if len(sys.argv) > 3 else 20)
