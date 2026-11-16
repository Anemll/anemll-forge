// GPU layer engine kernels (Qwen3.8-27B release target, M6). Activations channel-major [C][T], T = 64 tokens.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#define T 64
constant uint BITS [[function_constant(0)]];
constant uint GSUB [[function_constant(1)]];
constant uint CIN [[function_constant(2)]];
constant bool ACC [[function_constant(3)]];
constant uint CD [[function_constant(4)]];

inline float silu_f(float x) { return x / (1.0f + exp(-x)); }

// ---- RMSNorm over channels, per token: out32 = x * rsqrt(mean(x^2) + eps) * w, out16 = the same in half --------
kernel void rmsnorm(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]], device float *o32 [[buffer(2)]],
                    device half *o16 [[buffer(3)]], constant uint &C [[buffer(4)]], constant float &eps [[buffer(5)]],
                    uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                    uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[8];
    float s = 0;
    for (uint c = tid; c < C; c += 256) { float v = x[c * T + t]; s += v * v; }
    s = simd_sum(s);
    if (sl == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0;
    for (uint i = 0; i < 8; i++) tot += part[i];
    float r = rsqrt(tot / float(C) + eps);
    for (uint c = tid; c < C; c += 256) { float v = x[c * T + t] * r * w[c]; o32[c * T + t] = v; o16[c * T + t] = half(v); }
}

// ---- block Hadamard rotation: per 1024 channels y = H (s * x) / 32 (Sylvester order, as scipy.linalg.hadamard) -----
kernel void fwht(device const float *x [[buffer(0)]], device const float *signs [[buffer(1)]], device half *o [[buffer(2)]],
                 uint2 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float v[1024];
    uint b = g.x, t = g.y;
    for (uint i = tid; i < 1024; i += 512) v[i] = x[(b * 1024 + i) * T + t] * signs[b * 1024 + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint h = 1; h < 1024; h <<= 1) {
        uint i = (tid / h) * 2 * h + (tid % h), j = i + h;
        float a = v[i], c = v[j];
        v[i] = a + c; v[j] = a - c;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = tid; i < 1024; i += 512) o[(b * 1024 + i) * T + t] = half(v[i] * (1.0f / 32.0f));
}
kernel void to_half(device const float *x [[buffer(0)]], device half *o [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    o[i] = half(x[i]);
}

// ---- LUT GEMM from the ANE's compiled weights (HWX layout), y = fmul * scale * (W x) --------------------------------
// LR: the low-rank correction y += A t (A [cout][64] fp16, t = B x [64][T] fp16) as one more 64-deep step on the same tile
// PAIR (CD 1, 4-bit only): decode two indices (one byte) with one lookup in a 256-entry half2 pair table built per
// threadgroup from the palette: same values as the scalar path, half the index extractions and lookups.
// CF: decode items with the K column fastest across threads (default: the 8-row index group fastest, which puts
// consecutive threads GSUB / 8 rows x KS apart in the tile = bank conflicts for GSUB 16 / 32); same tile, bit-same.
template <uint TM, bool LR, uint KS = 64, bool DB = false, bool PAIR = false, bool CF = false>   // KS: K step; DB: double-buffered decoded tile (one barrier per step)
inline void lut_gemm_impl(device const uchar *k, device const uint *tab, device half *x, device float *y, float fmul, uint tg, uint tid,
                          threadgroup half *W0, threadgroup half *PAL, device const half *A = nullptr, device half *t16 = nullptr,
                          threadgroup half2 *PAL2 = nullptr) {
    threadgroup half *W = W0;
    const uint SPT = TM / (CD * GSUB), NP = 1u << BITS, MASK = NP - 1, PV = NP * CD;
    uint sub0 = tg * SPT;
    for (uint i = tid; i < SPT * PV; i += 128) {
        uint s = i / PV;
        PAL[s * 128 + i % PV] = ((device const half *)(k + tab[3 * (sub0 + s)]))[i % PV];
    }
    if (PAIR) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < SPT * 256; i += 128) { uint s = i / 256, p = i % 256; PAL2[i] = half2(PAL[s * 128 + (p & 15)], PAL[s * 128 + (p >> 4)]); }
    }
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tA(W, dextents<int32_t, 2>(KS, TM));
    constexpr auto d = matmul2d_descriptor(TM, T, KS, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<d, execution_simdgroups<4>> op;
    auto cT = op.template get_destination_cooperative_tensor<decltype(tA), tensor<device half, dextents<int32_t, 2>, tensor_inline>, float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
    const uint ITEMS = TM * KS / (8 * CD), PER = ITEMS / 128, IPS = KS * (GSUB / 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k0 = 0; k0 < CIN; k0 += KS) {
        if (DB) W = W0 + ((k0 / KS) & 1) * (TM * KS);
        for (uint it = 0; it < PER; it++) {
            uint item = tid + 128 * it, s = item / IPS, rem = item % IPS;
            uint c = CF ? rem % KS : rem / (GSUB / 8), h = CF ? rem / KS : rem % (GSUB / 8);
            device const uint *w = (device const uint *)(k + tab[3 * (sub0 + s) + 1] + 2 * GSUB * CD);
            uint b = ((k0 + c) * GSUB + 8 * h) * BITS, wi = b >> 5, sh = b & 31;
            ulong v = (ulong(w[wi]) | (ulong(w[wi + 1]) << 32)) >> sh;
            threadgroup half *pal = PAL + s * 128;
            if (PAIR) {
                threadgroup half2 *p2 = PAL2 + s * 256;
                for (uint r = 0; r < 8; r += 2) {
                    half2 w2 = p2[uint(v >> (r * 4)) & 0xFF];
                    W[(s * GSUB + 8 * h + r) * KS + c] = w2.x; W[(s * GSUB + 8 * h + r + 1) * KS + c] = w2.y;
                }
            } else
            for (uint r = 0; r < 8; r++) {
                uint ix = uint(v >> (r * BITS)) & MASK;
                if (CD == 2) { uint ch = s * 2 * GSUB + 2 * (8 * h + r); W[ch * KS + c] = pal[2 * ix]; W[(ch + 1) * KS + c] = pal[2 * ix + 1]; }
                else { W[(s * GSUB + 8 * h + r) * KS + c] = pal[ix]; }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tensor<device half, dextents<int32_t, 2>, tensor_inline> tB(x + k0 * T, dextents<int32_t, 2>(T, KS));
        tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tW(W, dextents<int32_t, 2>(KS, TM));
        op.run(tW, tB, cT);
        if (!DB) threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (DB) { threadgroup_barrier(mem_flags::mem_threadgroup); W = W0; }
    auto cA = op.template get_destination_cooperative_tensor<decltype(tA), tensor<device half, dextents<int32_t, 2>, tensor_inline>, float>();
    if (LR) {
        for (uint16_t i = 0; i < cA.get_capacity(); ++i) if (cA.is_valid_element(i)) cA[i] = 0;
        for (uint i = tid; i < TM * KS; i += 128) {
            uint chl = i / KS, c = i % KS, ss = chl / (CD * GSUB), q = chl % (CD * GSUB);
            W[chl * KS + c] = A[(CD * tab[3 * (sub0 + ss) + 2] + q) * 64 + c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tensor<device half, dextents<int32_t, 2>, tensor_inline> tT(t16, dextents<int32_t, 2>(T, KS));
        op.run(tA, tT, cA);
    }
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (!cT.is_valid_element(i)) continue;
        auto ix = cT.get_multidimensional_index(i);
        uint t = ix[0], chl = ix[1], ss = chl / (CD * GSUB), q = chl % (CD * GSUB);
        uint base = tab[3 * (sub0 + ss) + 1];
        float v = cT[i] * float(((device const half *)(k + base))[q]) * fmul;
        if (LR) v += cA[i];
        uint yi = (CD * tab[3 * (sub0 + ss) + 2] + q) * T + t;
        if (ACC) y[yi] += v; else y[yi] = v;   // ACC: straight into the residual stream
    }
}

kernel void lut_gemm(device const uchar *k [[buffer(0)]], device const uint *tab [[buffer(1)]], device half *x [[buffer(2)]],
                     device float *y [[buffer(3)]], constant float &fmul [[buffer(4)]],
                     uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W[64 * 64]; threadgroup half PAL[8 * 128];
    lut_gemm_impl<64, false>(k, tab, x, y, fmul, tg, tid, W, PAL);
}
kernel void lut_gemm32(device const uchar *k [[buffer(0)]], device const uint *tab [[buffer(1)]], device half *x [[buffer(2)]],
                       device float *y [[buffer(3)]], constant float &fmul [[buffer(4)]],
                       uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W[32 * 64]; threadgroup half PAL[8 * 128];
    lut_gemm_impl<32, false>(k, tab, x, y, fmul, tg, tid, W, PAL);
}
kernel void lut_gemm32c(device const uchar *k [[buffer(0)]], device const uint *tab [[buffer(1)]], device half *x [[buffer(2)]],
                        device float *y [[buffer(3)]], constant float &fmul [[buffer(4)]],
                        uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W[32 * 64]; threadgroup half PAL[8 * 128];
    lut_gemm_impl<32, false, 64, false, false, true>(k, tab, x, y, fmul, tg, tid, W, PAL);
}
// ---- W8A8 LUT GEMM (15 Nov): each subtable's palette quantized to int8 (scale max |entry| / 127, applied in the
// epilogue with the per-row scales: it factors out of the K sum); activations int8 with one scale per 1024-channel
// block and token (xs [CIN / 1024][T], quant8_blk); int8 x int8 -> int32 per K step, rescaled into an fp32 accumulator
// at the end of every 1024-channel block. TM 32 (T >= 128 calls).
kernel void lut_gemm32_i8(device const uchar *k [[buffer(0)]], device const uint *tab [[buffer(1)]], device int8_t *x8 [[buffer(2)]],
                          device float *y [[buffer(3)]], constant float &fmul [[buffer(4)]], device const float *xs [[buffer(5)]],
                          uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint TM = 32, KS = 64;
    threadgroup int8_t W[TM * KS], PAL8[8 * 128]; threadgroup float PSC[8];
    const uint SPT = TM / (CD * GSUB), NP = 1u << BITS, MASK = NP - 1, PV = NP * CD;
    uint sub0 = tg * SPT;
    if (tid < SPT) {
        device const half *pal = (device const half *)(k + tab[3 * (sub0 + tid)]);
        float m = 0;
        for (uint i = 0; i < PV; i++) m = max(m, abs(float(pal[i])));
        PSC[tid] = m > 0 ? m / 127.0f : 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < SPT * PV; i += 128) {
        uint s = i / PV;
        PAL8[s * 128 + i % PV] = int8_t(rint(float(((device const half *)(k + tab[3 * (sub0 + s)]))[i % PV]) / PSC[s]));
    }
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tA(W, dextents<int32_t, 2>(KS, TM));
    constexpr auto d = matmul2d_descriptor(TM, T, KS, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<d, execution_simdgroups<4>> op;
    auto cI = op.template get_destination_cooperative_tensor<decltype(tA), tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>, int32_t>();
    auto cF = op.template get_destination_cooperative_tensor<decltype(tA), tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>, float>();
    for (uint16_t i = 0; i < cF.get_capacity(); ++i) if (cF.is_valid_element(i)) { cF[i] = 0; cI[i] = 0; }
    const uint ITEMS = TM * KS / (8 * CD), PER = ITEMS / 128, IPS = KS * (GSUB / 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k0 = 0; k0 < CIN; k0 += KS) {
        for (uint it = 0; it < PER; it++) {
            uint item = tid + 128 * it, s = item / IPS, rem = item % IPS, c = rem / (GSUB / 8), h = rem % (GSUB / 8);
            device const uint *w = (device const uint *)(k + tab[3 * (sub0 + s) + 1] + 2 * GSUB * CD);
            uint b = ((k0 + c) * GSUB + 8 * h) * BITS, wi = b >> 5, sh = b & 31;
            ulong v = (ulong(w[wi]) | (ulong(w[wi + 1]) << 32)) >> sh;
            threadgroup int8_t *pal = PAL8 + s * 128;
            for (uint r = 0; r < 8; r++) {
                uint ix = uint(v >> (r * BITS)) & MASK;
                if (CD == 2) { uint ch = s * 2 * GSUB + 2 * (8 * h + r); W[ch * KS + c] = pal[2 * ix]; W[(ch + 1) * KS + c] = pal[2 * ix + 1]; }
                else { W[(s * GSUB + 8 * h + r) * KS + c] = pal[ix]; }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tB(x8 + k0 * T, dextents<int32_t, 2>(T, KS));
        op.run(tA, tB, cI);
        if ((k0 + KS) % 1024 == 0) {
            for (uint16_t i = 0; i < cI.get_capacity(); ++i) {
                if (!cI.is_valid_element(i)) continue;
                auto ix = cI.get_multidimensional_index(i);
                cF[i] += float(cI[i]) * xs[(k0 / 1024) * T + ix[0]]; cI[i] = 0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint16_t i = 0; i < cF.get_capacity(); ++i) {
        if (!cF.is_valid_element(i)) continue;
        auto ix = cF.get_multidimensional_index(i);
        uint t = ix[0], chl = ix[1], ss = chl / (CD * GSUB), q = chl % (CD * GSUB);
        uint base = tab[3 * (sub0 + ss) + 1];
        float v = cF[i] * PSC[ss] * float(((device const half *)(k + base))[q]) * fmul;
        uint yi = (CD * tab[3 * (sub0 + ss) + 2] + q) * T + t;
        if (ACC) y[yi] += v; else y[yi] = v;
    }
}
// lut_gemm32_i8 with the decoded tile channel-major (W[c][ch], read as a transposed left operand): each decode item
// (one K column, 8 indices) writes its 16 (CD 2) / 8 (CD 1) int8 weights with one vector store; K column fastest
// across threads. Same values and accumulation as lut_gemm32_i8.
// one K step of the channel-major int8 tile: decode KS columns x TM channels into W (lut_i8t_body)
template <uint TM, uint KS>
inline void i8t_decode(device const uchar *k, device const uint *tab, uint sub0, uint k0, uint tid, threadgroup int8_t *W,
                       threadgroup int8_t *PAL8, threadgroup ushort *PALP) {
    const uint NP = 1u << BITS, MASK = NP - 1, ITEMS = TM * KS / (8 * CD), PER = ITEMS / 128, IPS = KS * (GSUB / 8);
    for (uint it = 0; it < PER; it++) {
        uint item = tid + 128 * it, s = item / IPS, rem = item % IPS, c = rem % KS, h = rem / KS;
        device const uint *w = (device const uint *)(k + tab[3 * (sub0 + s) + 1] + 2 * GSUB * CD);
        uint b = ((k0 + c) * GSUB + 8 * h) * BITS, wi = b >> 5, sh = b & 31;
        ulong v = (ulong(w[wi]) | (ulong(w[wi + 1]) << 32)) >> sh;
        uint ix[8];
        for (uint r = 0; r < 8; r++) ix[r] = uint(v >> (r * BITS)) & MASK;
        if (CD == 2) {
            threadgroup ushort *pp = PALP + s * NP;
            uint4 o = uint4(uint(pp[ix[0]]) | (uint(pp[ix[1]]) << 16), uint(pp[ix[2]]) | (uint(pp[ix[3]]) << 16),
                            uint(pp[ix[4]]) | (uint(pp[ix[5]]) << 16), uint(pp[ix[6]]) | (uint(pp[ix[7]]) << 16));
            *(threadgroup uint4 *)(W + c * TM + s * 2 * GSUB + 16 * h) = o;
        } else {
            threadgroup int8_t *pal = PAL8 + s * 128;
            uint2 o = uint2(uint(uchar(pal[ix[0]])) | (uint(uchar(pal[ix[1]])) << 8) | (uint(uchar(pal[ix[2]])) << 16) | (uint(uchar(pal[ix[3]])) << 24),
                            uint(uchar(pal[ix[4]])) | (uint(uchar(pal[ix[5]])) << 8) | (uint(uchar(pal[ix[6]])) << 16) | (uint(uchar(pal[ix[7]])) << 24));
            *(threadgroup uint2 *)(W + c * TM + s * GSUB + 8 * h) = o;
        }
    }
}
// MODE (timing ablation only): 0 = full kernel, 1 = decode skipped (tile left as is), 2 = matmul skipped
// DB: double-buffered tile (W holds 2 x TM x KS): step s's matmul reads one half while step s + 1 decodes into the other,
// one barrier per K step. Same values and order (bit-same).
template <uint MODE, uint TM = 32, uint KS = 64, bool DB = false>
inline void lut_i8t_body(device const uchar *k, device const uint *tab, device int8_t *x8, device float *y, float fmul,
                         device const float *xs, uint tg, uint tid, threadgroup int8_t *W, threadgroup int8_t *PAL8,
                         threadgroup ushort *PALP, threadgroup float *PSC) {
    const uint SPT = TM / (CD * GSUB), NP = 1u << BITS, MASK = NP - 1, PV = NP * CD;
    uint sub0 = tg * SPT;
    if (tid < SPT) {
        device const half *pal = (device const half *)(k + tab[3 * (sub0 + tid)]);
        float m = 0;
        for (uint i = 0; i < PV; i++) m = max(m, abs(float(pal[i])));
        PSC[tid] = m > 0 ? m / 127.0f : 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < SPT * PV; i += 128) {
        uint s = i / PV;
        PAL8[s * 128 + i % PV] = int8_t(rint(float(((device const half *)(k + tab[3 * (sub0 + s)]))[i % PV]) / PSC[s]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (CD == 2) for (uint i = tid; i < SPT * NP; i += 128) {   // pair table: NP entries per subtable (<= 256 in all)
        uint s = i / NP, ix = i % NP;
        PALP[i] = ushort(uchar(PAL8[s * 128 + 2 * ix])) | ushort(ushort(uchar(PAL8[s * 128 + 2 * ix + 1])) << 8);
    }
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tA(W, dextents<int32_t, 2>(TM, KS));
    constexpr auto d = matmul2d_descriptor(TM, T, KS, true, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<d, execution_simdgroups<4>> op;
    auto cI = op.template get_destination_cooperative_tensor<decltype(tA), tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>, int32_t>();
    auto cF = op.template get_destination_cooperative_tensor<decltype(tA), tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>, float>();
    for (uint16_t i = 0; i < cF.get_capacity(); ++i) if (cF.is_valid_element(i)) { cF[i] = 0; cI[i] = 0; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (DB) {
        i8t_decode<TM, KS>(k, tab, sub0, 0, tid, W, PAL8, PALP);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k0 = 0, st = 0; k0 < CIN; k0 += KS, st++) {
            tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tAc(W + (st & 1) * (TM * KS), dextents<int32_t, 2>(TM, KS));
            tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tB(x8 + k0 * T, dextents<int32_t, 2>(T, KS));
            op.run(tAc, tB, cI);
            if (k0 + KS < CIN) i8t_decode<TM, KS>(k, tab, sub0, k0 + KS, tid, W + ((st + 1) & 1) * (TM * KS), PAL8, PALP);
            if ((k0 + KS) % 1024 == 0) {
                for (uint16_t i = 0; i < cI.get_capacity(); ++i) {
                    if (!cI.is_valid_element(i)) continue;
                    auto ix2 = cI.get_multidimensional_index(i);
                    cF[i] += float(cI[i]) * xs[(k0 / 1024) * T + ix2[0]]; cI[i] = 0;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else
    for (uint k0 = 0; k0 < CIN; k0 += KS) {
        if (MODE != 1) i8t_decode<TM, KS>(k, tab, sub0, k0, tid, W, PAL8, PALP);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tB(x8 + k0 * T, dextents<int32_t, 2>(T, KS));
        if (MODE != 2) op.run(tA, tB, cI);
        if ((k0 + KS) % 1024 == 0) {
            for (uint16_t i = 0; i < cI.get_capacity(); ++i) {
                if (!cI.is_valid_element(i)) continue;
                auto ix2 = cI.get_multidimensional_index(i);
                cF[i] += float(cI[i]) * xs[(k0 / 1024) * T + ix2[0]]; cI[i] = 0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint16_t i = 0; i < cF.get_capacity(); ++i) {
        if (!cF.is_valid_element(i)) continue;
        auto ix2 = cF.get_multidimensional_index(i);
        uint t = ix2[0], chl = ix2[1], ss = chl / (CD * GSUB), q = chl % (CD * GSUB);
        uint base = tab[3 * (sub0 + ss) + 1];
        float v = cF[i] * PSC[ss] * float(((device const half *)(k + base))[q]) * fmul;
        uint yi = (CD * tab[3 * (sub0 + ss) + 2] + q) * T + t;
        if (ACC) y[yi] += v; else y[yi] = v;
    }
}
#define LUT_I8T(NAME, MODE, TM_, KS_) \
kernel void NAME(device const uchar *k [[buffer(0)]], device const uint *tab [[buffer(1)]], device int8_t *x8 [[buffer(2)]], \
                 device float *y [[buffer(3)]], constant float &fmul [[buffer(4)]], device const float *xs [[buffer(5)]], \
                 uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup int8_t W[KS_ * TM_] __attribute__((aligned(16))), PAL8[8 * 128]; \
    threadgroup ushort PALP[256]; threadgroup float PSC[8]; \
    lut_i8t_body<MODE, TM_, KS_>(k, tab, x8, y, fmul, xs, tg, tid, W, PAL8, PALP, PSC); \
}
LUT_I8T(lut_gemm32_i8t, 0, 32, 64)
LUT_I8T(lut_gemm32_i8t_nodec, 1, 32, 64)
LUT_I8T(lut_gemm32_i8t_nomma, 2, 32, 64)
LUT_I8T(lut_gemm32_i8t_k128, 0, 32, 128)
LUT_I8T(lut_gemm32_i8t_k256, 0, 32, 256)
#define LUT_I8TD(NAME, TM_, KS_) \
kernel void NAME(device const uchar *k [[buffer(0)]], device const uint *tab [[buffer(1)]], device int8_t *x8 [[buffer(2)]], \
                 device float *y [[buffer(3)]], constant float &fmul [[buffer(4)]], device const float *xs [[buffer(5)]], \
                 uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup int8_t W[2 * KS_ * TM_] __attribute__((aligned(16))), PAL8[8 * 128]; \
    threadgroup ushort PALP[256]; threadgroup float PSC[8]; \
    lut_i8t_body<0, TM_, KS_, true>(k, tab, x8, y, fmul, xs, tg, tid, W, PAL8, PALP, PSC); \
}
LUT_I8TD(lut_gemm32_i8t_k256db, 32, 256)
LUT_I8TD(lut_gemm32_i8t_k128db, 32, 128)
LUT_I8T(lut_gemm32_i8t_k128_nodec, 1, 32, 128)
LUT_I8T(lut_gemm32_i8t_k512, 0, 32, 512)
LUT_I8T(lut_gemm32_i8t_m64, 0, 64, 64)
LUT_I8T(lut_gemm32_i8t_m64k128, 0, 64, 128)
LUT_I8T(lut_gemm32_i8t_m64_nodec, 1, 64, 64)
// int8 dense (attention k / v: the export's int8 weights w8 [M][K], per-row scales sc) x int8 activations (quant8_blk,
// scales xs per 1024-channel block and token): threadgroup = 32 rows x one 1024-channel block -> fp32 partial
// part[b][M][T] = sc * xs * (w8 x8); splitk_reduce sums the K / 1024 partials. Grid (M / 32, K / 1024), 128 threads.
kernel void dense_i8_splitk(device int8_t *w8 [[buffer(0)]], device int8_t *x8 [[buffer(1)]], device float *part [[buffer(2)]],
                            device const float *sc [[buffer(3)]], device const float *xs [[buffer(4)]],
                            constant uint &M [[buffer(5)]], constant uint &K [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]]) {
    uint o0 = tg.x * 32, b = tg.y, k0 = b * 1024;
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tA(w8 + o0 * K + k0, dextents<int32_t, 2>(1024, 32), array<int32_t, 2>{1, int(K)});
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tB(x8 + k0 * T, dextents<int32_t, 2>(T, 1024));
    constexpr auto d = matmul2d_descriptor(32, T, 1024, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    auto cI = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), int32_t>();
    op.run(tA, tB, cI);
    for (uint16_t i = 0; i < cI.get_capacity(); ++i) {
        if (!cI.is_valid_element(i)) continue;
        auto ix = cI.get_multidimensional_index(i);
        uint t = ix[0], r = ix[1];
        part[(b * M + o0 + r) * T + t] = float(cI[i]) * xs[b * T + t] * sc[o0 + r];
    }
}
// activations -> int8 per 1024-channel block and token: xs[b][t] = max |x| / 127. Grid (C / 1024, T / 32), 256 threads.
kernel void quant8_blk(device const half *x [[buffer(0)]], device int8_t *x8 [[buffer(1)]], device float *xs [[buffer(2)]],
                       uint2 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float m8[8][32];
    uint b = g.x, tt = tid % 32, t = g.y * 32 + tt, cg = tid / 32;
    float m = 0;
    for (uint j = 0; j < 128; j++) m = max(m, abs(float(x[(b * 1024 + cg + 8 * j) * T + t])));
    m8[cg][tt] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mm = 0;
    for (uint q = 0; q < 8; q++) mm = max(mm, m8[q][tt]);
    float sc = mm > 0 ? mm / 127.0f : 1.0f, inv = 1.0f / sc;
    if (cg == 0) xs[b * T + t] = sc;
    for (uint j = 0; j < 128; j++) { uint e = (b * 1024 + cg + 8 * j) * T + t; x8[e] = int8_t(rint(float(x[e]) * inv)); }
}
kernel void lut_gemm32p(device const uchar *k [[buffer(0)]], device const uint *tab [[buffer(1)]], device half *x [[buffer(2)]],
                        device float *y [[buffer(3)]], constant float &fmul [[buffer(4)]],
                        uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W[32 * 64]; threadgroup half PAL[8 * 128]; threadgroup half2 PAL2[2 * 256];   // PAIR: up to 2 subtables per tile
    lut_gemm_impl<32, false, 64, false, true>(k, tab, x, y, fmul, tg, tid, W, PAL, nullptr, nullptr, PAL2);
}
kernel void lut_gemm32_k128(device const uchar *k [[buffer(0)]], device const uint *tab [[buffer(1)]], device half *x [[buffer(2)]],
                            device float *y [[buffer(3)]], constant float &fmul [[buffer(4)]],
                            uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W[32 * 128]; threadgroup half PAL[8 * 128];
    lut_gemm_impl<32, false, 128>(k, tab, x, y, fmul, tg, tid, W, PAL);
}
kernel void lut_gemm32_db(device const uchar *k [[buffer(0)]], device const uint *tab [[buffer(1)]], device half *x [[buffer(2)]],
                          device float *y [[buffer(3)]], constant float &fmul [[buffer(4)]],
                          uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W[2 * 32 * 64]; threadgroup half PAL[8 * 128];
    lut_gemm_impl<32, false, 64, true>(k, tab, x, y, fmul, tg, tid, W, PAL);
}
kernel void lut_gemm32_k128db(device const uchar *k [[buffer(0)]], device const uint *tab [[buffer(1)]], device half *x [[buffer(2)]],
                              device float *y [[buffer(3)]], constant float &fmul [[buffer(4)]],
                              uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W[2 * 32 * 128]; threadgroup half PAL[8 * 128];
    lut_gemm_impl<32, false, 128, true>(k, tab, x, y, fmul, tg, tid, W, PAL);
}
kernel void lut_gemm64_db(device const uchar *k [[buffer(0)]], device const uint *tab [[buffer(1)]], device half *x [[buffer(2)]],
                          device float *y [[buffer(3)]], constant float &fmul [[buffer(4)]],
                          uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W[2 * 64 * 64]; threadgroup half PAL[8 * 128];
    lut_gemm_impl<64, false, 64, true>(k, tab, x, y, fmul, tg, tid, W, PAL);
}
kernel void lut_gemm_lr(device const uchar *k [[buffer(0)]], device const uint *tab [[buffer(1)]], device half *x [[buffer(2)]],
                        device float *y [[buffer(3)]], constant float &fmul [[buffer(4)]], device const half *A [[buffer(5)]],
                        device half *t16 [[buffer(6)]], uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W[64 * 64]; threadgroup half PAL[8 * 128];
    lut_gemm_impl<64, true>(k, tab, x, y, fmul, tg, tid, W, PAL, A, t16);
}
kernel void lut_gemm32_lr(device const uchar *k [[buffer(0)]], device const uint *tab [[buffer(1)]], device half *x [[buffer(2)]],
                          device float *y [[buffer(3)]], constant float &fmul [[buffer(4)]], device const half *A [[buffer(5)]],
                          device half *t16 [[buffer(6)]], uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W[32 * 64]; threadgroup half PAL[8 * 128];
    lut_gemm_impl<32, true>(k, tab, x, y, fmul, tg, tid, W, PAL, A, t16);
}

// ---- dense fp16 GEMM: y[M][64] (=|+=) W[M][K] x[K][64]; one threadgroup per 64 output rows ---------------------------
kernel void dense_gemm(device half *Wt [[buffer(0)]], device half *x [[buffer(1)]], device float *y [[buffer(2)]],
                       constant uint &M [[buffer(3)]], constant uint &K [[buffer(4)]], uint tg [[threadgroup_position_in_grid]]) {
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tA(Wt, dextents<int32_t, 2>(int(K), int(M)));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tB(x, dextents<int32_t, 2>(T, int(K)));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tC(y, dextents<int32_t, 2>(T, int(M)));
    constexpr auto dm = matmul2d_descriptor(64, T, static_cast<int>(dynamic_extent), false, false, false,
                                            matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto d0 = matmul2d_descriptor(64, T, static_cast<int>(dynamic_extent), false, false, false,
                                            matmul2d_descriptor::mode::multiply);
    auto mA = tA.slice(0, tg * 64);
    auto mC = tC.slice(0, tg * 64);
    if (ACC) { matmul2d<dm, execution_simdgroups<4>> op; op.run(mA, tB, mC); }
    else { matmul2d<d0, execution_simdgroups<4>> op; op.run(mA, tB, mC); }
}

kernel void add_into(device float *y [[buffer(0)]], device const float *a [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    y[i] += a[i];
}

// ---- SwiGLU: a = silu(g) * u ------------------------------------------------------------------------------------
kernel void swiglu(device const float *g [[buffer(0)]], device const float *u [[buffer(1)]], device float *a [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
    a[i] = silu_f(g[i]) * u[i];
}

// ---- DeltaNet prep: causal conv1d (4 taps, state of 3 rows) + SiLU on qkv, l2-normalized q (x dk^-1/2) and k, v,
// beta = sigmoid(b), decay = exp(softplus(a + dt) * neg_a); writes the new conv state ---------------------------------
kernel void gdn_conv(device const float *qkv [[buffer(0)]], device float *cs [[buffer(1)]], device const float *cw [[buffer(2)]],
                     device float *conv [[buffer(3)]], uint c [[thread_position_in_grid]]) {
    const uint CD = 10240;
    float r[T + 3];
    for (uint j = 0; j < 3; j++) r[j] = cs[j * CD + c];
    for (uint t = 0; t < T; t++) r[t + 3] = qkv[c * T + t];
    float w0 = cw[c], w1 = cw[CD + c], w2 = cw[2 * CD + c], w3 = cw[3 * CD + c];
    for (uint t = 0; t < T; t++) conv[c * T + t] = silu_f(r[t] * w0 + r[t + 1] * w1 + r[t + 2] * w2 + r[t + 3] * w3);
    for (uint j = 0; j < 3; j++) cs[j * CD + c] = r[T + j];
}
kernel void gdn_qkv(device const float *conv [[buffer(0)]], device float *Q [[buffer(1)]], device float *Kh [[buffer(2)]],
                    device float *V [[buffer(3)]], uint2 g [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]],
                    uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    // g.x: head slot 0..15 q, 16..31 k, 32..79 v; g.y: token; 128 threads = head dim
    threadgroup float part[4];
    uint h = g.x, t = g.y;
    float v = conv[(h * 128 + d) * T + t];
    if (h < 32) {
        float s = simd_sum(v * v);
        if (sl == 0) part[sg] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float r = rsqrt(part[0] + part[1] + part[2] + part[3] + 1e-6f);
        if (h < 16) Q[(h * T + t) * 128 + d] = v * r * rsqrt(128.0f);
        else Kh[((h - 16) * T + t) * 128 + d] = v * r;
    } else {
        V[((h - 32) * T + t) * 128 + d] = v;
    }
}
kernel void gdn_gates(device const float *ab [[buffer(0)]], device const float *dt [[buffer(1)]], device const float *nega [[buffer(2)]],
                      device float *beta [[buffer(3)]], device float *dec [[buffer(4)]], uint2 p [[thread_position_in_grid]]) {
    uint h = p.x, t = p.y;                                 // ab: rows 0..63 = b (48 used), 64..127 = a (48 used)
    float b = ab[h * T + t], a = ab[(64 + h) * T + t] + dt[h];
    float sp = a > 20.0f ? a : log(1.0f + exp(a));
    beta[h * T + t] = 1.0f / (1.0f + exp(-b));
    dec[h * T + t] = exp(sp * nega[h]);
}
// recurrence: S (dv rows x dk) per head in place; 4 lanes per value row
kernel void gdn_rec(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]], device const float *v [[buffer(2)]],
                    device const float *g [[buffer(3)]], device const float *beta [[buffer(4)]], device float *S [[buffer(5)]],
                    device float *o [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float kt[128], qt[128];
    uint h = tg.x, i = tg.y * 32 + tid / 4, l = tid % 4, kh = h / 3;
    float s[32];
    for (uint j = 0; j < 32; j++) s[j] = S[(h * 128 + i) * 128 + l * 32 + j];
    for (uint t = 0; t < T; t++) {
        kt[tid] = k[(kh * T + t) * 128 + tid]; qt[tid] = q[(kh * T + t) * 128 + tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float gt = g[h * T + t], dot = 0;
        for (uint j = 0; j < 32; j++) { s[j] *= gt; dot += s[j] * kt[l * 32 + j]; }
        dot = quad_sum(dot);
        float u = beta[h * T + t] * (v[(h * T + t) * 128 + i] - dot), out = 0;
        for (uint j = 0; j < 32; j++) { s[j] += u * kt[l * 32 + j]; out += s[j] * qt[l * 32 + j]; }
        out = quad_sum(out);
        if (l == 0) o[(h * T + t) * 128 + i] = out;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint j = 0; j < 32; j++) S[(h * 128 + i) * 128 + l * 32 + j] = s[j];
}
// recurrence, 8 lanes per value row (16 rows per threadgroup, grid (48, 8)); k / q of step t+1 load while step t runs
// (double-buffered: one barrier per step)
kernel void gdn_rec8(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]], device const float *v [[buffer(2)]],
                     device const float *g [[buffer(3)]], device const float *beta [[buffer(4)]], device float *S [[buffer(5)]],
                     device float *o [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float kt[2][128], qt[2][128];
    uint h = tg.x, i = tg.y * 16 + tid / 8, l = tid % 8, kh = h / 3;
    float s[16];
    for (uint j = 0; j < 16; j++) s[j] = S[(h * 128 + i) * 128 + l * 16 + j];
    kt[0][tid] = k[(kh * T) * 128 + tid]; qt[0][tid] = q[(kh * T) * 128 + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = 0; t < T; t++) {
        uint b = t & 1;
        if (t + 1 < T) { kt[b ^ 1][tid] = k[(kh * T + t + 1) * 128 + tid]; qt[b ^ 1][tid] = q[(kh * T + t + 1) * 128 + tid]; }
        float gt = g[h * T + t], bt = beta[h * T + t], vt = v[(h * T + t) * 128 + i], dot = 0;
        for (uint j = 0; j < 16; j++) { s[j] *= gt; dot += s[j] * kt[b][l * 16 + j]; }
        dot += simd_shuffle_xor(dot, 1); dot += simd_shuffle_xor(dot, 2); dot += simd_shuffle_xor(dot, 4);
        float u = bt * (vt - dot), out = 0;
        for (uint j = 0; j < 16; j++) { s[j] += u * kt[b][l * 16 + j]; out += s[j] * qt[b][l * 16 + j]; }
        out += simd_shuffle_xor(out, 1); out += simd_shuffle_xor(out, 2); out += simd_shuffle_xor(out, 4);
        if (l == 0) o[(h * T + t) * 128 + i] = out;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint j = 0; j < 16; j++) S[(h * 128 + i) * 128 + l * 16 + j] = s[j];
}
// recurrence, 4 lanes per value row as gdn_rec, but k / q / gate / beta of 16 steps staged in threadgroup memory and v
// prefetched into registers: two barriers per 16 steps instead of two per step
kernel void gdn_rec16(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]], device const float *v [[buffer(2)]],
                      device const float *g [[buffer(3)]], device const float *beta [[buffer(4)]], device float *S [[buffer(5)]],
                      device float *o [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint C = 16;
    threadgroup float kt[C][128], qt[C][128], gs[C], bs[C];
    uint h = tg.x, i = tg.y * 32 + tid / 4, l = tid % 4, kh = h / 3;
    float s[32];
    for (uint j = 0; j < 32; j++) s[j] = S[(h * 128 + i) * 128 + l * 32 + j];
    for (uint t0 = 0; t0 < T; t0 += C) {
        for (uint c = 0; c < C; c++) { kt[c][tid] = k[(kh * T + t0 + c) * 128 + tid]; qt[c][tid] = q[(kh * T + t0 + c) * 128 + tid]; }
        if (tid < C) { gs[tid] = g[h * T + t0 + tid]; bs[tid] = beta[h * T + t0 + tid]; }
        float vv[C];
        for (uint c = 0; c < C; c++) vv[c] = v[(h * T + t0 + c) * 128 + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = 0; c < C; c++) {
            float gt = gs[c], dot = 0;
            for (uint j = 0; j < 32; j++) { s[j] *= gt; dot += s[j] * kt[c][l * 32 + j]; }
            dot = quad_sum(dot);
            float u = bs[c] * (vv[c] - dot), out = 0;
            for (uint j = 0; j < 32; j++) { s[j] += u * kt[c][l * 32 + j]; out += s[j] * qt[c][l * 32 + j]; }
            out = quad_sum(out);
            if (l == 0) o[(h * T + t0 + c) * 128 + i] = out;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint j = 0; j < 32; j++) S[(h * 128 + i) * 128 + l * 32 + j] = s[j];
}
// recurrence, 4 lanes per value row as gdn_rec (grid (48, 4)), software-pipelined: k / q of step t+1 double-buffered in
// threadgroup memory and its gate / beta / v in registers, all loaded at the top of step t and stored after its math
// (one barrier per step); float4 reads of k / q. NP partial sums per dot: NP 1 = gdn_rec's order (bit-same), NP 4 =
// four interleaved chains of 8 (shorter latency chain, different fp32 rounding).
template <uint NP>
inline void gdn_rec_pipe(device const float *q, device const float *k, device const float *v, device const float *g,
                         device const float *beta, device float *S, device float *o, uint2 tg, uint tid,
                         threadgroup float4 (*kt)[32], threadgroup float4 (*qt)[32]) {
    uint h = tg.x, i = tg.y * 32 + tid / 4, l = tid % 4, kh = h / 3;
    float s[32];
    for (uint j = 0; j < 32; j++) s[j] = S[(h * 128 + i) * 128 + l * 32 + j];
    ((threadgroup float *)kt[0])[tid] = k[(kh * T) * 128 + tid]; ((threadgroup float *)qt[0])[tid] = q[(kh * T) * 128 + tid];
    float gt = g[h * T], bt = beta[h * T], vt = v[(h * T) * 128 + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = 0; t < T; t++) {
        uint b = t & 1, tn = min(t + 1, uint(T - 1));
        float kn = k[(kh * T + tn) * 128 + tid], qn = q[(kh * T + tn) * 128 + tid];
        float gn = g[h * T + tn], bn = beta[h * T + tn], vn = v[(h * T + tn) * 128 + i];
        threadgroup const float4 *kb = kt[b] + l * 8, *qb = qt[b] + l * 8;
        float d[NP];
        for (uint p = 0; p < NP; p++) d[p] = 0;
        for (uint j = 0; j < 8; j++) {
            float4 kk = kb[j];
            for (uint e = 0; e < 4; e++) { uint jj = 4 * j + e; s[jj] *= gt; d[jj % NP] += s[jj] * kk[e]; }
        }
        float dot = d[0];
        for (uint p = 1; p < NP; p++) dot += d[p];
        dot = quad_sum(dot);
        float u = bt * (vt - dot), r[NP];
        for (uint p = 0; p < NP; p++) r[p] = 0;
        for (uint j = 0; j < 8; j++) {
            float4 kk = kb[j], qq = qb[j];
            for (uint e = 0; e < 4; e++) { uint jj = 4 * j + e; s[jj] += u * kk[e]; r[jj % NP] += s[jj] * qq[e]; }
        }
        float out = r[0];
        for (uint p = 1; p < NP; p++) out += r[p];
        out = quad_sum(out);
        if (l == 0) o[(h * T + t) * 128 + i] = out;
        ((threadgroup float *)kt[b ^ 1])[tid] = kn; ((threadgroup float *)qt[b ^ 1])[tid] = qn;
        gt = gn; bt = bn; vt = vn;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint j = 0; j < 32; j++) S[(h * 128 + i) * 128 + l * 32 + j] = s[j];
}
kernel void gdn_rec_pf(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]], device const float *v [[buffer(2)]],
                       device const float *g [[buffer(3)]], device const float *beta [[buffer(4)]], device float *S [[buffer(5)]],
                       device float *o [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float4 kt[2][32], qt[2][32];
    gdn_rec_pipe<1>(q, k, v, g, beta, S, o, tg, tid, kt, qt);
}
kernel void gdn_rec_px(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]], device const float *v [[buffer(2)]],
                       device const float *g [[buffer(3)]], device const float *beta [[buffer(4)]], device float *S [[buffer(5)]],
                       device float *o [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float4 kt[2][32], qt[2][32];
    gdn_rec_pipe<4>(q, k, v, g, beta, S, o, tg, tid, kt, qt);
}
// gated RMSNorm: on[(h*128+d)][t] = rms(o[h][t][:]) * normw[d] * silu(z[(h*128+d)][t])
kernel void gdn_norm(device const float *o [[buffer(0)]], device const float *z [[buffer(1)]], device const float *nw [[buffer(2)]],
                     device float *on [[buffer(3)]], constant float &eps [[buffer(4)]], uint2 g [[threadgroup_position_in_grid]],
                     uint d [[thread_index_in_threadgroup]], uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[4];
    uint h = g.x, t = g.y;
    float v = o[(h * T + t) * 128 + d];
    float s = simd_sum(v * v);
    if (sl == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = rsqrt((part[0] + part[1] + part[2] + part[3]) / 128.0f + eps);
    on[(h * 128 + d) * T + t] = v * r * nw[d] * silu_f(z[(h * 128 + d) * T + t]);
}

// ---- full attention --------------------------------------------------------------------------------------------
// q heads (24 x [q 256 | gate 256] channels), k / v (4 x 256): per-head RMSNorm of q and k (gains 1 + w), partial RoPE on
// the first 64 dims; q -> Q[h][t][d]; k, v appended to the cache at positions p0 + t
kernel void attn_prep(device const float *qg [[buffer(0)]], device const float *kp [[buffer(1)]], device const float *vp [[buffer(2)]],
                      device const float *qn [[buffer(3)]], device const float *kn [[buffer(4)]], device float *Q [[buffer(5)]],
                      device float *Kc [[buffer(6)]], device float *Vc [[buffer(7)]], constant uint &p0 [[buffer(8)]],
                      constant uint &Lmax [[buffer(9)]], constant float &eps [[buffer(10)]], constant float *inv [[buffer(11)]],
                      uint2 g [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]],
                      uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[8], buf[256];
    uint h = g.x, t = g.y;                       // h 0..23 q heads, 24..27 k heads, 28..31 v heads
    float v;
    if (h < 24) v = qg[(h * 512 + d) * T + t];
    else if (h < 28) v = kp[((h - 24) * 256 + d) * T + t];
    else { Vc[((h - 28) * Lmax + p0 + t) * 256 + d] = vp[((h - 28) * 256 + d) * T + t]; return; }
    float s = simd_sum(v * v);
    if (sl == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0;
    for (uint i = 0; i < 8; i++) tot += part[i];
    v = v * rsqrt(tot / 256.0f + eps) * (h < 24 ? qn[d] : kn[d]);
    buf[d] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d < 64) {
        float ang = float(p0 + t) * inv[d % 32], c = cos(ang), sn = sin(ang);
        float rot = d < 32 ? -buf[d + 32] : buf[d - 32];
        v = buf[d] * c + rot * sn;
    }
    if (h < 24) Q[(h * T + t) * 256 + d] = v;
    else Kc[((h - 24) * Lmax + p0 + t) * 256 + d] = v;
}
// scores S[h][t][j] = Q[h][t] . K[h/6][j] / 16, j < p0 + t + 1 (else -inf)
kernel void attn_scores(device const float *Q [[buffer(0)]], device const float *Kc [[buffer(1)]], device float *S [[buffer(2)]],
                        constant uint &p0 [[buffer(3)]], constant uint &Lmax [[buffer(4)]], constant uint &L [[buffer(5)]],
                        uint3 p [[thread_position_in_grid]]) {
    uint j = p.x, t = p.y, h = p.z;
    if (j >= L) return;
    float s = -INFINITY;
    if (j <= p0 + t) {
        device const float4 *q = (device const float4 *)(Q + (h * T + t) * 256), *k = (device const float4 *)(Kc + ((h / 6) * Lmax + j) * 256);
        float a = 0;
        for (uint i = 0; i < 64; i++) a += dot(q[i], k[i]);
        s = a * 0.0625f;
    }
    S[(h * T + t) * Lmax + j] = s;
}
kernel void attn_softmax(device float *S [[buffer(0)]], constant uint &Lmax [[buffer(1)]], constant uint &L [[buffer(2)]],
                         uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[8];
    device float *row = S + r * Lmax;
    float m = -INFINITY;
    for (uint j = tid; j < L; j += 256) m = max(m, row[j]);
    m = simd_max(m);
    if (sl == 0) part[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = part[0];
    for (uint i = 1; i < 8; i++) m = max(m, part[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0;
    for (uint j = tid; j < L; j += 256) { float e = exp(row[j] - m); row[j] = e; s += e; }
    s = simd_sum(s);
    if (sl == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0;
    for (uint i = 0; i < 8; i++) tot += part[i];
    float inv = 1.0f / tot;
    for (uint j = tid; j < L; j += 256) row[j] *= inv;
}
// o[(h*256+d)][t] = sum_j P[h][t][j] V[h/6][j][d] * sigmoid(gate)
kernel void attn_pv(device const float *S [[buffer(0)]], device const float *Vc [[buffer(1)]], device const float *qg [[buffer(2)]],
                    device float *o [[buffer(3)]], constant uint &Lmax [[buffer(4)]], constant uint &L [[buffer(5)]],
                    uint3 p [[thread_position_in_grid]]) {
    uint d = p.x, t = p.y, h = p.z;
    device const float *pr = S + (h * T + t) * Lmax, *vv = Vc + (h / 6) * Lmax * 256 + d;
    float a = 0;
    for (uint j = 0; j < L; j++) a += pr[j] * vv[j * 256];
    float gt = qg[(h * 512 + 256 + d) * T + t];
    o[(h * 256 + d) * T + t] = a / (1.0f + exp(-gt));
}

kernel void from_half(device const half *x [[buffer(0)]], device float *o [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    o[i] = float(x[i]);
}

// split-K dense GEMM: part[s][M][64] = W[M][k-range s] x[k-range s][64]; grid (M / 64, splits)
kernel void dense_splitk(device half *Wt [[buffer(0)]], device half *x [[buffer(1)]], device float *part [[buffer(2)]],
                         constant uint &M [[buffer(3)]], constant uint &K [[buffer(4)]], constant uint &KS [[buffer(5)]],
                         uint2 tg [[threadgroup_position_in_grid]]) {
    uint s = tg.y, k0 = s * KS;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tA(Wt + k0, dextents<int32_t, 2>(int(KS), int(M)), array<int32_t, 2>{1, int(K)});
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tB(x + k0 * T, dextents<int32_t, 2>(T, int(KS)));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tC(part + s * M * T, dextents<int32_t, 2>(T, int(M)));
    constexpr auto d0 = matmul2d_descriptor(64, T, static_cast<int>(dynamic_extent), false, false, false,
                                            matmul2d_descriptor::mode::multiply);
    matmul2d<d0, execution_simdgroups<4>> op;
    auto mA = tA.slice(0, tg.x * 64);
    auto mC = tC.slice(0, tg.x * 64);
    op.run(mA, tB, mC);
}
kernel void splitk_reduce(device const float *part [[buffer(0)]], device float *y [[buffer(1)]], constant uint &n [[buffer(2)]],
                          constant uint &splits [[buffer(3)]], constant uint &acc [[buffer(4)]], device half *y16 [[buffer(5)]],
                          uint i [[thread_position_in_grid]]) {
    float s = acc == 1 ? y[i] : 0.0f;
    for (uint j = 0; j < splits; j++) s += part[j * n + i];
    y[i] = s;
    if (acc == 2) y16[i] = half(s);
}
// RMSNorm with coalesced reads: one threadgroup of 1024 threads, thread (g, t): channels g, g + 16, ...
kernel void rmsnorm2(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]], device float *o32 [[buffer(2)]],
                     device half *o16 [[buffer(3)]], constant uint &C [[buffer(4)]], constant float &eps [[buffer(5)]],
                     uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float part[16 * 64];
    uint t = tid % 64, g = tid / 64;
    float s = 0;
    for (uint c = g; c < C; c += 16) { float v = x[c * T + t]; s += v * v; }
    part[g * 64 + t] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0;
    for (uint j = 0; j < 16; j++) tot += part[j * 64 + t];
    float r = rsqrt(tot / float(C) + eps);
    for (uint c = g; c < C; c += 16) { float v = x[c * T + t] * r * w[c]; o32[c * T + t] = v; o16[c * T + t] = half(v); }
}

// ---- full attention on the matrix units (fp16 Q / K / V cache, f32 scores) ----------------------------------------
// prep as attn_prep, writing half Q [24][64][256] and half K / V cache [4][Lmax][256]
// QK8: attn_prep16 + int8 copies of the new q rows (Q8 / QS) and k rows (K8c / KSc, per kv head and key)
kernel void attn_prep16q8(device const float *qg [[buffer(0)]], device const float *kp [[buffer(1)]], device const float *vp [[buffer(2)]],
                          device const float *qn [[buffer(3)]], device const float *kn [[buffer(4)]], device half *Q [[buffer(5)]],
                          device half *Kc [[buffer(6)]], device half *Vc [[buffer(7)]], constant uint &p0 [[buffer(8)]],
                          constant uint &Lmax [[buffer(9)]], constant float &eps [[buffer(10)]], device const float *cs [[buffer(11)]],
                          device int8_t *Q8 [[buffer(12)]], device float *QS [[buffer(13)]], device int8_t *K8 [[buffer(14)]],
                          device float *KS [[buffer(15)]],
                          uint2 g [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]],
                          uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[8], buf[256], mx[8];
    uint h = g.x, t = g.y;
    float v;
    if (h < 24) v = qg[(h * 512 + d) * T + t];
    else if (h < 28) v = kp[((h - 24) * 256 + d) * T + t];
    else { Vc[((h - 28) * Lmax + p0 + t) * 256 + d] = half(vp[((h - 28) * 256 + d) * T + t]); return; }
    float s = simd_sum(v * v);
    if (sl == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0;
    for (uint i = 0; i < 8; i++) tot += part[i];
    v = v * rsqrt(tot / 256.0f + eps) * (h < 24 ? qn[d] : kn[d]);
    buf[d] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d < 64) {
        float c = cs[(t * 64 + d) * 2], sn = cs[(t * 64 + d) * 2 + 1];
        float rot = d < 32 ? -buf[d + 32] : buf[d - 32];
        v = buf[d] * c + rot * sn;
    }
    half hv = half(v);
    if (h < 24) Q[(h * T + t) * 256 + d] = hv;
    else Kc[((h - 24) * Lmax + p0 + t) * 256 + d] = hv;
    float m = simd_max(abs(float(hv)));
    if (sl == 0) mx[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mm = 0;
    for (uint i = 0; i < 8; i++) mm = max(mm, mx[i]);
    float sc = mm > 0 ? mm / 127.0f : 1.0f;
    int8_t q8 = int8_t(rint(float(hv) / sc));
    if (h < 24) { Q8[(h * T + t) * 256 + d] = q8; if (d == 0) QS[h * T + t] = sc; }
    else { K8[((h - 24) * Lmax + p0 + t) * 256 + d] = q8; if (d == 0) KS[(h - 24) * Lmax + p0 + t] = sc; }
}
// QK8: S tile = (Q8 K8[a ..]^T) x qs x ks (int32 accumulate), as attn_qk_h
kernel void attn_qk8i(device int8_t *Q8 [[buffer(0)]], device int8_t *K8 [[buffer(1)]], device half *S [[buffer(2)]],
                      constant uint &Lmax [[buffer(3)]], constant uint &a [[buffer(4)]], constant uint &TK [[buffer(5)]],
                      device const float *QS [[buffer(6)]], device const float *KS [[buffer(7)]],
                      uint3 tg [[threadgroup_position_in_grid]]) {
    uint kt = tg.x, rt = tg.y, gh = tg.z, r0 = gh * 6 * T + rt * 64, k0 = gh * Lmax + a + kt * 64;
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tQ(Q8 + r0 * 256, dextents<int32_t, 2>(256, 64));
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tK(K8 + k0 * 256, dextents<int32_t, 2>(256, 64));
    constexpr auto d = matmul2d_descriptor(64, 64, 256, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    auto cI = op.template get_destination_cooperative_tensor<decltype(tQ), decltype(tK), int32_t>();
    op.run(tQ, tK, cI);
    for (uint16_t i = 0; i < cI.get_capacity(); ++i) {
        if (!cI.is_valid_element(i)) continue;
        auto ix = cI.get_multidimensional_index(i);
        uint key = ix[0], row = ix[1];
        S[(r0 + row) * TK + kt * 64 + key] = half(float(cI[i]) * QS[r0 + row] * KS[k0 + key]);
    }
}
kernel void attn_prep16(device const float *qg [[buffer(0)]], device const float *kp [[buffer(1)]], device const float *vp [[buffer(2)]],
                        device const float *qn [[buffer(3)]], device const float *kn [[buffer(4)]], device half *Q [[buffer(5)]],
                        device half *Kc [[buffer(6)]], device half *Vc [[buffer(7)]], constant uint &p0 [[buffer(8)]],
                        constant uint &Lmax [[buffer(9)]], constant float &eps [[buffer(10)]], device const float *cs [[buffer(11)]],
                        uint2 g [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]],
                        uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[8], buf[256];
    uint h = g.x, t = g.y;
    float v;
    if (h < 24) v = qg[(h * 512 + d) * T + t];
    else if (h < 28) v = kp[((h - 24) * 256 + d) * T + t];
    else { Vc[((h - 28) * Lmax + p0 + t) * 256 + d] = half(vp[((h - 28) * 256 + d) * T + t]); return; }
    float s = simd_sum(v * v);
    if (sl == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0;
    for (uint i = 0; i < 8; i++) tot += part[i];
    v = v * rsqrt(tot / 256.0f + eps) * (h < 24 ? qn[d] : kn[d]);
    buf[d] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d < 64) {
        float c = cs[(t * 64 + d) * 2], sn = cs[(t * 64 + d) * 2 + 1];   // cos / sin of the position, computed in double on the host
        float rot = d < 32 ? -buf[d + 32] : buf[d - 32];
        v = buf[d] * c + rot * sn;
    }
    if (h < 24) Q[(h * T + t) * 256 + d] = half(v);
    else Kc[((h - 24) * Lmax + p0 + t) * 256 + d] = half(v);
}
// S[g rows][L] = Q_g (384 x 256) K_g^T: grid (L / 64 key tiles, 6 row tiles, 4 KV heads)
kernel void attn_qk(device half *Q [[buffer(0)]], device half *Kc [[buffer(1)]], device float *S [[buffer(2)]],
                    constant uint &Lmax [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]]) {
    uint kt = tg.x, rt = tg.y, gh = tg.z;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tQ(Q + (gh * 6 * T + rt * 64) * 256, dextents<int32_t, 2>(256, 64));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tK(Kc + (gh * Lmax + kt * 64) * 256, dextents<int32_t, 2>(256, 64));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tS(S + (gh * 6 * T + rt * 64) * Lmax + kt * 64, dextents<int32_t, 2>(64, 64),
                                                                 array<int32_t, 2>{1, int(Lmax)});
    constexpr auto d = matmul2d_descriptor(64, 64, 256, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    op.run(tQ, tK, tS);
}
// softmax of row (h, t) over keys j <= p0 + t (scale 1/16), probabilities in half
kernel void attn_softmax16(device const float *S [[buffer(0)]], device half *P [[buffer(1)]], constant uint &Lmax [[buffer(2)]],
                           constant uint &L [[buffer(3)]], constant uint &p0 [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]], uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[8];
    device const float *row = S + r * Lmax;
    device half *pr = P + r * Lmax;
    uint lim = p0 + (r % T) + 1;
    float m = -INFINITY;
    for (uint j = tid; j < lim; j += 256) m = max(m, row[j]);
    m = simd_max(m);
    if (sl == 0) part[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = part[0];
    for (uint i = 1; i < 8; i++) m = max(m, part[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0;
    for (uint j = tid; j < lim; j += 256) s += exp((row[j] - m) * 0.0625f);
    s = simd_sum(s);
    if (sl == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0;
    for (uint i = 0; i < 8; i++) tot += part[i];
    float inv = 1.0f / tot;
    for (uint j = tid; j < L; j += 256) pr[j] = half(j < lim ? exp((row[j] - m) * 0.0625f) * inv : 0.0f);
}
// O_g (384 x 256) = P_g (384 x L) V_g (L x 256): grid (4 dim tiles, 6 row tiles, 4 KV heads)
kernel void attn_pv16(device half *P [[buffer(0)]], device half *Vc [[buffer(1)]], device float *O [[buffer(2)]],
                      constant uint &Lmax [[buffer(3)]], constant uint &L [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]]) {
    uint dt = tg.x, rt = tg.y, gh = tg.z;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tP(P + (gh * 6 * T + rt * 64) * Lmax, dextents<int32_t, 2>(int(L), 64),
                                                                array<int32_t, 2>{1, int(Lmax)});
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tV(Vc + gh * Lmax * 256 + dt * 64, dextents<int32_t, 2>(64, int(L)),
                                                                array<int32_t, 2>{1, 256});
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tO(O + (gh * 6 * T + rt * 64) * 256 + dt * 64, dextents<int32_t, 2>(64, 64),
                                                                 array<int32_t, 2>{1, 256});
    constexpr auto d = matmul2d_descriptor(64, 64, static_cast<int>(dynamic_extent), false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    op.run(tP, tV, tO);
}
// o[(h*256+d)][t] = O[h][t][d] * sigmoid(gate)
kernel void attn_gate(device const float *O [[buffer(0)]], device const float *qg [[buffer(1)]], device float *o [[buffer(2)]],
                      uint3 p [[thread_position_in_grid]]) {
    uint d = p.x, t = p.y, h = p.z;
    float gt = qg[(h * 512 + 256 + d) * T + t];
    o[(h * 256 + d) * T + t] = O[(h * T + t) * 256 + d] / (1.0f + exp(-gt));
}

// RMSNorm in two passes: partial sums of squares per (320-channel chunk, token), coalesced over tokens; then apply
kernel void rms_partial(device const float *x [[buffer(0)]], device float *part [[buffer(1)]], uint ch [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float p[256];
    const uint NG = 256 / T; uint t = tid % T, g = tid / T;
    float s = 0;
    for (uint c = ch * 320 + g; c < ch * 320 + 320; c += NG) { float v = x[c * T + t]; s += v * v; }
    p[g * T + t] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (g == 0) { float a = 0; for (uint q = 0; q < NG; q++) a += p[q * T + t]; part[ch * T + t] = a; }
}
kernel void rms_apply(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]], device float *o32 [[buffer(2)]],
                      device half *o16 [[buffer(3)]], device const float *part [[buffer(4)]], constant uint &C [[buffer(5)]],
                      constant float &eps [[buffer(6)]], uint i [[thread_position_in_grid]]) {
    uint t = i % T, c = i / T;
    float s = 0;
    for (uint j = 0; j < C / 320; j++) s += part[j * T + t];
    float v = x[i] * rsqrt(s / float(C) + eps) * w[c];
    o32[i] = v; o16[i] = half(v);
}
// block Hadamard, 4 tokens per threadgroup (16-byte reads)
kernel void fwht4(device const float *x [[buffer(0)]], device const float *signs [[buffer(1)]], device half *o [[buffer(2)]],
                  uint2 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float v[4 * 1024];
    uint b = g.x, t0 = g.y * 4;
    for (uint idx = tid; idx < 4096; idx += 512) {
        uint i = idx / 4, tt = idx % 4;
        v[tt * 1024 + i] = x[(b * 1024 + i) * T + t0 + tt] * signs[b * 1024 + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint h = 1; h < 1024; h <<= 1) {
        uint i = (tid / h) * 2 * h + (tid % h), j = i + h;
        for (uint tt = 0; tt < 4; tt++) {
            float a = v[tt * 1024 + i], c = v[tt * 1024 + j];
            v[tt * 1024 + i] = a + c; v[tt * 1024 + j] = a - c;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint idx = tid; idx < 4096; idx += 512) {
        uint i = idx / 4, tt = idx % 4;
        o[(b * 1024 + i) * T + t0 + tt] = half(v[tt * 1024 + i] * (1.0f / 32.0f));
    }
}

// ---- FUSE (15 Nov): glue kernels merged, same fp32 expressions and reduction orders as the kernels they replace ----
// rms_apply + fwht4: v = x * rsqrt(sum / C + eps) * w (rms_apply's expression, part sums in its order); o16 = half(v)
// (rms_apply's unrotated half output); o = block Hadamard of v * signs (fwht4). Grid (C / 1024, T / 4), 512 threads.
kernel void rms_fwht4(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]], device const float *signs [[buffer(2)]],
                      device half *o [[buffer(3)]], device half *o16 [[buffer(4)]], device const float *part [[buffer(5)]],
                      constant uint &C [[buffer(6)]], constant float &eps [[buffer(7)]],
                      uint2 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float v[4 * 1024], rr[4];
    uint b = g.x, t0 = g.y * 4;
    if (tid < 4) { float s = 0; for (uint j = 0; j < C / 320; j++) s += part[j * T + t0 + tid]; rr[tid] = rsqrt(s / float(C) + eps); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < 4096; idx += 512) {
        uint i = idx / 4, tt = idx % 4, e = (b * 1024 + i) * T + t0 + tt;
        float hv = x[e] * rr[tt] * w[b * 1024 + i];
        o16[e] = half(hv);
        v[tt * 1024 + i] = hv * signs[b * 1024 + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint h = 1; h < 1024; h <<= 1) {
        uint i = (tid / h) * 2 * h + (tid % h), j = i + h;
        for (uint tt = 0; tt < 4; tt++) {
            float a = v[tt * 1024 + i], c = v[tt * 1024 + j];
            v[tt * 1024 + i] = a + c; v[tt * 1024 + j] = a - c;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint idx = tid; idx < 4096; idx += 512) {
        uint i = idx / 4, tt = idx % 4;
        o[(b * 1024 + i) * T + t0 + tt] = half(v[tt * 1024 + i] * (1.0f / 32.0f));
    }
}
// gdn_norm + fwht4 (mix_out): one 1024-channel block (8 heads) x 4 tokens per threadgroup (512 threads); per
// (head, token) sums of squares as gdn_norm (simd_sum over 32 consecutive dims, the 4 partials added in order), gate
// as gdn_norm, then fwht4's rotation. ON (fp32) is not written. Grid (6144 / 1024, T / 4).
kernel void gdn_norm_fwht4(device const float *O [[buffer(0)]], device const float *z [[buffer(1)]], device const float *nw [[buffer(2)]],
                           device const float *signs [[buffer(3)]], device half *o [[buffer(4)]], constant float &eps [[buffer(5)]],
                           uint2 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                           uint sl [[thread_index_in_simdgroup]]) {
    threadgroup float v[4 * 1024], part[32 * 4];
    uint b = g.x, t0 = g.y * 4;
    float ov[8];
    for (uint it = 0; it < 8; it++) {   // element it * 512 + tid: pair p = (head hl = p / 4, token tt = p % 4), dim d
        uint idx = it * 512 + tid, p = idx / 128, d = idx % 128, h = b * 8 + p / 4, tt = p % 4;
        float val = O[(h * T + t0 + tt) * 128 + d];
        ov[it] = val;
        float s = simd_sum(val * val);
        if (sl == 0) part[p * 4 + d / 32] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint it = 0; it < 8; it++) {
        uint idx = it * 512 + tid, p = idx / 128, d = idx % 128, h = b * 8 + p / 4, tt = p % 4, c = h * 128 + d;
        float r = rsqrt((part[p * 4] + part[p * 4 + 1] + part[p * 4 + 2] + part[p * 4 + 3]) / 128.0f + eps);
        float on = ov[it] * r * nw[d] * silu_f(z[c * T + t0 + tt]);
        v[tt * 1024 + (c - b * 1024)] = on * signs[c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint h = 1; h < 1024; h <<= 1) {
        uint i = (tid / h) * 2 * h + (tid % h), j = i + h;
        for (uint tt = 0; tt < 4; tt++) {
            float a = v[tt * 1024 + i], c = v[tt * 1024 + j];
            v[tt * 1024 + i] = a + c; v[tt * 1024 + j] = a - c;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint idx = tid; idx < 4096; idx += 512) {
        uint i = idx / 4, tt = idx % 4;
        o[(b * 1024 + i) * T + t0 + tt] = half(v[tt * 1024 + i] * (1.0f / 32.0f));
    }
}
// gdn_conv + gdn_qkv: one head slot (128 channels: 0..15 q, 16..31 k, 32..79 v) x 16 tokens per threadgroup; conv taps
// and SiLU as gdn_conv, l2 norms as gdn_qkv (same simdgroup partial order). Only the token-block-0 threadgroups touch
// the conv state: each thread reads its channel's old state, then writes the new one (the last 3 input rows), so
// no other threadgroup races on cs. Grid (80, T / 16), 128 threads.
kernel void gdn_conv_qkv(device const float *qkv [[buffer(0)]], device float *cs [[buffer(1)]], device const float *cw [[buffer(2)]],
                         device float *Q [[buffer(3)]], device float *Kh [[buffer(4)]], device float *V [[buffer(5)]],
                         uint2 g [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]],
                         uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint CDIM = 10240, TB = 16;
    threadgroup float part[TB][4];
    uint h = g.x, t0 = g.y * TB, c = h * 128 + d;
    float r[TB + 3], cv[TB];
    for (uint j = 0; j < 3; j++) r[j] = t0 == 0 ? cs[j * CDIM + c] : qkv[c * T + t0 - 3 + j];
    for (uint t = 0; t < TB; t++) r[t + 3] = qkv[c * T + t0 + t];
    float w0 = cw[c], w1 = cw[CDIM + c], w2 = cw[2 * CDIM + c], w3 = cw[3 * CDIM + c];
    for (uint t = 0; t < TB; t++) cv[t] = silu_f(r[t] * w0 + r[t + 1] * w1 + r[t + 2] * w2 + r[t + 3] * w3);
    if (t0 == 0) for (uint j = 0; j < 3; j++) cs[j * CDIM + c] = qkv[c * T + T - 3 + j];
    if (h < 32) {
        for (uint t = 0; t < TB; t++) { float s = simd_sum(cv[t] * cv[t]); if (sl == 0) part[t][sg] = s; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint t = 0; t < TB; t++) {
            float rn = rsqrt(part[t][0] + part[t][1] + part[t][2] + part[t][3] + 1e-6f);
            if (h < 16) Q[(h * T + t0 + t) * 128 + d] = cv[t] * rn * rsqrt(128.0f);
            else Kh[((h - 16) * T + t0 + t) * 128 + d] = cv[t] * rn;
        }
    } else {
        for (uint t = 0; t < TB; t++) V[((h - 32) * T + t0 + t) * 128 + d] = cv[t];
    }
}
// residual add folded into the output conversion: dst = half(x + y) (add_into then to_half / to_half_pair)
kernel void to_half_add(device const float *x [[buffer(0)]], device half *o [[buffer(1)]], device const float *y [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) {
    o[i] = half(x[i] + y[i]);
}
kernel void to_half_pair_add(device const float *x [[buffer(0)]], device half *a [[buffer(1)]], device half *b [[buffer(2)]],
                             device const float *y [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    const uint h = T / 2, c = i / T, t = i % T;
    half v = half(x[i] + y[i]);
    if (t < h) a[c * h + t] = v; else b[c * h + t - h] = v;
}

// MLP: a = silu(g) * u fused into the block Hadamard (mlp_mid), 4 tokens per threadgroup
kernel void fwht4_swiglu(device const float *g [[buffer(0)]], device const float *signs [[buffer(1)]], device half *o [[buffer(2)]],
                         device const float *u [[buffer(3)]], uint2 gr [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float v[4 * 1024];
    uint b = gr.x, t0 = gr.y * 4;
    for (uint idx = tid; idx < 4096; idx += 512) {
        uint i = idx / 4, tt = idx % 4, e = (b * 1024 + i) * T + t0 + tt;
        v[tt * 1024 + i] = silu_f(g[e]) * u[e] * signs[b * 1024 + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint h = 1; h < 1024; h <<= 1) {
        uint i = (tid / h) * 2 * h + (tid % h), j = i + h;
        for (uint tt = 0; tt < 4; tt++) {
            float a = v[tt * 1024 + i], c = v[tt * 1024 + j];
            v[tt * 1024 + i] = a + c; v[tt * 1024 + j] = a - c;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint idx = tid; idx < 4096; idx += 512) {
        uint i = idx / 4, tt = idx % 4;
        o[(b * 1024 + i) * T + t0 + tt] = half(v[tt * 1024 + i] * (1.0f / 32.0f));
    }
}
// residual add fused into the RMSNorm partial sums: x += y, then sums of squares of the new x
kernel void rms_partial_add(device float *x [[buffer(0)]], device float *part [[buffer(1)]], device const float *y [[buffer(2)]],
                            uint ch [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float p[256];
    const uint NG = 256 / T; uint t = tid % T, g = tid / T;
    float s = 0;
    for (uint c = ch * 320 + g; c < ch * 320 + 320; c += NG) { uint e = c * T + t; float v = x[e] + y[e]; x[e] = v; s += v * v; }
    p[g * T + t] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (g == 0) { float a = 0; for (uint q = 0; q < NG; q++) a += p[q * T + t]; part[ch * T + t] = a; }
}

// ---- tiled attention (key tiles of TK, online softmax across tiles) ----------------------------------------------
// S tile [24][T][TK] = Q K[a : a + TK]^T
kernel void attn_qk_t(device half *Q [[buffer(0)]], device half *Kc [[buffer(1)]], device float *S [[buffer(2)]],
                      constant uint &Lmax [[buffer(3)]], constant uint &a [[buffer(4)]], constant uint &TK [[buffer(5)]],
                      uint3 tg [[threadgroup_position_in_grid]]) {
    uint kt = tg.x, rt = tg.y, gh = tg.z;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tQ(Q + (gh * 6 * T + rt * 64) * 256, dextents<int32_t, 2>(256, 64));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tK(Kc + (gh * Lmax + a + kt * 64) * 256, dextents<int32_t, 2>(256, 64));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tS(S + (gh * 6 * T + rt * 64) * TK + kt * 64, dextents<int32_t, 2>(64, 64),
                                                                 array<int32_t, 2>{1, int(TK)});
    constexpr auto d = matmul2d_descriptor(64, 64, 256, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    op.run(tQ, tK, tS);
}
// per row: tile max mt (scaled), P = exp(s * scale - mt) over valid keys (j <= p0 + t), row sum dt
kernel void attn_softmax_t(device const float *S [[buffer(0)]], device half *P [[buffer(1)]], device float *st [[buffer(2)]],
                           constant uint &TK [[buffer(3)]], constant uint &a [[buffer(4)]], constant uint &n [[buffer(5)]],
                           constant uint &p0 [[buffer(6)]], uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                           uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[8];
    device const float *row = S + r * TK;
    device half *pr = P + r * TK;
    int lim = int(p0 + (r % T) + 1) - int(a);                // valid keys of this row in the tile
    uint nv = uint(clamp(lim, 0, int(n)));
    float m = -INFINITY;
    for (uint j = tid; j < nv; j += 256) m = max(m, row[j] * 0.0625f);
    m = simd_max(m);
    if (sl == 0) part[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = part[0];
    for (uint i = 1; i < 8; i++) m = max(m, part[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0;
    for (uint j = tid; j < n; j += 256) { float e = j < nv ? exp(row[j] * 0.0625f - m) : 0.0f; pr[j] = half(e); s += e; }
    s = simd_sum(s);
    if (sl == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float tot = 0; for (uint i = 0; i < 8; i++) tot += part[i]; st[2 * r] = m; st[2 * r + 1] = tot; }
}
// O tile [24 * T][256] = P (rows x n) V[a : a + n]
kernel void attn_pv_t(device half *P [[buffer(0)]], device half *Vc [[buffer(1)]], device float *O [[buffer(2)]],
                      constant uint &Lmax [[buffer(3)]], constant uint &a [[buffer(4)]], constant uint &TK [[buffer(5)]],
                      constant uint &n [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]]) {
    uint dt = tg.x, rt = tg.y, gh = tg.z;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tP(P + (gh * 6 * T + rt * 64) * TK, dextents<int32_t, 2>(int(n), 64),
                                                                array<int32_t, 2>{1, int(TK)});
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tV(Vc + (gh * Lmax + a) * 256 + dt * 64, dextents<int32_t, 2>(64, int(n)),
                                                                array<int32_t, 2>{1, 256});
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tO(O + (gh * 6 * T + rt * 64) * 256 + dt * 64, dextents<int32_t, 2>(64, 64),
                                                                 array<int32_t, 2>{1, 256});
    constexpr auto d = matmul2d_descriptor(64, 64, static_cast<int>(dynamic_extent), false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    op.run(tP, tV, tO);
}
// running (M, D, O) <- combine with the tile's (mt, dt, Ot); first tile initializes
kernel void attn_combine(device const float *Ot [[buffer(0)]], device const float *st [[buffer(1)]], device float *O [[buffer(2)]],
                         device float *MD [[buffer(3)]], constant uint &first [[buffer(4)]], uint2 p [[thread_position_in_grid]]) {
    uint d = p.x, r = p.y;
    float mt = st[2 * r], dt = st[2 * r + 1];
    float M = first ? -INFINITY : MD[2 * r], D = first ? 0.0f : MD[2 * r + 1];
    float mn = max(M, mt);
    float wa = (M == -INFINITY) ? 0.0f : exp(M - mn), wb = (mt == -INFINITY) ? 0.0f : exp(mt - mn);
    float o = first ? 0.0f : O[r * 256 + d];
    O[r * 256 + d] = o * wa + Ot[r * 256 + d] * wb;
    threadgroup_barrier(mem_flags::mem_device);
    if (d == 0) { MD[2 * r] = mn; MD[2 * r + 1] = D * wa + dt * wb; }
}
kernel void attn_finish(device const float *O [[buffer(0)]], device const float *MD [[buffer(1)]], device const float *qg [[buffer(2)]],
                        device float *o [[buffer(3)]], uint3 p [[thread_position_in_grid]]) {
    uint d = p.x, t = p.y, h = p.z, r = h * T + t;
    float gt = qg[(h * 512 + 256 + d) * T + t];
    o[(h * 256 + d) * T + t] = O[r * 256 + d] / MD[2 * r + 1] / (1.0f + exp(-gt));
}

// ---- fused flash attention: scores / probabilities stay on chip ---------------------------------------------------
// threadgroup (8 simdgroups) = 64 query rows of one KV group (one head, 64 consecutive tokens) x one key split;
// online softmax over 64-key tiles; partial (O unnormalized, m, l) per split, combined by attn_combine_splits.
// grid (6 T / 64 row tiles, 4 KV groups, NS splits)
kernel void attn_flash(device half *Q [[buffer(0)]], device half *Kc [[buffer(1)]], device half *Vc [[buffer(2)]],
                       device float *Op [[buffer(3)]], device float *stp [[buffer(4)]], constant uint &Lmax [[buffer(5)]],
                       constant uint &p0 [[buffer(6)]], constant uint &NS [[buffer(7)]],
                       uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float Sb[64 * 64];
    threadgroup half Pb[64 * 64];
    threadgroup float mrow[64], lrow[64], arow[64];
    uint rt = tg.x, gh = tg.y, sp = tg.z;
    uint row0 = gh * 6 * T + rt * 64;                 // global query row of this tile (q head = gh * 6 + (rt * 64) / T)
    uint t0 = (rt * 64) % T;                           // token of the tile's first row
    uint nk = p0 + t0 + 64;                            // keys any row of the tile can see
    uint ntile = (nk + 63) / 64, per = (ntile + NS - 1) / NS, k0 = sp * per, k1 = min(ntile, k0 + per);
    if (tid < 64) { mrow[tid] = -INFINITY; lrow[tid] = 0; }
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tQ(Q + row0 * 256, dextents<int32_t, 2>(256, 64));
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tS(Sb, dextents<int32_t, 2>(64, 64));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(Pb, dextents<int32_t, 2>(64, 64));
    constexpr auto dqk = matmul2d_descriptor(64, 64, 256, false, true, false, matmul2d_descriptor::mode::multiply);
    constexpr auto dpv = matmul2d_descriptor(64, 256, 64, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dqk, execution_simdgroups<8>> qk;
    matmul2d<dpv, execution_simdgroups<8>> pv;
    auto cO = pv.template get_destination_cooperative_tensor<decltype(tP), tensor<device half, dextents<int32_t, 2>, tensor_inline>, float>();
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) if (cO.is_valid_element(i)) cO[i] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint r = tid / 4, l4 = tid % 4;                   // softmax: 4 threads per row, 16 keys each
    for (uint kt = k0; kt < k1; kt++) {
        uint a = kt * 64;
        tensor<device half, dextents<int32_t, 2>, tensor_inline> tK(Kc + (gh * Lmax + a) * 256, dextents<int32_t, 2>(256, 64));
        qk.run(tQ, tK, tS);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int lim = int(p0 + t0 + r + 1) - int(a);       // visible keys of row r in this tile
        float mx = -INFINITY;
        for (uint j = l4 * 16; j < l4 * 16 + 16; j++) if (int(j) < lim) mx = max(mx, Sb[r * 64 + j] * 0.0625f);
        mx = quad_max(mx);
        float mold = mrow[r], mnew = max(mold, mx);
        float s = 0;
        for (uint j = l4 * 16; j < l4 * 16 + 16; j++) {
            float e = (int(j) < lim && mnew != -INFINITY) ? exp(Sb[r * 64 + j] * 0.0625f - mnew) : 0.0f;
            Pb[r * 64 + j] = half(e); s += e;
        }
        s = quad_sum(s);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (l4 == 0) {
            float al = (mold == -INFINITY) ? 0.0f : exp(mold - mnew);
            arow[r] = al; lrow[r] = lrow[r] * al + s; mrow[r] = mnew;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
            if (!cO.is_valid_element(i)) continue;
            auto ix = cO.get_multidimensional_index(i);
            cO[i] *= arow[ix[1]];
        }
        tensor<device half, dextents<int32_t, 2>, tensor_inline> tV(Vc + (gh * Lmax + a) * 256, dextents<int32_t, 2>(256, 64));
        pv.run(tP, tV, cO);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
        if (!cO.is_valid_element(i)) continue;
        auto ix = cO.get_multidimensional_index(i);
        Op[((sp * 24 * T) + row0 + ix[1]) * 256 + ix[0]] = cO[i];
    }
    if (tid < 64) { stp[2 * (sp * 24 * T + row0 + tid)] = mrow[tid]; stp[2 * (sp * 24 * T + row0 + tid) + 1] = lrow[tid]; }
}
// combine the key splits and normalize: OR[r][d], MD[r] = (m, 1)
kernel void attn_combine_splits(device const float *Op [[buffer(0)]], device const float *stp [[buffer(1)]], device float *O [[buffer(2)]],
                                device float *MD [[buffer(3)]], constant uint &NS [[buffer(4)]], uint2 p [[thread_position_in_grid]]) {
    uint d = p.x, r = p.y;
    float m = -INFINITY;
    for (uint s = 0; s < NS; s++) m = max(m, stp[2 * (s * 24 * T + r)]);
    float num = 0, den = 0;
    for (uint s = 0; s < NS; s++) {
        float ms = stp[2 * (s * 24 * T + r)];
        if (ms == -INFINITY) continue;
        float w = exp(ms - m);
        num += Op[(s * 24 * T + r) * 256 + d] * w; den += stp[2 * (s * 24 * T + r) + 1] * w;
    }
    O[r * 256 + d] = num / den;
    if (d == 0) { MD[2 * r] = m; MD[2 * r + 1] = 1.0f; }
}

// ---- FA2: flash attention with the score tile in registers (MPP cooperative tensors, one simdgroup per RM query rows):
// S = Q K^T (half, unscaled as attn_qk_h), online softmax in float on the cooperative elements, P (half) fed to P V as
// a cooperative left input, O accumulated in float registers. Threadgroup = NSG simdgroups x RM rows of one KV group
// (consecutive tokens of one head); grid (6 T / (NSG RM), 4 KV groups, NS key splits); per split the unnormalized O and
// (m, l) like attn_flash, combined by attn_combine_splits.
template <int RM, int KN, int NSG, int MODE = 0, bool STAGE = false>
inline void fa2_impl(device half *Q, device half *Kc, device half *Vc, device float *Op, device float *stp, uint Lmax, uint p0, uint NS,
                     uint3 tg, uint sg, uint sl, threadgroup float *st, threadgroup half *KT = nullptr, threadgroup half *VT = nullptr) {
    threadgroup float *mrow = st + sg * 4 * RM, *lrow = mrow + RM, *arow = lrow + RM, *red = arow + RM;
    uint gh = tg.y, sp = tg.z;
    uint lr0 = tg.x * NSG * RM + sg * RM;              // row within the KV group (6 heads x T tokens)
    uint row0 = gh * 6 * T + lr0, t0 = lr0 % T;
    uint nk = STAGE ? p0 + (tg.x * NSG * RM) % T + NSG * RM : p0 + t0 + RM;   // keys the last row sees (staged: of the threadgroup)
    uint ntile = (nk + KN - 1) / KN, per = (ntile + NS - 1) / NS, k0 = sp * per, k1 = min(ntile, k0 + per);
    if (sl < RM) { mrow[sl] = -INFINITY; lrow[sl] = 0; }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tQ(Q + row0 * 256, dextents<int32_t, 2>(256, RM));
    constexpr auto dqk = matmul2d_descriptor(RM, KN, 256, false, true, false, matmul2d_descriptor::mode::multiply);
    constexpr auto dpv = matmul2d_descriptor(RM, 256, KN, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dqk, execution_simdgroup> qk;
    matmul2d<dpv, execution_simdgroup> pv;
    using TK_t = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    auto cO = pv.template get_destination_cooperative_tensor<decltype(pv.template get_left_input_cooperative_tensor<half, half, float>()), TK_t, float>();
    #pragma unroll
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) if (cO.is_valid_element(i)) cO[i] = 0;
    for (uint kt = k0; kt < k1; kt++) {
        uint a = kt * KN;
        TK_t tK(Kc + (gh * Lmax + a) * 256, dextents<int32_t, 2>(256, KN));
        auto cS = qk.template get_destination_cooperative_tensor<decltype(tQ), TK_t, half>();
        if (STAGE) {
            uint tid = sg * 32 + sl;
            device const half4 *ks = (device const half4 *)(Kc + (gh * Lmax + a) * 256), *vs = (device const half4 *)(Vc + (gh * Lmax + a) * 256);
            threadgroup half4 *kd = (threadgroup half4 *)KT, *vd = (threadgroup half4 *)VT;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint j = tid; j < KN * 64; j += NSG * 32) { kd[j] = ks[j]; vd[j] = vs[j]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tKs(KT, dextents<int32_t, 2>(256, KN));
            qk.run(tQ, tKs, cS);
        } else qk.run(tQ, tK, cS);
        if (MODE != 2) {
        bool edge = a + KN > p0 + t0 + 1;               // some row of the tile cannot see some key
        if (edge) {
            #pragma unroll
            for (uint16_t i = 0; i < cS.get_capacity(); ++i) {
                if (!cS.is_valid_element(i)) continue;
                auto ix = cS.get_multidimensional_index(i);
                if (int(a + ix[0]) > int(p0 + t0 + ix[1])) cS[i] = -INFINITY;
            }
        }
        auto cM = qk.template get_row_reduction_destination_cooperative_tensor<decltype(tQ), TK_t, half>();
        reduce_rows(cS, cM, reduction_operation::max, half(-INFINITY));
        #pragma unroll
        for (uint16_t i = 0; i < cM.get_capacity(); ++i) if (cM.is_valid_element(i)) red[cM.get_multidimensional_index(i)[0]] = float(cM[i]);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (sl < RM) {
            float mo = mrow[sl], mn = max(mo, red[sl] * 0.0625f);
            arow[sl] = (mo == -INFINITY) ? 0.0f : exp(mo - mn);
            mrow[sl] = mn;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint16_t i = 0; i < cS.get_capacity(); ++i) {
            if (!cS.is_valid_element(i)) continue;
            float m = mrow[cS.get_multidimensional_index(i)[1]];
            cS[i] = half(m == -INFINITY ? 0.0f : exp(float(cS[i]) * 0.0625f - m));
        }
        auto cL = qk.template get_row_reduction_destination_cooperative_tensor<decltype(tQ), TK_t, half>();
        reduce_rows(cS, cL, reduction_operation::sum, half(0));
        #pragma unroll
        for (uint16_t i = 0; i < cO.get_capacity(); ++i) if (MODE == 0 && cO.is_valid_element(i)) cO[i] *= arow[cO.get_multidimensional_index(i)[1]];
        simdgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint16_t i = 0; i < cL.get_capacity(); ++i) if (cL.is_valid_element(i)) red[cL.get_multidimensional_index(i)[0]] = float(cL[i]);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (sl < RM) lrow[sl] = lrow[sl] * arow[sl] + red[sl];
        }
        auto cP = pv.template get_left_input_cooperative_tensor<half, half, float>(cS);
        if (STAGE) {
            tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tVs(VT, dextents<int32_t, 2>(256, KN));
            pv.run(cP, tVs, cO);
        } else {
            TK_t tV(Vc + (gh * Lmax + a) * 256, dextents<int32_t, 2>(256, KN));
            pv.run(cP, tV, cO);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
        if (!cO.is_valid_element(i)) continue;
        auto ix = cO.get_multidimensional_index(i);
        Op[((sp * 24 * T) + row0 + ix[1]) * 256 + ix[0]] = cO[i];
    }
    if (sl < RM) { stp[2 * (sp * 24 * T + row0 + sl)] = mrow[sl]; stp[2 * (sp * 24 * T + row0 + sl) + 1] = lrow[sl]; }
}
#define FA2K(NAME, RM, KN, NSG, MODE) \
kernel void NAME(device half *Q [[buffer(0)]], device half *Kc [[buffer(1)]], device half *Vc [[buffer(2)]], \
                 device float *Op [[buffer(3)]], device float *stp [[buffer(4)]], constant uint &Lmax [[buffer(5)]], \
                 constant uint &p0 [[buffer(6)]], constant uint &NS [[buffer(7)]], uint3 tg [[threadgroup_position_in_grid]], \
                 uint sg [[simdgroup_index_in_threadgroup]], uint sl [[thread_index_in_simdgroup]]) { \
    threadgroup float st[NSG * 4 * RM]; \
    fa2_impl<RM, KN, NSG, MODE>(Q, Kc, Vc, Op, stp, Lmax, p0, NS, tg, sg, sl, st); }
#define FA2S(NAME, RM, KN, NSG, MODE) \
kernel void NAME(device half *Q [[buffer(0)]], device half *Kc [[buffer(1)]], device half *Vc [[buffer(2)]], \
                 device float *Op [[buffer(3)]], device float *stp [[buffer(4)]], constant uint &Lmax [[buffer(5)]], \
                 constant uint &p0 [[buffer(6)]], constant uint &NS [[buffer(7)]], uint3 tg [[threadgroup_position_in_grid]], \
                 uint sg [[simdgroup_index_in_threadgroup]], uint sl [[thread_index_in_simdgroup]]) { \
    threadgroup float st[NSG * 4 * RM]; threadgroup half KT[KN * 256], VT[KN * 256]; \
    fa2_impl<RM, KN, NSG, MODE, true>(Q, Kc, Vc, Op, stp, Lmax, p0, NS, tg, sg, sl, st, KT, VT); }
FA2S(attn_fa2s_16_16_4, 16, 16, 4, 0)
FA2S(attn_fa2s_16_16_8, 16, 16, 8, 0)
FA2S(attn_fa2s_16_16_8_mm, 16, 16, 8, 2)
FA2K(attn_fa2_16_16, 16, 16, 4, 0)
FA2K(attn_fa2_16_32, 16, 32, 4, 0)
FA2K(attn_fa2_16_64, 16, 64, 4, 0)
FA2K(attn_fa2_16_32_norescale, 16, 32, 4, 1)
FA2K(attn_fa2_16_32_mm, 16, 32, 4, 2)
FA2K(attn_fa2_16_64_mm, 16, 64, 4, 2)

// ---- QX: two QK passes and no fp16 score round trip (QX=1): pass 1 (attn_qk_mx) the masked row maximum of each 64-key
// tile from the scores in registers, pass 2 (attn_qk_ex) the same QK again, P = exp(s / 16 - M) with M the row maximum
// over the whole key chunk, written once, plus each tile's row sums; attn_qx_st reduces (M, sum) per row for attn_combine.
// DRAM traffic per chunk: P written once and read once by attn_pv_w (two-pass: S written, read + written, read).
inline int f2o(float f) { int b = as_type<int>(f); return b >= 0 ? b : b ^ 0x7FFFFFFF; }
inline float o2f(int b) { return as_type<float>(b >= 0 ? b : b ^ 0x7FFFFFFF); }
kernel void attn_qk_mx(device half *Q [[buffer(0)]], device half *Kc [[buffer(1)]], device float *MX [[buffer(2)]],
                       constant uint &Lmax [[buffer(3)]], constant uint &a [[buffer(4)]], constant uint &TK [[buffer(5)]],
                       constant uint &p0 [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_int tmx[64];
    uint kt = tg.x, rt = tg.y, gh = tg.z, R0 = gh * 6 * T + rt * 64;
    if (tid < 64) atomic_store_explicit(&tmx[tid], f2o(-INFINITY), memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tQ(Q + R0 * 256, dextents<int32_t, 2>(256, 64));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tK(Kc + (gh * Lmax + a + kt * 64) * 256, dextents<int32_t, 2>(256, 64));
    constexpr auto d = matmul2d_descriptor(64, 64, 256, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    auto cS = op.get_destination_cooperative_tensor<decltype(tQ), decltype(tK), half>();
    op.run(tQ, tK, cS);
    int lim0 = int(p0 + (R0 % T) + 1) - int(a + kt * 64);   // visible keys of the tile's row r: lim0 + r
    int cur = -1; float m = -INFINITY;
    for (uint16_t i = 0; i < cS.get_capacity(); ++i) {
        if (!cS.is_valid_element(i)) continue;
        auto ix = cS.get_multidimensional_index(i);
        int r = ix[1];
        if (r != cur) { if (cur >= 0 && m != -INFINITY) atomic_fetch_max_explicit(&tmx[cur], f2o(m), memory_order_relaxed); cur = r; m = -INFINITY; }
        if (int(ix[0]) < lim0 + r) m = max(m, float(cS[i]));
    }
    if (cur >= 0 && m != -INFINITY) atomic_fetch_max_explicit(&tmx[cur], f2o(m), memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 64) MX[(R0 + tid) * (TK / 64) + kt] = o2f(atomic_load_explicit(&tmx[tid], memory_order_relaxed)) * 0.0625f;
}
kernel void attn_qk_ex(device half *Q [[buffer(0)]], device half *Kc [[buffer(1)]], device half *S [[buffer(2)]],
                       device const float *MX [[buffer(3)]], device float *SU [[buffer(4)]], constant uint &Lmax [[buffer(5)]],
                       constant uint &a [[buffer(6)]], constant uint &TK [[buffer(7)]], constant uint &p0 [[buffer(8)]],
                       constant uint &n [[buffer(9)]], uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float Mr[64];
    threadgroup atomic_float su[64];
    uint kt = tg.x, rt = tg.y, gh = tg.z, R0 = gh * 6 * T + rt * 64;
    if (tid < 64) {
        float m = -INFINITY;
        for (uint j = 0; j < n / 64; j++) m = max(m, MX[(R0 + tid) * (TK / 64) + j]);
        Mr[tid] = m; atomic_store_explicit(&su[tid], 0.0f, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tQ(Q + R0 * 256, dextents<int32_t, 2>(256, 64));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tK(Kc + (gh * Lmax + a + kt * 64) * 256, dextents<int32_t, 2>(256, 64));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tS(S + R0 * TK + kt * 64, dextents<int32_t, 2>(64, 64), array<int32_t, 2>{1, int(TK)});
    constexpr auto d = matmul2d_descriptor(64, 64, 256, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    auto cS = op.get_destination_cooperative_tensor<decltype(tQ), decltype(tK), half>();
    op.run(tQ, tK, cS);
    int lim0 = int(p0 + (R0 % T) + 1) - int(a + kt * 64);
    int cur = -1; float s = 0, m = 0;
    for (uint16_t i = 0; i < cS.get_capacity(); ++i) {
        if (!cS.is_valid_element(i)) continue;
        auto ix = cS.get_multidimensional_index(i);
        int r = ix[1];
        if (r != cur) { if (cur >= 0) atomic_fetch_add_explicit(&su[cur], s, memory_order_relaxed); cur = r; s = 0; m = Mr[r]; }
        float e = (int(ix[0]) < lim0 + r && m != -INFINITY) ? exp(float(cS[i]) * 0.0625f - m) : 0.0f;
        cS[i] = half(e); s += e;
    }
    if (cur >= 0) atomic_fetch_add_explicit(&su[cur], s, memory_order_relaxed);
    cS.store(tS);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 64) SU[(R0 + tid) * (TK / 64) + kt] = atomic_load_explicit(&su[tid], memory_order_relaxed);
}
kernel void attn_qx_st(device const float *MX [[buffer(0)]], device const float *SU [[buffer(1)]], device float *st [[buffer(2)]],
                       constant uint &TK [[buffer(3)]], constant uint &n [[buffer(4)]], uint r [[thread_position_in_grid]]) {
    float m = -INFINITY, s = 0;
    for (uint j = 0; j < n / 64; j++) { m = max(m, MX[r * (TK / 64) + j]); s += SU[r * (TK / 64) + j]; }
    st[2 * r] = m; st[2 * r + 1] = s;
}

// ---- tile-size variants of attn_qk_h / attn_pv_w (QKT=<rows>x<keys>, PVT=<dims>[s8]): RM query rows x KN keys per
// threadgroup; PV RM rows x DN dims. Same layouts and results as attn_qk_h / attn_pv_w.
template <int RM, int KN, int NSG>
inline void qk_t_impl(device half *Q, device half *Kc, device half *S, uint Lmax, uint a, uint TK, uint3 tg) {
    uint kt = tg.x, rt = tg.y, gh = tg.z;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tQ(Q + (gh * 6 * T + rt * RM) * 256, dextents<int32_t, 2>(256, RM));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tK(Kc + (gh * Lmax + a + kt * KN) * 256, dextents<int32_t, 2>(256, KN));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tS(S + (gh * 6 * T + rt * RM) * TK + kt * KN, dextents<int32_t, 2>(KN, RM),
                                                                array<int32_t, 2>{1, int(TK)});
    constexpr auto d = matmul2d_descriptor(RM, KN, 256, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<NSG>> op;
    op.run(tQ, tK, tS);
}
#define QKV_K(NAME, RM, KN, NSG) \
kernel void NAME(device half *Q [[buffer(0)]], device half *Kc [[buffer(1)]], device half *S [[buffer(2)]], \
                 constant uint &Lmax [[buffer(3)]], constant uint &a [[buffer(4)]], constant uint &TK [[buffer(5)]], \
                 uint3 tg [[threadgroup_position_in_grid]]) { qk_t_impl<RM, KN, NSG>(Q, Kc, S, Lmax, a, TK, tg); }
QKV_K(attn_qk_64x128, 64, 128, 4)
QKV_K(attn_qk_128x64, 128, 64, 4)
QKV_K(attn_qk_128x128, 128, 128, 8)
QKV_K(attn_qk_64x64s8, 64, 64, 8)
template <int RM, int DN, int NSG>
inline void pv_t_impl(device half *P, device half *Vc, device float *O, uint Lmax, uint a, uint TK, uint n, uint3 tg) {
    uint dt = tg.x, rt = tg.y, gh = tg.z;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tP(P + (gh * 6 * T + rt * RM) * TK, dextents<int32_t, 2>(int(n), RM),
                                                                array<int32_t, 2>{1, int(TK)});
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tV(Vc + (gh * Lmax + a) * 256 + dt * DN, dextents<int32_t, 2>(DN, int(n)),
                                                                array<int32_t, 2>{1, 256});
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tO(O + (gh * 6 * T + rt * RM) * 256 + dt * DN, dextents<int32_t, 2>(DN, RM),
                                                                 array<int32_t, 2>{1, 256});
    constexpr auto d = matmul2d_descriptor(RM, DN, static_cast<int>(dynamic_extent), false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<NSG>> op;
    op.run(tP, tV, tO);
}
#define PVV_K(NAME, RM, DN, NSG) \
kernel void NAME(device half *P [[buffer(0)]], device half *Vc [[buffer(1)]], device float *O [[buffer(2)]], \
                 constant uint &Lmax [[buffer(3)]], constant uint &a [[buffer(4)]], constant uint &TK [[buffer(5)]], \
                 constant uint &n [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]]) { pv_t_impl<RM, DN, NSG>(P, Vc, O, Lmax, a, TK, n, tg); }
PVV_K(attn_pv_64x256, 64, 256, 4)
PVV_K(attn_pv_64x256s8, 64, 256, 8)
PVV_K(attn_pv_128x128, 128, 128, 8)
PVV_K(attn_pv_64x128s8, 64, 128, 8)

// ---- two-pass tiled attention with fp16 scores (softmax in place) --------------------------------------------------
kernel void attn_qk_h(device half *Q [[buffer(0)]], device half *Kc [[buffer(1)]], device half *S [[buffer(2)]],
                      constant uint &Lmax [[buffer(3)]], constant uint &a [[buffer(4)]], constant uint &TK [[buffer(5)]],
                      uint3 tg [[threadgroup_position_in_grid]]) {
    uint kt = tg.x, rt = tg.y, gh = tg.z;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tQ(Q + (gh * 6 * T + rt * 64) * 256, dextents<int32_t, 2>(256, 64));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tK(Kc + (gh * Lmax + a + kt * 64) * 256, dextents<int32_t, 2>(256, 64));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tS(S + (gh * 6 * T + rt * 64) * TK + kt * 64, dextents<int32_t, 2>(64, 64),
                                                                array<int32_t, 2>{1, int(TK)});
    constexpr auto d = matmul2d_descriptor(64, 64, 256, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    op.run(tQ, tK, tS);
}
kernel void attn_softmax_h(device half *S [[buffer(0)]], device float *st [[buffer(1)]], constant uint &TK [[buffer(2)]],
                           constant uint &a [[buffer(3)]], constant uint &n [[buffer(4)]], constant uint &p0 [[buffer(5)]],
                           uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                           uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[8];
    device half *row = S + r * TK;
    int lim = int(p0 + (r % T) + 1) - int(a);
    uint nv = uint(clamp(lim, 0, int(n)));
    float m = -INFINITY;
    for (uint j = tid; j < nv; j += 256) m = max(m, float(row[j]) * 0.0625f);
    m = simd_max(m);
    if (sl == 0) part[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = part[0];
    for (uint i = 1; i < 8; i++) m = max(m, part[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0;
    for (uint j = tid; j < n; j += 256) { float e = j < nv ? exp(float(row[j]) * 0.0625f - m) : 0.0f; row[j] = half(e); s += e; }
    s = simd_sum(s);
    if (sl == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float tot = 0; for (uint i = 0; i < 8; i++) tot += part[i]; st[2 * r] = m; st[2 * r + 1] = tot; }
}
// PV with 64 rows x 128 dims per threadgroup (each probability row read twice instead of four times); grid (2, 6T/64, 4)
kernel void attn_pv_w(device half *P [[buffer(0)]], device half *Vc [[buffer(1)]], device float *O [[buffer(2)]],
                      constant uint &Lmax [[buffer(3)]], constant uint &a [[buffer(4)]], constant uint &TK [[buffer(5)]],
                      constant uint &n [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]]) {
    uint dt = tg.x, rt = tg.y, gh = tg.z;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tP(P + (gh * 6 * T + rt * 64) * TK, dextents<int32_t, 2>(int(n), 64),
                                                                array<int32_t, 2>{1, int(TK)});
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tV(Vc + (gh * Lmax + a) * 256 + dt * 128, dextents<int32_t, 2>(128, int(n)),
                                                                array<int32_t, 2>{1, 256});
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tO(O + (gh * 6 * T + rt * 64) * 256 + dt * 128, dextents<int32_t, 2>(128, 64),
                                                                 array<int32_t, 2>{1, 256});
    constexpr auto d = matmul2d_descriptor(64, 128, static_cast<int>(dynamic_extent), false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    op.run(tP, tV, tO);
}

// ---- attention with the softmax folded in (SMF=1): no separate softmax pass over the score tiles ------------------
// attn_qk_hm = attn_qk_h + each 64-key tile's masked row maximum (of the scaled scores) into MX[row][tile]
kernel void attn_qk_hm(device half *Q [[buffer(0)]], device half *Kc [[buffer(1)]], device half *S [[buffer(2)]],
                       device float *MX [[buffer(3)]], constant uint &Lmax [[buffer(4)]], constant uint &a [[buffer(5)]],
                       constant uint &TK [[buffer(6)]], constant uint &p0 [[buffer(7)]],
                       uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    uint kt = tg.x, rt = tg.y, gh = tg.z;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tQ(Q + (gh * 6 * T + rt * 64) * 256, dextents<int32_t, 2>(256, 64));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tK(Kc + (gh * Lmax + a + kt * 64) * 256, dextents<int32_t, 2>(256, 64));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tS(S + (gh * 6 * T + rt * 64) * TK + kt * 64, dextents<int32_t, 2>(64, 64),
                                                                array<int32_t, 2>{1, int(TK)});
    constexpr auto d = matmul2d_descriptor(64, 64, 256, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    op.run(tQ, tK, tS);
    threadgroup_barrier(mem_flags::mem_device);
    uint r = tid >> 1, kh = tid & 1, R = gh * 6 * T + rt * 64 + r;
    int lim = int(p0 + (R % T) + 1) - int(a + kt * 64 + kh * 32);   // valid keys in this thread's 32
    device half *row = S + R * TK + kt * 64 + kh * 32;
    float m = -INFINITY;
    for (int j = 0; j < 32; j++) if (j < lim) m = max(m, float(row[j]) * 0.0625f);
    m = max(m, simd_shuffle_xor(m, 1));
    if (kh == 0) MX[R * (TK / 64) + kt] = m;
}
// one-pass softmax (SMF=2): the row max comes from attn_qk_hm's tile maxima, so one read + one write of the scores
kernel void attn_softmax_hm(device half *S [[buffer(0)]], device float *st [[buffer(1)]], device const float *MX [[buffer(2)]],
                            constant uint &TK [[buffer(3)]], constant uint &a [[buffer(4)]], constant uint &n [[buffer(5)]],
                            constant uint &p0 [[buffer(6)]], uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                            uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[8];
    uint nt = n / 64;
    float m = tid < nt ? MX[r * (TK / 64) + tid] : -INFINITY;
    m = simd_max(m);
    if (sl == 0) part[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = part[0];
    for (uint i = 1; i < 8; i++) m = max(m, part[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int lim = int(p0 + (r % T) + 1) - int(a);
    uint nv = uint(clamp(lim, 0, int(n)));
    device half4 *row = (device half4 *)(S + r * TK);
    float s = 0;
    for (uint j4 = tid; j4 < n / 4; j4 += 256) {
        float4 v = float4(row[j4]) * 0.0625f - m;
        uint j = 4 * j4;
        float4 e = float4(j < nv ? exp(v.x) : 0.0f, j + 1 < nv ? exp(v.y) : 0.0f, j + 2 < nv ? exp(v.z) : 0.0f, j + 3 < nv ? exp(v.w) : 0.0f);
        row[j4] = half4(e); s += e.x + e.y + e.z + e.w;
    }
    s = simd_sum(s);
    if (sl == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float tot = 0; for (uint i = 0; i < 8; i++) tot += part[i]; st[2 * r] = m; st[2 * r + 1] = tot; }
}
// softmax with the row held in registers (SMF=3): n <= 4096 scores, 16 per thread, read once, written once
kernel void attn_softmax_hr(device half *S [[buffer(0)]], device float *st [[buffer(1)]], constant uint &TK [[buffer(2)]],
                            constant uint &a [[buffer(3)]], constant uint &n [[buffer(4)]], constant uint &p0 [[buffer(5)]],
                            uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                            uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[8];
    device half4 *row = (device half4 *)(S + r * TK);
    int lim = int(p0 + (r % T) + 1) - int(a);
    uint nv = uint(clamp(lim, 0, int(n))), n4 = n / 4;
    float4 v[4];
    float m = -INFINITY;
    for (uint k = 0; k < 4; k++) {
        uint j4 = tid + 256 * k, j = 4 * j4;
        float4 x = j4 < n4 ? float4(row[j4]) * 0.0625f : float4(-INFINITY);
        x.x = j < nv ? x.x : -INFINITY; x.y = j + 1 < nv ? x.y : -INFINITY; x.z = j + 2 < nv ? x.z : -INFINITY; x.w = j + 3 < nv ? x.w : -INFINITY;
        v[k] = x; m = max(m, max(max(x.x, x.y), max(x.z, x.w)));
    }
    m = simd_max(m);
    if (sl == 0) part[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = part[0];
    for (uint i = 1; i < 8; i++) m = max(m, part[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0;
    for (uint k = 0; k < 4; k++) {
        uint j4 = tid + 256 * k;
        if (j4 >= n4) break;
        float4 e = (m == -INFINITY) ? float4(0.0f) : exp(v[k] - m);
        row[j4] = half4(e); s += e.x + e.y + e.z + e.w;
    }
    s = simd_sum(s);
    if (sl == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float tot = 0; for (uint i = 0; i < 8; i++) tot += part[i]; st[2 * r] = m; st[2 * r + 1] = tot; }
}
// attn_pv_wm = attn_pv_w on exp(s - m) computed while loading each 64-key score tile (m: the row max over the tiles),
// row sums accumulated here; stats (m, sum) for attn_combine. grid (2, 6T/64, 4), 64 rows x 128 dims per threadgroup
kernel void attn_pv_wm(device half *S [[buffer(0)]], device half *Vc [[buffer(1)]], device float *O [[buffer(2)]],
                       device const float *MX [[buffer(3)]], device float *st [[buffer(4)]], constant uint &Lmax [[buffer(5)]],
                       constant uint &a [[buffer(6)]], constant uint &TK [[buffer(7)]], constant uint &n [[buffer(8)]],
                       constant uint &p0 [[buffer(9)]], uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half Pt[64 * 64];
    uint dt = tg.x, rt = tg.y, gh = tg.z;
    uint r = tid >> 1, kh = tid & 1, R = gh * 6 * T + rt * 64 + r, nt = n / 64;
    float m = -INFINITY;
    for (uint j = kh; j < nt; j += 2) m = max(m, MX[R * (TK / 64) + j]);
    m = max(m, simd_shuffle_xor(m, 1));
    int lim = int(p0 + (R % T) + 1) - int(a);
    float sum = 0;
    constexpr auto d = matmul2d_descriptor(64, 128, 64, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<d, execution_simdgroups<4>> op;
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(Pt, dextents<int32_t, 2>(64, 64));
    auto cO = op.template get_destination_cooperative_tensor<decltype(tP), tensor<device half, dextents<int32_t, 2>, tensor_inline>, float>();
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) if (cO.is_valid_element(i)) cO[i] = 0;
    device half *row = S + R * TK;
    for (uint j0 = 0; j0 < n; j0 += 64) {
        for (uint c = 0; c < 32; c++) {
            int j = int(j0 + kh * 32 + c);
            float e = (j < lim && m != -INFINITY) ? exp(float(row[j]) * 0.0625f - m) : 0.0f;
            sum += e; Pt[r * 64 + kh * 32 + c] = half(e);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tensor<device half, dextents<int32_t, 2>, tensor_inline> tV(Vc + (gh * Lmax + a + j0) * 256 + dt * 128, dextents<int32_t, 2>(128, 64),
                                                                    array<int32_t, 2>{1, 256});
        op.run(tP, tV, cO);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
        if (!cO.is_valid_element(i)) continue;
        auto ix = cO.get_multidimensional_index(i);
        O[(gh * 6 * T + rt * 64 + ix[1]) * 256 + dt * 128 + ix[0]] = cO[i];
    }
    sum += simd_shuffle_xor(sum, 1);
    if (dt == 0 && kh == 0) { st[2 * R] = m; st[2 * R + 1] = sum; }
}

// ---- KV8: attention over the ANE's own INT8 KV cache (kv8 build: keys dim_token [kvh][256][S], values [kvh][S][256],
// fp16 scales [kvh][S] per token / head; one buffer set per KV segment). Strides in bytes (scales: in halves).
// attn_prep8 = attn_prep16 with the new K / V rows quantized exactly as qwen38_kv_cache.quantize_values (fp16 values,
// scale = max(absmax / 127, 1e-6) stored fp16, codes rint(v / scale) clipped to +-127) into the segment at row r0 + t.
kernel void attn_prep8(device const float *qg [[buffer(0)]], device const float *kp [[buffer(1)]], device const float *vp [[buffer(2)]],
                       device const float *qn [[buffer(3)]], device const float *kn [[buffer(4)]], device half *Q [[buffer(5)]],
                       device int8_t *K8 [[buffer(6)]], device int8_t *V8 [[buffer(7)]], device half *KS [[buffer(8)]],
                       device half *VS [[buffer(9)]], constant uint4 &st [[buffer(10)]], constant uint2 &st2 [[buffer(11)]],
                       constant float &eps [[buffer(12)]], device const float *cs [[buffer(13)]],
                       uint2 g [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]],
                       uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    // st = (K head stride, K dim stride, V head stride, V row stride); st2 = (scale head stride, r0)
    threadgroup float part[8], buf[256];
    uint h = g.x, t = g.y, r = st2.y + t;
    float v;
    if (h < 24) v = qg[(h * 512 + d) * T + t];
    else if (h < 28) v = kp[((h - 24) * 256 + d) * T + t];
    else v = vp[((h - 28) * 256 + d) * T + t];
    if (h < 28) {
        float s = simd_sum(v * v);
        if (sl == 0) part[sg] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = 0;
        for (uint i = 0; i < 8; i++) tot += part[i];
        v = v * rsqrt(tot / 256.0f + eps) * (h < 24 ? qn[d] : kn[d]);
        buf[d] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (d < 64) {
            float c = cs[(t * 64 + d) * 2], sn = cs[(t * 64 + d) * 2 + 1];
            float rot = d < 32 ? -buf[d + 32] : buf[d - 32];
            v = buf[d] * c + rot * sn;
        }
        if (h < 24) { Q[(h * T + t) * 256 + d] = half(v * 0.0625f); return; }   // Q / 16 (scores vs INT8 codes stay in fp16; = the 1/sqrt(256) scale)
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float hv = float(half(v));                       // the cache quantizes the fp16 row
    float m = simd_max(fabs(hv));
    if (sl == 0) part[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = part[0];
    for (uint i = 1; i < 8; i++) m = max(m, part[i]);
    half sc = half(max(m / 127.0f, 1e-6f));
    int8_t code = int8_t(clamp(rint(hv / float(sc)), -127.0f, 127.0f));
    if (h < 28) { uint kh = h - 24; K8[kh * st.x + d * st.y + r] = code; if (d == 0) KS[kh * st2.x + r] = sc; }
    else { uint vh = h - 28; V8[vh * st.z + r * st.w + d] = code; if (d == 0) VS[vh * st2.x + r] = sc; }
}
kernel void attn_prep8q(device const float *qg [[buffer(0)]], device const float *kp [[buffer(1)]], device const float *vp [[buffer(2)]],
                       device const float *qn [[buffer(3)]], device const float *kn [[buffer(4)]], device half *Q [[buffer(5)]],
                       device int8_t *K8 [[buffer(6)]], device int8_t *V8 [[buffer(7)]], device half *KS [[buffer(8)]],
                       device half *VS [[buffer(9)]], constant uint4 &st [[buffer(10)]], constant uint2 &st2 [[buffer(11)]],
                       constant float &eps [[buffer(12)]], device const float *cs [[buffer(13)]],
                       device int8_t *Q8 [[buffer(14)]], device float *QS [[buffer(15)]],
                       uint2 g [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]],
                       uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    // st = (K head stride, K dim stride, V head stride, V row stride); st2 = (scale head stride, r0)
    threadgroup float part[8], buf[256];
    uint h = g.x, t = g.y, r = st2.y + t;
    float v;
    if (h < 24) v = qg[(h * 512 + d) * T + t];
    else if (h < 28) v = kp[((h - 24) * 256 + d) * T + t];
    else v = vp[((h - 28) * 256 + d) * T + t];
    if (h < 28) {
        float s = simd_sum(v * v);
        if (sl == 0) part[sg] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = 0;
        for (uint i = 0; i < 8; i++) tot += part[i];
        v = v * rsqrt(tot / 256.0f + eps) * (h < 24 ? qn[d] : kn[d]);
        buf[d] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (d < 64) {
            float c = cs[(t * 64 + d) * 2], sn = cs[(t * 64 + d) * 2 + 1];
            float rot = d < 32 ? -buf[d + 32] : buf[d - 32];
            v = buf[d] * c + rot * sn;
        }
        if (h < 24) {   // Q / 16 as before, and INT8 Q: code = rint(q / sq), QS = sq / 16 (score = acc x QS)
            Q[(h * T + t) * 256 + d] = half(v * 0.0625f);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float am = simd_max(fabs(v));
            if (sl == 0) part[sg] = am;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float mq = part[0];
            for (uint i = 1; i < 8; i++) mq = max(mq, part[i]);
            float sq = mq > 0 ? mq / 127.0f : 1.0f;
            Q8[(h * T + t) * 256 + d] = int8_t(clamp(rint(v / sq), -127.0f, 127.0f));
            if (d == 0) QS[h * T + t] = sq * 0.0625f;
            return;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float hv = float(half(v));                       // the cache quantizes the fp16 row
    float m = simd_max(fabs(hv));
    if (sl == 0) part[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = part[0];
    for (uint i = 1; i < 8; i++) m = max(m, part[i]);
    half sc = half(max(m / 127.0f, 1e-6f));
    int8_t code = int8_t(clamp(rint(hv / float(sc)), -127.0f, 127.0f));
    if (h < 28) { uint kh = h - 24; K8[kh * st.x + d * st.y + r] = code; if (d == 0) KS[kh * st2.x + r] = sc; }
    else { uint vh = h - 28; V8[vh * st.z + r * st.w + d] = code; if (d == 0) VS[vh * st2.x + r] = sc; }
}
// K8V16 (patch_k8v16.py): attn_prep8q + the fp16 V rows into the GPU's Vc
kernel void attn_prep8qv(device const float *qg [[buffer(0)]], device const float *kp [[buffer(1)]], device const float *vp [[buffer(2)]],
                       device const float *qn [[buffer(3)]], device const float *kn [[buffer(4)]], device half *Q [[buffer(5)]],
                       device int8_t *K8 [[buffer(6)]], device int8_t *V8 [[buffer(7)]], device half *KS [[buffer(8)]],
                       device half *VS [[buffer(9)]], constant uint4 &st [[buffer(10)]], constant uint2 &st2 [[buffer(11)]],
                       constant float &eps [[buffer(12)]], device const float *cs [[buffer(13)]],
                       device int8_t *Q8 [[buffer(14)]], device float *QS [[buffer(15)]],
                       device half *Vc [[buffer(16)]], constant uint2 &vl [[buffer(17)]],
                       uint2 g [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]],
                       uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    // st = (K head stride, K dim stride, V head stride, V row stride); st2 = (scale head stride, r0)
    threadgroup float part[8], buf[256];
    uint h = g.x, t = g.y, r = st2.y + t;
    float v;
    if (h < 24) v = qg[(h * 512 + d) * T + t];
    else if (h < 28) v = kp[((h - 24) * 256 + d) * T + t];
    else v = vp[((h - 28) * 256 + d) * T + t];
    if (h < 28) {
        float s = simd_sum(v * v);
        if (sl == 0) part[sg] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = 0;
        for (uint i = 0; i < 8; i++) tot += part[i];
        v = v * rsqrt(tot / 256.0f + eps) * (h < 24 ? qn[d] : kn[d]);
        buf[d] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (d < 64) {
            float c = cs[(t * 64 + d) * 2], sn = cs[(t * 64 + d) * 2 + 1];
            float rot = d < 32 ? -buf[d + 32] : buf[d - 32];
            v = buf[d] * c + rot * sn;
        }
        if (h < 24) {   // Q / 16 as before, and INT8 Q: code = rint(q / sq), QS = sq / 16 (score = acc x QS)
            Q[(h * T + t) * 256 + d] = half(v * 0.0625f);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float am = simd_max(fabs(v));
            if (sl == 0) part[sg] = am;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float mq = part[0];
            for (uint i = 1; i < 8; i++) mq = max(mq, part[i]);
            float sq = mq > 0 ? mq / 127.0f : 1.0f;
            Q8[(h * T + t) * 256 + d] = int8_t(clamp(rint(v / sq), -127.0f, 127.0f));
            if (d == 0) QS[h * T + t] = sq * 0.0625f;
            return;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float hv = float(half(v));                       // the cache quantizes the fp16 row
    float m = simd_max(fabs(hv));
    if (sl == 0) part[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = part[0];
    for (uint i = 1; i < 8; i++) m = max(m, part[i]);
    half sc = half(max(m / 127.0f, 1e-6f));
    int8_t code = int8_t(clamp(rint(hv / float(sc)), -127.0f, 127.0f));
    if (h < 28) { uint kh = h - 24; K8[kh * st.x + d * st.y + r] = code; if (d == 0) KS[kh * st2.x + r] = sc; }
    else { uint vh = h - 28; V8[vh * st.z + r * st.w + d] = code; if (d == 0) VS[vh * st2.x + r] = sc;
           Vc[(vh * vl.x + vl.y + t) * 256 + d] = half(hv); }   // + the fp16 row (vl = Lmax, p0)
}
// Vc rows [r0, r0 + n) of one attention layer from an ANE INT8 V segment (row r of the segment = position s0 + r);
// prm = (Lmax, s0, vh head stride bytes, vr row stride bytes), sh = scale head stride (halves). Grid (256, n, 4).
kernel void kv8_v_to_v16(device const int8_t *V8 [[buffer(0)]], device const half *VS [[buffer(1)]], device half *Vc [[buffer(2)]],
                         constant uint4 &prm [[buffer(3)]], constant uint &sh [[buffer(4)]], uint3 p [[thread_position_in_grid]]) {
    uint d = p.x, r = p.y, h = p.z;
    Vc[(h * prm.x + prm.y + r) * 256 + d] = half(float(V8[h * prm.z + r * prm.w + d]) * float(VS[h * sh + r]));
}
// QK8 on an ANE key segment: S tile = (Q8 K8^T) x QS, int32 accumulate; K8 as attn_qk8 (dim-major, bound at the
// tile's first key). Grid (n / 64, 6T / 64, 4).
kernel void attn_qk8is(device int8_t *Q8 [[buffer(0)]], device int8_t *K8 [[buffer(1)]], device half *S [[buffer(2)]],
                       constant uint2 &kst [[buffer(3)]], constant uint &TK [[buffer(4)]], device const float *QS [[buffer(5)]],
                       uint3 tg [[threadgroup_position_in_grid]]) {
    uint kt = tg.x, rt = tg.y, gh = tg.z, r0 = gh * 6 * T + rt * 64;
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tQ(Q8 + r0 * 256, dextents<int32_t, 2>(256, 64));
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tK(K8 + gh * kst.x + kt * 64, dextents<int32_t, 2>(64, 256),
                                                                  array<int32_t, 2>{1, int(kst.y)});
    constexpr auto d = matmul2d_descriptor(64, 64, 256, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    auto cI = op.template get_destination_cooperative_tensor<decltype(tQ), decltype(tK), int32_t>();
    op.run(tQ, tK, cI);
    for (uint16_t i = 0; i < cI.get_capacity(); ++i) {
        if (!cI.is_valid_element(i)) continue;
        auto ix = cI.get_multidimensional_index(i);
        uint key = ix[0], row = ix[1];
        S[(r0 + row) * TK + kt * 64 + key] = half(float(cI[i]) * QS[r0 + row]);
    }
}
// S[g rows][tile] = Q_g K8^T for one 64-key tile of a segment (K8 bound at the tile's first key); grid (n/64, 6T/64, 4)
kernel void attn_qk8(device half *Q [[buffer(0)]], device int8_t *K8 [[buffer(1)]], device half *S [[buffer(2)]],
                     constant uint2 &kst [[buffer(3)]], constant uint &TK [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]]) {
    uint kt = tg.x, rt = tg.y, gh = tg.z;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tQ(Q + (gh * 6 * T + rt * 64) * 256, dextents<int32_t, 2>(256, 64));
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tK(K8 + gh * kst.x + kt * 64, dextents<int32_t, 2>(64, 256),
                                                                  array<int32_t, 2>{1, int(kst.y)});
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tS(S + (gh * 6 * T + rt * 64) * TK + kt * 64, dextents<int32_t, 2>(64, 64),
                                                                array<int32_t, 2>{1, int(TK)});
    constexpr auto d = matmul2d_descriptor(64, 64, 256, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    op.run(tQ, tK, tS);
}
// softmax over a tile's scores x key scale; P = exp(s - m) x value scale (the PV matmul then reads INT8 V codes);
// stats (m, sum of exp(s - m)) as attn_softmax_h. KS / VS bound at the tile's first key.
kernel void attn_softmax8(device half *S [[buffer(0)]], device float *st [[buffer(1)]], constant uint &TK [[buffer(2)]],
                          constant uint &a [[buffer(3)]], constant uint &n [[buffer(4)]], constant uint &p0 [[buffer(5)]],
                          device const half *KS [[buffer(6)]], device const half *VS [[buffer(7)]], constant uint &ssh [[buffer(8)]],
                          uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                          uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[8];
    device half *row = S + r * TK;
    uint gh = r / (6 * T);
    device const half *ks = KS + gh * ssh, *vs = VS + gh * ssh;
    int lim = int(p0 + (r % T) + 1) - int(a);
    uint nv = uint(clamp(lim, 0, int(n)));
    float m = -INFINITY;
    for (uint j = tid; j < nv; j += 256) m = max(m, float(row[j]) * float(ks[j]));
    m = simd_max(m);
    if (sl == 0) part[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = part[0];
    for (uint i = 1; i < 8; i++) m = max(m, part[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0;
    for (uint j = tid; j < n; j += 256) {
        float e = j < nv ? exp(float(row[j]) * float(ks[j]) - m) : 0.0f;
        row[j] = half(e * float(vs[j]) * 4096.0f); s += e;   // x 4096: e x vs (~1e-5 .. 1e-6) stays out of fp16's subnormal range
    }
    s = simd_sum(s);
    if (sl == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float tot = 0; for (uint i = 0; i < 8; i++) tot += part[i]; st[2 * r] = m; st[2 * r + 1] = tot * 4096.0f; }   // the same 4096 in the sum: O / sum unchanged
}
// O = P' V8 (P' = probabilities x value scales, V8 bound at the tile's first key); grid (2, 6T/64, 4)
kernel void attn_pv8(device half *P [[buffer(0)]], device int8_t *V8 [[buffer(1)]], device float *O [[buffer(2)]],
                     constant uint2 &vst [[buffer(3)]], constant uint &TK [[buffer(4)]], constant uint &n [[buffer(5)]],
                     uint3 tg [[threadgroup_position_in_grid]]) {
    uint dt = tg.x, rt = tg.y, gh = tg.z;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tP(P + (gh * 6 * T + rt * 64) * TK, dextents<int32_t, 2>(int(n), 64),
                                                                array<int32_t, 2>{1, int(TK)});
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tV(V8 + gh * vst.x + dt * 128, dextents<int32_t, 2>(128, int(n)),
                                                                  array<int32_t, 2>{1, int(vst.y)});
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tO(O + (gh * 6 * T + rt * 64) * 256 + dt * 128, dextents<int32_t, 2>(128, 64),
                                                                 array<int32_t, 2>{1, 256});
    constexpr auto d = matmul2d_descriptor(64, 128, static_cast<int>(dynamic_extent), false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    op.run(tP, tV, tO);
}

// ---- V8 pair calls: two 64-token blocks (two ANE buffers [5120][T/2] fp16) <-> X [5120][T] fp32 ----------------------
kernel void from_half_pair(device const half *a [[buffer(0)]], device const half *b [[buffer(1)]], device float *x [[buffer(2)]],
                           uint i [[thread_position_in_grid]]) {
    const uint h = T / 2, c = i / T, t = i % T;
    x[i] = t < h ? float(a[c * h + t]) : float(b[c * h + t - h]);
}
kernel void to_half_pair(device const float *x [[buffer(0)]], device half *a [[buffer(1)]], device half *b [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
    const uint h = T / 2, c = i / T, t = i % T;
    if (t < h) a[c * h + t] = half(x[i]); else b[c * h + t - h] = half(x[i]);
}

// ---- A8 FFN (patch_a8.py): the GPU side of the ANE's runtime-INT8 FFN ----
// LUT -> INT8 codes of one chunk matrix (prep_a8.py tables), one thread per (column c, subgroup s): the subgroup's G
// index rows x 2 (cd 2) = 2G consecutive output rows of column c as 2G codes (pal8 lookup). prm = (cin, G, mode, xoff).
// mode 0 (gate / up): dst [17][5120][1024], (o, c) -> (o >> 10) * 5120 * 1024 + c * 1024 + (o & 1023);
// mode 1 (down): dst [17][1024][5120], (o, x = xoff + c) -> (x >> 10) * 1024 * 5120 + (x & 1023) * 5120 + o.
kernel void a8_expand(device const uchar *k [[buffer(0)]], device const uint *tab [[buffer(1)]], device const char2 *pal [[buffer(2)]],
                      device int8_t *dst [[buffer(3)]], constant uint4 &prm [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    uint c = gid.x, s = gid.y, cin = prm.x, G = prm.y;
    if (c >= cin) return;
    device const uchar *ix = k + tab[3 * s + 1] + 4 * G;   // indices follow the subgroup's 2G fp16 row scales
    uint o0 = 2 * tab[3 * s + 2], x = prm.w + c;
    device int8_t *d = prm.z == 0 ? dst + (o0 >> 10) * (5120u * 1024u) + c * 1024u + (o0 & 1023u)
                                  : dst + (x >> 10) * (1024u * 5120u) + (x & 1023u) * 5120u + o0;
    for (uint g0 = 0; g0 < G; g0 += 8) {   // 8 indices = 6 bytes at byte (c * G + g0) * 6 / 8 (LSB-first)
        uint b = (c * G + g0) * 6 / 8;
        ulong v = 0;
        for (uint q = 0; q < 6; q++) v |= ulong(ix[b + q]) << (8 * q);
        uint4 w = 0;
        for (uint r = 0; r < 8; r++) {
            char2 p = pal[uint(v >> (6 * r)) & 63u];
            uint u = uint(uchar(p.x)) | (uint(uchar(p.y)) << 8);
            w[r / 2] |= u << (16 * (r % 2));
        }
        *(device uint4 *)(d + 2 * g0) = w;
    }
}
// the FFN input of one call (h: fp16 channel-major [5120][T], rms_fwht4's output) -> INT8 rows x8[(r0 + t) * xs + ch]
// per (token, 2560-channel half), absmax scale; tx[(r0 + t) * ts + half] = scale x 128. prm = (r0, xs bytes, ts halves).
// Grid (2, T) threadgroups of 256.
kernel void a8_quant_tm(device const half *h [[buffer(0)]], device int8_t *x8 [[buffer(1)]], device half *tx [[buffer(2)]],
                        constant uint4 &prm [[buffer(3)]], uint2 g [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]], uint sl [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float mx[8];
    uint hf = g.x, t = g.y;
    const uint PW = prm.w ? prm.w : 2560, NJ = PW / 256;   // part width (channels per scale)
    float v[10], m = 0;
    for (uint j = 0; j < NJ; j++) { v[j] = float(h[(hf * PW + j * 256 + tid) * T + t]); m = max(m, abs(v[j])); }
    m = simd_max(m);
    if (sl == 0) mx[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mm = 0;
    for (uint i = 0; i < 8; i++) mm = max(mm, mx[i]);
    float sc = mm > 0 ? mm / 127.0f : 1.0f, inv = 1.0f / sc;
    device int8_t *row = x8 + (prm.x + t) * prm.y + hf * PW;
    for (uint j = 0; j < NJ; j++) row[j * 256 + tid] = int8_t(clamp(rint(v[j] * inv), -127.0f, 127.0f));
    if (tid == 0) tx[(prm.x + t) * prm.z + hf] = half(sc * 128.0f);
}
// x (fp32 channel-major [5120][T]) += y rows r0 .. r0 + T - 1 (token-major fp16, ys halves per row), 32 x 32 tiles.
// prm = (r0, ys). Grid (5120 / 32, T / 32) threadgroups of 256.
kernel void a8_add_tm(device float *x [[buffer(0)]], device const half *y [[buffer(1)]], constant uint4 &prm [[buffer(2)]],
                      uint2 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float tile[32][33];
    uint c0 = g.x * 32, t0 = g.y * 32;
    for (uint i = tid; i < 1024; i += 256) { uint tt = i / 32, cc = i % 32; tile[tt][cc] = float(y[(prm.x + t0 + tt) * prm.y + c0 + cc]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 1024; i += 256) { uint cc = i / 32, tt = i % 32; x[(c0 + cc) * T + t0 + tt] += tile[tt][cc]; }
}

// A8 tiled expander (patch_a8b.py): tile = 32 row-consecutive subgroups (G 16) x 32 columns, 1024 threads.
// prm = (cin, G, mode, xoff) as a8_expand; grid (cin / 32, nsub / 32).
kernel void a8_expand_t(device const uchar *k [[buffer(0)]], device const uint *tab [[buffer(1)]], device const char2 *pal [[buffer(2)]],
                        device int8_t *dst [[buffer(3)]], constant uint4 &prm [[buffer(4)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uchar ti[32 * 32 * 12];   // [subgroup][column][12 bytes = 16 indices]
    threadgroup char2 tp[64];
    uint c0 = tg.x * 32, s0 = tg.y * 32;
    if (tid < 64) tp[tid] = pal[tid];
    {   // load: lanes over columns
        uint s = tid / 32, c = tid % 32;
        device const uchar *src = k + tab[3 * (s0 + s) + 1] + 64 + (c0 + c) * 12;
        threadgroup uchar *d = ti + (s * 32 + c) * 12;
        for (uint q = 0; q < 12; q++) d[q] = src[q];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint c = tid / 32, s = tid % 32;      // decode + write: lanes over subgroups
    threadgroup const uchar *b = ti + (s * 32 + c) * 12;
    uint o0 = 2 * tab[3 * (s0 + s) + 2], x = prm.w + c0 + c, cc = c0 + c;
    device int8_t *d = prm.z == 0 ? dst + (o0 >> 10) * (5120u * 1024u) + cc * 1024u + (o0 & 1023u)
                                  : dst + (x >> 10) * (1024u * 5120u) + (x & 1023u) * 5120u + o0;
    for (uint h = 0; h < 2; h++) {
        ulong v = 0;
        for (uint q = 0; q < 6; q++) v |= ulong(b[6 * h + q]) << (8 * q);
        uint4 w = 0;
        for (uint r = 0; r < 8; r++) {
            char2 p = tp[uint(v >> (6 * r)) & 63u];
            w[r / 2] |= (uint(uchar(p.x)) | (uint(uchar(p.y)) << 8)) << (16 * (r % 2));
        }
        *(device uint4 *)(d + 16 * h) = w;
    }
}

// A8 general tiled expander (patch_a8c.py): any subgroup of 32 output rows (G x cd = 32), BITS 4 / 6, cd 1 / 2; tile =
// 32 row-consecutive subgroups x 32 columns, 1024 threads. prm = (cin, G, mode, xoff), fm = (bits, cd, bytes per column).
kernel void a8_expand_g(device const uchar *k [[buffer(0)]], device const uint *tab [[buffer(1)]], device const char2 *pal [[buffer(2)]],
                        device int8_t *dst [[buffer(3)]], constant uint4 &prm [[buffer(4)]], constant uint4 &fm [[buffer(5)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uchar ti[32 * 32 * 16];
    threadgroup char2 tp[64];
    const uint BITS = fm.x, CD = fm.y, BPC = fm.z, NI = 32 / CD, MASK = (1u << BITS) - 1;
    uint c0 = tg.x * 32, s0 = tg.y * 32;
    if (tid < 64) tp[tid] = pal[tid];
    {
        uint s = tid / 32, c = tid % 32;
        device const uchar *src = k + tab[3 * (s0 + s) + 1] + 64 + (c0 + c) * BPC;
        threadgroup uchar *d = ti + (s * 32 + c) * 16;
        for (uint q = 0; q < BPC; q++) d[q] = src[q];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint c = tid / 32, s = tid % 32;
    threadgroup const uchar *b = ti + (s * 32 + c) * 16;
    uint o0 = CD * tab[3 * (s0 + s) + 2], x = prm.w + c0 + c, cc = c0 + c;
    device int8_t *d = prm.z == 0 ? dst + (o0 >> 10) * (5120u * 1024u) + cc * 1024u + (o0 & 1023u)
                                  : dst + (x >> 10) * (1024u * 5120u) + (x & 1023u) * 5120u + o0;
    uchar ob[32];
    for (uint i = 0; i < NI; i++) {
        uint bit = i * BITS, by = bit >> 3, sh = bit & 7;
        uint v = (uint(b[by]) | (by + 1 < BPC ? uint(b[by + 1]) << 8 : 0u)) >> sh;
        char2 p = tp[v & MASK];
        if (CD == 2) { ob[2 * i] = uchar(p.x); ob[2 * i + 1] = uchar(p.y); } else { ob[i] = uchar(p.x); }
    }
    for (uint h = 0; h < 2; h++) {
        uint4 w;
        for (uint q = 0; q < 4; q++) w[q] = uint(ob[16 * h + 4 * q]) | (uint(ob[16 * h + 4 * q + 1]) << 8) | (uint(ob[16 * h + 4 * q + 2]) << 16) | (uint(ob[16 * h + 4 * q + 3]) << 24);
        *(device uint4 *)(d + 16 * h) = w;
    }
}
// x (fp32 channel-major [C][T]) = src (the ANE's token-major output [C / 1024 blocks][Tb rows][1024] fp16) rows r0 ..;
// prm = (r0, Tb). Grid (C / 32, T / 32) threadgroups of 256.
kernel void a8_load_tm(device float *x [[buffer(0)]], device const half *y [[buffer(1)]], constant uint4 &prm [[buffer(2)]],
                       uint2 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float tile[32][33];
    uint c0 = g.x * 32, t0 = g.y * 32;
    device const half *yb = y + ulong(c0 >> 10) * prm.y * 1024u;
    for (uint i = tid; i < 1024; i += 256) { uint tt = i / 32, cc = i % 32; tile[tt][cc] = float(yb[(prm.x + t0 + tt) * 1024u + ((c0 + cc) & 1023u)]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 1024; i += 256) { uint cc = i / 32, tt = i % 32; x[(c0 + cc) * T + t0 + tt] = tile[tt][cc]; }
}

// ---- KV8 attention, fp16-rate path (patch_kv8b.py, KV8D=1) ----
// S tile = (Q8 K8^T) x QS x KS: INT8 scores on an ANE key segment with the key scales folded in (KS bound at the
// tile's first key, head stride ssh halves). Grid (n / 64, 6T / 64, 4).
kernel void attn_qk8isk(device int8_t *Q8 [[buffer(0)]], device int8_t *K8 [[buffer(1)]], device half *S [[buffer(2)]],
                        constant uint2 &kst [[buffer(3)]], constant uint &TK [[buffer(4)]], device const float *QS [[buffer(5)]],
                        device const half *KS [[buffer(6)]], constant uint &ssh [[buffer(7)]],
                        uint3 tg [[threadgroup_position_in_grid]]) {
    uint kt = tg.x, rt = tg.y, gh = tg.z, r0 = gh * 6 * T + rt * 64;
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tQ(Q8 + r0 * 256, dextents<int32_t, 2>(256, 64));
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tK(K8 + gh * kst.x + kt * 64, dextents<int32_t, 2>(64, 256),
                                                                  array<int32_t, 2>{1, int(kst.y)});
    constexpr auto d = matmul2d_descriptor(64, 64, 256, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    auto cI = op.template get_destination_cooperative_tensor<decltype(tQ), decltype(tK), int32_t>();
    op.run(tQ, tK, cI);
    device const half *ks = KS + gh * ssh + kt * 64;
    for (uint16_t i = 0; i < cI.get_capacity(); ++i) {
        if (!cI.is_valid_element(i)) continue;
        auto ix = cI.get_multidimensional_index(i);
        uint key = ix[0], row = ix[1];
        S[(r0 + row) * TK + kt * 64 + key] = half(float(cI[i]) * QS[r0 + row] * float(ks[key]));
    }
}
// softmax over a tile's (already scaled) scores; P = exp(s - m); stats (m, sum) as attn_softmax8 (no per-key scales)
kernel void attn_softmax8n(device half *S [[buffer(0)]], device float *st [[buffer(1)]], constant uint &TK [[buffer(2)]],
                           constant uint &a [[buffer(3)]], constant uint &n [[buffer(4)]], constant uint &p0 [[buffer(5)]],
                           uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                           uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[8];
    device half *row = S + r * TK;
    int lim = int(p0 + (r % T) + 1) - int(a);
    uint nv = uint(clamp(lim, 0, int(n)));
    float m = -INFINITY;
    for (uint j = tid; j < nv; j += 256) m = max(m, float(row[j]));
    m = simd_max(m);
    if (sl == 0) part[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = part[0];
    for (uint i = 1; i < 8; i++) m = max(m, part[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0;
    for (uint j = tid; j < n; j += 256) {
        float e = j < nv ? exp(float(row[j]) - m) : 0.0f;
        row[j] = half(e); s += e;
    }
    s = simd_sum(s);
    if (sl == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float tot = 0; for (uint i = 0; i < 8; i++) tot += part[i]; st[2 * r] = m; st[2 * r + 1] = tot; }
}
// O tile (64 rows x 128 dims) = P (64 rows x n keys) x V, V dequantized on chip per 64-key chunk (INT8 code x the
// key's value scale -> fp16 threadgroup tile). V8 / VS bound at the tile's first key (V8: head stride vst.x, row
// stride vst.y bytes; VS: head stride ssh halves). Grid (2, 6T / 64, 4), 128 threads.
kernel void attn_pv8d(device half *P [[buffer(0)]], device int8_t *V8 [[buffer(1)]], device float *O [[buffer(2)]],
                      constant uint2 &vst [[buffer(3)]], constant uint &TK [[buffer(4)]], constant uint &n [[buffer(5)]],
                      device const half *VS [[buffer(6)]], constant uint &ssh [[buffer(7)]],
                      uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half Vt[64 * 128];
    uint dt = tg.x, rt = tg.y, gh = tg.z, r0 = gh * 6 * T + rt * 64;
    device const int8_t *vb = V8 + gh * vst.x + dt * 128;
    device const half *vsb = VS + gh * ssh;
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tV(Vt, dextents<int32_t, 2>(128, 64));
    constexpr auto d = matmul2d_descriptor(64, 128, 64, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<d, execution_simdgroups<4>> op;
    using TPt = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    auto cO = op.template get_destination_cooperative_tensor<TPt, decltype(tV), float>();
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) if (cO.is_valid_element(i)) cO[i] = 0;
    for (uint k0 = 0; k0 < n; k0 += 64) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 64 * 128 / 4; i += 128) {
            uint key = (i * 4) / 128, c = (i * 4) % 128;
            char4 v4 = *(device const char4 *)(vb + (k0 + key) * vst.y + c);
            float sc = float(vsb[k0 + key]);
            Vt[key * 128 + c] = half(float(v4.x) * sc); Vt[key * 128 + c + 1] = half(float(v4.y) * sc);
            Vt[key * 128 + c + 2] = half(float(v4.z) * sc); Vt[key * 128 + c + 3] = half(float(v4.w) * sc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        TPt tP(P + r0 * TK + k0, dextents<int32_t, 2>(64, 64), array<int32_t, 2>{1, int(TK)});
        op.run(tP, tV, cO);
    }
    for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
        if (!cO.is_valid_element(i)) continue;
        auto ix = cO.get_multidimensional_index(i);
        O[(r0 + ix[1]) * 256 + dt * 128 + ix[0]] = cO[i];
    }
}

// ---- PV over all 256 dims per threadgroup (patch_pvw.py, PVWW=1): P read once ----
kernel void attn_pv_ww(device half *P [[buffer(0)]], device half *Vc [[buffer(1)]], device float *O [[buffer(2)]],
                       constant uint &Lmax [[buffer(3)]], constant uint &a [[buffer(4)]], constant uint &TK [[buffer(5)]],
                       constant uint &n [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]]) {
    uint rt = tg.y, gh = tg.z;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tP(P + (gh * 6 * T + rt * 64) * TK, dextents<int32_t, 2>(int(n), 64),
                                                                array<int32_t, 2>{1, int(TK)});
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tV(Vc + (gh * Lmax + a) * 256, dextents<int32_t, 2>(256, int(n)),
                                                                array<int32_t, 2>{1, 256});
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tO(O + (gh * 6 * T + rt * 64) * 256, dextents<int32_t, 2>(256, 64),
                                                                 array<int32_t, 2>{1, 256});
    constexpr auto d = matmul2d_descriptor(64, 256, static_cast<int>(dynamic_extent), false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<8>> op;
    op.run(tP, tV, tO);
}
kernel void attn_pv8w(device half *P [[buffer(0)]], device int8_t *V8 [[buffer(1)]], device float *O [[buffer(2)]],
                      constant uint2 &vst [[buffer(3)]], constant uint &TK [[buffer(4)]], constant uint &n [[buffer(5)]],
                      uint3 tg [[threadgroup_position_in_grid]]) {
    uint rt = tg.y, gh = tg.z;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tP(P + (gh * 6 * T + rt * 64) * TK, dextents<int32_t, 2>(int(n), 64),
                                                                array<int32_t, 2>{1, int(TK)});
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tV(V8 + gh * vst.x, dextents<int32_t, 2>(256, int(n)),
                                                                  array<int32_t, 2>{1, int(vst.y)});
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tO(O + (gh * 6 * T + rt * 64) * 256, dextents<int32_t, 2>(256, 64),
                                                                 array<int32_t, 2>{1, 256});
    constexpr auto d = matmul2d_descriptor(64, 256, static_cast<int>(dynamic_extent), false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<8>> op;
    op.run(tP, tV, tO);
}

// ---- valid rows in a pair call (patch_nv.py): copies of gdn_gates / gdn_conv_qkv for the prompt's last pair ----
kernel void gdn_gates_nv(device const float *ab [[buffer(0)]], device const float *dt [[buffer(1)]], device const float *nega [[buffer(2)]],
                      device float *beta [[buffer(3)]], device float *dec [[buffer(4)]], constant uint &nv [[buffer(5)]],
                         uint2 p [[thread_position_in_grid]]) {
    uint h = p.x, t = p.y;                                 // ab: rows 0..63 = b (48 used), 64..127 = a (48 used)
    if (t >= nv) { beta[h * T + t] = 0.0f; dec[h * T + t] = 1.0f; return; }   // a pad row: the state stays as it is
    float b = ab[h * T + t], a = ab[(64 + h) * T + t] + dt[h];
    float sp = a > 20.0f ? a : log(1.0f + exp(a));
    beta[h * T + t] = 1.0f / (1.0f + exp(-b));
    dec[h * T + t] = exp(sp * nega[h]);
}
kernel void gdn_conv_qkv_nv(device const float *qkv [[buffer(0)]], device float *cs [[buffer(1)]], device const float *cw [[buffer(2)]],
                         device float *Q [[buffer(3)]], device float *Kh [[buffer(4)]], device float *V [[buffer(5)]],
                         constant uint &nv [[buffer(6)]],
                         uint2 g [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]],
                         uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint CDIM = 10240, TB = 16;
    threadgroup float part[TB][4];
    uint h = g.x, t0 = g.y * TB, c = h * 128 + d;
    float r[TB + 3], cv[TB];
    for (uint j = 0; j < 3; j++) r[j] = t0 == 0 ? cs[j * CDIM + c] : qkv[c * T + t0 - 3 + j];
    for (uint t = 0; t < TB; t++) r[t + 3] = qkv[c * T + t0 + t];
    float w0 = cw[c], w1 = cw[CDIM + c], w2 = cw[2 * CDIM + c], w3 = cw[3 * CDIM + c];
    for (uint t = 0; t < TB; t++) cv[t] = silu_f(r[t] * w0 + r[t + 1] * w1 + r[t + 2] * w2 + r[t + 3] * w3);
    if (t0 == 0) {   // the history after the last valid row (nv < 3: partly the old history, r[0..2])
        float nc[3];
        for (uint j = 0; j < 3; j++) { int ix = int(nv) - 3 + int(j); nc[j] = ix >= 0 ? qkv[c * T + uint(ix)] : r[3 + ix]; }
        for (uint j = 0; j < 3; j++) cs[j * CDIM + c] = nc[j];
    }
    if (h < 32) {
        for (uint t = 0; t < TB; t++) { float s = simd_sum(cv[t] * cv[t]); if (sl == 0) part[t][sg] = s; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint t = 0; t < TB; t++) {
            float rn = rsqrt(part[t][0] + part[t][1] + part[t][2] + part[t][3] + 1e-6f);
            if (h < 16) Q[(h * T + t0 + t) * 128 + d] = cv[t] * rn * rsqrt(128.0f);
            else Kh[((h - 16) * T + t0 + t) * 128 + d] = cv[t] * rn;
        }
    } else {
        for (uint t = 0; t < TB; t++) V[((h - 32) * T + t0 + t) * 128 + d] = cv[t];
    }
}

// ---- DeltaNet state <-> the bridge's arrays on the GPU (patch_state_gpu.py) ----
// n = (conv elements 11 x 10240, rec 48 x 128 x 128, pend elements); one thread per output element
kernel void gdn_state_out(device const float *S [[buffer(0)]], device const float *CS [[buffer(1)]], device half *conv [[buffer(2)]],
                          device half *rec [[buffer(3)]], device half *pend [[buffer(4)]], constant uint4 &n [[buffer(5)]],
                          uint i [[thread_position_in_grid]]) {
    if (i < n.x) { conv[i] = i < 3 * 10240 ? half(CS[i]) : half(0.0f); return; }
    uint j = i - n.x;
    if (j < n.y) { uint h = j / 16384, kk = (j / 128) % 128, v = j % 128; rec[j] = half(64.0f * S[(h * 128 + v) * 128 + kk]); return; }
    j -= n.y;
    if (j < n.z) pend[j] = half(0.0f);
}
kernel void gdn_state_in(device float *S [[buffer(0)]], device float *CS [[buffer(1)]], device const half *conv [[buffer(2)]],
                         device const half *rec [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < 3 * 10240) { CS[i] = float(conv[i]); return; }
    uint j = i - 3 * 10240;
    if (j < 48 * 16384) { uint h = j / 16384, v = (j / 128) % 128, kk = j % 128; S[j] = float(rec[(h * 128 + kk) * 128 + v]) / 64.0f; }
}
