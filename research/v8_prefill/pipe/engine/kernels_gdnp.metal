// ---- DeltaNet projection GEMM prototypes (queue 9, 19 Nov 2026; appended after kernels.metal by gdn_gemm_proto) ----
// lut_gemm32_i8t_k256lr: lut_gemm32_i8t_k256 (TM 32, KS 256, single buffer) with the low-rank correction folded in:
// after the INT8 K loop, the threadgroup gathers its 32 output rows of A (lra, [cout][64] fp16, rows in the kernel's
// channel order via tab) and runs A_tile x t (t = B x, [64][T] fp16, from dense_splitk / splitk_reduce) as one more
// matmul2d (fp16, fp32 accumulate) into a threadgroup [32][T] fp32 tile; the epilogue writes LUT + low-rank in one store
// (replaces dense_gemm, which re-read and re-wrote y). T = 128 only (threadgroup memory: 8 + 1.5 + 4 + 16 KB).
template <uint TM, uint KS>
inline void lut_i8t_lr_body(device const uchar *k, device const uint *tab, device int8_t *x8, device float *y, float fmul,
                            device const float *xs, device const half *A, device const half *tt, uint tg, uint tid,
                            threadgroup int8_t *W, threadgroup int8_t *PAL8, threadgroup ushort *PALP, threadgroup float *PSC,
                            threadgroup half *Alr, threadgroup float *L) {
    const uint SPT = TM / (CD * GSUB), NP = 1u << BITS, PV = NP * CD;
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
    if (CD == 2) for (uint i = tid; i < SPT * NP; i += 128) {
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
    for (uint k0 = 0; k0 < CIN; k0 += KS) {
        i8t_decode<TM, KS>(k, tab, sub0, k0, tid, W, PAL8, PALP);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tB(x8 + k0 * T, dextents<int32_t, 2>(T, KS));
        op.run(tA, tB, cI);
        if ((k0 + KS) % 1024 == 0) {
            for (uint16_t i = 0; i < cI.get_capacity(); ++i) {
                if (!cI.is_valid_element(i)) continue;
                auto ix2 = cI.get_multidimensional_index(i);
                cF[i] += float(cI[i]) * xs[(k0 / 1024) * T + ix2[0]]; cI[i] = 0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // low-rank: gather this threadgroup's A rows (local channel chl -> global CD * tab row + q), A_tile x t -> L
    for (uint i = tid; i < TM * 16; i += 128) {
        uint chl = i / 16, r4 = (i % 16) * 4, ss = chl / (CD * GSUB), q = chl % (CD * GSUB);
        uint o = CD * tab[3 * (sub0 + ss) + 2] + q;
        *(threadgroup half4 *)(Alr + chl * 64 + r4) = *(device const half4 *)(A + o * 64 + r4);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tAl(Alr, dextents<int32_t, 2>(64, TM));
        tensor<device half, dextents<int32_t, 2>, tensor_inline> tT((device half *)tt, dextents<int32_t, 2>(T, 64));
        constexpr auto dl = matmul2d_descriptor(TM, T, 64, false, false, false, matmul2d_descriptor::mode::multiply);
        matmul2d<dl, execution_simdgroups<4>> opl;
        auto cL = opl.template get_destination_cooperative_tensor<decltype(tAl), decltype(tT), float>();
        opl.run(tAl, tT, cL);
        tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tL(L, dextents<int32_t, 2>(T, TM));
        cL.store(tL);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint16_t i = 0; i < cF.get_capacity(); ++i) {
        if (!cF.is_valid_element(i)) continue;
        auto ix2 = cF.get_multidimensional_index(i);
        uint t = ix2[0], chl = ix2[1], ss = chl / (CD * GSUB), q = chl % (CD * GSUB);
        uint base = tab[3 * (sub0 + ss) + 1];
        float v = cF[i] * PSC[ss] * float(((device const half *)(k + base))[q]) * fmul;
        uint yi = (CD * tab[3 * (sub0 + ss) + 2] + q) * T + t;
        y[yi] = (ACC ? y[yi] : 0.0f) + v + L[chl * T + t];
    }
}
#if T <= 128
kernel void lut_gemm32_i8t_k256lr(device const uchar *k [[buffer(0)]], device const uint *tab [[buffer(1)]], device int8_t *x8 [[buffer(2)]],
                                  device float *y [[buffer(3)]], constant float &fmul [[buffer(4)]], device const float *xs [[buffer(5)]],
                                  device const half *A [[buffer(6)]], device const half *tt [[buffer(7)]],
                                  uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup int8_t W[256 * 32] __attribute__((aligned(16))), PAL8[8 * 128];
    threadgroup ushort PALP[256]; threadgroup float PSC[8];
    threadgroup half Alr[32 * 64] __attribute__((aligned(16)));
    threadgroup float L[32 * T] __attribute__((aligned(16)));
    lut_i8t_lr_body<32, 256>(k, tab, x8, y, fmul, xs, A, tt, tg, tid, W, PAL8, PALP, PSC, Alr, L);
}
#endif

// lut_gemm32_i8t_k256ws: warp-specialized hidden decode. 256 threads: simdgroups 0-3 run the INT8 matmul on chunk c
// (double-buffered W) while simdgroups 4-7 decode chunk c + 1 into the other buffer; one barrier per chunk. Same values
// and order as lut_gemm32_i8t_k256 (bit-same outputs expected).
kernel void lut_gemm32_i8t_k256ws(device const uchar *k [[buffer(0)]], device const uint *tab [[buffer(1)]], device int8_t *x8 [[buffer(2)]],
                                  device float *y [[buffer(3)]], constant float &fmul [[buffer(4)]], device const float *xs [[buffer(5)]],
                                  uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint TM = 32, KS = 256;
    threadgroup int8_t W[2 * KS * TM] __attribute__((aligned(16))), PAL8[8 * 128];
    threadgroup ushort PALP[256]; threadgroup float PSC[8];
    const uint SPT = TM / (CD * GSUB), NP = 1u << BITS, PV = NP * CD;
    uint sub0 = tg * SPT;
    bool prod = tid >= 128; uint pt = tid - 128;
    if (tid < SPT) {
        device const half *pal = (device const half *)(k + tab[3 * (sub0 + tid)]);
        float m = 0;
        for (uint i = 0; i < PV; i++) m = max(m, abs(float(pal[i])));
        PSC[tid] = m > 0 ? m / 127.0f : 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < SPT * PV; i += 256) {
        uint s = i / PV;
        PAL8[s * 128 + i % PV] = int8_t(rint(float(((device const half *)(k + tab[3 * (sub0 + s)]))[i % PV]) / PSC[s]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (CD == 2) for (uint i = tid; i < SPT * NP; i += 256) {
        uint s = i / NP, ix = i % NP;
        PALP[i] = ushort(uchar(PAL8[s * 128 + 2 * ix])) | ushort(ushort(uchar(PAL8[s * 128 + 2 * ix + 1])) << 8);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (prod) i8t_decode<TM, KS>(k, tab, sub0, 0, pt, W, PAL8, PALP);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr auto d = matmul2d_descriptor(TM, T, KS, true, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<d, execution_simdgroups<4>> op;
    using TA = tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline>;
    auto cI = op.template get_destination_cooperative_tensor<TA, tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>, int32_t>();
    auto cF = op.template get_destination_cooperative_tensor<TA, tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>, float>();
    if (!prod) for (uint16_t i = 0; i < cF.get_capacity(); ++i) if (cF.is_valid_element(i)) { cF[i] = 0; cI[i] = 0; }
    for (uint k0 = 0, st = 0; k0 < CIN; k0 += KS, st++) {
        if (!prod) {
            TA tAc(W + (st & 1) * (TM * KS), dextents<int32_t, 2>(TM, KS));
            tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tB(x8 + k0 * T, dextents<int32_t, 2>(T, KS));
            op.run(tAc, tB, cI);
            if ((k0 + KS) % 1024 == 0) {
                for (uint16_t i = 0; i < cI.get_capacity(); ++i) {
                    if (!cI.is_valid_element(i)) continue;
                    auto ix2 = cI.get_multidimensional_index(i);
                    cF[i] += float(cI[i]) * xs[(k0 / 1024) * T + ix2[0]]; cI[i] = 0;
                }
            }
        } else if (k0 + KS < CIN) {
            i8t_decode<TM, KS>(k, tab, sub0, k0 + KS, pt, W + ((st + 1) & 1) * (TM * KS), PAL8, PALP);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (prod) return;
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
