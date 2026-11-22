// ---- 21 Nov 2026 (M6 ANE bot, item 4): chunked (WY / UT) gated delta rule, prototype; appended to the served
// kernels.metal (T = 128) by gdn_chunk_proto.swift. Same buffers and layouts as gdn_rec:
//   q, k [16 key heads][T][128] fp32 (q l2-normed x 128^-1/2, k l2-normed), v / o [48][T][128], g (decay, already exp)
//   and beta [48][T], S [48][dv 128][dk 128] fp32 in place; value head h reads key head h / 3.
// Per chunk of GC tokens (Gamma_t = prod_{s<=t} g_s, L = log Gamma):
//   A[t][s] = beta_t e^(L_t - L_s) k_t.k_s (s < t),  X = (I + A)^-1 (fp32 forward substitution)
//   W = X diag(beta Gamma) K,  U~ = X diag(beta) V,  U = U~ - W S0^T
//   O = diag(Gamma) Q S0^T + (P o U),  P[t][s] = e^(L_t - L_s) q_t.k_s (s <= t)
//   S = Gamma_end S0 + U^T diag(e^(L_end - L_s)) K
// Matrix products on the GPU matrix units (matmul2d, 4 simdgroups): fp16 operands except S0, P and X (fp32), fp32
// accumulate; the state stays fp32. Variants (macros): S0H fp16 copy of S0 for the two S0 products, XH X in fp16,
// PH P in fp16, NOSOLVE (timing diagnostic only, wrong output) skips the triangular solve. One threadgroup per value head; scratch in device memory (threadgroup memory
// holds X only). sh: 7 x GC x 128 halves per head, sf: GC x GC + GC x 128 floats per head.
#ifndef GC
#define GC 64
#endif
kernel void gdn_chunk(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]], device const float *v [[buffer(2)]],
                      device const float *g [[buffer(3)]], device const float *beta [[buffer(4)]], device float *S [[buffer(5)]],
                      device float *o [[buffer(6)]], device half *sh [[buffer(7)]], device float *sf [[buffer(8)]], device half *s16 [[buffer(9)]],
                      uint h [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    constexpr int C = GC, D = 128;
    threadgroup float L[C], Bt[C], Xm[C * C], row[C];
#ifdef XH
    threadgroup half Xh[C * C];
#endif
#ifdef PH
    device half *Ph = (device half *)(sf + h * (C * C + C * D));
#endif
    device half *S16 = s16 + h * D * D;
    const uint kh = h / 3;
    device half *Kh = sh + h * (7 * C * D), *Qh = Kh + C * D, *Kb = Qh + C * D, *Vb = Kb + C * D, *Kd = Vb + C * D,
                *Wm = Kd + C * D, *U = Wm + C * D;
    device float *P = sf + h * (C * C + C * D), *Ut = P + C * C;
    device float *Sh = S + h * D * D, *oh = o + h * T * D;
    for (uint c0 = 0; c0 < T; c0 += C) {
        if (tid < uint(C)) { L[tid] = log(g[h * T + c0 + tid]); Bt[tid] = beta[h * T + c0 + tid]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) { float s = 0; for (int t = 0; t < C; t++) { s += L[t]; L[t] = s; } }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float Lend = L[C - 1];
        for (uint e = tid; e < uint(C * D); e += 128) {
            uint t = e / D, j = e % D;
            float kk = k[(kh * T + c0 + t) * D + j], qq = q[(kh * T + c0 + t) * D + j], vv = v[(h * T + c0 + t) * D + j];
            Kh[e] = half(kk); Qh[e] = half(qq); Kb[e] = half(Bt[t] * exp(L[t]) * kk); Vb[e] = half(Bt[t] * vv);
            Kd[e] = half(exp(Lend - L[t]) * kk);
        }
#ifdef S0H
        for (uint e = tid; e < uint(D * D); e += 128) S16[e] = half(Sh[e]);
#endif
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        {   // A (into Xm) and P from K K^T and Q K^T
            tensor<device half, dextents<int32_t, 2>, tensor_inline> tK(Kh, dextents<int32_t, 2>(D, C)), tQ(Qh, dextents<int32_t, 2>(D, C));
            constexpr auto d = matmul2d_descriptor(C, C, D, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
            matmul2d<d, execution_simdgroups<4>> op;
            auto cG = op.template get_destination_cooperative_tensor<decltype(tK), decltype(tK), float>();
            for (uint16_t i = 0; i < cG.get_capacity(); ++i) if (cG.is_valid_element(i)) cG[i] = 0;
            op.run(tK, tK, cG);
            for (uint16_t i = 0; i < cG.get_capacity(); ++i) {
                if (!cG.is_valid_element(i)) continue;
                auto ix = cG.get_multidimensional_index(i);
                int s = ix[0], t = ix[1];
                Xm[t * C + s] = s < t ? Bt[t] * exp(L[t] - L[s]) * cG[i] : 0.0f;
            }
            auto cP = op.template get_destination_cooperative_tensor<decltype(tQ), decltype(tK), float>();
            for (uint16_t i = 0; i < cP.get_capacity(); ++i) if (cP.is_valid_element(i)) cP[i] = 0;
            op.run(tQ, tK, cP);
            for (uint16_t i = 0; i < cP.get_capacity(); ++i) {
                if (!cP.is_valid_element(i)) continue;
                auto ix = cP.get_multidimensional_index(i);
                int s = ix[0], t = ix[1];
#ifdef PH
                Ph[t * C + s] = half(s <= t ? exp(L[t] - L[s]) * cP[i] : 0.0f);
#else
                P[t * C + s] = s <= t ? exp(L[t] - L[s]) * cP[i] : 0.0f;
#endif
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        // X = (I + A)^-1, in place: row t = e_t - sum_{s<t} A[t][s] X[s]; X[s][s] = 1 (stored 0 until the end)
#ifndef NOSOLVE
        for (int t = 1; t < C; t++) {
            if (tid < uint(t)) row[tid] = Xm[t * C + tid];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < uint(t)) {
                int c = tid;
                float acc = row[c];
                for (int s = c + 1; s < t; s++) acc += row[s] * Xm[s * C + c];
                Xm[t * C + c] = -acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
#endif
        if (tid < uint(C)) Xm[tid * C + tid] = 1.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
#ifdef XH
        for (uint e = tid; e < uint(C * C); e += 128) Xh[e] = half(Xm[e]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
        {   // W = X Kb (fp16 out), U~ = X Vb (fp32 out)
#ifdef XH
            tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tX(Xh, dextents<int32_t, 2>(C, C));
#else
            tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tX(Xm, dextents<int32_t, 2>(C, C));
#endif
            tensor<device half, dextents<int32_t, 2>, tensor_inline> tKb(Kb, dextents<int32_t, 2>(D, C)), tVb(Vb, dextents<int32_t, 2>(D, C));
            constexpr auto d = matmul2d_descriptor(C, D, C, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
            matmul2d<d, execution_simdgroups<4>> op;
            auto cW = op.template get_destination_cooperative_tensor<decltype(tX), decltype(tKb), float>();
            for (uint16_t i = 0; i < cW.get_capacity(); ++i) if (cW.is_valid_element(i)) cW[i] = 0;
            op.run(tX, tKb, cW);
            for (uint16_t i = 0; i < cW.get_capacity(); ++i) {
                if (!cW.is_valid_element(i)) continue;
                auto ix = cW.get_multidimensional_index(i);
                Wm[ix[1] * D + ix[0]] = half(cW[i]);
            }
            auto cU = op.template get_destination_cooperative_tensor<decltype(tX), decltype(tVb), float>();
            for (uint16_t i = 0; i < cU.get_capacity(); ++i) if (cU.is_valid_element(i)) cU[i] = 0;
            op.run(tX, tVb, cU);
            for (uint16_t i = 0; i < cU.get_capacity(); ++i) {
                if (!cU.is_valid_element(i)) continue;
                auto ix = cU.get_multidimensional_index(i);
                Ut[ix[1] * D + ix[0]] = cU[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
        {   // U = U~ - W S0^T (fp16 out); O = Gamma_t Q S0^T (fp32, into o)
            tensor<device half, dextents<int32_t, 2>, tensor_inline> tW(Wm, dextents<int32_t, 2>(D, C)), tQ(Qh, dextents<int32_t, 2>(D, C));
#ifdef S0H
            tensor<device half, dextents<int32_t, 2>, tensor_inline> tS(S16, dextents<int32_t, 2>(D, D));
#else
            tensor<device float, dextents<int32_t, 2>, tensor_inline> tS(Sh, dextents<int32_t, 2>(D, D));
#endif
            constexpr auto d = matmul2d_descriptor(C, D, D, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
            matmul2d<d, execution_simdgroups<4>> op;
            auto c1 = op.template get_destination_cooperative_tensor<decltype(tW), decltype(tS), float>();
            for (uint16_t i = 0; i < c1.get_capacity(); ++i) if (c1.is_valid_element(i)) c1[i] = 0;
            op.run(tW, tS, c1);
            for (uint16_t i = 0; i < c1.get_capacity(); ++i) {
                if (!c1.is_valid_element(i)) continue;
                auto ix = c1.get_multidimensional_index(i);
                int vi = ix[0], t = ix[1];
                U[t * D + vi] = half(Ut[t * D + vi] - c1[i]);
            }
            auto c2 = op.template get_destination_cooperative_tensor<decltype(tQ), decltype(tS), float>();
            for (uint16_t i = 0; i < c2.get_capacity(); ++i) if (c2.is_valid_element(i)) c2[i] = 0;
            op.run(tQ, tS, c2);
            for (uint16_t i = 0; i < c2.get_capacity(); ++i) {
                if (!c2.is_valid_element(i)) continue;
                auto ix = c2.get_multidimensional_index(i);
                int vi = ix[0], t = ix[1];
                oh[(c0 + t) * D + vi] = exp(L[t]) * c2[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
        {   // O += P U
#ifdef PH
            tensor<device half, dextents<int32_t, 2>, tensor_inline> tP(Ph, dextents<int32_t, 2>(C, C));
#else
            tensor<device float, dextents<int32_t, 2>, tensor_inline> tP(P, dextents<int32_t, 2>(C, C));
#endif
            tensor<device half, dextents<int32_t, 2>, tensor_inline> tU(U, dextents<int32_t, 2>(D, C));
            constexpr auto d = matmul2d_descriptor(C, D, C, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
            matmul2d<d, execution_simdgroups<4>> op;
            auto c = op.template get_destination_cooperative_tensor<decltype(tP), decltype(tU), float>();
            for (uint16_t i = 0; i < c.get_capacity(); ++i) if (c.is_valid_element(i)) c[i] = 0;
            op.run(tP, tU, c);
            for (uint16_t i = 0; i < c.get_capacity(); ++i) {
                if (!c.is_valid_element(i)) continue;
                auto ix = c.get_multidimensional_index(i);
                oh[(c0 + ix[1]) * D + ix[0]] += c[i];
            }
        }
        {   // S = Gamma_end S0 + U^T Kd, in two 64-row halves of dv
            tensor<device half, dextents<int32_t, 2>, tensor_inline> tKd(Kd, dextents<int32_t, 2>(D, C));
            constexpr auto d = matmul2d_descriptor(64, D, C, true, false, false, matmul2d_descriptor::mode::multiply_accumulate);
            matmul2d<d, execution_simdgroups<4>> op;
            const float ge = exp(Lend);
            for (int i0 = 0; i0 < D; i0 += 64) {
                tensor<device half, dextents<int32_t, 2>, tensor_inline> tUT(U + i0, dextents<int32_t, 2>(64, C), array<int32_t, 2>{1, D});
                auto c = op.template get_destination_cooperative_tensor<decltype(tUT), decltype(tKd), float>();
                for (uint16_t i = 0; i < c.get_capacity(); ++i) if (c.is_valid_element(i)) c[i] = 0;
                op.run(tUT, tKd, c);
                for (uint16_t i = 0; i < c.get_capacity(); ++i) {
                    if (!c.is_valid_element(i)) continue;
                    auto ix = c.get_multidimensional_index(i);
                    uint si = (i0 + ix[1]) * D + ix[0];
                    Sh[si] = ge * Sh[si] + c[i];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
}

// ---- gdn_chunk2: the same chunked form with each head's value rows split over NP threadgroups (grid (48, NP)).
// The rows of S are independent given k / q / g / beta, so part p owns value rows [p DS, (p + 1) DS), DS = 128 / NP:
// its slice of U~, U, O and S. The per-head pieces (K K^T, Q K^T, X, W) are recomputed in every part (redundant).
// Scratch per (head, part): sh 5 x GC x 128 + 2 x GC x DS halves, sf GC x GC + GC x DS floats (allocated as for NP 1).
#ifndef NP
#define NP 2
#endif
kernel void gdn_chunk2(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]], device const float *v [[buffer(2)]],
                       device const float *g [[buffer(3)]], device const float *beta [[buffer(4)]], device float *S [[buffer(5)]],
                       device float *o [[buffer(6)]], device half *sh [[buffer(7)]], device float *sf [[buffer(8)]],
                       uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    constexpr int C = GC, D = 128, DS = D / NP;
    threadgroup float L[C], Bt[C], Xm[C * C], row[C];
    const uint h = tgp.x, part = tgp.y, kh = h / 3, i0 = part * DS, id = h * NP + part;
    device half *Kh = sh + id * (7 * C * D), *Qh = Kh + C * D, *Kb = Qh + C * D, *Kd = Kb + C * D, *Wm = Kd + C * D,
                *Vb = Wm + C * D, *U = Vb + C * DS;
    device float *P = sf + id * (C * C + C * D), *Ut = P + C * C;
    device float *Sh = S + h * D * D, *oh = o + h * T * D;
    for (uint c0 = 0; c0 < T; c0 += C) {
        if (tid < uint(C)) { L[tid] = log(g[h * T + c0 + tid]); Bt[tid] = beta[h * T + c0 + tid]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) { float s = 0; for (int t = 0; t < C; t++) { s += L[t]; L[t] = s; } }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float Lend = L[C - 1];
        for (uint e = tid; e < uint(C * D); e += 128) {
            uint t = e / D, j = e % D;
            float kk = k[(kh * T + c0 + t) * D + j], qq = q[(kh * T + c0 + t) * D + j];
            Kh[e] = half(kk); Qh[e] = half(qq); Kb[e] = half(Bt[t] * exp(L[t]) * kk); Kd[e] = half(exp(Lend - L[t]) * kk);
        }
        for (uint e = tid; e < uint(C * DS); e += 128) {
            uint t = e / DS, j = e % DS;
            Vb[e] = half(Bt[t] * v[(h * T + c0 + t) * D + i0 + j]);
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        {
            tensor<device half, dextents<int32_t, 2>, tensor_inline> tK(Kh, dextents<int32_t, 2>(D, C)), tQ(Qh, dextents<int32_t, 2>(D, C));
            constexpr auto d = matmul2d_descriptor(C, C, D, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
            matmul2d<d, execution_simdgroups<4>> op;
            auto cG = op.template get_destination_cooperative_tensor<decltype(tK), decltype(tK), float>();
            for (uint16_t i = 0; i < cG.get_capacity(); ++i) if (cG.is_valid_element(i)) cG[i] = 0;
            op.run(tK, tK, cG);
            for (uint16_t i = 0; i < cG.get_capacity(); ++i) {
                if (!cG.is_valid_element(i)) continue;
                auto ix = cG.get_multidimensional_index(i);
                int s = ix[0], t = ix[1];
                Xm[t * C + s] = s < t ? Bt[t] * exp(L[t] - L[s]) * cG[i] : 0.0f;
            }
            auto cP = op.template get_destination_cooperative_tensor<decltype(tQ), decltype(tK), float>();
            for (uint16_t i = 0; i < cP.get_capacity(); ++i) if (cP.is_valid_element(i)) cP[i] = 0;
            op.run(tQ, tK, cP);
            for (uint16_t i = 0; i < cP.get_capacity(); ++i) {
                if (!cP.is_valid_element(i)) continue;
                auto ix = cP.get_multidimensional_index(i);
                int s = ix[0], t = ix[1];
                P[t * C + s] = s <= t ? exp(L[t] - L[s]) * cP[i] : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        for (int t = 1; t < C; t++) {
            if (tid < uint(t)) row[tid] = Xm[t * C + tid];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < uint(t)) {
                int c = tid;
                float acc = row[c];
                for (int s = c + 1; s < t; s++) acc += row[s] * Xm[s * C + c];
                Xm[t * C + c] = -acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid < uint(C)) Xm[tid * C + tid] = 1.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   // W = X Kb (full), U~ = X Vb (slice)
            tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tX(Xm, dextents<int32_t, 2>(C, C));
            tensor<device half, dextents<int32_t, 2>, tensor_inline> tKb(Kb, dextents<int32_t, 2>(D, C)), tVb(Vb, dextents<int32_t, 2>(DS, C));
            constexpr auto d = matmul2d_descriptor(C, D, C, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
            matmul2d<d, execution_simdgroups<4>> op;
            auto cW = op.template get_destination_cooperative_tensor<decltype(tX), decltype(tKb), float>();
            for (uint16_t i = 0; i < cW.get_capacity(); ++i) if (cW.is_valid_element(i)) cW[i] = 0;
            op.run(tX, tKb, cW);
            for (uint16_t i = 0; i < cW.get_capacity(); ++i) {
                if (!cW.is_valid_element(i)) continue;
                auto ix = cW.get_multidimensional_index(i);
                Wm[ix[1] * D + ix[0]] = half(cW[i]);
            }
            constexpr auto ds = matmul2d_descriptor(C, DS, C, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
            matmul2d<ds, execution_simdgroups<4>> ops;
            auto cU = ops.template get_destination_cooperative_tensor<decltype(tX), decltype(tVb), float>();
            for (uint16_t i = 0; i < cU.get_capacity(); ++i) if (cU.is_valid_element(i)) cU[i] = 0;
            ops.run(tX, tVb, cU);
            for (uint16_t i = 0; i < cU.get_capacity(); ++i) {
                if (!cU.is_valid_element(i)) continue;
                auto ix = cU.get_multidimensional_index(i);
                Ut[ix[1] * DS + ix[0]] = cU[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
        {   // U = U~ - W S0[slice]^T ; O[slice] = Gamma Q S0[slice]^T
            tensor<device half, dextents<int32_t, 2>, tensor_inline> tW(Wm, dextents<int32_t, 2>(D, C)), tQ(Qh, dextents<int32_t, 2>(D, C));
            tensor<device float, dextents<int32_t, 2>, tensor_inline> tS(Sh + i0 * D, dextents<int32_t, 2>(D, DS));
            constexpr auto d = matmul2d_descriptor(C, DS, D, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
            matmul2d<d, execution_simdgroups<4>> op;
            auto c1 = op.template get_destination_cooperative_tensor<decltype(tW), decltype(tS), float>();
            for (uint16_t i = 0; i < c1.get_capacity(); ++i) if (c1.is_valid_element(i)) c1[i] = 0;
            op.run(tW, tS, c1);
            for (uint16_t i = 0; i < c1.get_capacity(); ++i) {
                if (!c1.is_valid_element(i)) continue;
                auto ix = c1.get_multidimensional_index(i);
                int vi = ix[0], t = ix[1];
                U[t * DS + vi] = half(Ut[t * DS + vi] - c1[i]);
            }
            auto c2 = op.template get_destination_cooperative_tensor<decltype(tQ), decltype(tS), float>();
            for (uint16_t i = 0; i < c2.get_capacity(); ++i) if (c2.is_valid_element(i)) c2[i] = 0;
            op.run(tQ, tS, c2);
            for (uint16_t i = 0; i < c2.get_capacity(); ++i) {
                if (!c2.is_valid_element(i)) continue;
                auto ix = c2.get_multidimensional_index(i);
                oh[(c0 + ix[1]) * D + i0 + ix[0]] = exp(L[ix[1]]) * c2[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
        {   // O[slice] += P U ; S[slice] = Gamma_end S0[slice] + U^T Kd
            tensor<device float, dextents<int32_t, 2>, tensor_inline> tP(P, dextents<int32_t, 2>(C, C));
            tensor<device half, dextents<int32_t, 2>, tensor_inline> tU(U, dextents<int32_t, 2>(DS, C)), tKd(Kd, dextents<int32_t, 2>(D, C));
            constexpr auto d = matmul2d_descriptor(C, DS, C, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
            matmul2d<d, execution_simdgroups<4>> op;
            auto c = op.template get_destination_cooperative_tensor<decltype(tP), decltype(tU), float>();
            for (uint16_t i = 0; i < c.get_capacity(); ++i) if (c.is_valid_element(i)) c[i] = 0;
            op.run(tP, tU, c);
            for (uint16_t i = 0; i < c.get_capacity(); ++i) {
                if (!c.is_valid_element(i)) continue;
                auto ix = c.get_multidimensional_index(i);
                oh[(c0 + ix[1]) * D + i0 + ix[0]] += c[i];
            }
            constexpr auto d2 = matmul2d_descriptor(DS, D, C, true, false, false, matmul2d_descriptor::mode::multiply_accumulate);
            matmul2d<d2, execution_simdgroups<4>> op2;
            const float ge = exp(Lend);
            auto cs = op2.template get_destination_cooperative_tensor<decltype(tU), decltype(tKd), float>();
            for (uint16_t i = 0; i < cs.get_capacity(); ++i) if (cs.is_valid_element(i)) cs[i] = 0;
            op2.run(tU, tKd, cs);
            for (uint16_t i = 0; i < cs.get_capacity(); ++i) {
                if (!cs.is_valid_element(i)) continue;
                auto ix = cs.get_multidimensional_index(i);
                uint si = (i0 + ix[1]) * D + ix[0];
                Sh[si] = ge * Sh[si] + cs[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
}

// ---- gdn_chunk3: gdn_chunk with stacked operands, 5 tensor ops per chunk instead of 9 (one threadgroup per head):
//   [K; Q] K^T (2C x C) -> A and P;  X [Kb | Vb] (C x 2D) -> W and U~;  [Q; W] S0^T (2C x D) -> O and U;
//   P U -> O;  U^T Kd (D x D) -> S.   Scratch per head: sh = Kh, Qh, Wm (C x D each, contiguous), KV (C x 2D), Kd, U.
kernel void gdn_chunk3(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]], device const float *v [[buffer(2)]],
                       device const float *g [[buffer(3)]], device const float *beta [[buffer(4)]], device float *S [[buffer(5)]],
                       device float *o [[buffer(6)]], device half *sh [[buffer(7)]], device float *sf [[buffer(8)]],
                       uint h [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    constexpr int C = GC, D = 128;
    threadgroup float L[C], Bt[C], Xm[C * C], row[C];
    const uint kh = h / 3;
    device half *Kh = sh + h * (7 * C * D), *Qh = Kh + C * D, *Wm = Qh + C * D, *KV = Wm + C * D, *Kd = KV + 2 * C * D, *U = Kd + C * D;
    device float *P = sf + h * (C * C + C * D), *Ut = P + C * C;
    device float *Sh = S + h * D * D, *oh = o + h * T * D;
    for (uint c0 = 0; c0 < T; c0 += C) {
        if (tid < uint(C)) { L[tid] = log(g[h * T + c0 + tid]); Bt[tid] = beta[h * T + c0 + tid]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) { float s = 0; for (int t = 0; t < C; t++) { s += L[t]; L[t] = s; } }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float Lend = L[C - 1];
        for (uint e = tid; e < uint(C * D); e += 128) {
            uint t = e / D, j = e % D;
            float kk = k[(kh * T + c0 + t) * D + j], qq = q[(kh * T + c0 + t) * D + j], vv = v[(h * T + c0 + t) * D + j];
            Kh[e] = half(kk); Qh[e] = half(qq); KV[t * 2 * D + j] = half(Bt[t] * exp(L[t]) * kk); KV[t * 2 * D + D + j] = half(Bt[t] * vv);
            Kd[e] = half(exp(Lend - L[t]) * kk);
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        {   // [K; Q] K^T
            tensor<device half, dextents<int32_t, 2>, tensor_inline> tKQ(Kh, dextents<int32_t, 2>(D, 2 * C)), tK(Kh, dextents<int32_t, 2>(D, C));
            constexpr auto d = matmul2d_descriptor(2 * C, C, D, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
            matmul2d<d, execution_simdgroups<4>> op;
            auto c = op.template get_destination_cooperative_tensor<decltype(tKQ), decltype(tK), float>();
            for (uint16_t i = 0; i < c.get_capacity(); ++i) if (c.is_valid_element(i)) c[i] = 0;
            op.run(tKQ, tK, c);
            for (uint16_t i = 0; i < c.get_capacity(); ++i) {
                if (!c.is_valid_element(i)) continue;
                auto ix = c.get_multidimensional_index(i);
                int s = ix[0], r = ix[1];
                if (r < C) Xm[r * C + s] = s < r ? Bt[r] * exp(L[r] - L[s]) * c[i] : 0.0f;
                else { int t = r - C; P[t * C + s] = s <= t ? exp(L[t] - L[s]) * c[i] : 0.0f; }
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        for (int t = 1; t < C; t++) {
            if (tid < uint(t)) row[tid] = Xm[t * C + tid];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < uint(t)) {
                int c = tid;
                float acc = row[c];
                for (int s = c + 1; s < t; s++) acc += row[s] * Xm[s * C + c];
                Xm[t * C + c] = -acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid < uint(C)) Xm[tid * C + tid] = 1.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   // X [Kb | Vb] -> W (half), U~ (float)
            tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tX(Xm, dextents<int32_t, 2>(C, C));
            tensor<device half, dextents<int32_t, 2>, tensor_inline> tKV(KV, dextents<int32_t, 2>(2 * D, C));
            constexpr auto d = matmul2d_descriptor(C, 2 * D, C, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
            matmul2d<d, execution_simdgroups<4>> op;
            auto c = op.template get_destination_cooperative_tensor<decltype(tX), decltype(tKV), float>();
            for (uint16_t i = 0; i < c.get_capacity(); ++i) if (c.is_valid_element(i)) c[i] = 0;
            op.run(tX, tKV, c);
            for (uint16_t i = 0; i < c.get_capacity(); ++i) {
                if (!c.is_valid_element(i)) continue;
                auto ix = c.get_multidimensional_index(i);
                int col = ix[0], t = ix[1];
                if (col < D) Wm[t * D + col] = half(c[i]); else Ut[t * D + col - D] = c[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
        {   // [Q; W] S0^T -> O (rows < C), U (rows >= C)
            tensor<device half, dextents<int32_t, 2>, tensor_inline> tQW(Qh, dextents<int32_t, 2>(D, 2 * C));
            tensor<device float, dextents<int32_t, 2>, tensor_inline> tS(Sh, dextents<int32_t, 2>(D, D));
            constexpr auto d = matmul2d_descriptor(2 * C, D, D, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
            matmul2d<d, execution_simdgroups<4>> op;
            auto c = op.template get_destination_cooperative_tensor<decltype(tQW), decltype(tS), float>();
            for (uint16_t i = 0; i < c.get_capacity(); ++i) if (c.is_valid_element(i)) c[i] = 0;
            op.run(tQW, tS, c);
            for (uint16_t i = 0; i < c.get_capacity(); ++i) {
                if (!c.is_valid_element(i)) continue;
                auto ix = c.get_multidimensional_index(i);
                int vi = ix[0], r = ix[1];
                if (r < C) oh[(c0 + r) * D + vi] = exp(L[r]) * c[i];
                else { int t = r - C; U[t * D + vi] = half(Ut[t * D + vi] - c[i]); }
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
        {   // O += P U ; S = Gamma_end S0 + U^T Kd
            tensor<device float, dextents<int32_t, 2>, tensor_inline> tP(P, dextents<int32_t, 2>(C, C));
            tensor<device half, dextents<int32_t, 2>, tensor_inline> tU(U, dextents<int32_t, 2>(D, C)), tKd(Kd, dextents<int32_t, 2>(D, C));
            constexpr auto d = matmul2d_descriptor(C, D, C, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
            matmul2d<d, execution_simdgroups<4>> op;
            auto c = op.template get_destination_cooperative_tensor<decltype(tP), decltype(tU), float>();
            for (uint16_t i = 0; i < c.get_capacity(); ++i) if (c.is_valid_element(i)) c[i] = 0;
            op.run(tP, tU, c);
            for (uint16_t i = 0; i < c.get_capacity(); ++i) {
                if (!c.is_valid_element(i)) continue;
                auto ix = c.get_multidimensional_index(i);
                oh[(c0 + ix[1]) * D + ix[0]] += c[i];
            }
            constexpr auto d2 = matmul2d_descriptor(D, D, C, true, false, false, matmul2d_descriptor::mode::multiply_accumulate);
            matmul2d<d2, execution_simdgroups<4>> op2;
            const float ge = exp(Lend);
            auto cs = op2.template get_destination_cooperative_tensor<decltype(tU), decltype(tKd), float>();
            for (uint16_t i = 0; i < cs.get_capacity(); ++i) if (cs.is_valid_element(i)) cs[i] = 0;
            op2.run(tU, tKd, cs);
            for (uint16_t i = 0; i < cs.get_capacity(); ++i) {
                if (!cs.is_valid_element(i)) continue;
                auto ix = cs.get_multidimensional_index(i);
                uint si = ix[1] * D + ix[0];
                Sh[si] = ge * Sh[si] + cs[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
}
