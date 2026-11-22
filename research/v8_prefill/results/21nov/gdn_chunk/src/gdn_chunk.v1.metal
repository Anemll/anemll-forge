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
// accumulate; the state stays fp32. One threadgroup per value head; scratch in device memory (threadgroup memory
// holds X only). sh: 7 x GC x 128 halves per head, sf: GC x GC + GC x 128 floats per head.
#ifndef GC
#define GC 64
#endif
kernel void gdn_chunk(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]], device const float *v [[buffer(2)]],
                      device const float *g [[buffer(3)]], device const float *beta [[buffer(4)]], device float *S [[buffer(5)]],
                      device float *o [[buffer(6)]], device half *sh [[buffer(7)]], device float *sf [[buffer(8)]],
                      uint h [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    constexpr int C = GC, D = 128;
    threadgroup float L[C], Bt[C], Xm[C * C], row[C];
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
                P[t * C + s] = s <= t ? exp(L[t] - L[s]) * cP[i] : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        // X = (I + A)^-1, in place: row t = e_t - sum_{s<t} A[t][s] X[s]; X[s][s] = 1 (stored 0 until the end)
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
        {   // W = X Kb (fp16 out), U~ = X Vb (fp32 out)
            tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tX(Xm, dextents<int32_t, 2>(C, C));
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
            tensor<device float, dextents<int32_t, 2>, tensor_inline> tS(Sh, dextents<int32_t, 2>(D, D));
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
            tensor<device float, dextents<int32_t, 2>, tensor_inline> tP(P, dextents<int32_t, 2>(C, C));
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
