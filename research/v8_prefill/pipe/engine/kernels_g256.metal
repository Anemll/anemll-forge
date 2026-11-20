// ---- 256-row DeltaNet calls (queue 9b, 19 Nov 2026; appended by gpu_v8_g256.swift, compiled with T = 256) ----
// Four 64-row ANE blocks <-> the mixer's [C][T] fp32 activations: block j = columns 64 j .. 64 j + 63.
kernel void from_half_quad(device const half *a [[buffer(0)]], device const half *b [[buffer(1)]], device const half *c [[buffer(2)]],
                           device const half *d [[buffer(3)]], device float *x [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    const uint q = T / 4, ch = i / T, t = i % T, j = t / q, r = t % q;
    device const half *s = j == 0 ? a : j == 1 ? b : j == 2 ? c : d;
    x[i] = float(s[ch * q + r]);
}
kernel void to_half_quad(device const float *x [[buffer(0)]], device half *a [[buffer(1)]], device half *b [[buffer(2)]],
                         device half *c [[buffer(3)]], device half *d [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    const uint q = T / 4, ch = i / T, t = i % T, j = t / q, r = t % q;
    device half *o = j == 0 ? a : j == 1 ? b : j == 2 ? c : d;
    o[ch * q + r] = half(x[i]);
}
kernel void to_half_quad_add(device const float *x [[buffer(0)]], device half *a [[buffer(1)]], device half *b [[buffer(2)]],
                             device half *c [[buffer(3)]], device half *d [[buffer(4)]], device const float *y [[buffer(5)]],
                             uint i [[thread_position_in_grid]]) {
    const uint q = T / 4, ch = i / T, t = i % T, j = t / q, r = t % q;
    device half *o = j == 0 ? a : j == 1 ? b : j == 2 ? c : d;
    o[ch * q + r] = half(x[i] + y[i]);
}
// rms_partial at T = 256 with the T = 128 summation order (rms_partial: NG = 256 / T interleaved partial sums, NG = 2
// at T = 128, then added in order): one thread per token, the even and the odd channels of the 320-channel chunk summed
// separately and added as (0 + even) + odd, so the RMS scales (and the INT8 activations after them) are bit-identical.
kernel void rms_partial_g256(device const float *x [[buffer(0)]], device float *part [[buffer(1)]], uint ch [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]]) {
    uint t = tid;
    float s0 = 0, s1 = 0;
    for (uint c = ch * 320; c < ch * 320 + 320; c += 2) { float v = x[c * T + t]; s0 += v * v; }
    for (uint c = ch * 320 + 1; c < ch * 320 + 320; c += 2) { float v = x[c * T + t]; s1 += v * v; }
    float a = 0; a += s0; a += s1;
    part[ch * T + t] = a;
}
