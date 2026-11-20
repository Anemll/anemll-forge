// GPU load generator for the GPU / ANE coupling tests (18 Nov 2026, M6). One tool, four kinds of work, each with a
// working-set size and a duty cycle, so the Neural Engine's slowdown can be compared per watt and per cache footprint:
//   alu    FMA chain in registers (no memory traffic)
//   bw     streaming reads of an MB-sized buffer (small MB: served by the GPU / system caches; large: DRAM)
//   mm8    INT8 x INT8 -> INT32 matmul2d (the GPU's matrix units), weights from an MB-sized buffer
//   mm16   FP16 x FP16 -> FP32 matmul2d, same shape
// mm*: each threadgroup copies a 32-row x KB weight tile into threadgroup memory once and multiplies it by REP
// activation tiles (64 tokens each, a small shared buffer), so the compute per weight byte is fixed (REP x 128 ops/B
// for INT8) and only the weight footprint changes: logical rows M = 16384 always; physical rows wrap modulo the
// footprint (MB << 20) / (K * bytes). DUTY < 1 sleeps between command buffers to lower the average power.
// Prints one line every 0.5 s: "<epoch> GPU <mode> <rate> <unit> busy <pct>", then a final median line.
//   gpu_load MODE SECONDS [MB=1024] [REP=8] [DUTY=1]
import Foundation
import Metal

let a = CommandLine.arguments
let mode = a[1], secs = Double(a[2])!, mb = a.count > 3 ? Int(a[3])! : 1024
let rep = a.count > 4 ? UInt32(a[4])! : 8, duty = a.count > 5 ? Double(a[5])! : 1.0
let dev = MTLCreateSystemDefaultDevice()!, q = dev.makeCommandQueue()!
let src = """
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
#define T 64
kernel void alu(device float *o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    float4 a = float4(i) * 1e-6f, b = 1.0001f, c = 0.9999f;
    for (uint j = 0; j < 4096; j++) { a = fma(a, b, c); b = fma(b, c, a); }
    if (a.x == 1234.5f) o[i] = b.y;
}
kernel void bw(device const uint4 *a [[buffer(0)]], device uint *o [[buffer(1)]], constant uint &n [[buffer(2)]],
               uint i [[thread_position_in_grid]], uint g [[threads_per_grid]]) {
    uint4 s = 0;
    for (uint k = i; k < n; k += g) s += a[k];
    if ((s.x ^ s.y ^ s.z ^ s.w) == 0x12345678u) o[i] = s.x;
}
// P: [0] physical rows, [1] K, [2] REP
template <typename E, typename A, uint KB>
inline void mm_body(device const E *w, device const E *x, device int *o, constant uint *P, uint2 tg, uint tid, threadgroup E *W) {
    const uint Mp = P[0], K = P[1], REP = P[2];
    const uint o0 = (tg.x * 32) % Mp, k0 = tg.y * KB;
    const uint V = 16 / sizeof(E);                       // elements per 16-byte copy
    for (uint i = tid; i < 32 * KB / V; i += 128) {
        uint e = i * V, r = e / KB, c = e % KB;
        *(threadgroup uint4 *)(W + e) = *(device const uint4 *)(w + (o0 + r) * K + k0 + c);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tensor<threadgroup E, dextents<int32_t, 2>, tensor_inline> tA(W, dextents<int32_t, 2>(KB, 32));
    constexpr auto d = matmul2d_descriptor(32, T, KB, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<d, execution_simdgroups<4>> op;
    auto c = op.template get_destination_cooperative_tensor<decltype(tA), tensor<device E, dextents<int32_t, 2>, tensor_inline>, A>();
    for (uint16_t i = 0; i < c.get_capacity(); ++i) if (c.is_valid_element(i)) c[i] = 0;
    for (uint r = 0; r < REP; r++) {
        tensor<device E, dextents<int32_t, 2>, tensor_inline> tB((device E *)x + (r * K + k0) * T, dextents<int32_t, 2>(T, KB));
        op.run(tA, tB, c);
    }
    A s = 0;
    for (uint16_t i = 0; i < c.get_capacity(); ++i) if (c.is_valid_element(i)) s += c[i];
    if (s == A(123456789)) o[tg.x] = int(s);
}
kernel void mm8(device const int8_t *w [[buffer(0)]], device const int8_t *x [[buffer(1)]], device int *o [[buffer(2)]],
                constant uint *P [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup int8_t W[32 * 512];
    mm_body<int8_t, int32_t, 512>(w, x, o, P, tg, tid, W);
}
kernel void mm16(device const half *w [[buffer(0)]], device const half *x [[buffer(1)]], device int *o [[buffer(2)]],
                 constant uint *P [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W[32 * 256];
    mm_body<half, float, 256>(w, x, o, P, tg, tid, W);
}
"""
let opts = MTLCompileOptions(); opts.languageVersion = .version4_1
let lib = try! dev.makeLibrary(source: src, options: opts)
let pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: mode)!)
let o = dev.makeBuffer(length: 1 << 24, options: .storageModeShared)!
let M = 16384, K = 8192
var work = 0.0, unit = "", encode: (MTLComputeCommandEncoder) -> Void
switch mode {
case "alu":
    unit = "TFLOPS"; work = Double(1 << 18) * 4096 * 2 * 4 * 2 / 1e12
    encode = { e in e.setBuffer(o, offset: 0, index: 0)
        e.dispatchThreads(MTLSize(width: 1 << 18, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)) }
case "bw":
    let bytes = mb << 20, big = dev.makeBuffer(length: bytes, options: .storageModePrivate)!
    var n = UInt32(bytes / 16); unit = "GB/s"
    // small buffers: several passes per command buffer so each lasts milliseconds
    let passes = max(1, (256 << 20) / bytes); work = Double(bytes * passes) / 1e9
    encode = { e in for _ in 0..<passes { e.setBuffer(big, offset: 0, index: 0); e.setBuffer(o, offset: 0, index: 1); e.setBytes(&n, length: 4, index: 2)
        e.dispatchThreads(MTLSize(width: 1 << 18, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)) } }
case "mm8", "mm16":
    let eb = mode == "mm8" ? 1 : 2, kb = mode == "mm8" ? 512 : 256
    let mp = max(32, min(M, ((mb << 20) / (K * eb)) / 32 * 32))
    let w = dev.makeBuffer(length: mp * K * eb, options: .storageModePrivate)!
    let x = dev.makeBuffer(length: Int(rep) * K * 64 * eb, options: .storageModeShared)!
    if mode == "mm8" { let p = x.contents().bindMemory(to: Int8.self, capacity: x.length); for i in 0..<x.length { p[i] = Int8(truncatingIfNeeded: (i * 7) % 5 - 2) } }
    else { let p = x.contents().bindMemory(to: UInt16.self, capacity: x.length / 2); for i in 0..<(x.length / 2) { p[i] = 0x3000 } }
    var P: [UInt32] = [UInt32(mp), UInt32(K), rep]
    unit = mode == "mm8" ? "TOPS" : "TFLOPS"; work = 2.0 * Double(M) * Double(K) * 64 * Double(rep) / 1e12
    FileHandle.standardError.write(String(format: "%@: weights %.1f MB (%d physical rows of %d), REP %d, %.0f ops per weight byte\n",
                                          mode, Double(mp * K * eb) / 1048576, mp, M, rep, 2.0 * 64 * Double(rep) / Double(eb)).data(using: .utf8)!)
    encode = { e in e.setBuffer(w, offset: 0, index: 0); e.setBuffer(x, offset: 0, index: 1); e.setBuffer(o, offset: 0, index: 2)
        e.setBytes(&P, length: 12, index: 3)
        e.dispatchThreadgroups(MTLSize(width: M / 32, height: K / kb, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)) }
default: fatalError("mode alu|bw|mm8|mm16")
}
// one warm-up command buffer, then "ready"
do { let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!; e.setComputePipelineState(pso); encode(e); e.endEncoding(); cb.commit(); cb.waitUntilCompleted() }
print(String(format: "ready %.3f", Date().timeIntervalSince1970)); fflush(stdout)
let t0 = Date(); var wStart = Date(), wWork = 0.0, wGpu = 0.0, rates: [Double] = []
var gpuAll = 0.0
while Date().timeIntervalSince(t0) < secs {
    let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pso); encode(e); e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    let g = cb.gpuEndTime - cb.gpuStartTime
    wWork += work; wGpu += g; gpuAll += g
    if duty < 1 { usleep(useconds_t(g * (1 / duty - 1) * 1e6)) }
    let el = Date().timeIntervalSince(wStart)
    if el >= 0.5 {
        let r = wWork / wGpu; rates.append(r)
        print(String(format: "%.3f GPU %@ %.2f %@ busy %.0f", Date().timeIntervalSince1970, mode, r, unit, 100 * wGpu / el)); fflush(stdout)
        wStart = Date(); wWork = 0; wGpu = 0
    }
}
let s = rates.sorted()
print(String(format: "GPU %@ median %.2f %@ (MB %d REP %d duty %.2f, busy %.0f%%)", mode, s.isEmpty ? 0 : s[s.count / 2], unit, mb, rep, duty,
             100 * gpuAll / Date().timeIntervalSince(t0)))
