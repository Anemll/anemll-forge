// INT8 matmul2d rate at the DeltaNet LUT GEMM's shapes (19 Nov 2026, M6): each threadgroup computes C[TM x N] over
// K = 5120 in KS steps, A (the decoded weights) from THREADGROUP memory, B (the activations, [K][N] int8) from DEVICE
// memory, as lut_gemm32_i8t; no decode (A is a constant tile), so this is the matmul + B-load ceiling for the shape.
// Grid = COUT / TM threadgroups (COUT = 10240, the qkv projection).   gpu_mm_shape TM N KS SG [SECONDS]
import Foundation
import Metal
let a = CommandLine.arguments
let TM = Int(a[1])!, N = Int(a[2])!, KS = Int(a[3])!, SG = Int(a[4])!, secs = a.count > 5 ? Double(a[5])! : 1.0
let K = 5120, COUT = 10240, NTG = COUT / TM
let src = """
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
kernel void mm(device int8_t *B [[buffer(0)]], device int *o [[buffer(1)]], uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup int8_t W[\(TM) * \(KS)] __attribute__((aligned(16)));
    for (uint i = tid; i < \(TM * KS) / 4; i += \(SG * 32)) ((threadgroup uint *)W)[i] = 0x01010101u ^ (tg & 3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tA(W, dextents<int32_t, 2>(\(TM), \(KS)));
    constexpr auto d = matmul2d_descriptor(\(TM), \(N), \(KS), true, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<d, execution_simdgroups<\(SG)>> op;
    auto c = op.template get_destination_cooperative_tensor<decltype(tA), tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>, int32_t>();
    for (uint16_t i = 0; i < c.get_capacity(); ++i) if (c.is_valid_element(i)) c[i] = 0;
    for (uint k0 = 0; k0 < \(K); k0 += \(KS)) {
        tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tB(B + k0 * \(N), dextents<int32_t, 2>(\(N), \(KS)));
        op.run(tA, tB, c);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    int s = 0;
    for (uint16_t i = 0; i < c.get_capacity(); ++i) if (c.is_valid_element(i)) s += c[i];
    if (s == 123456789) o[tg] = s;
}
"""
let dev = MTLCreateSystemDefaultDevice()!, q = dev.makeCommandQueue()!
let opts = MTLCompileOptions(); opts.languageVersion = .version4_1
let lib: MTLLibrary
do { lib = try dev.makeLibrary(source: src, options: opts) } catch { print("TM \(TM) N \(N) KS \(KS) SG \(SG): COMPILE FAILED"); exit(0) }
let pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "mm")!)
let B = dev.makeBuffer(length: K * N, options: .storageModeShared)!, o = dev.makeBuffer(length: NTG * 4, options: .storageModeShared)!
memset(B.contents(), 1, B.length)
let work = 2.0 * Double(COUT) * Double(N) * Double(K) / 1e12
func run() -> Double {
    let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pso); e.setBuffer(B, offset: 0, index: 0); e.setBuffer(o, offset: 0, index: 1)
    for _ in 0..<8 { e.dispatchThreadgroups(MTLSize(width: NTG, height: 128 / N, depth: 1), threadsPerThreadgroup: MTLSize(width: SG * 32, height: 1, depth: 1)) }
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    return (cb.gpuEndTime - cb.gpuStartTime) / 8
}
let t0 = Date(); while Date().timeIntervalSince(t0) < 1.5 { _ = run() }
var r: [Double] = []; let t1 = Date()
while Date().timeIntervalSince(t1) < secs { r.append(run()) }
r.sort(); let med = r[r.count / 2]
// 128 tokens per call: N < 128 runs 128 / N threadgroup rows
print(String(format: "TM %3d N %3d KS %3d SG %d: %4d tgs, %.3f ms per 10240x5120x128 GEMM, %.1f TOPS, max threads %d", TM, N, KS, SG, NTG * (128 / N), med * 1e3, work * 128 / Double(N) / Double(128 / N) / med, pso.maxTotalThreadsPerThreadgroup))
