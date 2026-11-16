// GPU stress for the ANE coupling test: `bw` streams a 1 GiB buffer (float4 reads, almost no math), `alu` runs an FMA
// chain in registers (no memory traffic). Prints the achieved GB/s or TFLOPS.   gpu_stress bw|alu SECONDS
import Foundation
import Metal
let mode = CommandLine.arguments[1], secs = Double(CommandLine.arguments[2])!
let dev = MTLCreateSystemDefaultDevice()!, q = dev.makeCommandQueue()!
let src = """
#include <metal_stdlib>
using namespace metal;
kernel void bw(device const float4 *a [[buffer(0)]], device float *o [[buffer(1)]], constant uint &n [[buffer(2)]],
               uint i [[thread_position_in_grid]], uint g [[threads_per_grid]]) {
    float4 s = 0;
    for (uint j = i; j < n; j += g) s += a[j];
    if (s.x == 1234.5f) o[i] = s.y;
}
kernel void alu(device float *o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    float4 a = float4(i) * 1e-6f, b = 1.0001f, c = 0.9999f;
    for (uint j = 0; j < 4096; j++) { a = fma(a, b, c); b = fma(b, c, a); }
    if (a.x == 1234.5f) o[i] = b.y;
}
"""
let lib = try! dev.makeLibrary(source: src, options: nil)
let p = try! dev.makeComputePipelineState(function: lib.makeFunction(name: mode)!)
let n = 1 << 26, big = dev.makeBuffer(length: n * 16, options: .storageModePrivate)!, o = dev.makeBuffer(length: 1 << 24, options: .storageModeShared)!
var nn = UInt32(n), work = 0.0, gpu = 0.0
let t0 = Date()
while Date().timeIntervalSince(t0) < secs {
    let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(p)
    if mode == "bw" { e.setBuffer(big, offset: 0, index: 0); e.setBuffer(o, offset: 0, index: 1); e.setBytes(&nn, length: 4, index: 2)
        e.dispatchThreads(MTLSize(width: 1 << 18, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)); work += Double(n * 16) }
    else { e.setBuffer(o, offset: 0, index: 0)
        e.dispatchThreads(MTLSize(width: 1 << 18, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)); work += Double(1 << 18) * 4096 * 2 * 4 * 2 }
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted(); gpu += cb.gpuEndTime - cb.gpuStartTime
}
print(String(format: "GPU %@ %.1f %@ (GPU busy %.0f%%)", mode, mode == "bw" ? work / gpu / 1e9 : work / gpu / 1e12, mode == "bw" ? "GB/s" : "TFLOPS", 100 * gpu / Date().timeIntervalSince(t0)))
