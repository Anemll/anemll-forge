// GPU streaming-read bandwidth (17 Nov 2026): a kernel that reads a large shared buffer once per command buffer
// (uint4 loads, grid-stride, one partial sum per thread), back to back for SECONDS; prints the rate per 0.5 s window
// with a wall-clock timestamp, so a concurrent ANE load can be lined up.   bw_gpu SECONDS [MB]
import Foundation
import Metal
let secs = Double(CommandLine.arguments[1])!, mb = Int(CommandLine.arguments.count > 2 ? CommandLine.arguments[2] : "1024")!
let dev = MTLCreateSystemDefaultDevice()!, q = dev.makeCommandQueue()!
let src = """
#include <metal_stdlib>
using namespace metal;
kernel void rd(device const uint4 *a [[buffer(0)]], device uint *o [[buffer(1)]], constant uint &n [[buffer(2)]],
               uint i [[thread_position_in_grid]], uint g [[threads_per_grid]]) {
    uint4 s = 0;
    for (uint k = i; k < n; k += g) s += a[k];
    o[i] = s.x ^ s.y ^ s.z ^ s.w;
}
"""
let lib = try! dev.makeLibrary(source: src, options: nil)
let pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "rd")!)
let bytes = mb << 20
let a = dev.makeBuffer(length: bytes, options: .storageModeShared)!
memset(a.contents(), 1, bytes)
let threads = 65536 * 4
let o = dev.makeBuffer(length: threads * 4, options: .storageModePrivate)!
var n = UInt32(bytes / 16)
let t0 = Date().timeIntervalSince1970
var win: [Double] = [], wstart = t0
while Date().timeIntervalSince1970 - t0 < secs {
    let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pso); e.setBuffer(a, offset: 0, index: 0); e.setBuffer(o, offset: 0, index: 1)
    e.setBytes(&n, length: 4, index: 2)
    e.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    win.append(Double(bytes) / (cb.gpuEndTime - cb.gpuStartTime) / 1e9)
    let now = Date().timeIntervalSince1970
    if now - wstart >= 0.5 { win.sort(); print(String(format: "%.3f GPU %.1f GB/s", now, win[win.count / 2])); fflush(stdout); win = []; wstart = now }
}
