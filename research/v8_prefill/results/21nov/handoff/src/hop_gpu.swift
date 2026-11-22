// 21 Nov 2026 (M6 ANE bot, item 1): the GPU side of the standalone ANE <-> GPU handoff probe (hop_probe.py).
// Every buffer is an ANE IOSurface buffer wrapped zero-copy (hop_wrap), as the served split verify does (dec_gpu.swift).
// Kernels: hop_add (y = x + 1e-3 over the 16 x 5120 fp16 tensor, ~0.01 ms) and hop_busy (the same, then a dependent
// FMA loop of `iters` per thread: a stand-in for a drafter-sized GPU matrix of a chosen duration).
//   swiftc -O -emit-library -o libhop_gpu.dylib hop_gpu.swift -framework Metal -framework QuartzCore
import Foundation
import Metal
import QuartzCore

let dev = MTLCreateSystemDefaultDevice()!
let queue = dev.makeCommandQueue()!
var psos: [String: MTLComputePipelineState] = [:]
var wrapped: [MTLBuffer] = []
var inflight: [Int: MTLCommandBuffer] = [:]
var ticket = 0

let SRC = """
#include <metal_stdlib>
using namespace metal;
kernel void hop_add(device const half *x [[buffer(0)]], device half *y [[buffer(1)]], constant uint &n [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] = x[i] + half(1e-3);
}
kernel void hop_busy(device const half *x [[buffer(0)]], device half *y [[buffer(1)]], constant uint &n [[buffer(2)]],
                     constant uint &iters [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float a = float(x[i]), b = 1.0f;
    for (uint k = 0; k < iters; k++) { b = fma(b, 0.999999f, a * 1e-7f); }
    y[i] = half(float(x[i]) + 1e-3f + b * 1e-9f);
}
"""

@_cdecl("hop_init")
public func hop_init() -> Int32 {
    do {
        let lib = try dev.makeLibrary(source: SRC, options: nil)
        for n in ["hop_add", "hop_busy"] { psos[n] = try dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }
    } catch { print("hop_init: \(error)"); return -1 }
    return 0
}

/// Wrap an IOSurface buffer (its CPU address, page aligned; nbytes a page multiple) zero-copy. Returns its index.
@_cdecl("hop_wrap")
public func hop_wrap(_ addr: UnsafeMutableRawPointer, _ nbytes: Int) -> Int32 {
    guard let b = dev.makeBuffer(bytesNoCopy: addr, length: nbytes, options: .storageModeShared, deallocator: nil) else { return -1 }
    wrapped.append(b)
    return Int32(wrapped.count - 1)
}

func encode(_ cb: MTLCommandBuffer, _ xi: Int, _ yi: Int, _ n: Int, _ iters: Int) {
    let e = cb.makeComputeCommandEncoder()!
    var nn = UInt32(n), it = UInt32(iters)
    e.setComputePipelineState(psos[iters > 0 ? "hop_busy" : "hop_add"]!)
    e.setBuffer(wrapped[xi], offset: 0, index: 0); e.setBuffer(wrapped[yi], offset: 0, index: 1)
    e.setBytes(&nn, length: 4, index: 2)
    if iters > 0 { e.setBytes(&it, length: 4, index: 3) }
    e.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    e.endEncoding()
}

/// Synchronous (the served pattern): encode, commit, waitUntilCompleted. out = host times (s): before encode, after
/// commit, GPU start, GPU end, after the wait returns. Returns GPU ms, or < 0 on error.
@_cdecl("hop_run")
public func hop_run(_ xi: Int32, _ yi: Int32, _ n: Int32, _ iters: Int32, _ out: UnsafeMutablePointer<Double>) -> Double {
    let t0 = CACurrentMediaTime()
    let cb = queue.makeCommandBuffer()!
    encode(cb, Int(xi), Int(yi), Int(n), Int(iters))
    cb.commit()
    let t1 = CACurrentMediaTime()
    cb.waitUntilCompleted()
    let t4 = CACurrentMediaTime()
    if cb.status != .completed { return -2 }
    out[0] = t0; out[1] = t1; out[2] = cb.gpuStartTime; out[3] = cb.gpuEndTime; out[4] = t4
    return (cb.gpuEndTime - cb.gpuStartTime) * 1e3
}

/// Asynchronous (fork-join): commit now, return a ticket; hop_wait joins. out[0] = time after commit.
@_cdecl("hop_submit")
public func hop_submit(_ xi: Int32, _ yi: Int32, _ n: Int32, _ iters: Int32, _ out: UnsafeMutablePointer<Double>) -> Int32 {
    let cb = queue.makeCommandBuffer()!
    encode(cb, Int(xi), Int(yi), Int(n), Int(iters))
    cb.commit()
    out[0] = CACurrentMediaTime()
    ticket += 1; inflight[ticket] = cb
    return Int32(ticket)
}

/// Join: out = GPU start, GPU end, after the wait returns. Returns GPU ms.
@_cdecl("hop_wait")
public func hop_wait(_ t: Int32, _ out: UnsafeMutablePointer<Double>) -> Double {
    guard let cb = inflight.removeValue(forKey: Int(t)) else { return -1 }
    cb.waitUntilCompleted()
    out[2] = CACurrentMediaTime()
    if cb.status != .completed { return -2 }
    out[0] = cb.gpuStartTime; out[1] = cb.gpuEndTime
    return (cb.gpuEndTime - cb.gpuStartTime) * 1e3
}

@_cdecl("hop_now")
public func hop_now() -> Double { CACurrentMediaTime() }
