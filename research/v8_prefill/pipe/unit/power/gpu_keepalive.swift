// GPU clock keep-alive (18 Nov 2026, M6): back-to-back spin dispatches of one tiny kernel (TG threadgroups of one
// simdgroup each, an FMA chain of ~US microseconds at full clock), N serial dispatches per command buffer, two command
// buffers in flight, so the GPU never idles and its power controller holds the top P-state for other processes' short
// bursts (gpu_clk.swift: the decode attention after a 5 ms CPU gap 3.2 -> 1.2 ms at 64K keys). Short dispatches
// interfere less with the other work; N per command buffer keeps the CPU wake-ups low. The submitting thread runs at
// QOS (ut = utility, default; ui = user-interactive). Exits on SIGTERM / SIGINT (handlers installed, so an inherited
// SIG_IGN from a background shell does not keep it alive) or after MAX_SECONDS.
//   gpu_keepalive MAX_SECONDS [US=20] [N=250] [TG=1] [QOS=ut]
// Prints "ready <epoch>", then at exit "keepalive: <s> s, <n> command buffers, GPU busy <pct>% (spin time / wall)".
// Build: swiftc -O -o gpu_keepalive gpu_keepalive.swift -framework Metal
import Foundation
import Metal

let a = CommandLine.arguments
let maxS = Double(a[1])!, us = a.count > 2 ? Double(a[2])! : 20, nd = a.count > 3 ? Int(a[3])! : 250, tg = a.count > 4 ? Int(a[4])! : 1
let qos = a.count > 5 ? a[5] : "ut"
var stop: sig_atomic_t = 0
signal(SIGTERM) { _ in stop = 1 }
signal(SIGINT) { _ in stop = 1 }
let dev = MTLCreateSystemDefaultDevice()!, q = dev.makeCommandQueue()!
let src = """
#include <metal_stdlib>
using namespace metal;
kernel void ka_spin(device float *o [[buffer(0)]], constant uint &n [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    float a = float(i) * 1e-6f, b = 1.0001f, c = 0.9999f;
    for (uint j = 0; j < n; j++) { a = fma(a, b, c); b = fma(b, c, a); }
    if (a == 1234.5f) o[i & 1023] = b;
}
"""
let lib = try! dev.makeLibrary(source: src, options: MTLCompileOptions())
let pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "ka_spin")!)
let buf = dev.makeBuffer(length: 4096, options: .storageModePrivate)!
func spin(_ n: UInt32, _ count: Int = 1) -> MTLCommandBuffer {
    let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pso); e.setBuffer(buf, offset: 0, index: 0)
    var nn = n; e.setBytes(&nn, length: 4, index: 1)
    for _ in 0..<count { e.dispatchThreads(MTLSize(width: 32 * tg, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)) }
    e.endEncoding(); cb.commit(); return cb
}
// calibration (iterations per us at the loaded clock): a long burst, measured back to back
var itPerUs = 90.0
for _ in 0..<3 { let w = spin(400_000); w.waitUntilCompleted(); let c = spin(200_000); c.waitUntilCompleted()
    itPerUs = max(itPerUs, 200_000 / ((c.gpuEndTime - c.gpuStartTime) * 1e6)) }
let n = UInt32(us * itPerUs)
print(String(format: "ready %.3f (%d x %.0f us x %d simdgroups per command buffer, %.1f it/us, QoS %@)", Date().timeIntervalSince1970, nd, us, tg, itPerUs, qos)); fflush(stdout)
let t0 = Date(); var cbs = 0, busy = 0.0
let done = DispatchSemaphore(value: 0)
let th = Thread {
    var prev: MTLCommandBuffer? = nil
    while stop == 0 && Date().timeIntervalSince(t0) < maxS {
        let cb = spin(n, nd); cbs += 1
        if let p = prev { p.waitUntilCompleted(); busy += p.gpuEndTime - p.gpuStartTime }
        prev = cb
    }
    prev?.waitUntilCompleted()
    done.signal()
}
th.qualityOfService = qos == "ui" ? .userInteractive : qos == "bg" ? .background : qos == "def" ? .default : .utility
th.start(); done.wait()
let el = Date().timeIntervalSince(t0)
print(String(format: "keepalive: %.1f s, %d command buffers, GPU busy %.1f%%", el, cbs, 100 * busy / el)); fflush(stdout)
