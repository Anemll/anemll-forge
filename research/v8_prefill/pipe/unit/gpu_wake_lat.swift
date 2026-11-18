// GPU start latency after an idle gap (16 Nov 2026): commit -> gpuStartTime of a small kernel after the GPU idled for
// GAP ms, plain and with a warm-up kick (an empty-ish command buffer committed LEAD ms before the real one).
//   swiftc -O -o gpu_wake_lat gpu_wake_lat.swift -framework Metal -framework QuartzCore && ./gpu_wake_lat
import Foundation
import Metal
import QuartzCore
let dev = MTLCreateSystemDefaultDevice()!, q = dev.makeCommandQueue()!
let src = "#include <metal_stdlib>\nusing namespace metal;\nkernel void fill(device float *x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i] = x[i] * 0.5f + 1.0f; }"
let lib = try! dev.makeLibrary(source: src, options: nil)
let pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "fill")!)
let buf = dev.makeBuffer(length: 1 << 22, options: .storageModePrivate)!
func cmd(_ n: Int) -> MTLCommandBuffer {
    let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pso); e.setBuffer(buf, offset: 0, index: 0)
    e.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    e.endEncoding(); return cb
}
func spinUntil(_ t: Double) { while CACurrentMediaTime() < t {} }
func med(_ a: [Double]) -> Double { a.sorted()[a.count / 2] }
for gap in [0.2, 0.5, 1.0, 2.0, 4.0, 6.0] {
    var lat: [Double] = [], wake: [Double] = []
    for _ in 0..<40 {
        let t0 = CACurrentMediaTime(); spinUntil(t0 + gap * 1e-3)
        let cb = cmd(1 << 20); let tc = CACurrentMediaTime(); cb.commit(); cb.waitUntilCompleted()
        lat.append((cb.gpuStartTime - tc) * 1e3); wake.append((CACurrentMediaTime() - cb.gpuEndTime) * 1e3)
    }
    var row = String(format: "gap %4.1f ms: commit->start %.3f, end->wake %.3f", gap, med(lat), med(wake))
    for lead in [0.1, 0.2, 0.3, 0.5] where lead < gap {
        var lk: [Double] = []
        for _ in 0..<40 {
            let t0 = CACurrentMediaTime(); spinUntil(t0 + (gap - lead) * 1e-3)
            let k = cmd(32); k.commit()
            spinUntil(t0 + gap * 1e-3)
            let cb = cmd(1 << 20); let tc = CACurrentMediaTime(); cb.commit(); cb.waitUntilCompleted()
            lk.append((cb.gpuStartTime - tc) * 1e3)
        }
        row += String(format: "; kick %.1f ms ahead -> %.3f", lead, med(lk))
    }
    print(row); fflush(stdout)
}
