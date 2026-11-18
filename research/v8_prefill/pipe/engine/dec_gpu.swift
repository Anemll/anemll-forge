// Decode-verify attention on the GPU (16 Nov 2026), a small dylib for the split verify (part A on the ANE -> this ->
// part B on the ANE). Kernels: dec_kernels.metal (dec_prep, dec_attn, dec_combine_gate). Every buffer is an ANE IOSurface
// buffer wrapped zero-copy (dec_wrap): part A's q / g / kt / vt outputs, the attention layer's INT8 KV cache segments
// (the same buffers the ANE verify reads) and part B's o input.
//   swiftc -O -emit-library -o libdec_gpu.dylib dec_gpu.swift -framework Metal
// Env: DEC_KERNELS (path of dec_kernels.metal; default next to the dylib's working dir), DEC_CK (keys per chunk, 2048).
import Foundation
import Metal
import QuartzCore

let dev = MTLCreateSystemDefaultDevice()!
let queue = dev.makeCommandQueue()!
var psos: [String: MTLComputePipelineState] = [:]
var wrapped: [MTLBuffer] = []
let CK = Int(ProcessInfo.processInfo.environment["DEC_CK"] ?? "2048")!
let NCMAX = (131072 / 2048 * 2048 / CK + 1) * 4   // x4: research per-simdgroup partials (dec_attn_sg)
let Q8 = dev.makeBuffer(length: 24 * 8 * 256, options: .storageModePrivate)!
let QS = dev.makeBuffer(length: 24 * 8 * 4, options: .storageModePrivate)!
let OP = dev.makeBuffer(length: 4 * NCMAX * 3 * 16 * 256 * 4, options: .storageModePrivate)!
let MP = dev.makeBuffer(length: 4 * NCMAX * 3 * 16 * 4, options: .storageModePrivate)!
let LP = dev.makeBuffer(length: 4 * NCMAX * 3 * 16 * 4, options: .storageModePrivate)!

struct DPrm { var n, p0, ck, nc: UInt32; var start: (UInt32, UInt32, UInt32, UInt32, UInt32); var rows: (UInt32, UInt32, UInt32, UInt32, UInt32) }
struct DSt { var qh, qt, kh, kt, vh, vt, gh, gt, oc: UInt32 }
struct Seg { var start: Int; var rows: Int; var k, ks, v, vs: MTLBuffer }
var layers: [Int: [Seg]] = [:]
var inflight: [Int: MTLCommandBuffer] = [:]
var ticket = 0

@_cdecl("dec_init")
public func dec_init(_ path: UnsafePointer<CChar>) -> Int32 {
    let text: String
    do { text = try String(contentsOfFile: String(cString: path), encoding: .utf8) } catch { print("dec_init: \(error)"); return -1 }
    let opts = MTLCompileOptions(); opts.languageVersion = .version4_1
    var src = text
    // research variants go INTO the source text (a compile option alone can hit a cached binary of the same source)
    if let v = ProcessInfo.processInfo.environment["DEC_SUB"], let n = Int(v) { src = "#define DEC_SUB \(n)\n" + src }   // research
    if let v = ProcessInfo.processInfo.environment["DEC_PF"] { let a = v.split(separator: ",").map { String($0) }   // research: "mode,dist,at,cur"
        src = "#define PF_MODE \(a[0])\n#define PF_DIST \(a.count > 1 ? a[1] : "1")\n#define PF_AT \(a.count > 2 ? a[2] : "0")\n#define PF_CUR \(a.count > 3 ? a[3] : "0")\n" + src }
    if let v = ProcessInfo.processInfo.environment["DEC_SUBR"], let n = Int(v) { src = "#define SUBR \(n)\n" + src }
    if let v = ProcessInfo.processInfo.environment["DEC_MCFG"] { let a = v.split(separator: ",").map { String($0) }   // research: "M,NSG,QSP"
        src = "#define DEC_M \(a[0])\n#define DEC_NSG \(a[1])\n#define DEC_QSP \(a[2])\n" + src }
    if let v = ProcessInfo.processInfo.environment["DEC_M48"], v == "1" { src = "#define DEC_M 48\n#define DEC_NSG 6\n#define DEC_QSP 1\n" + src }
    if let v = ProcessInfo.processInfo.environment["DEC_SM"], let n = Int(v) { src = "#define DEC_SM \(n)\n" + src; print("dec_init: DEC_SM=\(n), source \(src.count) chars"); fflush(stdout) }
    do {
        let lib = try dev.makeLibrary(source: src, options: opts)
        for n in ["dec_prep", "dec_attn", "dec_combine_gate", "dec_combine_part", "dec_flag_wait", "dec_flag_set"] {
            psos[n] = try dev.makeComputePipelineState(function: lib.makeFunction(name: n)!)
        }
        // attention kernel: dec_attn_r4 (scores in registers, identical output, ~3% faster) when the source has it;
        // DEC_ATTN=<name> picks another (dec_attn: the previous one; research kernels)
        let a = ProcessInfo.processInfo.environment["DEC_ATTN"] ?? "dec_attn_r4"
        if let f = lib.makeFunction(name: a) {
            psos["dec_attn"] = try dev.makeComputePipelineState(function: f); print("dec_init: attention kernel \(a)"); fflush(stdout)
        }
    } catch { print("dec_init: \(error)".prefix(3000)); return -2 }
    return Int32(CK)
}

/// A CPU-visible, page-aligned region (an ANE IOSurface buffer) as a Metal buffer, zero-copy. Returns its handle.
@_cdecl("dec_wrap")
public func dec_wrap(_ p: UnsafeMutableRawPointer, _ n: Int) -> Int32 {
    let page = Int(getpagesize())
    guard Int(bitPattern: p) % page == 0, let b = dev.makeBuffer(bytesNoCopy: p, length: (n + page - 1) / page * page,
                                                                 options: .storageModeShared, deallocator: nil) else { return -1 }
    wrapped.append(b); return Int32(wrapped.count - 1)
}

/// One INT8 KV segment of attention layer `layer`: positions [start, start + rows), K [4][256][rows], KS [4][rows],
/// V [4][rows][256], VS [4][rows] (dense; the caller checks the strides).
@_cdecl("dec_seg")
public func dec_seg(_ layer: Int32, _ start: Int32, _ rows: Int32, _ k: Int32, _ ks: Int32, _ v: Int32, _ vs: Int32) -> Int32 {
    var s = layers[Int(layer)] ?? []
    s.removeAll { $0.start == Int(start) }
    s.append(Seg(start: Int(start), rows: Int(rows), k: wrapped[Int(k)], ks: wrapped[Int(ks)], v: wrapped[Int(v)], vs: wrapped[Int(vs)]))
    s.sort { $0.start < $1.start }
    guard s.count <= 5 else { return -1 }
    layers[Int(layer)] = s
    return Int32(s.count)
}

/// Forget attention layer `layer`'s segments (before registering the runtime's current ones again).
@_cdecl("dec_clear")
public func dec_clear(_ layer: Int32) -> Int32 {
    layers[Int(layer)] = nil
    return 0
}

func encode(_ cb: MTLCommandBuffer, _ layer: Int, _ h: UnsafePointer<Int32>, _ st: UnsafePointer<UInt32>, _ p0: Int,
            part: Int = -1) -> Bool {
    guard let segs = layers[layer], !segs.isEmpty else { return false }
    let n = p0 + 8
    var s5 = [UInt32](repeating: 0, count: 5), r5 = [UInt32](repeating: 0, count: 5)
    for (i, s) in segs.enumerated() { s5[i] = UInt32(s.start); r5[i] = UInt32(s.rows) }
    guard let last = segs.last, n <= last.start + last.rows else { return false }
    var prm = DPrm(n: UInt32(n), p0: UInt32(p0), ck: UInt32(CK), nc: UInt32((n - segs[0].start + CK - 1) / CK),
                   start: (s5[0], s5[1], s5[2], s5[3], s5[4]), rows: (r5[0], r5[1], r5[2], r5[3], r5[4]))
    var dst = DSt(qh: st[0], qt: st[1], kh: st[2], kt: st[3], vh: st[4], vt: st[5], gh: st[6], gt: st[7], oc: st[8])
    let sb = (0..<5).map { segs[min($0, segs.count - 1)] }
    let e = cb.makeComputeCommandEncoder()!
    // h: q, g, kt, vt, o (wrapped handles)
    e.setComputePipelineState(psos["dec_prep"]!)
    e.setBuffer(wrapped[Int(h[0])], offset: 0, index: 0); e.setBuffer(wrapped[Int(h[2])], offset: 0, index: 1)
    e.setBuffer(wrapped[Int(h[3])], offset: 0, index: 2); e.setBuffer(Q8, offset: 0, index: 3); e.setBuffer(QS, offset: 0, index: 4)
    for i in 0..<5 {
        e.setBuffer(sb[i].k, offset: 0, index: 5 + i); e.setBuffer(sb[i].ks, offset: 0, index: 10 + i)
        e.setBuffer(sb[i].v, offset: 0, index: 15 + i); e.setBuffer(sb[i].vs, offset: 0, index: 20 + i)
    }
    e.setBytes(&prm, length: MemoryLayout<DPrm>.size, index: 25); e.setBytes(&dst, length: MemoryLayout<DSt>.size, index: 26)
    e.dispatchThreadgroups(MTLSize(width: 32, height: 8, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    e.memoryBarrier(scope: .buffers)
    e.setComputePipelineState(psos["dec_attn"]!)
    e.setBuffer(Q8, offset: 0, index: 0); e.setBuffer(QS, offset: 0, index: 1)
    for i in 0..<5 {
        e.setBuffer(sb[i].k, offset: 0, index: 2 + i); e.setBuffer(sb[i].ks, offset: 0, index: 7 + i)
        e.setBuffer(sb[i].v, offset: 0, index: 12 + i); e.setBuffer(sb[i].vs, offset: 0, index: 17 + i)
    }
    e.setBuffer(OP, offset: 0, index: 22); e.setBuffer(MP, offset: 0, index: 23); e.setBuffer(LP, offset: 0, index: 24)
    e.setBytes(&prm, length: MemoryLayout<DPrm>.size, index: 25)
    let qy = Int(ProcessInfo.processInfo.environment["DEC_QGROUPS"] ?? "3")!   // research: 1 = only the first 16 query rows (timing)
    e.dispatchThreadgroups(MTLSize(width: Int(prm.nc), height: qy, depth: 4), threadsPerThreadgroup: MTLSize(width: Int(ProcessInfo.processInfo.environment["DEC_TPT"] ?? "128")!, height: 1, depth: 1))
    e.memoryBarrier(scope: .buffers)
    if ProcessInfo.processInfo.environment["DEC_ATTN"] == "dec_attn_sg" { prm.nc *= 4 }   // research: 4 partials per chunk
    if part >= 0 {   // partial result (normalized output + log-sum-exp, FP32) into wrapped[part], no gate
        e.setComputePipelineState(psos["dec_combine_part"]!)
        e.setBuffer(OP, offset: 0, index: 0); e.setBuffer(MP, offset: 0, index: 1); e.setBuffer(LP, offset: 0, index: 2)
        e.setBuffer(wrapped[part], offset: 0, index: 3); e.setBytes(&prm, length: MemoryLayout<DPrm>.size, index: 4)
    } else {
    e.setComputePipelineState(psos["dec_combine_gate"]!)
    e.setBuffer(OP, offset: 0, index: 0); e.setBuffer(MP, offset: 0, index: 1); e.setBuffer(LP, offset: 0, index: 2)
    e.setBuffer(wrapped[Int(h[4])], offset: 0, index: 3); e.setBytes(&prm, length: MemoryLayout<DPrm>.size, index: 4)
    e.setBuffer(wrapped[Int(h[1])], offset: 0, index: 5); e.setBytes(&dst, length: MemoryLayout<DSt>.size, index: 6)
    }
    e.dispatchThreads(MTLSize(width: 256, height: 48, depth: 4), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    e.endEncoding()
    return true
}

/// One attention layer's decode verify step at p0 (8 rows): prep (INT8 Q; the new K / V rows into the cache), the
/// attention over [0, p0 + 8), x sigmoid(gate) -> o. h: wrapped q, g, kt, vt, o; st: DSt strides (9 x UInt32).
/// Blocks; returns the GPU time in ms (negative on error).
@_cdecl("dec_run")
public func dec_run(_ layer: Int32, _ h: UnsafePointer<Int32>, _ st: UnsafePointer<UInt32>, _ p0: Int32) -> Double {
    let cb = queue.makeCommandBuffer()!
    guard encode(cb, Int(layer), h, st, Int(p0)) else { return -1 }
    cb.commit(); cb.waitUntilCompleted()
    if cb.status != .completed { return -2 }
    return (cb.gpuEndTime - cb.gpuStartTime) * 1e3
}

/// Asynchronous form: returns a ticket for dec_wait.
@_cdecl("dec_submit")
public func dec_submit(_ layer: Int32, _ h: UnsafePointer<Int32>, _ st: UnsafePointer<UInt32>, _ p0: Int32) -> Int32 {
    let cb = queue.makeCommandBuffer()!
    guard encode(cb, Int(layer), h, st, Int(p0)) else { return -1 }
    ticket += 1; inflight[ticket] = cb
    cb.commit()
    return Int32(ticket)
}

/// Asynchronous partial form (split attention: this GPU part covers the keys of the layer's registered segments
/// from the first one's start): writes the partial output + log-sum-exp into wrapped[out]. Returns a ticket for dec_wait.
@_cdecl("dec_submit_part")
public func dec_submit_part(_ layer: Int32, _ h: UnsafePointer<Int32>, _ st: UnsafePointer<UInt32>, _ p0: Int32, _ out: Int32) -> Int32 {
    let cb = queue.makeCommandBuffer()!
    guard encode(cb, Int(layer), h, st, Int(p0), part: Int(out)) else { return -1 }
    ticket += 1; inflight[ticket] = cb
    cb.commit()
    return Int32(ticket)
}
/// Steady-state timing (research): `iters` copies of the layer's work in ONE command buffer, no CPU gaps, so the GPU
/// stays at its loaded clock. Returns GPU ms per copy.
@_cdecl("dec_bench")
public func dec_bench(_ layer: Int32, _ h: UnsafePointer<Int32>, _ st: UnsafePointer<UInt32>, _ p0: Int32, _ iters: Int32) -> Double {
    let cb = queue.makeCommandBuffer()!
    for _ in 0..<Int(iters) { guard encode(cb, Int(layer), h, st, Int(p0)) else { return -1 } }
    cb.commit(); cb.waitUntilCompleted()
    if cb.status != .completed { return -2 }
    return (cb.gpuEndTime - cb.gpuStartTime) * 1e3 / Double(iters)
}
@_cdecl("dec_wait")
public func dec_wait(_ t: Int32) -> Double {
    guard let cb = inflight.removeValue(forKey: Int(t)) else { return -1 }
    cb.waitUntilCompleted()
    if cb.status != .completed { return -2 }
    return (cb.gpuEndTime - cb.gpuStartTime) * 1e3
}

// ---- low-latency handoff: the GPU work is encoded and committed ahead (waiting on a shared event), the CPU fires it
// when part A returns and spins on a second event for the end (no commit / scheduling / wake-up on the critical path).
let evGo = dev.makeSharedEvent()!, evDone = dev.makeSharedEvent()!
var armed: [Int: (UInt64, MTLCommandBuffer)] = [:]
var evN: UInt64 = 0

@_cdecl("dec_arm")
public func dec_arm(_ layer: Int32, _ h: UnsafePointer<Int32>, _ st: UnsafePointer<UInt32>, _ p0: Int32) -> Int32 {
    evN += 1
    let v = evN, cb = queue.makeCommandBuffer()!
    cb.encodeWaitForEvent(evGo, value: v)
    guard encode(cb, Int(layer), h, st, Int(p0)) else { return -1 }
    cb.encodeSignalEvent(evDone, value: v)
    cb.commit()
    ticket += 1; armed[ticket] = (v, cb)
    return Int32(ticket)
}

@_cdecl("dec_fire")
public func dec_fire(_ t: Int32) -> Int32 {
    guard let (v, _) = armed[Int(t)] else { return -1 }
    evGo.signaledValue = v
    return 0
}

/// Spin until the armed work is done; returns the GPU time of its kernels' command buffer in ms (gpuStart counts from
/// the event wait's end only approximately, so callers time the wall themselves).
@_cdecl("dec_spin")
public func dec_spin(_ t: Int32) -> Double {
    guard let (v, cb) = armed.removeValue(forKey: Int(t)) else { return -1 }
    while evDone.signaledValue < v { if cb.status == .error { return -2 } }
    return 0
}

/// Commit now, spin for the end (no event wait): the plain path without the wake-up latency.
@_cdecl("dec_run_spin")
public func dec_run_spin(_ layer: Int32, _ h: UnsafePointer<Int32>, _ st: UnsafePointer<UInt32>, _ p0: Int32) -> Double {
    let cb = queue.makeCommandBuffer()!
    guard encode(cb, Int(layer), h, st, Int(p0)) else { return -1 }
    cb.commit()
    while cb.status != .completed { if cb.status == .error { return -2 } }
    return (cb.gpuEndTime - cb.gpuStartTime) * 1e3
}

// ---- latency breakdown (host time, s): before encode, after commit, GPU start, GPU end, after the wait returns
var lat = [Double](repeating: 0, count: 5)
@_cdecl("dec_run_lat")
public func dec_run_lat(_ layer: Int32, _ h: UnsafePointer<Int32>, _ st: UnsafePointer<UInt32>, _ p0: Int32, _ out: UnsafeMutablePointer<Double>) -> Double {
    let t0 = CACurrentMediaTime()
    let cb = queue.makeCommandBuffer()!
    guard encode(cb, Int(layer), h, st, Int(p0)) else { return -1 }
    cb.commit()
    let t1 = CACurrentMediaTime()
    cb.waitUntilCompleted()
    let t4 = CACurrentMediaTime()
    out[0] = t0; out[1] = t1; out[2] = cb.gpuStartTime; out[3] = cb.gpuEndTime; out[4] = t4
    return (cb.gpuEndTime - cb.gpuStartTime) * 1e3
}

// ---- GPU-side flag wait (dec_flag_wait): armed before part A, released by a plain CPU store
let flagIn = dev.makeBuffer(length: 16384, options: .storageModeShared)!, flagOut = dev.makeBuffer(length: 16384, options: .storageModeShared)!
let flagRes = dev.makeBuffer(length: 64, options: .storageModeShared)!
var fN: UInt32 = 0
var armed2: [UInt32: MTLCommandBuffer] = [:]

@_cdecl("dec_arm2")
public func dec_arm2(_ layer: Int32, _ h: UnsafePointer<Int32>, _ st: UnsafePointer<UInt32>, _ p0: Int32) -> Int32 {
    fN += 1
    var v = fN
    let cb = queue.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(psos["dec_flag_wait"]!)
    e.setBuffer(flagIn, offset: 0, index: 0); e.setBytes(&v, length: 4, index: 1); e.setBuffer(flagRes, offset: 0, index: 2)
    e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    e.endEncoding()
    guard encode(cb, Int(layer), h, st, Int(p0)) else { return -1 }
    let e2 = cb.makeComputeCommandEncoder()!
    e2.setComputePipelineState(psos["dec_flag_set"]!)
    e2.setBuffer(flagOut, offset: 0, index: 0); e2.setBytes(&v, length: 4, index: 1)
    e2.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    e2.endEncoding()
    cb.commit()
    armed2[v] = cb
    return Int32(v)
}

@_cdecl("dec_fire2")
public func dec_fire2(_ v: Int32) {
    let p = flagIn.contents().bindMemory(to: UInt32.self, capacity: 1)
    OSMemoryBarrier()
    p.pointee = UInt32(v)
    OSMemoryBarrier()
}

/// mode 0: waitUntilCompleted; mode 1: spin on the GPU-written flag (plain loads), then return (the command buffer is
/// retired later). Returns 1 if the GPU-side wait timed out, -2 on a command buffer error.
@_cdecl("dec_wait2")
public func dec_wait2(_ v: Int32, _ mode: Int32) -> Int32 {
    guard let cb = armed2.removeValue(forKey: UInt32(v)) else { return -1 }
    if mode == 1 {
        let p = flagOut.contents().bindMemory(to: UInt32.self, capacity: 1)
        var n = 0
        while true {
            OSMemoryBarrier()
            if p.pointee >= UInt32(v) { break }
            n += 1
            if n & 0xFFFFF == 0 && (cb.status == .error || cb.status == .completed) { break }   // rare Metal calls only
        }
        if cb.status == .error { return -2 }
        return 0
    }
    cb.waitUntilCompleted()
    if cb.status != .completed { return -2 }
    return Int32(flagRes.contents().bindMemory(to: UInt32.self, capacity: 2)[1])
}

// ---- warm-up kick: the GPU takes ~0.25-0.3 ms to start work after idling >= 2 ms (pipe/unit/gpu_wake_lat.swift);
// a tiny command buffer committed ~0.25 ms before the real one hides that. dec_kick_after(us): commit the kick `us`
// microseconds from now on a background thread (sleep, then spin for the last 0.2 ms).
let kickBuf = dev.makeBuffer(length: 4096, options: .storageModePrivate)!
// one persistent kick thread (QoS user-interactive): sleeps on a semaphore, then until 1.2 ms before the target
// (usleep overshoots by ~0.6 ms), then spins; a detached thread per kick landed ~0.7 ms late (16 Nov).
let kickSem = DispatchSemaphore(value: 0)
let kickSpinUs = Double(ProcessInfo.processInfo.environment["DEC_KICK_SPIN_US"] ?? "0")!
let kickEarlyUs = Double(ProcessInfo.processInfo.environment["DEC_KICK_EARLY_US"] ?? "0")!
var kickTarget = 0.0
let kickThread: Thread = {
    let th = Thread {
        while true {
            kickSem.wait()
            let t = kickTarget
            let rem = t - CACurrentMediaTime()
            if kickSpinUs > 0 {
                if rem > kickSpinUs * 1e-6 { usleep(UInt32(rem * 1e6 - kickSpinUs)) }
                while CACurrentMediaTime() < t {}
            } else if rem > kickEarlyUs * 1e-6 { usleep(UInt32(rem * 1e6 - kickEarlyUs)) }   // no spin (a spinning core slows the ANE call)
            let cb = queue.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
            e.setComputePipelineState(psos["dec_flag_set"]!); e.setBuffer(kickBuf, offset: 0, index: 0)
            var v: UInt32 = 1; e.setBytes(&v, length: 4, index: 1)
            e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            e.endEncoding(); kickT[0] = CACurrentMediaTime(); cb.commit(); cb.waitUntilCompleted()
            kickT[1] = cb.gpuStartTime; kickT[2] = cb.gpuEndTime; kickT[3] = t
        }
    }
    th.qualityOfService = .userInteractive
    th.start()
    return th
}()
@_cdecl("dec_kick_after")
public func dec_kick_after(_ us: Int32) {
    _ = kickThread
    kickTarget = CACurrentMediaTime() + Double(us) * 1e-6
    kickSem.signal()
}
var kickT = [Double](repeating: 0, count: 4)
/// the last kick: [commit, GPU start, GPU end, target] (host s)
@_cdecl("dec_kick_info")
public func dec_kick_info(_ out: UnsafeMutablePointer<Double>) { for i in 0..<4 { out[i] = kickT[i] } }

// ---- residency: every wrapped buffer in one MTLResidencySet attached to the queue (no per-commit residency work)
var rset: MTLResidencySet? = nil
@_cdecl("dec_residency")
public func dec_residency() -> Int32 {
    let d = MTLResidencySetDescriptor(); d.initialCapacity = wrapped.count + 8
    guard let s = try? dev.makeResidencySet(descriptor: d) else { return -1 }
    for b in wrapped { s.addAllocation(b) }
    for b in [Q8, QS, OP, MP, LP, kickBuf] { s.addAllocation(b) }
    s.commit(); s.requestResidency()
    queue.addResidencySet(s)
    rset = s
    return Int32(s.allocationCount)
}
