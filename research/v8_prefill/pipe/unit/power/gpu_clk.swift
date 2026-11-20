// GPU clock for short bursts (18 Nov 2026, M6): the served decode attention (dec_kernels.metal: dec_prep, dec_attn_r4,
// dec_combine_gate, encoded exactly as engine/dec_gpu.swift does) on one synthetic INT8 layer (segments
// 8K+8K+16K+32K+64K, random codes), run in regimes that try to keep the GPU at full clock across CPU gaps (the
// server's ~5-6 ms ANE calls between its 16 GPU attention calls). Every regime runs in ROUNDS interleaved rounds (the
// order rotates each round); each phase writes one JSON line (times for the IOReport / powermetrics windows, per-call
// GPU ms, wall ms commit -> return, start latency commit -> GPU start).
//   gpu_clk OUT.jsonl ROUNDS SECS REGIME [REGIME ...]
// REGIME = comma-separated options:
//   steady | b2b | gap=MS | idle             the base (idle: no attention calls, the keep-alive alone)
//   pat=G1:G2:..[,dg=MS]                     gaps cycling G1, G2, .. ms (the server: 5 / 9 ms), + MS before every 16th
//                                            call (the draft between verify steps)
//   p=KEYS                                   keys (default 65536)
//   ka=cont:US:TG                            keep-alive, another queue: back-to-back spin dispatches of US us (at full
//                                            clock), TG simdgroup-sized threadgroups each, 2 command buffers in flight
//   ka=contm:N:US:TG                         as cont, N serial dispatches of US us per command buffer
//   ka=per:MS:US:TG                          a spin dispatch of US us every MS ms (not waited on)
//   ka=full:US:DUTY                          full-GPU ALU bursts of US us at duty DUTY
//   pw=LEAD_US:tiny | pw=LEAD_US:spin:US:TG | pw=LEAD_US:full:US
//                                            pre-warm on the keep-alive queue LEAD_US before each attention commit
//   ramp=MS[,idlems=MS]                      the keep-alive's ramp: GPU idle (150 ms), keep-alive cont:200:1 started,
//                                            one attention call MS later, keep-alive stopped (repeat)
//   kaqos=ui|def|ut|bg                       QoS of the keep-alive thread (default ui)
//   ev                                       the attention command buffer committed at the gap start, waiting on an
//                                            MTLSharedEvent; the CPU signals it at the gap end (times from the signal)
//   qos=ui|ui2|ut|bg                         QoS of the submitting thread (default .default)
//   lc                                       ProcessInfo activity .latencyCritical + .userInitiated during the phase
//   dl=US                                    PRIVATE: -[MTLCommandBuffer commitWithDeadline:] (mach time + US us)
//   sdl=US                                   PRIVATE: -[MTLCommandBuffer setDeadline:] then commit
//   prio=N                                   PRIVATE: -[MTLCommandQueue setGPUPriority:] on the attention queue
// Build: swiftc -O -o gpu_clk gpu_clk.swift -framework Metal
import Foundation
import Metal
import QuartzCore
import ObjectiveC

let A = CommandLine.arguments
guard A.count >= 5 else { print("gpu_clk OUT.jsonl ROUNDS SECS REGIME..."); exit(2) }
let outPath = A[1], ROUNDS = Int(A[2])!, SECS = Double(A[3])!, REGIMES = Array(A[4...])
let SETTLE = Double(ProcessInfo.processInfo.environment["CLK_SETTLE"] ?? "0.4")!   // discarded start of each phase, s
let REST = Double(ProcessInfo.processInfo.environment["CLK_REST"] ?? "1.0")!       // idle between phases, s
let here = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
let kpath = ProcessInfo.processInfo.environment["DEC_KERNELS"] ?? here.appendingPathComponent("../../engine/dec_kernels.metal").path

let dev = MTLCreateSystemDefaultDevice()!
let queue = dev.makeCommandQueue()!     // attention (as dec_gpu.swift's single queue)
let kq = dev.makeCommandQueue()!        // keep-alive / pre-warm
var tb = mach_timebase_info_data_t(); mach_timebase_info(&tb)
func machNow() -> UInt64 { mach_absolute_time() }
func us2mach(_ us: Double) -> UInt64 { UInt64(us * 1e3 * Double(tb.denom) / Double(tb.numer)) }
func sleepUntil(_ t: UInt64) { mach_wait_until(t) }
func sleepUs(_ us: Double) { if us > 0 { mach_wait_until(machNow() + us2mach(us)) } }

// ---- kernels
let opts = MTLCompileOptions(); opts.languageVersion = .version4_1
let lib = try! dev.makeLibrary(source: try! String(contentsOfFile: kpath, encoding: .utf8), options: opts)
var psos: [String: MTLComputePipelineState] = [:]
for n in ["dec_prep", "dec_combine_gate"] { psos[n] = try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }
psos["dec_attn"] = try! dev.makeComputePipelineState(function: lib.makeFunction(name: ProcessInfo.processInfo.environment["DEC_ATTN"] ?? "dec_attn_r4")!)
let kaSrc = """
#include <metal_stdlib>
using namespace metal;
kernel void ka_spin(device float *o [[buffer(0)]], constant uint &n [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    float a = float(i) * 1e-6f, b = 1.0001f, c = 0.9999f;
    for (uint j = 0; j < n; j++) { a = fma(a, b, c); b = fma(b, c, a); }
    if (a == 1234.5f) o[i & 1023] = b;
}
kernel void ka_tiny(device uint *o [[buffer(0)]], uint i [[thread_position_in_grid]]) { if (i == 0) o[0] += 1; }
"""
let klib = try! dev.makeLibrary(source: kaSrc, options: MTLCompileOptions())
let spinPSO = try! dev.makeComputePipelineState(function: klib.makeFunction(name: "ka_spin")!)
let tinyPSO = try! dev.makeComputePipelineState(function: klib.makeFunction(name: "ka_tiny")!)
let kaBuf = dev.makeBuffer(length: 1 << 16, options: .storageModePrivate)!
let FULL_THREADS = 1 << 16

func encSpin(_ cb: MTLCommandBuffer, _ n: UInt32, threads: Int) {
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(spinPSO); e.setBuffer(kaBuf, offset: 0, index: 0)
    var nn = n; e.setBytes(&nn, length: 4, index: 1)
    e.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(threads, threads >= 256 ? 256 : 32), height: 1, depth: 1))
    e.endEncoding()
}
func encTiny(_ cb: MTLCommandBuffer) {
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(tinyPSO); e.setBuffer(kaBuf, offset: 0, index: 0)
    e.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    e.endEncoding()
}

// ---- synthetic layer (as dec_clock.py)
struct DPrm { var n, p0, ck, nc: UInt32; var start: (UInt32, UInt32, UInt32, UInt32, UInt32); var rows: (UInt32, UInt32, UInt32, UInt32, UInt32) }
struct DSt { var qh, qt, kh, kt, vh, vt, gh, gt, oc: UInt32 }
struct Seg { var start: Int; var rows: Int; var k, ks, v, vs: MTLBuffer }
let CK = 2048
var rng: UInt64 = 0x9E3779B97F4A7C15
func fillBytes(_ b: MTLBuffer) {
    let p = b.contents().bindMemory(to: UInt64.self, capacity: b.length / 8)
    for i in 0..<(b.length / 8) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; p[i] = rng }
}
func fillHalf(_ b: MTLBuffer, _ lo: Float, _ hi: Float) {
    let p = b.contents().bindMemory(to: Float16.self, capacity: b.length / 2)
    for i in 0..<(b.length / 2) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; p[i] = Float16(lo + (hi - lo) * Float(rng >> 40) / Float(1 << 24)) }
}
var segs: [Seg] = []
var start = 0
for L in [8192, 8192, 16384, 32768, 65536] {
    let k = dev.makeBuffer(length: 4 * 256 * L, options: .storageModeShared)!, v = dev.makeBuffer(length: 4 * 256 * L, options: .storageModeShared)!
    let ks = dev.makeBuffer(length: 4 * L * 2, options: .storageModeShared)!, vs = dev.makeBuffer(length: 4 * L * 2, options: .storageModeShared)!
    fillBytes(k); fillBytes(v); fillHalf(ks, 0.01, 0.03); fillHalf(vs, 0.003, 0.013)
    segs.append(Seg(start: start, rows: L, k: k, ks: ks, v: v, vs: vs)); start += L
}
let qb = dev.makeBuffer(length: 24 * 8 * 256 * 2, options: .storageModeShared)!; fillHalf(qb, -2, 2)
let gb = dev.makeBuffer(length: 24 * 8 * 256 * 2, options: .storageModeShared)!
let ktb = dev.makeBuffer(length: 4 * 8 * 256 * 2, options: .storageModeShared)!, vtb = dev.makeBuffer(length: 4 * 8 * 256 * 2, options: .storageModeShared)!
let ob = dev.makeBuffer(length: 6144 * 8 * 2, options: .storageModeShared)!
let NCMAX = (131072 / CK + 1)
let Q8 = dev.makeBuffer(length: 24 * 8 * 256, options: .storageModePrivate)!, QS = dev.makeBuffer(length: 24 * 8 * 4, options: .storageModePrivate)!
let OP = dev.makeBuffer(length: 4 * NCMAX * 3 * 16 * 256 * 4, options: .storageModePrivate)!
let MP = dev.makeBuffer(length: 4 * NCMAX * 3 * 16 * 4, options: .storageModePrivate)!, LP = dev.makeBuffer(length: 4 * NCMAX * 3 * 16 * 4, options: .storageModePrivate)!

func encodeAttn(_ cb: MTLCommandBuffer, _ p0: Int) {
    let n = p0 + 8
    var s5 = [UInt32](repeating: 0, count: 5), r5 = [UInt32](repeating: 0, count: 5)
    for (i, s) in segs.enumerated() { s5[i] = UInt32(s.start); r5[i] = UInt32(s.rows) }
    var prm = DPrm(n: UInt32(n), p0: UInt32(p0), ck: UInt32(CK), nc: UInt32((n - segs[0].start + CK - 1) / CK),
                   start: (s5[0], s5[1], s5[2], s5[3], s5[4]), rows: (r5[0], r5[1], r5[2], r5[3], r5[4]))
    var dst = DSt(qh: 2048, qt: 256, kh: 2048, kt: 256, vh: 2048, vt: 256, gh: 2048, gt: 256, oc: 8)
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(psos["dec_prep"]!)
    e.setBuffer(qb, offset: 0, index: 0); e.setBuffer(ktb, offset: 0, index: 1)
    e.setBuffer(vtb, offset: 0, index: 2); e.setBuffer(Q8, offset: 0, index: 3); e.setBuffer(QS, offset: 0, index: 4)
    for i in 0..<5 {
        e.setBuffer(segs[i].k, offset: 0, index: 5 + i); e.setBuffer(segs[i].ks, offset: 0, index: 10 + i)
        e.setBuffer(segs[i].v, offset: 0, index: 15 + i); e.setBuffer(segs[i].vs, offset: 0, index: 20 + i)
    }
    e.setBytes(&prm, length: MemoryLayout<DPrm>.size, index: 25); e.setBytes(&dst, length: MemoryLayout<DSt>.size, index: 26)
    e.dispatchThreadgroups(MTLSize(width: 32, height: 8, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    e.memoryBarrier(scope: .buffers)
    e.setComputePipelineState(psos["dec_attn"]!)
    e.setBuffer(Q8, offset: 0, index: 0); e.setBuffer(QS, offset: 0, index: 1)
    for i in 0..<5 {
        e.setBuffer(segs[i].k, offset: 0, index: 2 + i); e.setBuffer(segs[i].ks, offset: 0, index: 7 + i)
        e.setBuffer(segs[i].v, offset: 0, index: 12 + i); e.setBuffer(segs[i].vs, offset: 0, index: 17 + i)
    }
    e.setBuffer(OP, offset: 0, index: 22); e.setBuffer(MP, offset: 0, index: 23); e.setBuffer(LP, offset: 0, index: 24)
    e.setBytes(&prm, length: MemoryLayout<DPrm>.size, index: 25)
    e.dispatchThreadgroups(MTLSize(width: Int(prm.nc), height: 3, depth: 4), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    e.memoryBarrier(scope: .buffers)
    e.setComputePipelineState(psos["dec_combine_gate"]!)
    e.setBuffer(OP, offset: 0, index: 0); e.setBuffer(MP, offset: 0, index: 1); e.setBuffer(LP, offset: 0, index: 2)
    e.setBuffer(ob, offset: 0, index: 3); e.setBytes(&prm, length: MemoryLayout<DPrm>.size, index: 4)
    e.setBuffer(gb, offset: 0, index: 5); e.setBytes(&dst, length: MemoryLayout<DSt>.size, index: 6)
    e.dispatchThreads(MTLSize(width: 256, height: 48, depth: 4), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    e.endEncoding()
}

// ---- private selectors (research only)
typealias VoidU64 = @convention(c) (AnyObject, Selector, UInt64) -> Void
typealias BoolU64 = @convention(c) (AnyObject, Selector, UInt64) -> Bool
func callVU(_ o: AnyObject, _ s: String, _ v: UInt64) { let sel = NSSelectorFromString(s); unsafeBitCast(class_getMethodImplementation(object_getClass(o)!, sel)!, to: VoidU64.self)(o, sel, v) }
func callBU(_ o: AnyObject, _ s: String, _ v: UInt64) -> Bool { let sel = NSSelectorFromString(s); return unsafeBitCast(class_getMethodImplementation(object_getClass(o)!, sel)!, to: BoolU64.self)(o, sel, v) }
typealias U64F = @convention(c) (AnyObject, Selector) -> UInt64
func callU(_ o: AnyObject, _ s: String) -> UInt64 { let sel = NSSelectorFromString(s); return unsafeBitCast(class_getMethodImplementation(object_getClass(o)!, sel)!, to: U64F.self)(o, sel) }

// ---- spin calibration: iterations per us at full clock (after a long full-GPU burst, back to back)
func calib(threads: Int) -> Double {
    var best = 0.0
    for _ in 0..<3 {
        let warm = kq.makeCommandBuffer()!; encSpin(warm, 400_000, threads: FULL_THREADS); warm.commit()
        let cb = kq.makeCommandBuffer()!; encSpin(cb, 200_000, threads: threads); cb.commit(); cb.waitUntilCompleted()
        best = max(best, 200_000 / ((cb.gpuEndTime - cb.gpuStartTime) * 1e6))
    }
    return best
}
let itPerUsTG = calib(threads: 32), itPerUsFull = calib(threads: FULL_THREADS)
FileHandle.standardError.write(String(format: "calib: spin 1 simdgroup %.1f it/us, full GPU (%d threads) %.1f it/us\n", itPerUsTG, FULL_THREADS, itPerUsFull).data(using: .utf8)!)

// ---- keep-alive thread
final class KA {
    var mode: [String]
    var run = true
    let done = DispatchSemaphore(value: 0)
    var busy = 0.0, cbs = 0
    var qos: QualityOfService = .userInteractive
    init(_ spec: String, qos: String? = nil) {
        mode = spec.split(separator: ":").map(String.init)
        switch qos ?? "ui" { case "ut": self.qos = .utility; case "bg": self.qos = .background; case "def": self.qos = .default; default: self.qos = .userInteractive }
    }
    func start() {
        let th = Thread { [self] in
            var prev: MTLCommandBuffer? = nil
            var next = machNow()
            while run {
                switch mode[0] {
                case "cont":
                    let us = Double(mode[1])!, tg = Int(mode[2])!
                    let cb = kq.makeCommandBuffer()!; encSpin(cb, UInt32(us * itPerUsTG), threads: 32 * tg); cb.commit(); cbs += 1
                    if let p = prev { p.waitUntilCompleted(); busy += p.gpuEndTime - p.gpuStartTime }
                    prev = cb
                case "contm":   // N serial dispatches of US us per command buffer (fewer CPU wake-ups), 2 in flight
                    let nd = Int(mode[1])!, us = Double(mode[2])!, tg = Int(mode[3])!
                    let cb = kq.makeCommandBuffer()!
                    let e = cb.makeComputeCommandEncoder()!
                    e.setComputePipelineState(spinPSO); e.setBuffer(kaBuf, offset: 0, index: 0)
                    var nn = UInt32(us * itPerUsTG); e.setBytes(&nn, length: 4, index: 1)
                    for _ in 0..<nd { e.dispatchThreads(MTLSize(width: 32 * tg, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)) }
                    e.endEncoding(); cb.commit(); cbs += 1
                    if let p = prev { p.waitUntilCompleted(); busy += p.gpuEndTime - p.gpuStartTime }
                    prev = cb
                case "per":
                    let ms = Double(mode[1])!, us = Double(mode[2])!, tg = Int(mode[3])!
                    next += us2mach(ms * 1e3); sleepUntil(next)
                    let cb = kq.makeCommandBuffer()!
                    if us > 0 { encSpin(cb, UInt32(us * itPerUsTG), threads: 32 * tg) } else { encTiny(cb) }
                    cb.addCompletedHandler { [self] c in busy += c.gpuEndTime - c.gpuStartTime }
                    cb.commit(); cbs += 1
                case "full":
                    let us = Double(mode[1])!, duty = Double(mode[2])!
                    let cb = kq.makeCommandBuffer()!; encSpin(cb, UInt32(us * itPerUsFull), threads: FULL_THREADS); cb.commit(); cb.waitUntilCompleted(); cbs += 1
                    let g = cb.gpuEndTime - cb.gpuStartTime; busy += g
                    sleepUs(g * 1e6 * (1 / duty - 1))
                default: fatalError("ka mode")
                }
            }
            prev?.waitUntilCompleted()
            done.signal()
        }
        th.qualityOfService = qos
        th.start()
    }
    func stop() { run = false; done.wait(); kq.makeCommandBuffer().map { $0.commit(); $0.waitUntilCompleted() } }
}

// ---- one phase
let ev = dev.makeSharedEvent()!
var evN: UInt64 = 0
func pct(_ a: [Double], _ q: Double) -> Double { if a.isEmpty { return .nan }; let s = a.sorted(); return s[min(s.count - 1, Int(q * Double(s.count - 1) + 0.5))] }
func r4(_ x: Double) -> Double { (x * 1e4).rounded() / 1e4 }

func runPhase(_ spec: String, _ round: Int) -> [String: Any] {
    var o: [String: String] = [:]
    for part in spec.split(separator: ",") { let kv = part.split(separator: "=", maxSplits: 1).map(String.init); o[kv[0]] = kv.count > 1 ? kv[1] : "1" }
    let P = Int(o["p"] ?? "65536")!
    let gapMs = Double(o["gap"] ?? "0")!
    let pat = (o["pat"] ?? "").split(separator: ":").compactMap { Double($0) }, dg = Double(o["dg"] ?? "0")!
    var callIdx = 0
    var ka: KA? = nil
    if let k = o["ka"] { ka = KA(k, qos: o["kaqos"]); ka!.start() }
    var act: NSObjectProtocol? = nil
    if o["lc"] != nil { act = ProcessInfo.processInfo.beginActivity(options: [.latencyCritical, .userInitiated], reason: "gpu_clk") }
    if let pr = o["prio"] { _ = callBU(queue, "setGPUPriority:", UInt64(pr)!) }
    var res: [String: Any] = ["name": spec, "round": round, "P": P]
    var gms: [Double] = [], wms: [Double] = [], lms: [Double] = []
    let pw = o["pw"]?.split(separator: ":").map(String.init)
    let done = DispatchSemaphore(value: 0)
    var t0 = 0.0, t1 = 0.0
    let th = Thread {
        let tStart = CACurrentMediaTime(); t0 = Date().timeIntervalSince1970 + SETTLE
        while CACurrentMediaTime() < tStart + SETTLE + SECS {
            let keep = CACurrentMediaTime() >= tStart + SETTLE
            if o["idle"] != nil { sleepUs(20_000); continue }
            if let rl = o["ramp"] {   // ramp: GPU idle IDLE ms, keep-alive (cont:200:1) started, one attention call RAMP ms later
                sleepUs(Double(o["idlems"] ?? "150")! * 1e3)
                let k = KA("cont:200:1"); k.start(); sleepUs(Double(rl)! * 1e3)
                let cb = queue.makeCommandBuffer()!; encodeAttn(cb, P)
                let tc = CACurrentMediaTime(); cb.commit(); cb.waitUntilCompleted(); let tr = CACurrentMediaTime()
                k.stop()
                if keep { gms.append((cb.gpuEndTime - cb.gpuStartTime) * 1e3); wms.append((tr - tc) * 1e3); lms.append((cb.gpuStartTime - tc) * 1e3) }
                continue
            }
            if o["steady"] != nil {
                let cb = queue.makeCommandBuffer()!; for _ in 0..<50 { encodeAttn(cb, P) }; cb.commit(); cb.waitUntilCompleted()
                if keep { gms.append((cb.gpuEndTime - cb.gpuStartTime) * 1e3 / 50) }
                continue
            }
            var tc = 0.0
            let cb: MTLCommandBuffer
            var gapMs = gapMs
            if !pat.isEmpty { gapMs = pat[callIdx % pat.count] + (callIdx % 16 == 0 ? dg : 0) }
            callIdx += 1
            if o["ev"] != nil {
                evN += 1; let v = evN
                cb = queue.makeCommandBuffer()!; cb.encodeWaitForEvent(ev, value: v); encodeAttn(cb, P); cb.commit()
                sleepUs(gapMs * 1e3)
                tc = CACurrentMediaTime(); ev.signaledValue = v
            } else {
                if let pw = pw {
                    let lead = Double(pw[0])!
                    sleepUs(gapMs * 1e3 - lead)
                    let w = kq.makeCommandBuffer()!
                    switch pw[1] {
                    case "tiny": encTiny(w)
                    case "spin": encSpin(w, UInt32(Double(pw[2])! * itPerUsTG), threads: 32 * Int(pw[3])!)
                    case "full": encSpin(w, UInt32(Double(pw[2])! * itPerUsFull), threads: FULL_THREADS)
                    default: fatalError("pw kind")
                    }
                    w.commit()
                    sleepUs(lead)
                } else { sleepUs(gapMs * 1e3) }
                cb = queue.makeCommandBuffer()!; encodeAttn(cb, P)
                tc = CACurrentMediaTime()
                if let d = o["dl"] { callVU(cb, "commitWithDeadline:", machNow() + us2mach(Double(d)!)) }
                else if let d = o["sdl"] { callVU(cb, "setDeadline:", machNow() + us2mach(Double(d)!)); cb.commit() }
                else { cb.commit() }
            }
            cb.waitUntilCompleted()
            let tr = CACurrentMediaTime()
            if keep {
                gms.append((cb.gpuEndTime - cb.gpuStartTime) * 1e3); wms.append((tr - tc) * 1e3); lms.append((cb.gpuStartTime - tc) * 1e3)
            }
        }
        t1 = Date().timeIntervalSince1970
        done.signal()
    }
    switch o["qos"] ?? "def" {
    case "ui": th.qualityOfService = .userInteractive
    case "ut": th.qualityOfService = .utility
    case "bg": th.qualityOfService = .background
    case "ui2": th.qualityOfService = .userInitiated
    default: th.qualityOfService = .default
    }
    th.start(); done.wait()
    if let k = ka { k.stop(); res["ka_busy_pct"] = r4(100 * k.busy / (SETTLE + SECS)); res["ka_cbs"] = k.cbs }
    if let a = act { ProcessInfo.processInfo.endActivity(a) }
    if o["prio"] != nil { _ = callBU(queue, "setGPUPriority:", 1) }
    res["t0"] = t0; res["t1"] = t1; res["n"] = gms.count
    if !gms.isEmpty { res["gpu_ms"] = [pct(gms, 0.1), pct(gms, 0.5), pct(gms, 0.9)].map(r4) }
    if !wms.isEmpty { res["wall_ms"] = [pct(wms, 0.1), pct(wms, 0.5), pct(wms, 0.9)].map(r4); res["lat_ms"] = [pct(lms, 0.1), pct(lms, 0.5), pct(lms, 0.9)].map(r4) }
    res["gpu_ms_all"] = gms.map(r4)
    if !wms.isEmpty { res["wall_ms_all"] = wms.map(r4) }
    return res
}

// warm-up (pipelines, residency)
for _ in 0..<3 { let cb = queue.makeCommandBuffer()!; encodeAttn(cb, 65536); cb.commit(); cb.waitUntilCompleted() }
FileManager.default.createFile(atPath: outPath, contents: nil)
let fh = FileHandle(forWritingAtPath: outPath)!
print(String(format: "ready %.3f queue gpuPriority %llu", Date().timeIntervalSince1970, callU(queue, "getGPUPriority"))); fflush(stdout)
for r in 0..<ROUNDS {
    for i in 0..<REGIMES.count {
        let spec = REGIMES[(i + r) % REGIMES.count]
        sleepUs(REST * 1e6)
        let res = runPhase(spec, r)
        let d = try! JSONSerialization.data(withJSONObject: res)
        fh.write(d); fh.write("\n".data(using: .utf8)!)
        let g = res["gpu_ms"] as? [Double] ?? [0, 0, 0], w = res["wall_ms"] as? [Double]
        print(String(format: "r%d %-34@ gpu %.3f ms  wall %@  n %d%@", r, spec as NSString, g[1], w.map { String(format: "%.3f", $0[1]) } ?? "-",
                     res["n"] as! Int, res["ka_busy_pct"].map { String(format: "  ka busy %.1f%%", $0 as! Double) } ?? "")); fflush(stdout)
    }
}
fh.closeFile()
print("done"); fflush(stdout)
