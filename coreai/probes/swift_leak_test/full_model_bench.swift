// Full Qwen3.8-27B Core AI target (16 decoder chunks) timed from Swift: verify-8 (v8_<ctx>) and prefill-64
// (p64_<ctx>) with IOSurface-backed inputs / outputs (zeros; x of chunk i+1 is the y surface of chunk i),
// everything inside Task.detached(priority: .userInitiated).
//
//   build: DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer \
//          xcrun swiftc -O -swift-version 5 full_model_bench.swift -o full_model_bench
//   run:   full_model_bench <model dir with manifest.json> [verify entry] [prefill entry]
//
// Memory guard: chunks are loaded one at a time. Before each load the free + purgeable + file-backed memory must
// stay above AVAIL_MIN_GB (default 0.75) after the expected per-chunk increment, and swap use must not have grown
// by more than SWAP_MAX_GB (default 0.5) since start. If not every chunk fits, the chunks are timed in rotating
// groups (load a group, time it as a sequence, unload it, next group) inside this one process and the full-model
// numbers are the per-call sums over the groups. GROUP=n forces a group size.
// env: WARM (5), NV (60 verify calls), WARM_P (3), NP (30 prefill calls)

import CoreAI
import Darwin
import Foundation
import IOSurface

setvbuf(stdout, nil, _IOLBF, 0)

// ---- memory ---------------------------------------------------------------------------------------------------
struct Mem { var wired: Double; var avail: Double; var swap: Double; var compressor: Double }

func mem() -> Mem {
    var s = vm_statistics64()
    var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
    _ = withUnsafeMutablePointer(to: &s) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
        }
    }
    let pg = Double(getpagesize()) / 1_073_741_824
    var sw = xsw_usage()
    var sz = MemoryLayout<xsw_usage>.size
    sysctlbyname("vm.swapusage", &sw, &sz, nil, 0)
    return Mem(wired: Double(s.wire_count) * pg,
               avail: Double(UInt64(s.free_count) + UInt64(s.purgeable_count) + UInt64(s.external_page_count)) * pg,
               swap: Double(sw.xsu_used) / 1_073_741_824,
               compressor: Double(s.compressor_page_count) * pg)
}

func fmt(_ x: Double, _ d: Int = 2) -> String { String(format: "%.\(d)f", x) }
func memStr(_ m: Mem) -> String {
    "wired \(fmt(m.wired)) GB, avail \(fmt(m.avail)) GB, swap used \(fmt(m.swap)) GB, compressor \(fmt(m.compressor)) GB"
}
func nowMs() -> Double { Double(clock_gettime_nsec_np(CLOCK_UPTIME_RAW)) / 1e6 }
func pct(_ a: [Double], _ p: Double) -> Double {
    let s = a.sorted()
    guard !s.isEmpty else { return 0 }
    let r = p * Double(s.count - 1)
    let lo = Int(r.rounded(.down)), hi = min(lo + 1, s.count - 1)
    return s[lo] + (s[hi] - s[lo]) * (r - Double(lo))
}

// ---- IOSurface-backed values ----------------------------------------------------------------------------------
func elemSize(_ t: NDArray.ScalarType) -> Int {
    switch t {
    case .float16, .bfloat16, .int16, .uint16: return 2
    case .float32, .int32, .uint32: return 4
    case .float64, .int64, .uint64: return 8
    default: return 1
    }
}

func ndDesc(_ v: InferenceValue.Descriptor?, _ name: String) -> NDArrayDescriptor {
    guard let v else { fatalError("no descriptor for \(name)") }
    switch v {
    case .ndArray(let d): return d
    case .image: fatalError("\(name) is an image")
    @unknown default: fatalError("\(name): unknown kind")
    }
}

/// An IOSurface shaped like the tensor (Core AI requires width = last dim; rows = product of the leading dims).
final class Surf {
    let s: IOSurface
    let d: NDArrayDescriptor
    init(_ d: NDArrayDescriptor) {
        self.d = d
        let es = elemSize(d.scalarType), shape = d.shape, st = d.preferredStrides
        let width = shape.last ?? 1, height = max(1, shape.dropLast().reduce(1, *))
        let rowBytes = (shape.count >= 2 ? st[shape.count - 2] : width) * es
        let props: [IOSurfacePropertyKey: Any] = [.width: width, .height: height, .bytesPerElement: es,
                                                  .bytesPerRow: rowBytes,
                                                  .allocSize: max(d.minimumByteCount, rowBytes * height, 16),
                                                  .pixelFormat: 0]
        guard let s = IOSurface(properties: props) else { fatalError("IOSurface for \(shape) failed") }
        s.lock(options: [], seed: nil)
        memset(s.baseAddress, 0, s.allocationSize)
        s.unlock(options: [], seed: nil)
        self.s = s
    }
    var bytes: Int { s.allocationSize }
}

/// The bound values of one entry of one chunk: input and output surfaces by name.
final class Binding {
    let inNames: [String], outNames: [String]
    let ins: [Surf], outs: [Surf]
    init(inNames: [String], ins: [Surf], outNames: [String], outs: [Surf]) {
        self.inNames = inNames; self.ins = ins; self.outNames = outNames; self.outs = outs
    }
}

func run(_ fn: InferenceFunction, _ b: Binding) async throws {
    var inputs = InferenceFunction.Inputs()
    for i in 0..<b.inNames.count {
        let x = b.ins[i], s = x.s
        inputs.insert(_overrideLifetime(NDArray.RawView(ioSurface: s, scalarType: x.d.scalarType, shape: x.d.shape,
                                                        strides: x.d.preferredStrides), borrowing: b),
                      for: b.inNames[i])
    }
    var views = InferenceFunction.MutableViews()
    for i in 0..<b.outNames.count {
        let x = b.outs[i], s = x.s
        views.insert(_overrideLifetime(NDArray.MutableRawView(ioSurface: s, scalarType: x.d.scalarType, shape: x.d.shape,
                                                              strides: x.d.preferredStrides), borrowing: b),
                     for: b.outNames[i])
    }
    let outs = try await fn.run(inputs: inputs, outputViews: views)
    precondition(outs.count == 0, "runtime allocated \(outs.count) outputs")
}

// ---- manifest -------------------------------------------------------------------------------------------------
let args = CommandLine.arguments
guard args.count >= 2 else { print("usage: full_model_bench <model dir> [verify entry] [prefill entry]"); exit(2) }
let root = URL(fileURLWithPath: (args[1] as NSString).expandingTildeInPath)
let env = ProcessInfo.processInfo.environment
let WARM = Int(env["WARM"] ?? "5")!, NV = Int(env["NV"] ?? "60")!
let WARM_P = Int(env["WARM_P"] ?? "3")!, NP = Int(env["NP"] ?? "30")!
let AVAIL_MIN = Double(env["AVAIL_MIN_GB"] ?? "0.75")!, SWAP_MAX = Double(env["SWAP_MAX_GB"] ?? "0.5")!
let FORCE_GROUP = env["GROUP"].flatMap { Int($0) }

let man = try JSONSerialization.jsonObject(with: Data(contentsOf: root.appendingPathComponent("manifest.json")))
    as! [String: Any]
let chunks = man["chunks"] as! [[String: Any]]
let ctx = ((man["ctxs"] as? [Int])?.max()) ?? 16384
let vEntry = args.count > 2 ? args[2] : "v8_\(ctx / 1024)k"
let pEntry = args.count > 3 ? args[3] : "p64_\((((man["pctxs"] as? [Int])?.max()) ?? ctx) / 1024)k"
// .aimodelc vs .aimodel: a .aimodelc whose MPSGraph package is newer than this OS can read makes the load crash
// (SIGSEGV after "No valid MPSGraph Package Version ... This OS supports MPSGraph Package Version up to 7.0.80";
// Xcode-beta coreai-build writes 7.1.2 while macOS 27.0 / 26A428 reads <= 7.0.80). PREFER=auto (default) picks the
// .aimodelc only when its package version <= OS_MPSGRAPH_MAX (default 7.0.80), else the .aimodel source, which the OS
// specializes on first load (~40 s per chunk, cached per executable name under ~/Library/Caches/coreai-cache).
let PREFER = env["PREFER"] ?? "auto"
let OS_MAX = env["OS_MPSGRAPH_MAX"] ?? "7.0.80"
func versionLE(_ a: String, _ b: String) -> Bool {
    let x = a.split(separator: ".").map { Int($0) ?? 0 }, y = b.split(separator: ".").map { Int($0) ?? 0 }
    for i in 0..<max(x.count, y.count) {
        let u = i < x.count ? x[i] : 0, v = i < y.count ? y[i] : 0
        if u != v { return u < v }
    }
    return true
}
/// Highest MPSGraph package version inside a compiled package (nil if none found).
func packageVersion(_ c: URL) -> String? {
    guard let e = FileManager.default.enumerator(at: c, includingPropertiesForKeys: nil) else { return nil }
    var best: String? = nil
    for case let u as URL in e where u.lastPathComponent == "manifest.plist" && u.path.contains(".mpsgraphpackage") {
        guard let d = NSDictionary(contentsOf: u), let pv = d["Package Version"] as? [String: Any] else { continue }
        for k in pv.keys where best == nil || !versionLE(k, best!) { best = k }
    }
    return best
}
func pick(_ entry: [String: Any]) -> URL {
    let src = root.appendingPathComponent(entry["file"] as! String)
    guard let cn = entry["compiled"] as? String, PREFER != "aimodel" else { return src }
    let comp = root.appendingPathComponent(cn)
    guard FileManager.default.fileExists(atPath: comp.path) else { return src }
    if PREFER == "aimodelc" { return comp }
    if let v = packageVersion(comp), !versionLE(v, OS_MAX) {
        print("  \(cn): MPSGraph package \(v) > OS max \(OS_MAX) -> loading \(entry["file"] as! String) instead")
        return src
    }
    return comp
}
let paths: [URL] = chunks.map(pick)
let names = chunks.map { ($0["file"] as! String).replacingOccurrences(of: ".aimodel", with: "") }
let headPath: URL? = (man["head"] as? [String: Any]).map(pick)
print("\(chunks.count) chunks, entries \(vEntry) / \(pEntry); loading \(paths.map { $0.pathExtension }.reduce(into: [String: Int]()) { $0[$1, default: 0] += 1 })"
      + (headPath.map { "; head \($0.lastPathComponent)" } ?? ""))
if env["DRY_RUN"] == "1" {
    for (n, p) in zip(names, paths) { print("  \(n): \(p.lastPathComponent)") }
    print("dry run: \(memStr(mem()))")
    exit(0)
}

// ---- one loaded chunk -----------------------------------------------------------------------------------------
final class Chunk {
    let idx: Int
    var model: AIModel?
    var fv: InferenceFunction?, fp: InferenceFunction?
    var bv: Binding?, bp: Binding?
    init(idx: Int) { self.idx = idx }
}

/// Surfaces shared across chunks (per-call inputs: cos, sin, mask, commit, ...) keyed by entry|name|shape; x/y ping-pong
/// per entry; per-chunk state / KV (conv0, rec1, pend2, k3, v3, ...) keyed by chunk|name|shape and shared by both entries.
var shared: [String: Surf] = [:]
func isPerChunk(_ n: String) -> Bool { n.range(of: #"^(conv|rec|pend|k|v)\d+$"#, options: .regularExpression) != nil }

func bind(_ fn: InferenceFunction, entry: String, chunk: Int, local: inout [String: Surf]) -> Binding {
    let d = fn.descriptor
    let parity = chunk % 2
    var ins: [Surf] = [], outs: [Surf] = []
    for n in d.inputNames {
        let desc = ndDesc(d.inputDescriptor(of: n), n)
        if n == "x" {
            let k = "\(entry)|xy\(parity)|\(desc.shape)"
            ins.append(shared[k] ?? { let s = Surf(desc); shared[k] = s; return s }())
        } else if isPerChunk(n) {
            let k = "\(n)|\(desc.shape)"
            ins.append(local[k] ?? { let s = Surf(desc); local[k] = s; return s }())
        } else {
            let k = "\(entry)|\(n)|\(desc.shape)"
            ins.append(shared[k] ?? { let s = Surf(desc); shared[k] = s; return s }())
        }
    }
    for n in d.outputNames {
        let desc = ndDesc(d.outputDescriptor(of: n), n)
        if n == "y" {
            let k = "\(entry)|xy\(1 - parity)|\(desc.shape)"
            outs.append(shared[k] ?? { let s = Surf(desc); shared[k] = s; return s }())
        } else {
            outs.append(Surf(desc))
        }
    }
    return Binding(inNames: d.inputNames, ins: ins, outNames: d.outputNames, outs: outs)
}

// ---- main -----------------------------------------------------------------------------------------------------
let prio: TaskPriority = .userInitiated
try await Task.detached(priority: prio) {
    let m0 = mem()
    print("start: \(memStr(m0))")
    if env["SPECIALIZE_ONLY"] == "1" {
        // one chunk at a time: specialize (compile + cache) and release; no functions loaded, nothing timed
        let todo = paths.filter { $0.pathExtension == "aimodel" } + (env["HEAD"] == "1" ? [headPath].compactMap { $0 }.filter { $0.pathExtension == "aimodel" } : [])
        for (i, p) in todo.enumerated() {
            let m = mem()
            if m.swap - m0.swap > SWAP_MAX { print("STOP: swap grew by \(fmt(m.swap - m0.swap)) GB"); break }
            if m.avail < AVAIL_MIN + 1.5 { print("STOP: low memory (\(memStr(m)))"); break }
            let t = nowMs()
            do {
                _ = try await AIModel.specialize(contentsOf: p, options: SpecializationOptions(preferredComputeUnitKind: .neuralEngine))
            }
            print("specialized \(i + 1)/\(todo.count) \(p.lastPathComponent) in \(fmt((nowMs() - t) / 1000, 1)) s | \(memStr(mem()))")
        }
        print("specialize pass done: \(memStr(mem()))")
        return
    }
    var perChunkInc = 1.6    // GB, updated from the first measured chunk
    var tv = [[Double]](repeating: [], count: chunks.count)   // [chunk][call] ms, verify
    var tp = [[Double]](repeating: [], count: chunks.count)   // prefill
    var groupTotalsV: [[Double]] = [], groupTotalsP: [[Double]] = []
    var groupSizes: [Int] = []
    var loadLog: [String] = []
    var next = 0
    var firstChunkMem: (model: Double, v: Double, p: Double, calls: Double)? = nil
    var aborted: String? = nil

    while next < chunks.count && aborted == nil {
        // ---- load a group
        var group: [Chunk] = []
        let mg = mem()
        while next < chunks.count {
            if let g = FORCE_GROUP, group.count >= g { break }
            let m = mem()
            if m.swap - m0.swap > SWAP_MAX { aborted = "swap grew by \(fmt(m.swap - m0.swap)) GB"; break }
            if m.avail - perChunkInc < AVAIL_MIN {
                if group.isEmpty { aborted = "not enough memory for one chunk (\(memStr(m)))" }
                break
            }
            let c = Chunk(idx: next)
            let a = mem()
            var t = nowMs()
            c.model = try await AIModel(contentsOf: paths[next],
                                        options: SpecializationOptions(preferredComputeUnitKind: .neuralEngine))
            let tm = nowMs() - t
            let b = mem()
            t = nowMs()
            c.fv = try c.model!.loadFunction(named: vEntry)
            let bv = mem()
            c.fp = try c.model!.loadFunction(named: pEntry)
            let tf = nowMs() - t
            let bp = mem()
            guard c.fv != nil, c.fp != nil else { fatalError("\(names[next]): missing \(vEntry) or \(pEntry)") }
            var local: [String: Surf] = [:]
            c.bv = bind(c.fv!, entry: vEntry, chunk: next, local: &local)
            c.bp = bind(c.fp!, entry: pEntry, chunk: next, local: &local)
            let surfGB = Double(local.values.map(\.bytes).reduce(0, +) + (c.bv!.outs + c.bp!.outs).map(\.bytes).reduce(0, +)) / 1_073_741_824
            // one call per entry: forces any lazy program load
            try await run(c.fv!, c.bv!)
            try await run(c.fp!, c.bp!)
            let bc = mem()
            let inc = bc.wired - a.wired
            if firstChunkMem == nil {
                firstChunkMem = (b.wired - a.wired, bv.wired - b.wired, bp.wired - bv.wired, bc.wired - bp.wired)
            }
            perChunkInc = max(perChunkInc * 0.5, inc + 0.1)
            let line = "load \(names[next]) (\(paths[next].pathExtension)): model \(fmt(tm / 1000, 1)) s, functions "
                + "\(fmt(tf / 1000, 1)) s | wired +\(fmt(b.wired - a.wired)) model, +\(fmt(bv.wired - b.wired)) "
                + "\(vEntry), +\(fmt(bp.wired - bv.wired)) \(pEntry), +\(fmt(bc.wired - bp.wired)) first calls "
                + "(chunk total +\(fmt(inc)) GB; its KV/state/output surfaces \(fmt(surfGB, 3)) GB) | \(memStr(bc))"
            print(line)
            loadLog.append(line)
            group.append(c)
            next += 1
        }
        if group.isEmpty { break }
        groupSizes.append(group.count)
        let ml = mem()
        print("group \(groupSizes.count): chunks \(names[group.first!.idx]) ... \(names[group.last!.idx]) "
              + "(\(group.count)), wired +\(fmt(ml.wired - mg.wired)) GB for the group | \(memStr(ml))")

        // ---- time the group as a sequence
        var gv: [Double] = [], gp: [Double] = []
        for it in 0..<(WARM + NV) {
            var tot = 0.0
            for c in group {
                let t = nowMs()
                try await run(c.fv!, c.bv!)
                let dt = nowMs() - t
                tot += dt
                if it >= WARM { tv[c.idx].append(dt) }
            }
            if it >= WARM { gv.append(tot) }
        }
        for it in 0..<(WARM_P + NP) {
            var tot = 0.0
            for c in group {
                let t = nowMs()
                try await run(c.fp!, c.bp!)
                let dt = nowMs() - t
                tot += dt
                if it >= WARM_P { tp[c.idx].append(dt) }
            }
            if it >= WARM_P { gp.append(tot) }
        }
        groupTotalsV.append(gv)
        groupTotalsP.append(gp)
        let mt = mem()
        print("group \(groupSizes.count) timed: verify median \(fmt(pct(gv, 0.5))) ms, prefill median "
              + "\(fmt(pct(gp, 0.5))) ms | \(memStr(mt))")
        if mt.swap - m0.swap > SWAP_MAX { aborted = "swap grew by \(fmt(mt.swap - m0.swap)) GB during timing" }

        // ---- unload the group (unless everything is loaded)
        if next < chunks.count {
            for c in group { c.bv = nil; c.bp = nil; c.fv = nil; c.fp = nil; c.model = nil }
            group.removeAll()
            try await Task.sleep(nanoseconds: 1_500_000_000)
            print("group unloaded: \(memStr(mem()))")
        }
    }

    // ---- report
    let done = tv.filter { !$0.isEmpty }.count
    print("\n==== results (\(done) of \(chunks.count) chunks timed; group sizes \(groupSizes)"
          + (aborted != nil ? "; STOPPED: \(aborted!)" : "") + ")")
    if let f = firstChunkMem {
        print("first chunk wired: AIModel +\(fmt(f.model)) GB, loadFunction(\(vEntry)) +\(fmt(f.v)) GB, "
              + "loadFunction(\(pEntry)) +\(fmt(f.p)) GB, first calls +\(fmt(f.calls)) GB")
    }
    print("per-chunk median ms (verify \(vEntry) | prefill \(pEntry)), p90 in brackets:")
    for i in 0..<chunks.count where !tv[i].isEmpty {
        print("  \(names[i]): \(fmt(pct(tv[i], 0.5))) [\(fmt(pct(tv[i], 0.9)))] | \(fmt(pct(tp[i], 0.5))) [\(fmt(pct(tp[i], 0.9)))]")
    }
    // full-model call j = sum over groups of the group's call j
    let nv = groupTotalsV.map(\.count).min() ?? 0, np = groupTotalsP.map(\.count).min() ?? 0
    let fullV = (0..<nv).map { j in groupTotalsV.reduce(0) { $0 + $1[j] } }
    let fullP = (0..<np).map { j in groupTotalsP.reduce(0) { $0 + $1[j] } }
    let exact = groupSizes.count == 1 && done == chunks.count
    print("full-model verify-8 (\(done) chunks, \(exact ? "one resident sequence" : "sum over groups")): median "
          + "\(fmt(pct(fullV, 0.5))) ms, p90 \(fmt(pct(fullV, 0.9))) ms over \(fullV.count) calls; "
          + "sum of per-chunk medians \(fmt(tv.filter { !$0.isEmpty }.map { pct($0, 0.5) }.reduce(0, +))) ms")
    let pm = pct(fullP, 0.5)
    print("full-model prefill-64 (\(done) chunks): median \(fmt(pm)) ms, p90 \(fmt(pct(fullP, 0.9))) ms over "
          + "\(fullP.count) calls -> \(fmt(64_000 / pm, 1)) tok/s; sum of per-chunk medians "
          + "\(fmt(tp.filter { !$0.isEmpty }.map { pct($0, 0.5) }.reduce(0, +))) ms")
    // ---- head (optional, HEAD=1): loaded alone after the chunks, timed on its own
    if env["HEAD"] == "1", aborted == nil, let h = man["head"] as? [String: Any] {
        _ = h
        let hp = headPath!
        let m = mem()
        if m.avail - 1.2 < AVAIL_MIN || m.swap - m0.swap > SWAP_MAX {
            print("head skipped: \(memStr(m))")
        } else {
            let hm = try await AIModel(contentsOf: hp, options: SpecializationOptions(preferredComputeUnitKind: .neuralEngine))
            let hname = env["HEAD_ENTRY"] ?? hm.functionNames.first!
            let hf = try hm.loadFunction(named: hname)!
            var local: [String: Surf] = [:]
            let hb = bind(hf, entry: "head", chunk: 0, local: &local)
            var ht: [Double] = []
            for it in 0..<(WARM + NV) {
                let t = nowMs(); try await run(hf, hb); if it >= WARM { ht.append(nowMs() - t) }
            }
            let hmed = pct(ht, 0.5)
            print("head \(hp.lastPathComponent) (\(hname), inputs \(hf.descriptor.inputNames), outputs "
                  + "\(hf.descriptor.outputNames)): median \(fmt(hmed)) ms, p90 \(fmt(pct(ht, 0.9))) ms, wired +"
                  + "\(fmt(mem().wired - m.wired)) GB")
            print("verify-8 16 chunks + head: median \(fmt(pct(fullV, 0.5) + hmed)) ms (chunk-sum median + head median)")
        }
    }
    print("end: \(memStr(mem()))")
}.value
