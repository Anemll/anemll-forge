// Core AI native Swift runtime: does a long inference loop leak output buffers?
//
//   build: DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer \
//          xcrun swiftc -O -swift-version 5 main.swift -o coreai_leak_test
//   run:   coreai_leak_test <mode> <calls> <entry> <model.aimodel>     (or ./run_mode.sh <mode> <calls> [entry])
//
// modes (loop modes run in Task.detached(priority: PRIO), default userInitiated)
//   info  load, print the function's inputs / outputs / strides, check NDArray copy-on-write behaviour
//   A     run(inputs:) with runtime-allocated outputs, outputs dropped every call
//   AF    run(inputs:) with runtime-allocated outputs, state outputs (<name>_out) fed back as inputs
//         (the pattern of the Python leak probe "plain" variant)
//   B     run(inputs:outputViews:) with one preallocated NDArray per output, the same buffers every call
//   AI    IOSurface inputs (Inputs + RawView(ioSurface:)), runtime-allocated outputs
//   BI    IOSurface inputs + IOSurface output views (MutableRawView(ioSurface:)), same surfaces every call
//   C/CI  one A-style call vs one B / BI-style call on the same inputs, compare every output
//   AB/ABI/RE/RQ  interleaved timing: A vs B / A,B,AI,BI / run vs encode(to: ComputeStream) / task priority
// env: INPUT_LAYOUT=packed (numpy-style packed input strides), PRIO, REPORT, GUARD_GB
//
// Every `REPORT` (default 100) calls: median / mean call time of the window, system wired memory
// (host_statistics64 wire_count) and the process physical footprint (task_info phys_footprint).
// A guard stops the loop if the footprint or wired memory grow by more than GUARD_GB (default 4 GB).

import CoreAI
import Darwin
import Foundation
import IOSurface

setvbuf(stdout, nil, _IOLBF, 0)

// ---- memory probes --------------------------------------------------------------------------------------------
func wiredGB() -> Double {
    var stats = vm_statistics64()
    var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &stats) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
        }
    }
    guard kr == KERN_SUCCESS else { return -1 }
    return Double(stats.wire_count) * Double(getpagesize()) / 1_073_741_824
}

func footprintGB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    guard kr == KERN_SUCCESS else { return -1 }
    return Double(info.phys_footprint) / 1_073_741_824
}

func nowMs() -> Double { Double(clock_gettime_nsec_np(CLOCK_UPTIME_RAW)) / 1e6 }

func median(_ a: [Double]) -> Double {
    let s = a.sorted()
    return s.isEmpty ? 0 : (s.count % 2 == 1 ? s[s.count / 2] : 0.5 * (s[s.count / 2 - 1] + s[s.count / 2]))
}

func fmt(_ x: Double, _ d: Int = 2) -> String { String(format: "%.\(d)f", x) }

// ---- NDArray helpers ------------------------------------------------------------------------------------------
func ndDesc(_ v: InferenceValue.Descriptor?, _ name: String) -> NDArrayDescriptor {
    guard let v else { fatalError("no descriptor for \(name)") }
    switch v {
    case .ndArray(let d): return d
    case .image: fatalError("\(name) is an image")
    @unknown default: fatalError("\(name): unknown kind")
    }
}

func elemSize(_ t: NDArray.ScalarType) -> Int {
    switch t {
    case .float16, .bfloat16, .int16, .uint16: return 2
    case .float32, .int32, .uint32: return 4
    case .float64, .int64, .uint64: return 8
    default: return 1
    }
}

/// A new NDArray laid out like `desc`, every byte of its storage set to `byte`; if `seed` is given and the
/// type is fp16, filled with small pseudo-random values instead.
func makeArray(_ desc: NDArrayDescriptor, byte: UInt8 = 0, seed: UInt64? = nil) -> NDArray {
    var a = NDArray(descriptor: desc)
    let n = desc.minimumByteCount
    a.mutableRawView().withUnsafeMutableBytes { p, _, _ in
        memset(p, Int32(byte), n)
        if var s = seed, desc.scalarType == .float16 {
            let f = p.bindMemory(to: Float16.self, capacity: n / 2)
            for i in 0..<(n / 2) {
                s = s &* 6364136223846793005 &+ 1442695040888963407
                f[i] = Float16((Double(s >> 11) / Double(1 << 53) * 2 - 1) * 0.2)
            }
        }
    }
    return a
}

func basePointer(_ a: NDArray) -> UInt {
    a.rawView().withUnsafeBytes { p, _, _ in UInt(bitPattern: p) }
}

/// Logical elements of an fp16 array as Float (strides taken in elements; checked in `info`).
func logicalF16(_ a: NDArray) -> [Float] {
    a.rawView().withUnsafeBytes { p, shapeSpan, strideSpan in
        let shape = (0..<shapeSpan.count).map { shapeSpan[$0] }
        let strides = (0..<strideSpan.count).map { strideSpan[$0] }
        let total = shape.reduce(1, *)
        var out = [Float](repeating: 0, count: total)
        let f = p.assumingMemoryBound(to: Float16.self)
        var idx = [Int](repeating: 0, count: shape.count)
        for k in 0..<total {
            var off = 0
            for d in 0..<shape.count { off += idx[d] * strides[d] }
            out[k] = Float(f[off])
            var d = shape.count - 1
            while d >= 0 {
                idx[d] += 1
                if idx[d] < shape[d] { break }
                idx[d] = 0
                d -= 1
            }
        }
        return out
    }
}

func shapeStrides(_ a: NDArray) -> String { "shape \(a.shape) strides \(a.strides)" }

// ---- preallocated output buffers ------------------------------------------------------------------------------
/// Owns the output NDArrays. Only this object holds them (no copies elsewhere), so the mutating
/// `mutableRawView()` never has a reason to copy-on-write.
final class OutputBuffers {
    let names: [String]
    var arrays: [NDArray]
    init(names: [String], descs: [NDArrayDescriptor], byte: UInt8 = 0) {
        self.names = names
        self.arrays = descs.map { makeArray($0, byte: byte) }
    }
}

/// The pattern that compiles with Swift 6.4 lifetime checking: build the ~Escapable MutableViews from a
/// dynamic list by re-anchoring each element's MutableRawView on the owning class instance.
func runInto(_ fn: InferenceFunction, _ inputs: [String: NDArray], _ bufs: OutputBuffers)
    async throws -> InferenceFunction.Outputs
{
    var views = InferenceFunction.MutableViews()
    for i in 0..<bufs.names.count {
        views.insert(_overrideLifetime(bufs.arrays[i].mutableRawView(), borrowing: bufs), for: bufs.names[i])
    }
    return try await fn.run(inputs: inputs, outputViews: views)
}

/// Packed-stride NDArray (NDArray(shape:scalarType:)), zeroed or filled like makeArray.
func makePacked(_ desc: NDArrayDescriptor, seed: UInt64? = nil) -> NDArray {
    var a = NDArray(shape: desc.shape, scalarType: desc.scalarType)
    let n = desc.shape.reduce(1, *) * elemSize(desc.scalarType)
    a.mutableRawView().withUnsafeMutableBytes { p, _, _ in fill(p, n, seed: desc.scalarType == .float16 ? seed : nil) }
    return a
}

func fill(_ p: UnsafeMutableRawPointer, _ n: Int, byte: UInt8 = 0, seed: UInt64?) {
    memset(p, Int32(byte), n)
    if var s = seed {
        let f = p.bindMemory(to: Float16.self, capacity: n / 2)
        for i in 0..<(n / 2) {
            s = s &* 6364136223846793005 &+ 1442695040888963407
            f[i] = Float16((Double(s >> 11) / Double(1 << 53) * 2 - 1) * 0.2)
        }
    }
}

// ---- IOSurface-backed buffers (the zero-copy pattern for a Python wrapper: numpy can view baseAddress) ----------
/// An IOSurface shaped like the tensor, as Core AI's IOSurface views require: width = last dimension,
/// height = product of the leading dimensions, bytesPerRow = the descriptor's row stride.
func makeSurface(_ d: NDArrayDescriptor, _ strides: [Int]) -> IOSurface {
    let es = elemSize(d.scalarType), shape = d.shape
    let width = shape.last ?? 1
    let height = max(1, shape.dropLast().reduce(1, *))
    let rowBytes = (shape.count >= 2 ? strides[shape.count - 2] : width) * es
    let props: [IOSurfacePropertyKey: Any] = [.width: width, .height: height, .bytesPerElement: es,
                                              .bytesPerRow: rowBytes, .allocSize: max(d.minimumByteCount, rowBytes * height, 16),
                                              .pixelFormat: 0]
    guard let s = IOSurface(properties: props) else { fatalError("IOSurface for \(shape) failed") }
    return s
}

/// One IOSurface per named value, laid out with the descriptor's preferred strides.
final class SurfaceSet {
    let names: [String]
    let descs: [NDArrayDescriptor]
    let surfaces: [IOSurface]
    let strides: [[Int]]
    /// packed: packed (numpy-style) strides instead of the descriptor's preferred ANE-aligned strides
    init(names: [String], descs: [NDArrayDescriptor], byte: UInt8 = 0, packed: Bool = false,
         seed: (String) -> UInt64? = { _ in nil }) {
        self.names = names
        self.descs = descs
        self.strides = descs.map { d in
            guard packed else { return d.preferredStrides }
            var st = [Int](repeating: 1, count: d.shape.count)
            for j in stride(from: d.shape.count - 2, through: 0, by: -1) { st[j] = st[j + 1] * d.shape[j + 1] }
            return st
        }
        self.surfaces = zip(descs, strides).map { makeSurface($0, $1) }
        for (i, s) in surfaces.enumerated() {
            s.lock(options: [], seed: nil)
            fill(s.baseAddress, s.allocationSize, byte: byte,
                 seed: descs[i].scalarType == .float16 ? seed(names[i]) : nil)
            s.unlock(options: [], seed: nil)
        }
    }
    func logical(_ i: Int) -> [Float] {
        let d = descs[i], shape = d.shape, strides = self.strides[i]
        let f = surfaces[i].baseAddress.assumingMemoryBound(to: Float16.self)
        let total = shape.reduce(1, *)
        var out = [Float](repeating: 0, count: total), idx = [Int](repeating: 0, count: shape.count)
        for k in 0..<total {
            var off = 0
            for j in 0..<shape.count { off += idx[j] * strides[j] }
            out[k] = Float(f[off])
            var j = shape.count - 1
            while j >= 0 { idx[j] += 1; if idx[j] < shape[j] { break }; idx[j] = 0; j -= 1 }
        }
        return out
    }
}

/// IOSurface inputs through InferenceFunction.Inputs (RawView(ioSurface:)); outputs either runtime-allocated
/// (outs == nil) or written into IOSurface MutableRawViews.
func runSurfaces(_ fn: InferenceFunction, _ ins: SurfaceSet, _ outs: SurfaceSet, useOutputViews: Bool)
    async throws -> InferenceFunction.Outputs
{
    var inputs = InferenceFunction.Inputs()
    for i in 0..<ins.names.count {
        let d = ins.descs[i], s = ins.surfaces[i]
        inputs.insert(_overrideLifetime(NDArray.RawView(ioSurface: s, scalarType: d.scalarType,
                                                        shape: d.shape, strides: ins.strides[i]),
                                        borrowing: ins), for: ins.names[i])
    }
    var views = InferenceFunction.MutableViews()
    if useOutputViews {
        for i in 0..<outs.names.count {
            let d = outs.descs[i], s = outs.surfaces[i]
            views.insert(_overrideLifetime(NDArray.MutableRawView(ioSurface: s, scalarType: d.scalarType,
                                                                  shape: d.shape, strides: outs.strides[i]),
                                           borrowing: outs), for: outs.names[i])
        }
    }
    return try await fn.run(inputs: inputs, outputViews: views)
}

// ---- main -----------------------------------------------------------------------------------------------------
let args = CommandLine.arguments
guard args.count >= 5 else {
    print("usage: coreai_leak_test <info|A|AF|B|C|AI|BI|CI|AB|ABI> <calls> <entry> <model.aimodel>")
    exit(2)
}
let mode = args[1], N = Int(args[2])!, entry = args[3], path = args[4]
let env = ProcessInfo.processInfo.environment
let REPORT = Int(env["REPORT"] ?? "100")!
let GUARD = Double(env["GUARD_GB"] ?? "4")!
let prioName = env["PRIO"] ?? "userInitiated"
let prio: TaskPriority = ["background": .background, "utility": .utility, "medium": .medium,
                          "userInitiated": .userInitiated, "high": .high][prioName]!

print("mode \(mode), \(N) calls, entry \(entry), model \(path)")
print("start: wired \(fmt(wiredGB())) GB, footprint \(fmt(footprintGB(), 3)) GB")
var t = nowMs()
let model = try await AIModel(contentsOf: URL(fileURLWithPath: path),
                              options: SpecializationOptions(preferredComputeUnitKind: .neuralEngine))
print("AIModel loaded in \(fmt((nowMs() - t) / 1000, 1)) s; functions \(model.functionNames)")
t = nowMs()
guard let fn = try model.loadFunction(named: entry) else { fatalError("no function \(entry)") }
let d = fn.descriptor
print("loadFunction(\(entry)) in \(fmt((nowMs() - t) / 1000, 1)) s: \(d.inputCount) inputs, "
      + "\(d.outputCount) outputs, states \(d.stateNames)")
print("after load: wired \(fmt(wiredGB())) GB, footprint \(fmt(footprintGB(), 3)) GB")

let inNames = d.inputNames, outNames = d.outputNames
let inDescs = inNames.map { ndDesc(d.inputDescriptor(of: $0), $0) }
let outDescs = outNames.map { ndDesc(d.outputDescriptor(of: $0), $0) }

// INPUT_LAYOUT=packed: inputs made with NDArray(shape:scalarType:) (packed strides, like numpy arrays handed to
// the Python binding) instead of the descriptor's preferred (ANE-aligned) strides
let packedInputs = env["INPUT_LAYOUT"] == "packed"
var inputs: [String: NDArray] = [:]
for (n, desc) in zip(inNames, inDescs) {
    inputs[n] = packedInputs ? makePacked(desc, seed: n == "x" ? 1 : nil) : makeArray(desc, seed: n == "x" ? 1 : nil)
}
if packedInputs { print("inputs use packed strides, e.g. commit \(shapeStrides(inputs["commit"] ?? inputs[inNames[0]]!))") }
// state outputs that map back onto an input (conv0_out -> conv0, ...)
let feedback = outNames.filter { $0.hasSuffix("_out") && inputs[String($0.dropLast(4))] != nil }

if mode == "info" {
    for (n, desc) in zip(inNames, inDescs) {
        print("  in  \(n): \(desc.scalarType) \(desc.shape) preferredStrides \(desc.preferredStrides) "
              + "bytes \(desc.minimumByteCount) | array \(shapeStrides(inputs[n]!))")
    }
    for (n, desc) in zip(outNames, outDescs) {
        print("  out \(n): \(desc.scalarType) \(desc.shape) preferredStrides \(desc.preferredStrides) "
              + "bytes \(desc.minimumByteCount)")
    }
    print("feedback pairs: \(feedback)")
    // copy-on-write check: a second copy of an NDArray, then mutableRawView() on the original
    var a = makeArray(outDescs[0])
    let b = a
    let p0 = basePointer(a)
    a.mutableRawView().withUnsafeMutableBytes { p, _, _ in p.storeBytes(of: 7, as: UInt8.self) }
    print("CoW check: base before \(String(p0, radix: 16)), after mutableRawView with a live copy "
          + "\(String(basePointer(a), radix: 16)), copy sees write: "
          + "\(b.rawView().withUnsafeBytes { p, _, _ in p.load(as: UInt8.self) } == 7)")
    var outs = try await fn.run(inputs: inputs)
    print("run(inputs:) returned \(outs.count) outputs: \(Array(outs.names))")
    if let v = outs.remove(outNames[0]), let nd = v.ndArray {
        print("  runtime-allocated \(outNames[0]): \(shapeStrides(nd)) type \(nd.scalarType)")
    }
    exit(0)
}

if mode == "C" {
    // A-style: runtime-allocated outputs
    var outsA = try await fn.run(inputs: inputs)
    var ref: [String: [Float]] = [:]
    for n in outNames {
        if let v = outsA.remove(n), let nd = v.ndArray, nd.scalarType == .float16 { ref[n] = logicalF16(nd) }
    }
    // B-style: preallocated buffers pre-filled with 0xFF bytes (fp16 NaN) so untouched elements show up
    let bufs = OutputBuffers(names: outNames, descs: outDescs, byte: 0xFF)
    let ptrs = bufs.arrays.map(basePointer)
    let outsB = try await runInto(fn, inputs, bufs)
    print("run(inputs:outputViews:) returned \(outsB.count) outputs: \(Array(outsB.names))")
    for (i, n) in outNames.enumerated() {
        let arr = bufs.arrays[i]
        guard arr.scalarType == .float16 else { print("  \(n): \(arr.scalarType), skipped"); continue }
        let got = logicalF16(arr)
        let nan = got.filter { $0.isNaN }.count
        let nz = got.filter { !$0.isNaN && $0 != 0 }.count
        var maxd: Float = 0
        if let r = ref[n] { for k in 0..<r.count where !got[k].isNaN { maxd = max(maxd, abs(r[k] - got[k])) } }
        let refnz = ref[n]?.filter { $0 != 0 }.count ?? -1
        print("  \(n): \(got.count) elems, untouched(NaN) \(nan), nonzero \(nz) (ref nonzero \(refnz)), "
              + "max|A-B| \(maxd), same base ptr \(basePointer(arr) == ptrs[i])")
    }
    exit(0)
}

let sIn = SurfaceSet(names: inNames, descs: inDescs, packed: packedInputs, seed: { $0 == "x" ? 1 : nil })
if packedInputs { print("IOSurface inputs use packed strides, e.g. \(inNames[5]) \(sIn.strides[5]) bytesPerRow \(sIn.surfaces[5].bytesPerRow)") }
let sOut = SurfaceSet(names: outNames, descs: outDescs, byte: 0xFF)

if mode == "CI" {
    // IOSurface inputs + IOSurface output views vs NDArray inputs + runtime outputs (same input values)
    var outsA = try await fn.run(inputs: inputs)
    var ref: [String: [Float]] = [:]
    for n in outNames { if let v = outsA.remove(n), let nd = v.ndArray { ref[n] = logicalF16(nd) } }
    let o = try await runSurfaces(fn, sIn, sOut, useOutputViews: true)
    print("IOSurface run returned \(o.count) runtime outputs")
    for (i, n) in outNames.enumerated() {
        let got = sOut.logical(i), r = ref[n]!
        var maxd: Float = 0
        for k in 0..<r.count where !got[k].isNaN { maxd = max(maxd, abs(r[k] - got[k])) }
        print("  \(n): untouched(NaN) \(got.filter { $0.isNaN }.count), nonzero \(got.filter { !$0.isNaN && $0 != 0 }.count)"
              + " (ref \(r.filter { $0 != 0 }.count)), max|A-BI| \(maxd)")
    }
    exit(0)
}

if mode == "ABI" {
    // interleaved timing of the four variants, blocks of N calls, 5 rounds, in a task at PRIO
    let inputsABI = inputs
    try await Task.detached(priority: prio) {
    let inputs = inputsABI
    print("ABI at priority \(prioName)")
    let bufsAB = OutputBuffers(names: outNames, descs: outDescs)
    var tot: [String: [Double]] = [:]
    let variants = ["A", "B", "AI", "BI"]
    for r in 1...5 {
        var line = "round \(r):"
        for v in variants {
            var ts: [Double] = []
            for _ in 0..<N {
                let t0 = nowMs()
                switch v {
                case "A": _ = try await fn.run(inputs: inputs)
                case "B": _ = try await runInto(fn, inputs, bufsAB)
                case "AI": _ = try await runSurfaces(fn, sIn, sOut, useOutputViews: false)
                default: _ = try await runSurfaces(fn, sIn, sOut, useOutputViews: true)
                }
                ts.append(nowMs() - t0)
            }
            if r > 1 { tot[v, default: []] += ts }   // round 1 = warm-up
            line += " \(v) \(fmt(median(ts)))"
        }
        print(line + " ms")
    }
    print("ABI medians (rounds 2-5): " + variants.map { "\($0) \(fmt(median(tot[$0]!))) ms" }.joined(separator: ", "))
    }.value
    exit(0)
}

// ---- encode(... to: ComputeStream) path: one stream reused for every call -------------------------------------
let stream = ComputeStream()
var asyncIns: [String: InferenceFunction.AsyncValue] = [:]
for n in inNames { asyncIns[n] = InferenceFunction.AsyncValue(inputs[n]!) }

/// E: encode with runtime-allocated outputs, then wait for every output value (full completion)
func encodeWait(_ fn: InferenceFunction) async throws {
    let outs = try fn.encode(inputs: asyncIns, to: stream)
    for (_, v) in outs { _ = try await v.ndArray }
}
/// ES: encode, then stream.currentWorkCompleted()
func encodeStream(_ fn: InferenceFunction) async throws {
    let outs = try fn.encode(inputs: asyncIns, to: stream)
    await stream.currentWorkCompleted()
    _ = outs
}

if mode == "RE" {
    // interleaved timing: run() vs encode()+wait vs encode()+currentWorkCompleted, blocks of N calls, 5 rounds
    var tot: [String: [Double]] = [:]
    let variants = ["run", "E", "ES"]
    for r in 1...5 {
        var line = "round \(r):"
        for v in variants {
            var ts: [Double] = []
            for _ in 0..<N {
                let t0 = nowMs()
                switch v {
                case "run": _ = try await fn.run(inputs: inputs)
                case "E": try await encodeWait(fn)
                default: try await encodeStream(fn)
                }
                ts.append(nowMs() - t0)
            }
            if r > 1 { tot[v, default: []] += ts }
            line += " \(v) \(fmt(median(ts)))"
        }
        print(line + " ms")
    }
    print("RE medians (rounds 2-5): " + variants.map { "\($0) \(fmt(median(tot[$0]!))) ms" }.joined(separator: ", "))
    exit(0)
}

if mode == "RQ" {
    // run() timing inside detached tasks of different priorities (QoS), blocks of N calls, 3 rounds
    print("main thread QoS \(qos_class_self().rawValue), Task.currentPriority \(Task.currentPriority)")
    let prios: [(String, TaskPriority)] = [("background", .background), ("utility", .utility), ("medium", .medium),
                                           ("userInitiated", .userInitiated), ("high", .high)]
    let inputsCopy = inputs
    for r in 1...3 {
        var line = "round \(r):"
        for (name, p) in prios {
            let med = try await Task.detached(priority: p) { () async throws -> Double in
                var ts: [Double] = []
                for _ in 0..<N { let t0 = nowMs(); _ = try await fn.run(inputs: inputsCopy); ts.append(nowMs() - t0) }
                return median(ts)
            }.value
            line += " \(name) \(fmt(med))"
        }
        print(line + " ms")
    }
    exit(0)
}

if mode == "AB" {
    // interleaved timing: blocks of `calls` A-style then B-style calls, 6 rounds, same process / conditions
    let bufsAB = OutputBuffers(names: outNames, descs: outDescs)
    var ta: [Double] = [], tb: [Double] = []
    for _ in 0..<5 { _ = try await fn.run(inputs: inputs); _ = try await runInto(fn, inputs, bufsAB) }
    for r in 1...6 {
        var a: [Double] = [], b: [Double] = []
        for _ in 0..<N { let t0 = nowMs(); _ = try await fn.run(inputs: inputs); a.append(nowMs() - t0) }
        for _ in 0..<N { let t0 = nowMs(); _ = try await runInto(fn, inputs, bufsAB); b.append(nowMs() - t0) }
        print("round \(r): A median \(fmt(median(a))) ms | B median \(fmt(median(b))) ms")
        ta += a; tb += b
    }
    print("AB overall: A median \(fmt(median(ta))) ms, B median \(fmt(median(tb))) ms, "
          + "A-B \(fmt(median(ta) - median(tb))) ms")
    exit(0)
}

// ---- loops ----------------------------------------------------------------------------------------------------
// The loop runs in a detached task at PRIO (default userInitiated). Top-level code runs at .medium, and the
// runtime's work then lands on lower-QoS threads: ~14.5 ms/call instead of ~9 ms (see mode RQ).

func runLoop(_ fn: InferenceFunction, _ startInputs: [String: NDArray], _ sIn: SurfaceSet, _ sOut: SurfaceSet)
    async throws
{
    var inputs = startInputs
    let bufs = OutputBuffers(names: outNames, descs: outDescs)
    let ptrs = bufs.arrays.map(basePointer)
    var window: [Double] = [], all: [Double] = []
    let w0 = wiredGB(), f0 = footprintGB()
    print("loop start (priority \(prioName)): wired \(fmt(w0)) GB, footprint \(fmt(f0, 3)) GB")
    var stopped = false
    for i in 1...N {
        let t0 = nowMs()
        switch mode {
        case "A":
            let outs = try await fn.run(inputs: inputs)
            _ = consume outs
        case "AF":
            var outs = try await fn.run(inputs: inputs)
            for n in feedback {
                if let v = outs.remove(n), let nd = v.ndArray { inputs[String(n.dropLast(4))] = nd }
            }
        case "B":
            let outs = try await runInto(fn, inputs, bufs)
            if i == 1 { print("B: run returned \(outs.count) runtime outputs besides the views") }
        case "AI":
            _ = try await runSurfaces(fn, sIn, sOut, useOutputViews: false)
        case "BI":
            let outs = try await runSurfaces(fn, sIn, sOut, useOutputViews: true)
            if i == 1 { print("BI: run returned \(outs.count) runtime outputs besides the views") }
        default:
            fatalError("unknown mode \(mode)")
        }
        let dt = nowMs() - t0
        window.append(dt)
        all.append(dt)
        if i % REPORT == 0 || i == N {
            let w = wiredGB(), f = footprintGB()
            print("calls \(String(format: "%5d", i)): median \(fmt(median(window))) ms, mean "
                  + "\(fmt(window.reduce(0, +) / Double(window.count))) ms, max \(fmt(window.max()!)) ms | "
                  + "wired \(fmt(w)) GB (\(w - w0 >= 0 ? "+" : "")\(fmt(w - w0))) | footprint \(fmt(f, 3)) GB "
                  + "(\(f - f0 >= 0 ? "+" : "")\(fmt(f - f0, 3)))")
            window.removeAll(keepingCapacity: true)
            if f - f0 > GUARD || w - w0 > GUARD {
                print("GUARD: memory grew by more than \(GUARD) GB, stopping at call \(i)")
                stopped = true
                break
            }
        }
    }
    if mode == "B" {
        let moved = zip(bufs.arrays.map(basePointer), ptrs).filter { $0 != $1 }.count
        print("B: output buffers whose storage moved during the run: \(moved)")
    }
    print("done\(stopped ? " (guard)" : ""): \(all.count) calls, overall median \(fmt(median(all))) ms, "
          + "first-100 median \(fmt(median(Array(all.prefix(100))))) ms, last-100 median "
          + "\(fmt(median(Array(all.suffix(100))))) ms")
}

let loopInputs = inputs
try await Task.detached(priority: prio) { try await runLoop(fn, loopInputs, sIn, sOut) }.value
