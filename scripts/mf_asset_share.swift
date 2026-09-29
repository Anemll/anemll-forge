// Do two functions of one multifunction model share weight memory when loaded from ONE MLModelAsset?
// Wired memory (host_statistics64) after load and first prediction of each function, for:
//   asset    : MLModelAsset(url:) once, MLModel.load(asset:configuration:) per function
//   separate : MLModel(contentsOf:configuration:) per function (as coremltools does)
// swiftc -O mf_asset_share.swift -o /tmp/mf_asset_share && /tmp/mf_asset_share <model.mlmodelc> ctx2048 ctx8192 [asset|separate]
import CoreML
import Foundation

func wiredGB() -> Double {
    var stats = vm_statistics64()
    var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &stats) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count) }
    }
    guard kr == KERN_SUCCESS else { return -1 }
    return Double(stats.wire_count) * Double(vm_kernel_page_size) / 1_073_741_824
}

func zeros(_ model: MLModel) throws -> MLFeatureProvider {
    var d: [String: MLFeatureValue] = [:]
    for (name, desc) in model.modelDescription.inputDescriptionsByName {
        guard let c = desc.multiArrayConstraint else { continue }
        let a = try MLMultiArray(shape: c.shape, dataType: .float16)
        memset(a.dataPointer, 0, a.count * 2)
        d[name] = MLFeatureValue(multiArray: a)
    }
    return try MLDictionaryFeatureProvider(dictionary: d)
}

@main
struct Main {
    static func main() async throws {
        let args = CommandLine.arguments
        let url = URL(fileURLWithPath: args[1])
        let fns = [args[2], args[3]]
        let mode = args.count > 4 ? args[4] : "asset"
        let cfg0 = MLModelConfiguration()
        cfg0.computeUnits = .cpuAndNeuralEngine
        let w0 = wiredGB()
        print(String(format: "[%@] wired before %.2f GB", mode, w0))
        var models: [MLModel] = []
        let asset: MLModelAsset? = mode == "asset" ? try MLModelAsset(url: url) : nil
        for f in fns {
            let cfg = cfg0.copy() as! MLModelConfiguration
            cfg.functionName = f
            let t = Date()
            let m: MLModel
            if let asset { m = try await MLModel.load(asset: asset, configuration: cfg) } else { m = try MLModel(contentsOf: url, configuration: cfg) }
            models.append(m)
            print(String(format: "   load %@: %.1fs, wired +%.2f GB", f, Date().timeIntervalSince(t), wiredGB() - w0))
        }
        for (f, m) in zip(fns, models) {
            let x = try zeros(m)
            let t = Date()
            _ = try await m.prediction(from: x)
            print(String(format: "   first use %@: %.0f ms, wired +%.2f GB", f, 1000 * Date().timeIntervalSince(t), wiredGB() - w0))
        }
        models.removeAll()
        sleep(1)
        print(String(format: "   released: wired +%.2f GB", wiredGB() - w0))
    }
}
