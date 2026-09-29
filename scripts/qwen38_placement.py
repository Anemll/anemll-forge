"""Per-op device placement (MLComputePlan) of a compiled model: prints ops not preferring the ANE.
    python qwen38_placement.py model.mlmodelc"""
import sys
from collections import Counter

import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan
from coremltools.models.compute_device import MLNeuralEngineComputeDevice

plan = MLComputePlan.load_from_path(sys.argv[1], compute_units=ct.ComputeUnit.CPU_AND_NE)
prog = plan.model_structure.program
counts = Counter()
for fname, fn in prog.functions.items():
    for op in fn.block.operations:
        u = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if u is None:
            continue
        dev = type(u.preferred_compute_device).__name__.replace("ML", "").replace("ComputeDevice", "")
        counts[dev] += 1
        if not isinstance(u.preferred_compute_device, MLNeuralEngineComputeDevice):
            outs = [f"{o.name}" for o in op.outputs][:1]
            print(f"  {dev:12s} {op.operator_name:28s} {outs} supported={[type(d).__name__[2:-13] for d in u.supported_compute_devices]}")
print(dict(counts))
