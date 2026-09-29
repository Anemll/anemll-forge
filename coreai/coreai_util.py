"""Small Core AI helpers shared by the probes (was bench_stacked.specialization_for in fp8-mlp-metal41-bench)."""
from __future__ import annotations

from coreai.runtime import ComputeUnitKind, SpecializationOptions


def specialization_for(compute: str) -> SpecializationOptions:
    """Specialization options preferring the ANE ("ane") or the GPU ("gpu")."""
    kind = {"gpu": ComputeUnitKind.gpu(), "ane": ComputeUnitKind.neural_engine()}[compute]
    return SpecializationOptions.from_preferred_compute_unit_kind(kind)
