"""Minimal local helpers extracted from the first-party Core AI benchmarks.
See provenance/LOCAL_EXPERIMENT_IMPORTS.json for original sources and hashes.
No runtime is imported until specialization_for is called; unset USE_LOCAL_COREAI
before launching Core AI experiments (this helper never mutates the environment).
"""
from pathlib import Path
import shutil
import numpy as np

SEED = 42


def to_numpy(value):
    for attr in ("numpy", "to_numpy"):
        if hasattr(value, attr):
            return np.asarray(getattr(value, attr)())
    return np.asarray(value)


def remove_path(path: Path):
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def specialization_for(compute: str):
    from coreai.runtime import ComputeUnitKind, SpecializationOptions
    if not SpecializationOptions.is_supported():
        raise RuntimeError("SpecializationOptions not supported. Unset USE_LOCAL_COREAI before starting; these experiments require the OS Core AI framework.")
    kind = {"gpu": ComputeUnitKind.gpu(), "ane": ComputeUnitKind.neural_engine()}[compute]
    return SpecializationOptions.from_preferred_compute_unit_kind(kind)
