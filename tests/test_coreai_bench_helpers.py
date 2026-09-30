"""CPU-only helper fixtures; no Core AI framework, compilation or model loading."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('bench_helpers', ROOT / 'scripts/coreai_bench_helpers.py')
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)

class HelpersTest(unittest.TestCase):
    def test_numpy_conversion_protocol_and_fallback(self):
        class Both:
            def numpy(self): return [1, 2]
            def to_numpy(self): raise AssertionError('numpy takes precedence')
        np.testing.assert_array_equal(H.to_numpy(Both()), [1, 2])
        np.testing.assert_array_equal(H.to_numpy(SimpleNamespace(to_numpy=lambda: [3])), [3])
        np.testing.assert_array_equal(H.to_numpy([4]), [4])

    def test_remove_only_requested_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); keep = root/'keep'; keep.write_text('keep')
            target = root/'target'; target.mkdir(); (target/'nested').write_text('x')
            H.remove_path(target); self.assertFalse(target.exists()); self.assertTrue(keep.exists())
            H.remove_path(target); H.remove_path(keep); self.assertFalse(keep.exists())

    def test_requested_compute_and_unsupported_runtime(self):
        runtime = ModuleType('coreai.runtime')
        runtime.ComputeUnitKind = SimpleNamespace(gpu=lambda:'gpu', neural_engine=lambda:'ane')
        runtime.SpecializationOptions = SimpleNamespace(is_supported=lambda:True, from_preferred_compute_unit_kind=lambda k:('preferred',k))
        with patch.dict(sys.modules, {'coreai.runtime':runtime}):
            self.assertEqual(H.specialization_for('ane'), ('preferred','ane'))
            self.assertEqual(H.specialization_for('gpu'), ('preferred','gpu'))
            with self.assertRaises(KeyError): H.specialization_for('cpu')
            runtime.SpecializationOptions.is_supported=lambda:False
            with self.assertRaisesRegex(RuntimeError,'Unset USE_LOCAL_COREAI'): H.specialization_for('ane')

if __name__ == '__main__': unittest.main()
