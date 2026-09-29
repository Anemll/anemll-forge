import contextlib
import io
from unittest.mock import patch
import importlib.util
import json
import plistlib
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('audit', Path(__file__).resolve().parents[1] / 'coreai' / 'inspect_coreai_cache.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class CacheAuditTests(unittest.TestCase):
    def fixture(self, root, attrs, graph=b'', entry='v8_8k_hash_0'):
        package = root / 'chunk.aimodel'
        package.mkdir()
        (package / 'main.hash').write_bytes(b'hash')
        cache = root / 'cache'
        p = cache / 'TEST' / 'my-python' / b'hash'.hex() / 'spec' / 'model.aimodelx' / 'x.mpsgraphpackage'
        p.mkdir(parents=True)
        manifest = {'Package Version': {'7.0': {'GPU adapter present': 'NO', 'Optimized Modules': {'preferredANE_mode2': {
            'Entry Function Attributes': {entry: attrs}, 'File Name': 'model.mpsgraph'}}}}}
        (p / 'manifest.plist').write_bytes(plistlib.dumps(manifest))
        (p / 'model.mpsgraph').write_bytes(graph)
        return package, cache

    def test_fully_ane(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, c = self.fixture(Path(tmp), ['mps.fullyPlacedOnANE', 'mps.noGPUActivity'], b'v8_8k_hash_0_ANE_region_0')
            self.assertEqual(audit.inspect_package(p, ['v8_8k'], c, 'TEST', '/bin/my_python')['status'], 'fully_ane')

    def test_gpu_despite_preferred_ane_and_adapter_no(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, c = self.fixture(Path(tmp), ['mps.aneAlignedIO'], b'v8_8k_hash_0_GPU_region_0')
            self.assertEqual(audit.inspect_package(p, ['v8_8k'], c, 'TEST', '/bin/my_python')['status'], 'gpu_regions_present')

    def test_gpu_symbols_override_fully_ane_attributes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, c = self.fixture(Path(tmp), ['mps.fullyPlacedOnANE', 'mps.noGPUActivity'], b'v8_8k_hash_0_GPU_region_0')
            self.assertEqual(audit.inspect_package(p, ['v8_8k'], c, 'TEST', 'my_python')['status'], 'gpu_regions_present')

    def test_missing_graph_is_unknown_despite_full_attributes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, c = self.fixture(Path(tmp), ['mps.fullyPlacedOnANE', 'mps.noGPUActivity'])
            next(c.rglob('model.mpsgraph')).unlink()
            self.assertEqual(audit.inspect_package(p, ['v8_8k'], c, 'TEST', 'my_python')['status'], 'unknown')

    def test_missing_expected_entry_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, c = self.fixture(Path(tmp), ['mps.fullyPlacedOnANE', 'mps.noGPUActivity'])
            self.assertEqual(audit.inspect_package(p, ['v8_16k'], c, 'TEST', 'my_python')['status'], 'unknown')
            self.assertEqual(audit.inspect_package(p, ['v8_8k'], c, 'OTHER', 'my_python')['status'], 'unknown')

    def test_strict_missing_returns_one_without_loading_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(audit, 'package_entries', return_value=[(root / 'absent.aimodel', ['v8_8k'])]):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    code = audit.main(['--model-dir', tmp, '--os-build', 'TEST', '--strict'])
                self.assertEqual(code, 1)
                self.assertEqual(json.loads(output.getvalue())['packages'][0]['status'], 'unknown')

    def test_drafter_sidecar_dict_uses_actual_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'manifest.json').write_text(json.dumps({'chunks': [], 'head': {'file': 'head.aimodel'}}))
            drafter = root / 'drafter.aimodel'
            drafter.with_suffix('.json').write_text(json.dumps({'entries': {'draft': {}, 'ctx64': {}}}))
            self.assertEqual(audit.package_entries(root, drafter)[-1][1], ['draft', 'ctx64'])


if __name__ == '__main__':
    unittest.main()
