import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import qwen38_hardware_profile as hp


class HardwareProfileTests(unittest.TestCase):
    def host(self, chip='Apple M5 Pro', gib=24):
        return patch.object(hp.subprocess, 'check_output', side_effect=[chip + '\n', str(gib * 1024**3)])

    def manifest(self):
        return {'hardware_profile': hp.M5PRO_24GB_PROFILE.copy(),
                'ctxs': [8192, 16384, 24576, 31744], 'pctxs': [8192, 16384, 24576]}

    @patch.object(hp.platform, 'system', return_value='Darwin')
    def test_exact_tested_hardware_allowed(self, _):
        with self.host():
            hp.validate_hardware_profile(self.manifest())

    @patch.object(hp.platform, 'system', return_value='Darwin')
    def test_other_chips_and_memory_capacities_rejected(self, _):
        for chip, memory in [('Apple M6', 32), ('Apple M6', 24), ('Apple M6 Pro', 32),
                             ('Apple M5 Pro', 32), ('Apple M5 Pro', 48), ('Apple M5 Pro', 16),
                             ('Apple M5', 24), ('Apple M5 Max', 24), ('unknown', 24)]:
            with self.subTest(chip=chip, memory=memory), self.host(chip, memory):
                with self.assertRaisesRegex(ValueError, 'exactly Apple M5 Pro'):
                    hp.validate_hardware_profile(self.manifest())

    @patch.object(hp.platform, 'system', return_value='Linux')
    def test_non_macos_fails_without_querying_sysctl(self, _):
        with patch.object(hp.subprocess, 'check_output') as query:
            with self.assertRaises(ValueError):
                hp.require_m5pro_24gb()
            query.assert_not_called()

    @patch.object(hp.platform, 'system', return_value='Darwin')
    def test_unavailable_or_malformed_detection_fails_closed(self, _):
        for error in (OSError('unavailable'), subprocess.TimeoutExpired('sysctl', 5)):
            with patch.object(hp.subprocess, 'check_output', side_effect=error):
                with self.assertRaisesRegex(ValueError, 'Cannot verify'):
                    hp.require_m5pro_24gb()
        with patch.object(hp.subprocess, 'check_output', side_effect=['Apple M5 Pro', 'unknown']):
            with self.assertRaisesRegex(ValueError, 'Cannot verify'):
                hp.require_m5pro_24gb()

    def test_standard_m6_build_never_queries_or_selects_workaround(self):
        with patch.object(hp, 'require_m5pro_24gb') as gate:
            hp.validate_hardware_profile({'ctxs': [8192, 16384, 32768],
                                          'kv_cache': {'format': 'selectable', 'default': 'v8'}})
            gate.assert_not_called()

    def test_unknown_profiles_and_incompatible_contracts_fail(self):
        for change in ({'hardware_profile': None}, {'hardware_profile': {'id': 'unknown'}},
                       {'kv_cache': {'format': 'v8'}}, {'ctxs': [32768]}, {'pctxs': [31744]}):
            m = self.manifest()
            m.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                hp.validate_hardware_profile(m)

    @patch.object(hp.platform, 'system', return_value='Darwin')
    def test_both_runtime_backends_reject_m6_before_model_allocation(self, _):
        import qwen38_coreai_model as runtime
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'manifest.json').write_text(json.dumps(self.manifest()))
            for cls in (runtime.CoreAIQwenPy, runtime.CoreAIQwenBridge):
                with self.subTest(backend=cls.__name__), self.host('Apple M6', 32):
                    with self.assertRaisesRegex(ValueError, 'exactly Apple M5 Pro'):
                        cls(root=root)
