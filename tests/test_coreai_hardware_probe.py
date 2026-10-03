"""Hardware-profile rejection before the generic package probe loads a model."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import qwen38_hardware_profile as hardware

spec = importlib.util.spec_from_file_location('hardware_probe_greedy', ROOT / 'coreai/qwen38_coreai_greedy.py')
greedy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(greedy)


class HardwareProbeTests(unittest.TestCase):
    def test_marked_package_rejected_on_m6_before_load_or_logging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'manifest.json').write_text(json.dumps({
                'hardware_profile': hardware.M5PRO_24GB_PROFILE,
                'ctxs': [8192, 16384, 24576, 31744], 'pctxs': [8192, 16384, 24576]}))
            with patch.object(hardware.platform, 'system', return_value='Darwin'), \
                 patch.object(hardware.subprocess, 'check_output', side_effect=['Apple M6', str(32 * 1024**3)]), \
                 patch.object(greedy.subprocess, 'Popen') as logger:
                with self.assertRaisesRegex(ValueError, 'exactly Apple M5 Pro'):
                    greedy.probe(root / 'unused.aimodel')
                logger.assert_not_called()


if __name__ == '__main__':
    unittest.main()
