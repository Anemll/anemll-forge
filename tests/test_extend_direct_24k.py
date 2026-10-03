import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

COREAI = Path(__file__).resolve().parents[1] / "coreai"
sys.path.insert(0, str(COREAI))
spec = importlib.util.spec_from_file_location("extend24", COREAI / "extend_direct_24k.py")
extend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extend)


class ExtendDirectTests(unittest.TestCase):
    def manifest(self):
        return dict(ctxs=[8192, 16384], pctxs=[8192, 16384], T=8, TP=64,
                    kv_len={"8192": 8192, "16384": 16384},
                    pkv_len={"8192": 8192, "16384": 16384},
                    chunks=[dict(entries=["v8_8k", "v8_16k", "p64_8k", "p64_16k"],
                                 taps=[5], compiled="old")], head=dict(compiled="old"))

    def test_context_rewrite(self):
        self.assertEqual(extend.rewrite('v8_16k tensor<4x16384x256xf16> tensor<4x32x16392xf16>', 8),
                         'v8_24k tensor<4x24576x256xf16> tensor<4x32x24584xf16>')
        self.assertEqual(extend.rewrite('tensor<4x256x16448xf16>', 64), 'tensor<4x256x24640xf16>')
        self.assertEqual(extend.rewrite('163840 1.16384 16384.5 8192 2147483647', 8),
                         '163840 1.16384 16384.5 8192 2147483647')
        with self.assertRaises(ValueError):
            extend.rewrite('16384', 16)

    def test_manifest_preserves_source_and_non_context_state(self):
        source = self.manifest()
        result = extend.extend_manifest(source)
        self.assertEqual(source, self.manifest())
        self.assertEqual(result['ctxs'], [8192, 16384, 24576])
        self.assertEqual(result['pkv_len']['24576'], 24576)
        self.assertEqual(result['chunks'][0]['entries'],
                         ['v8_8k', 'v8_16k', 'v8_24k', 'p64_8k', 'p64_16k', 'p64_24k'])
        self.assertEqual(result['chunks'][0]['taps'], [5])
        self.assertNotIn('compiled', result['chunks'][0])
        self.assertEqual(result['hardware_profile'], extend.M5PRO_24GB_PROFILE)

    def test_cli_requires_explicit_opt_in_before_touching_assets(self):
        with patch.object(extend, 'require_m5pro_24gb') as gate:
            with self.assertRaises(SystemExit) as exc:
                extend.main(['--source', 'unused-source', '--output', 'unused-output'])
            self.assertEqual(exc.exception.code, 2)
            gate.assert_not_called()

    def test_cli_opt_in_does_not_override_hardware_gate(self):
        with patch.object(extend, 'require_m5pro_24gb', side_effect=ValueError('incompatible hardware')):
            with self.assertRaises(SystemExit) as exc:
                extend.main(['--source', 'unused-source', '--output', 'unused-output', '--m5pro-24gb'])
            self.assertEqual(exc.exception.code, 2)

    def test_rejects_incompatible_sources(self):
        for key, value in [('ctxs', [16384]), ('pctxs', [8192]), ('TP', 32),
                           ('kv_len', {'16384': 16320})]:
            source = self.manifest()
            source[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                extend.extend_manifest(source)

    def test_32k_shape_extension_and_multiple_contexts(self):
        self.assertEqual(extend.rewrite('v8_16k tensor<4x16384x256xf16> tensor<4x32x16392xf16>', 8, 32768),
                         'v8_32k tensor<4x32768x256xf16> tensor<4x32x32776xf16>')
        self.assertEqual(extend.rewrite('p64_16k tensor<4x256x16448xf16>', 64, 32768),
                         'p64_32k tensor<4x256x32832xf16>')
        result = extend.extend_manifest(self.manifest(), [32768, 24576, 32768])
        self.assertEqual(result['ctxs'], [8192, 16384, 24576, 32768])
        self.assertEqual(result['kv_len']['32768'], 32768)

    def test_rejects_invalid_extensions_and_compressed_sources(self):
        for contexts in ([], [16384], [24577], [65536], ['32768']):
            with self.subTest(contexts=contexts), self.assertRaises(ValueError):
                extend.extend_manifest(self.manifest(), contexts)
        source = self.manifest()
        source['kv_cache'] = dict(format='selectable')
        with self.assertRaisesRegex(ValueError, 'legacy FP16'):
            extend.extend_manifest(source)

    def test_decode_only_extension(self):
        result = extend.extend_manifest(self.manifest(), [24576, 31744], [24576])
        self.assertEqual(result['ctxs'], [8192, 16384, 24576, 31744])
        self.assertEqual(result['pctxs'], [8192, 16384, 24576])
        self.assertNotIn('31744', result['pkv_len'])
        self.assertIn('v8_31k', result['chunks'][0]['entries'])
        self.assertNotIn('p64_31k', result['chunks'][0]['entries'])
        self.assertEqual(result['chunks'][0]['entries_ctx'][1], result['pctxs'])
        with self.assertRaisesRegex(ValueError, 'subset'):
            extend.extend_manifest(self.manifest(), [31744], [24576])


if __name__ == '__main__':
    unittest.main()
