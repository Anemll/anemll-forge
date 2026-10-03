import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location("select_context", Path(__file__).resolve().parents[1] / "coreai/select_context.py")
select = importlib.util.module_from_spec(spec)
spec.loader.exec_module(select)


class SelectContextTests(unittest.TestCase):
    def test_context_selection_preserves_hardware_profile(self):
        source = self.manifest()
        source['hardware_profile'] = {'id': 'test-profile'}
        result = select.context_manifest(source, [8192])
        self.assertEqual(result['hardware_profile'], source['hardware_profile'])
        self.assertIsNot(result['hardware_profile'], source['hardware_profile'])

    def test_named_resource_verification(self):
        def module(body):
            return SimpleNamespace(operation=SimpleNamespace(get_asm=lambda **kw: body))
        head = 'dense_resource<weight> : tensor<1xf16>\n{-#\n'
        first = select.named_resource_digests(module(head + 'weight: "0x10000000ABCD"\nother: "0x0000"\n#-}'))
        second = select.named_resource_digests(module(head + 'other: "0x0000"\nweight: "0x10000000abcd"\n#-}'))
        self.assertEqual(first, second)
        changed = select.named_resource_digests(module(head + 'weight: "0x10000000ABCE"\nother: "0x0000"\n#-}'))
        self.assertNotEqual(first, changed)
        for body in ('no table', head + 'other: "0x0000"\n#-}',
                     head + 'weight: "0x00"\nweight: "0x00"\n#-}'):
            with self.assertRaises(ValueError):
                select.named_resource_digests(module(body))

    def manifest(self):
        return dict(ctxs=[8192, 65536], pctxs=[8192, 65536],
                    kv_len={"8192": 8192, "65536": 65472},
                    pkv_len={"8192": 8192, "65536": 65472},
                    chunks=[dict(file="chunk.aimodel", compiled="old.aimodelc",
                                 entries=["v8_8k", "v8_64k", "p64_8k", "p64_64k"],
                                 entries_ctx=[[8192, 65536], [8192, 65536]], taps=[5])],
                    head=dict(file="head.aimodel", compiled="old-head.aimodelc"))

    def test_selects_both_entries_and_preserves_original(self):
        source = self.manifest()
        result = select.context_manifest(source, 8192)
        self.assertEqual(result["ctxs"], [8192])
        self.assertEqual(result["pctxs"], [8192])
        self.assertEqual(result["kv_len"], {"8192": 8192})
        self.assertEqual(result["chunks"][0]["entries"], ["v8_8k", "p64_8k"])
        self.assertEqual(result["chunks"][0]["entries_ctx"], [[8192], [8192]])
        self.assertEqual(result["chunks"][0]["taps"], [5])
        self.assertNotIn("compiled", result["chunks"][0])
        self.assertNotIn("compiled", result["head"])
        self.assertEqual(source, self.manifest())

    def test_rejects_unsupported_context_or_missing_entry(self):
        with self.assertRaises(ValueError):
            select.context_manifest(self.manifest(), 16384)
        source = self.manifest()
        source["chunks"][0]["entries"].remove("p64_8k")
        with self.assertRaises(ValueError):
            select.context_manifest(source, 8192)

    def test_ladder_is_sorted_unique_and_keeps_both_entry_types(self):
        source = self.manifest()
        result = select.context_manifest(source, [65536, 8192, 8192])
        self.assertEqual(result["ctxs"], [8192, 65536])
        self.assertEqual(result["pctxs"], [8192, 65536])
        self.assertEqual(result["kv_len"], {"8192": 8192, "65536": 65472})
        self.assertEqual(result["chunks"][0]["entries"], ["v8_8k", "v8_64k", "p64_8k", "p64_64k"])
        self.assertEqual(result["chunks"][0]["entries_ctx"], [[8192, 65536], [8192, 65536]])
        self.assertEqual(source, self.manifest())

    def test_rejects_empty_ladder_or_one_invalid_context(self):
        for contexts in ([], [8192, 16384], [0]):
            with self.subTest(contexts=contexts), self.assertRaises(ValueError):
                select.context_manifest(self.manifest(), contexts)

    def test_rejects_selectable_graphs_instead_of_publishing_stale_aliases(self):
        source = self.manifest()
        source['kv_cache'] = dict(format='selectable')
        with self.assertRaisesRegex(ValueError, 'format-aware'):
            select.context_manifest(source, 8192)

    def test_decoder_ladder_can_have_a_smaller_prefill_ladder(self):
        source = self.manifest()
        result = select.context_manifest(source, [8192, 65536], [8192])
        self.assertEqual(result['pctxs'], [8192])
        self.assertEqual(result['ctxs'], [8192, 65536])
        self.assertEqual(result['pkv_len'], {'8192': 8192})
        self.assertEqual(result['chunks'][0]['entries'], ['v8_8k', 'v8_64k', 'p64_8k'])
        self.assertEqual(result['chunks'][0]['entries_ctx'], [[8192, 65536], [8192]])
        self.assertEqual(source, self.manifest())

    def test_prefill_subset_must_belong_to_verification_ladder(self):
        with self.assertRaises(ValueError):
            select.context_manifest(self.manifest(), [8192], [65536])
        result = select.context_manifest(self.manifest(), [8192], [])
        self.assertEqual(result['TP'], 0)
        self.assertEqual(result['chunks'][0]['entries'], ['v8_8k'])

    def test_artifact_must_stay_inside_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(select.artifact_path(root, "chunk.aimodel"), root / "chunk.aimodel")
            for filename in ("../escape.aimodel", "/escape.aimodel"):
                with self.assertRaises(ValueError):
                    select.artifact_path(root, filename)

    def test_resource_hash_is_independent_of_other_sections(self):
        def vint(value):
            if value < 128:
                return bytes([(value << 1) | 1])
            raise ValueError("test encoder only supports small values")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "main.mlirb"
            prefix = b"ML\xefR" + vint(6) + b"test\x00"
            resource = b"exact weight bytes"
            path.write_bytes(prefix + b"\x05" + vint(len(resource)) + resource)
            original = select.resource_digest(path)
            path.write_bytes(prefix + b"\x01" + vint(3) + b"abc" + b"\x05" + vint(len(resource)) + resource)
            self.assertEqual(select.resource_digest(path), original)
            path.write_bytes(prefix + b"\x05" + vint(len(resource)) + b"different weights!")
            self.assertNotEqual(select.resource_digest(path), original)


if __name__ == "__main__":
    unittest.main()
