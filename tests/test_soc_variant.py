"""Per-chip function sets (scripts/soc_variant.py): the derived manifest and the one-time M5 build, with the Core AI
strip replaced by a stub (the real one needs coreai-core and a package)."""
import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import soc_variant  # noqa: E402


def manifest():
    chunk = lambda i: {"file": f"chunk_{i}.aimodel", "entries": ["v8_8k", "v8_8k_m5", "p64_8k", "p64_8k_m5"],
                       "entries_by_soc": {"m5": {"v8_8k": "v8_8k_m5", "p64_8k": "p64_8k_m5"}},
                       "numerics": {"ATT_INT8MM": "s8,s8b,sm8,pvf8", "ATT_INT8MM_M5": "s8,s8b",
                                    "ATT_PF8_UNIT": 0.015625, "KV_KEYS_T": True}}
    return {"ctxs": [8192], "pctxs": [8192], "chunks": [chunk(0), chunk(1)], "head": {"file": "head_T8.aimodel"}}


class SocVariantTests(unittest.TestCase):
    def test_derived_manifest(self):
        m = soc_variant.derived_manifest(manifest(), "m5", Path("/b"))
        c = m["chunks"][0]
        self.assertEqual(c["entries"], ["v8_8k", "p64_8k"])
        self.assertNotIn("entries_by_soc", c)
        self.assertEqual(c["numerics"]["ATT_INT8MM"], "s8,s8b")
        self.assertNotIn("ATT_INT8MM_M5", c["numerics"])
        self.assertNotIn("ATT_PF8_UNIT", c["numerics"])  # no FP8 form left
        self.assertTrue(c["numerics"]["KV_KEYS_T"])
        self.assertEqual(m["derived"], {"soc": "m5", "from": "/b"})
        self.assertEqual(soc_variant.soc_sets(manifest()), ["m5"])
        with self.assertRaises(ValueError):
            soc_variant.derived_manifest(manifest(), "m7")

    def test_prepare_derives_once_and_passes_other_chips_through(self):
        calls = []

        def strip(src, dst, keep):
            calls.append((src.name, dict(keep)))
            shutil.copytree(src, dst)
            return list(keep), []
        stub = types.ModuleType("strip_functions")
        stub.strip = strip
        sys.modules["strip_functions"] = stub
        try:
            with tempfile.TemporaryDirectory() as d:
                root, state = Path(d) / "build", Path(d) / "state"
                root.mkdir()
                for name in ("chunk_0.aimodel", "chunk_1.aimodel", "head_T8.aimodel"):
                    (root / name).mkdir()
                    (root / name / "main.mlirb").write_bytes(b"x" * 100)
                (root / "manifest.json").write_text(json.dumps(manifest()))
                self.assertEqual(soc_variant.prepare(root, "m6", log=lambda m: None, state=state), root)
                out = soc_variant.prepare(root, "m5", log=lambda m: None, state=state)
                self.assertNotEqual(out, root)
                self.assertEqual(calls, [("chunk_0.aimodel", {"v8_8k_m5": "v8_8k", "p64_8k_m5": "p64_8k"}),
                                         ("chunk_1.aimodel", {"v8_8k_m5": "v8_8k", "p64_8k_m5": "p64_8k"})])
                self.assertTrue((out / "head_T8.aimodel").is_symlink())
                self.assertEqual(json.loads((out / "manifest.json").read_text())["chunks"][1]["entries"],
                                 ["v8_8k", "p64_8k"])
                self.assertFalse(any(p.name.endswith(".partial.aimodel") for p in out.iterdir()))
                self.assertEqual(soc_variant.prepare(root, "m5", log=lambda m: None, state=state), out)
                self.assertEqual(len(calls), 2)  # reused, not derived again
        finally:
            sys.modules.pop("strip_functions", None)


if __name__ == "__main__":
    unittest.main()
