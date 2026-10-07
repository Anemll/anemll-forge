"""Startup log / health report of the target graph options (GDN_FAST, ATT_BLOCK) from a build manifest."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import qwen38_coreai_model as runtime


def manifest(*numerics):
    return {"chunks": [{"file": f"c{i}", "numerics": n} for i, n in enumerate(numerics)]}


class TargetGraphTests(unittest.TestCase):
    def test_recorded_fast_build(self):
        man = manifest({"GDN_FAST": True, "ATT_BLOCK": 2048}, {"GDN_FAST": True, "ATT_BLOCK": 2048})
        self.assertEqual(runtime.target_graph(man), {"gdn_fast": True, "att_block": 2048, "att_block_prefill": 2048,
                                                     "recorded": True, "att_int8mm": "", "att_int8mm_by_layer": {},
                                                     "att_pf8_unit": None})
        line = runtime.graph_line(man, Path("/b"))
        self.assertIn("GDN_FAST=1 ATT_BLOCK=2048 | build /b", line)
        self.assertNotIn("not recorded", line)

    def test_older_manifest_reports_release_defaults(self):
        man = manifest({"SILU": "tanh"})
        self.assertEqual(runtime.target_graph(man), {"gdn_fast": False, "att_block": 16384,
                                                     "att_block_prefill": 16384, "recorded": False, "att_int8mm": "",
                                                     "att_int8mm_by_layer": {}, "att_pf8_unit": None})
        self.assertIn("GDN_FAST=0 ATT_BLOCK=16384 (not recorded in manifest: release defaults)",
                      runtime.graph_line(man, Path("/b")))

    def test_separate_prefill_tile_is_reported(self):
        man = manifest({"GDN_FAST": True, "ATT_BLOCK": 2048, "ATT_BLOCK_PREFILL": 4096})
        self.assertEqual(runtime.target_graph(man)["att_block_prefill"], 4096)
        self.assertIn("ATT_BLOCK=2048 ATT_BLOCK_PREFILL=4096 |", runtime.graph_line(man, Path("/b")))

    def test_8bit_attention_forms_are_reported(self):
        n = {"GDN_FAST": True, "ATT_BLOCK": 2048, "ATT_INT8MM": "s8,s8b,sm8,pvf8", "ATT_PF8_UNIT": 0.015625}
        man = manifest(n, {**n, "ATT_INT8MM_BY_LAYER": {"63": "s8,s8b"}})
        g = runtime.target_graph(man)
        self.assertEqual(g["att_int8mm"], "s8,s8b,sm8,pvf8")
        self.assertEqual(g["att_int8mm_by_layer"], {"63": "s8,s8b"})
        line = runtime.graph_line(man, Path("/b"))
        self.assertIn("| 8-bit attention: INT8 scores, FP8 softmax (probabilities and sum, FP8 scale 1/64), FP8 PV probabilities "
                      "[ATT_INT8MM=s8,s8b,sm8,pvf8] per-layer 63:s8,s8b |", line)
        n2 = {**n, "ATT_S8_UNIT": 0.25, "ATT_S8B_UNIT": 0.25}
        self.assertIn("INT8 scores (step 1/4), FP8 softmax", runtime.graph_line(manifest(n2), Path("/b")))
        self.assertNotIn("8-bit attention", runtime.graph_line(manifest({"GDN_FAST": True, "ATT_BLOCK": 2048}), Path("/b")))

    def test_mixed_chunks_are_not_hidden(self):
        man = manifest({"GDN_FAST": True, "ATT_BLOCK": 2048}, {"GDN_FAST": False, "ATT_BLOCK": 16384})
        g = runtime.target_graph(man)
        self.assertEqual(g["gdn_fast"], [False, True])
        self.assertEqual(g["att_block"], [2048, 16384])


if __name__ == "__main__":
    unittest.main()
