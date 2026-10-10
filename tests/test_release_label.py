"""The startup line and /health field that name the ANEMLL release of a build's weights."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import qwen38_coreai_model as M  # noqa: E402


class ReleaseLabelTests(unittest.TestCase):
    def test_published_exports(self):
        self.assertEqual(M.model_release({"export": "mix25in_mixr_lr64mix"}), "0.1")
        self.assertEqual(M.model_release({"export": "release_vq3pA_mixh_s600_k1_mat"}), "0.2")
        self.assertIn("release 0.2", M.release_line({"export": "release_vq3pA_mixh_s600_k1_mat"}))
        self.assertEqual(M.model_release({"export": "/any/path/release_vq3pA_mixh_s600_k1_mat"}), "0.2")  # local build

    def test_explicit_field_and_unknown_builds(self):
        self.assertEqual(M.model_release({"export": "anything", "release": "0.3"}), "0.3")
        self.assertIsNone(M.model_release({"export": "my_export"}))
        self.assertEqual(M.release_line({"export": "my_export"}), "model: custom build (weights my_export)")
        self.assertEqual(M.release_line({}), "model: weights not recorded in the manifest")


if __name__ == "__main__":
    unittest.main()
