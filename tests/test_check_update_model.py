"""Model update checks use inventories and pin remote reads; no model loads or downloads."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_update_model as C


def inventory():
    return {"upstream_model": {"id": "Qwen/example", "revision": "base"},
            "runtimes": {"coreai": {"path": "coreai", "contexts": [8192]}},
            "drafter": {"target_export": "old"},
            "files": [{"path": "coreai/weights", "component": "coreai", "bytes": 10, "sha256": "a" * 64},
                      {"path": "drafter/weights", "component": "drafter", "bytes": 3, "sha256": "b" * 64},
                      {"path": "README.md", "component": "documentation", "bytes": 5, "sha256": "c" * 64}]}


class UpdateTests(unittest.TestCase):
    def test_identical_inventory_ignores_order(self):
        before = inventory()
        after = copy.deepcopy(before)
        after["files"].reverse()
        # Inventory order is metadata only, never a model update.
        result = C.compare(before, after, "coreai")
        self.assertFalse(result["model_update_available"])
        self.assertEqual(C.compare(before, before, "coreai")["status"], "up_to_date")

    def test_changed_added_removed_target_and_drafter_files(self):
        before = inventory()
        after = copy.deepcopy(before)
        after["files"][0]["sha256"] = "d" * 64
        after["files"].pop(1)
        after["files"].append({"path": "model/new", "component": "model", "bytes": 1, "sha256": "e" * 64})
        result = C.compare(before, after, "coreai")
        self.assertEqual(result["status"], "model_update_available")
        self.assertEqual(result["changed_model_files"], ["coreai/weights", "drafter/weights", "model/new"])

    def test_documentation_only_and_incompatible_repository(self):
        before = inventory()
        after = copy.deepcopy(before)
        after["files"][-1]["sha256"] = "d" * 64
        self.assertEqual(C.compare(before, after, "coreai")["status"], "metadata_update_available")
        after["upstream_model"]["id"] = "different"
        with self.assertRaisesRegex(ValueError, "different upstream"):
            C.compare(before, after, "coreai")
        with self.assertRaisesRegex(ValueError, "no coreml"):
            C.compare(before, before, "coreml")

    def test_remote_manifest_uses_resolved_commit(self):
        hub = types.ModuleType("huggingface_hub")
        api = Mock()
        api.repo_info.return_value.sha = "f" * 40
        hub.HfApi = Mock(return_value=api)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "release.json"
            path.write_text(json.dumps(inventory()))
            hub.hf_hub_download = Mock(return_value=str(path))
            with patch.dict(sys.modules, {"huggingface_hub": hub}), patch.object(C, "validate_manifest", side_effect=lambda m: m):
                commit, _ = C.remote_manifest("owner/model", "main")
            self.assertEqual(commit, "f" * 40)
            hub.hf_hub_download.assert_called_once_with(repo_id="owner/model", filename="release.json", revision=commit)

    def test_json_and_fresh_download_destination(self):
        before, after = inventory(), inventory()
        after["files"][0]["sha256"] = "d" * 64
        with tempfile.TemporaryDirectory(prefix="forge with spaces ") as d:
            bundle = Path(d) / "bundle"
            bundle.mkdir()
            bundle.with_name("bundle-ffffffff").mkdir()
            out = io.StringIO()
            with patch.object(C, "load_manifest", return_value=before), patch.object(C, "remote_manifest", return_value=("f" * 40, after)), contextlib.redirect_stdout(out):
                self.assertEqual(C.main(["--bundle", str(bundle), "--json"]), 0)
            result = json.loads(out.getvalue())
            self.assertIn("bundle-ffffffff-2", result["download_command"])
            self.assertIn("f" * 40, result["download_command"])
            self.assertEqual(list(bundle.iterdir()), [])

    def test_failure_never_reports_current(self):
        out = io.StringIO()
        with patch.object(C, "load_manifest", side_effect=ValueError("missing inventory")), contextlib.redirect_stdout(out):
            self.assertEqual(C.main(["--json"]), 1)
        self.assertEqual(json.loads(out.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
