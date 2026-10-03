import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import qwen38_pi_config as pc


class PiConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pi = Path(self.tmp.name).resolve()
        self.models = {"providers": {"ane-qwen38": {"models": [
            {"id": "qwen38-27b-ane", "contextWindow": 16384, "maxTokens": 4096,
             "name": "Qwen3.8 27B VQ (M6 ANE, 16K, DFlash)"}]}}}
        self.settings = {"compaction": {"modelOverrides": {
            "ane-qwen38/qwen38-27b-ane": {"reserveTokens": 8192, "keepRecentTokens": 2048}}}}
        (self.pi / "models.json").write_text(json.dumps(self.models))
        (self.pi / "settings.json").write_text(json.dumps(self.settings))

    def entry(self):
        return json.loads((self.pi / "models.json").read_text())["providers"]["ane-qwen38"]["models"][0]

    def override(self):
        s = json.loads((self.pi / "settings.json").read_text())
        return s["compaction"]["modelOverrides"]["ane-qwen38/qwen38-27b-ane"]

    def test_budget_ladder(self):
        self.assertEqual(pc.budget(16384), (4096, 8192, 2048))
        self.assertEqual(pc.budget(32768), (8192, 12288, 4096))
        self.assertEqual(pc.budget(31744), (7936, 12032, 3968))
        self.assertEqual(pc.budget(65536), (16384, 20480, 8192))
        self.assertEqual(pc.budget(8192), (2048, 6144, 1024))

    def test_31k_profile_rejected_before_writing_on_other_hardware(self):
        before = {name: (self.pi / name).read_bytes() for name in ('models.json', 'settings.json')}
        with patch.object(pc, 'require_m5pro_24gb', side_effect=ValueError('incompatible hardware')):
            with self.assertRaisesRegex(ValueError, 'incompatible hardware'):
                pc.sync(self.pi, 31744, 'experimental', True)
        for name, content in before.items():
            self.assertEqual((self.pi / name).read_bytes(), content)
        self.assertFalse((self.pi / 'models.json.orig-qwen38').exists())

    def test_standard_32k_pi_profile_does_not_use_m5_gate(self):
        with patch.object(pc, 'require_m5pro_24gb') as gate:
            pc.sync(self.pi, 32768, 'standard', True)
            gate.assert_not_called()

    def test_31k_profile_checks_hardware_and_sets_budget(self):
        with patch.object(pc, 'require_m5pro_24gb') as gate:
            pc.sync(self.pi, 31744, 'experimental', True)
            gate.assert_called_once_with()
        self.assertEqual(self.entry()['contextWindow'], 31744)
        self.assertEqual(self.override(), {'reserveTokens': 12032, 'keepRecentTokens': 3968})

    def test_invalid_context_does_not_modify_profile(self):
        before = {name: (self.pi / name).read_bytes() for name in ("models.json", "settings.json")}
        for ctx in (-1, 0, 2048, 4096, 12000, 131072):
            with self.subTest(ctx=ctx), self.assertRaisesRegex(ValueError, "unsupported context"):
                pc.sync(self.pi, ctx, "coreai", True)
            for name, content in before.items():
                self.assertEqual((self.pi / name).read_bytes(), content)
        self.assertFalse((self.pi / "models.json.orig-qwen38").exists())

    def test_sync_64k_updates_and_backs_up(self):
        changes = pc.sync(self.pi, 65536, "coreai_mixr12", True)
        self.assertEqual(len(changes), 2)
        self.assertEqual(self.entry()["contextWindow"], 65536)
        self.assertEqual(self.entry()["maxTokens"], 16384)
        self.assertEqual(self.entry()["name"], "Qwen3.8 27B VQ coreai_mixr12 (ANE, 64K, DFlash)")
        self.assertEqual(self.override(), {"reserveTokens": 20480, "keepRecentTokens": 8192})
        self.assertTrue((self.pi / "models.json.orig-qwen38").exists())
        self.assertTrue((self.pi / "settings.json.prev-qwen38").exists())

    def test_sync_is_idempotent(self):
        pc.sync(self.pi, 65536, "coreai_mixr12", True)
        self.assertEqual(pc.sync(self.pi, 65536, "coreai_mixr12", True), [])

    def test_dry_run_writes_nothing(self):
        changes = pc.sync(self.pi, 65536, "coreai_mixr12", True, dry_run=True)
        self.assertEqual(len(changes), 2)
        self.assertEqual(self.entry()["contextWindow"], 16384)
        self.assertFalse((self.pi / "models.json.orig-qwen38").exists())

    def test_missing_entry_is_reported(self):
        self.models["providers"]["ane-qwen38"]["models"][0]["id"] = "other"
        (self.pi / "models.json").write_text(json.dumps(self.models))
        with self.assertRaisesRegex(ValueError, "no ane-qwen38/qwen38-27b-ane"):
            pc.sync(self.pi, 65536, "", True)

    def test_missing_or_malformed_settings_does_not_modify_models(self):
        model = self.pi / "models.json"
        before = model.read_bytes()
        settings = self.pi / "settings.json"
        settings.unlink()
        for content in (None, "invalid json", '[]', '{"compaction": null}'):
            with self.subTest(content=content):
                if content is not None:
                    settings.write_text(content)
                with self.assertRaises(ValueError):
                    pc.sync(self.pi, 65536, "coreai", True)
                self.assertEqual(model.read_bytes(), before)
                self.assertFalse((self.pi / "models.json.orig-qwen38").exists())

    def test_second_replace_failure_rolls_back_models(self):
        before = {name: (self.pi / name).read_bytes() for name in ("models.json", "settings.json")}
        original_replace = pc.os.replace

        def fail_settings(source, dest):
            if dest == self.pi / "settings.json":
                raise OSError("synthetic replacement failure")
            return original_replace(source, dest)

        with patch.object(pc.os, "replace", side_effect=fail_settings):
            with self.assertRaisesRegex(OSError, "synthetic replacement failure"):
                pc.sync(self.pi, 65536, "coreai", True)
        for name, content in before.items():
            self.assertEqual((self.pi / name).read_bytes(), content)
        self.assertFalse(list(self.pi.glob(".*.qwen38-*")))

    def test_sync_preserves_other_fields_and_file_permissions(self):
        self.models["providers"]["another-provider"] = {"apiKey": "synthetic-placeholder", "models": []}
        (self.pi / "models.json").write_text(json.dumps(self.models))
        key = "ane-qwen38/qwen38-27b-ane"
        self.settings["compaction"]["modelOverrides"][key]["enabled"] = False
        self.settings["unrelated"] = {"example": True}
        (self.pi / "settings.json").write_text(json.dumps(self.settings))
        (self.pi / "models.json").chmod(0o600)
        pc.sync(self.pi, 65536, "coreai", True)
        models = json.loads((self.pi / "models.json").read_text())
        settings = json.loads((self.pi / "settings.json").read_text())
        self.assertEqual(models["providers"]["another-provider"], self.models["providers"]["another-provider"])
        self.assertFalse(settings["compaction"]["modelOverrides"][key]["enabled"])
        self.assertEqual(settings["unrelated"], {"example": True})
        self.assertEqual((self.pi / "models.json").stat().st_mode & 0o777, 0o600)

    def test_cli_dry_run_exit_code(self):
        self.assertEqual(pc.main(["--ctx", "65536", "--pi-dir", str(self.pi), "--dry-run"]), 0)
        self.assertEqual(self.entry()["contextWindow"], 16384)


if __name__ == "__main__":
    unittest.main()
