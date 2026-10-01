"""Release defaults, target/drafter association and speculative state-commit checks."""
import argparse
import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import struct
import sys
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import forge
try:
    from . import test_hf_release as release_fixture
except ImportError:  # unittest discovery loads files as top-level modules
    import test_hf_release as release_fixture
hf = release_fixture.hf

try:
    import numpy as np
except ImportError:
    np = None


class DrafterReleaseTests(unittest.TestCase):
    def setUp(self):
        self.fixture = release_fixture.ReleaseTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.build = self.root / "coreai"
        self.directory = self.root / "drafter"
        self.directory.mkdir()
        self.pkg = self.directory / hf.DEFAULT_DRAFTER
        self.pkg.mkdir()
        (self.pkg / "resources.bin").write_bytes(b"paired DFlash2 asset")
        (self.directory / "LICENSE").write_text("Unmodified upstream license")
        self.cfg = dict(vocab_size=1, hidden_size=5120, num_target_layers=64, num_hidden_layers=5,
                        head_dim=128, num_key_value_heads=8, sliding_window=2048,
                        dflash_config=dict(block_size=8, selector_rank=256, target_layer_ids=[5, 19, 33, 47, 61]))
        self.meta = dict(T=8, R=8, RP=64, W=2048, target_export="mix25in_mixr_lr64mix",
                         head_export="mix25in_mixr_lr64mix/lm_head.safetensors",
                         entries={"draft": {"outputs": ["logits"]}, "ctx64": {}})
        self.write_metadata()
        target = json.loads((self.build / "manifest.json").read_text())
        target.update(export="mix25in_mixr_lr64mix", taps=self.cfg["dflash_config"]["target_layer_ids"])
        (self.build / "manifest.json").write_text(json.dumps(target))
        self.selector()

    def write_metadata(self):
        (self.directory / "config.json").write_text(json.dumps(self.cfg))
        self.pkg.with_suffix(".json").write_text(json.dumps(self.meta))

    def selector(self, *, shape=None, extra=False):
        shape = shape or [self.cfg["vocab_size"], 256]
        size = shape[0] * shape[1] * 2
        header = {key: dict(dtype="BF16", shape=shape, data_offsets=[i * size, (i + 1) * size])
                  for i, key in enumerate(hf.SELECTOR_KEYS)}
        if extra:
            header["unrelated.weight"] = dict(dtype="BF16", shape=[1], data_offsets=[2 * size, 2 * size + 2])
        raw = json.dumps(header).encode()
        (self.directory / hf.SELECTOR_FILE).write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * (2 * size + 2 * extra))

    def manifest(self):
        return self.fixture.manifest()

    def test_default_inventory_and_integrity_include_drafter(self):
        m = self.manifest()
        self.assertEqual(m["drafter"]["model"], "dflash2_lut4_gptq.aimodel")
        self.assertEqual(m["drafter"]["codebooks"], "selector.safetensors")
        self.assertEqual(m["drafter"]["head_export"], "mix25in_mixr_lr64mix")
        result = hf.verify(self.root, m, "coreai")
        files = [f for f in m["files"] if f["component"] in {"model", "documentation", "coreai", "drafter"}]
        self.assertEqual(result["verified_files"], len(files))
        license_entry = next(f for f in files if f["path"] == "drafter/LICENSE")
        self.assertNotIn("modification_notice", license_entry)

    def test_target_only_requires_explicit_plain_diagnostic(self):
        import shutil
        shutil.rmtree(self.directory)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "requires Core AI"):
            hf.make_manifest(argparse.Namespace(bundle=self.root, model_id="Qwen/test", model_revision="a" * 40))
        m = self.manifest()
        with self.assertRaisesRegex(ValueError, "requires Core AI"):
            hf.verify(self.root, m, "coreai")
        hf.verify(self.root, m, "coreai", plain=True)

    def test_pairing_rejects_wrong_head_target_taps_or_width(self):
        for key, value in (("head_export", "wrong/lm_head.safetensors"), ("target_export", "wrong"), ("T", 4)):
            original = copy.deepcopy(self.meta)
            self.meta[key] = value
            self.write_metadata()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.manifest()
            self.meta = original
        self.cfg["dflash_config"]["target_layer_ids"] = [5, 19, 33, 47, 60]
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "taps"):
            self.manifest()

    def test_selector_rejects_wrong_shape_or_full_checkpoint(self):
        for kw in ({"shape": [2, 256]}, {"extra": True}):
            self.selector(**kw)
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self.manifest()

    def test_selector_same_size_corruption_fails_hash(self):
        m = self.manifest()
        p = self.directory / hf.SELECTOR_FILE
        data = p.read_bytes()
        p.write_bytes(data[:-1] + b"X")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            hf.verify(self.root, m, "coreai")

    def test_association_and_extra_package_files_rejected(self):
        m = self.manifest()
        bad = copy.deepcopy(m)
        bad["drafter"]["head_export"] = "wrong"
        with self.assertRaisesRegex(ValueError, "association"):
            hf.verify(self.root, bad, "coreai")
        (self.pkg / "untracked.bin").write_bytes(b"unhashed")
        with self.assertRaisesRegex(ValueError, "inventoried"):
            hf.verify(self.root, m, "coreai")

    def test_download_selects_drafter_by_default_and_can_omit_for_plain(self):
        m = self.manifest()
        for plain in (False, True):
            hub, _, revision = self.fixture.mock_hub()
            args = argparse.Namespace(repo="owner/model", revision="main", runtime="coreai", include_export=False,
                                      output=self.fixture.base / ("plain" if plain else "spec"), plain=plain)
            with patch.dict(sys.modules, {"huggingface_hub": hub}), contextlib.redirect_stdout(io.StringIO()):
                hf.download(args)
            patterns = hub.snapshot_download.call_args.kwargs["allow_patterns"]
            self.assertEqual(any(p.startswith("drafter/") for p in patterns), not plain)
            self.assertEqual(hub.snapshot_download.call_args.kwargs["revision"], revision)

    def test_launcher_selects_tested_coreai_drafter_and_portable_support(self):
        a = forge.parser().parse_args(["serve", "--model", str(self.root / "model"), "--build", str(self.build), "--ctx", "8192"])
        cmd, env = forge.prepare(a)
        self.assertEqual(env["RUNTIME"], "coreai")
        self.assertEqual(cmd[cmd.index("--draft") + 1], str(self.pkg.resolve()))
        self.assertEqual(cmd[cmd.index("--drafter") + 1], str(self.directory.resolve()))
        self.assertEqual(cmd[cmd.index("--runtime") + 1], "coreai")
        self.assertNotIn("rtn", " ".join(cmd))
        with patch("forge.subprocess.call") as run, contextlib.redirect_stdout(io.StringIO()):
            forge.main(["serve", "--model", str(self.root / "model"), "--build", str(self.build), "--ctx", "8192", "--dry-run"])
        run.assert_not_called()

    def test_launcher_plain_is_opt_in_and_coreml_requires_it(self):
        args = ["serve", "--model", str(self.root / "model"), "--build", str(self.build), "--ctx", "8192"]
        cmd, _ = forge.prepare(forge.parser().parse_args(args + ["--plain"]))
        self.assertNotIn("--draft", cmd)
        self.assertIn("--plain", cmd)
        (self.build / "manifest_ctx8192_v4.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "requires --plain"):
            forge.prepare(forge.parser().parse_args(args + ["--runtime", "coreml"]))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            forge.parser().parse_args(args + ["--draft"])

    def test_compact_selector_loader_never_opens_full_checkpoint_when_present(self):
        source = Path(__file__).resolve().parents[1] / "scripts/dflash2_ane_drafter.py"
        tree = ast.parse(source.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load_codebooks")
        namespace = dict(Path=Path, DRAFTER=self.directory)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        opened = Mock()
        opened.__enter__ = Mock(return_value=opened)
        opened.__exit__ = Mock(return_value=False)
        safetensors = types.ModuleType("safetensors")
        safetensors.safe_open = Mock(return_value=opened)
        with patch.dict(sys.modules, {"safetensors": safetensors}):
            namespace["load_codebooks"](self.directory)
        self.assertEqual(safetensors.safe_open.call_args.args[0], self.directory / "selector.safetensors")
        self.assertEqual([c.args[0] for c in opened.get_tensor.call_args_list], list(hf.SELECTOR_KEYS))

    @unittest.skipIf(np is None, "server CLI tests require numpy")
    def test_server_defaults_and_mismatch_fail_before_target_load(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("release_server", Path(__file__).resolve().parents[1] / "scripts/qwen38_server.py")
        server = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(server)
        args = server.parse(["--hf", str(self.root / "model"), "--model-dir", str(self.build), "--ctx", "8192"])
        self.assertEqual(args.runtime, "coreai")
        self.assertFalse(args.plain)
        self.assertIsNone(args.draft)
        self.meta["head_export"] = "wrong/lm_head.safetensors"
        self.write_metadata()
        module = types.ModuleType("qwen38_coreai_model")
        module.CoreAIQwen = Mock()
        with patch.dict(sys.modules, {"qwen38_coreai_model": module}), self.assertRaisesRegex(ValueError, "head export mismatch"):
            server.Engine(args)
        module.CoreAIQwen.assert_not_called()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            server.parse(["--hf", "model", "--model-dir", "build", "--draft"])

    @unittest.skipIf(np is None, "smoke integration tests require numpy")
    def test_default_smoke_loads_paired_drafter_and_uses_external_bridge(self):
        self.cfg["vocab_size"] = 20
        self.write_metadata()
        self.selector()
        cfg_path = self.root / "model/config.json"
        config = json.loads(cfg_path.read_text())
        config["text_config"]["vocab_size"] = 20
        cfg_path.write_text(json.dumps(config))
        self.fixture.embedding(shape=(20, 5120))
        m = self.manifest()
        model = Mock()
        model.pos = 1
        model.emb = np.zeros((20, 5120), dtype=np.float16)
        model.feed.return_value = SpeculativeTests.row(1)
        model.call.return_value = np.stack([SpeculativeTests.row(i) for i in [2, 3, 9, 0, 0, 0, 0, 0]])
        model.features.side_effect = lambda n: np.zeros((n, 25600), dtype=np.float16)
        drafter = Mock()
        drafter.propose.return_value = ([2, 3, 4, 5, 6, 7, 8], dict(
            logits=np.zeros((7, 20)), hp=np.zeros((7, 256)), hidden=np.zeros((8, 5120))))
        tokenizer = Mock()
        tokenizer.encode.return_value = types.SimpleNamespace(ids=[0])
        tokenizer.token_to_id.side_effect = {"<|im_end|>": 19, "<|endoftext|>": None}.get
        tokenizer.decode.return_value = "visible speculative output"
        tok_module = types.ModuleType("tokenizers")
        tok_module.Tokenizer = Mock()
        tok_module.Tokenizer.from_file.return_value = tokenizer
        target_module = types.ModuleType("qwen38_coreai_model")
        target_module.CoreAIQwen = Mock(return_value=model)
        support_module = types.ModuleType("dflash2_ane_drafter")
        support_module.load_codebooks = Mock(return_value={"selector": "fixture"})
        draft_module = types.ModuleType("dflash2_coreai_drafter")
        draft_module.CoreAIDrafter = Mock(return_value=drafter)
        bridge = self.fixture.base / "external-bridge"
        bridge.mkdir()
        (bridge / "libcoreai_bridge.dylib").write_bytes(b"not loaded by mocked runtime")
        with patch.dict(sys.modules, {"tokenizers": tok_module, "qwen38_coreai_model": target_module,
                                     "dflash2_ane_drafter": support_module, "dflash2_coreai_drafter": draft_module}), \
                patch.object(hf.sys, "platform", "darwin"), \
                patch.dict(hf.os.environ, {"COREAI_BRIDGE_DIR": str(bridge),
                                           "COREAI_BRIDGE_LIB": str(bridge / "libcoreai_bridge.dylib")}):
            result = hf.smoke(self.root, m, "coreai", 8192, "short prompt", 4)
            self.assertEqual(hf.os.environ["COREAI_BRIDGE_DIR"], str(bridge.resolve()))
        target_module.CoreAIQwen.assert_called_once_with(ctx=8192, ladder=[8192])
        support_module.load_codebooks.assert_called_once_with(self.directory)
        args = draft_module.CoreAIDrafter.call_args.args
        self.assertEqual(args[:2], (self.pkg, self.cfg))
        self.assertIs(args[3], model.emb)
        model.feed.assert_called_once_with([0], on_features=drafter.add_context)
        model.accept.assert_called_once_with(3)
        tokenizer.decode.assert_called_once_with([1, 2, 3, 9], skip_special_tokens=True)
        self.assertTrue(result["speculative"])
        self.assertEqual(result["draft_calls"], 1)

    @unittest.skipIf(np is None, "server CLI tests require numpy")
    def test_server_missing_bridge_fails_before_target_import(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("release_server_missing_bridge", Path(__file__).resolve().parents[1] / "scripts/qwen38_server.py")
        server = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(server)
        args = server.parse(["--hf", str(self.root / "model"), "--model-dir", str(self.build), "--ctx", "8192"])
        target = types.ModuleType("qwen38_coreai_model")
        target.CoreAIQwen = Mock()
        with patch.dict(sys.modules, {"qwen38_coreai_model": target}), \
                patch.dict(hf.os.environ, {"COREAI_BRIDGE_LIB": str(self.fixture.base / "missing.dylib")}):
            with self.assertRaisesRegex(ValueError, "Build the Swift bridge first"):
                server.Engine(args)
        target.CoreAIQwen.assert_not_called()

    @unittest.skipIf(np is None, "server CLI tests require numpy")
    def test_server_unloadable_bridge_fails_before_target_import(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("release_server_bad_bridge", Path(__file__).resolve().parents[1] / "scripts/qwen38_server.py")
        server = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(server)
        args = server.parse(["--hf", str(self.root / "model"), "--model-dir", str(self.build), "--ctx", "8192"])
        library = self.fixture.base / "invalid.dylib"
        library.write_bytes(b"invalid native library")
        bridge = types.ModuleType("coreai_bridge")
        bridge.lib = Mock(side_effect=OSError("incompatible native library"))
        target = types.ModuleType("qwen38_coreai_model")
        target.CoreAIQwen = Mock()
        with patch.dict(sys.modules, {"coreai_bridge": bridge, "qwen38_coreai_model": target}), \
                patch.dict(hf.os.environ, {"COREAI_BRIDGE_LIB": str(library)}):
            with self.assertRaisesRegex(OSError, "incompatible native library"):
                server.Engine(args)
        bridge.lib.assert_called_once_with()
        target.CoreAIQwen.assert_not_called()


@unittest.skipIf(np is None, "speculative orchestration tests require numpy")
class SpeculativeTests(unittest.TestCase):
    def setUp(self):
        self.model = Mock()
        self.model.pos = 10
        self.drafter = Mock()
        self.model.feed.return_value = self.row(1)
        self.drafter.propose.return_value = ([2, 3, 4, 5, 6, 7, 8], dict(
            logits=np.zeros((7, 20)), hp=np.zeros((7, 256)), hidden=np.zeros((8, 5120))))
        self.model.call.return_value = np.stack([self.row(i) for i in [2, 3, 9, 0, 0, 0, 0, 0]])
        self.model.features.side_effect = lambda n: np.zeros((n, 25600), dtype=np.float16)

    @staticmethod
    def row(token):
        row = np.zeros(20)
        row[token] = 1
        return row

    def test_rejection_commits_only_anchor_and_accepted_rows(self):
        out, stats = hf.speculative_greedy(self.model, self.drafter, [0], 20, 4, {19}, gap=0)
        self.assertEqual(out, [1, 2, 3, 9])
        self.model.feed.assert_called_once_with([0], on_features=self.drafter.add_context)
        self.model.call.assert_called_once_with([1, 2, 3, 4, 5, 6, 7, 8])
        self.model.accept.assert_called_once_with(3)
        self.model.features.assert_called_once_with(3)
        self.assertEqual(stats, dict(draft_calls=1, verify_calls=1, accepted_draft_tokens=2))
        np.testing.assert_array_equal(self.drafter.add_context.call_args.args[1], [10, 11, 12])

    def test_completion_cap_or_stop_does_not_commit_unused_rows(self):
        self.model.call.return_value = np.stack([self.row(i) for i in [2, 3, 4, 5, 6, 7, 8, 9]])
        out, _ = hf.speculative_greedy(self.model, self.drafter, [0], 20, 3, {19}, gap=0)
        self.assertEqual(out, [1, 2, 3])
        self.model.accept.assert_called_once_with(2)
        self.model.accept.reset_mock()
        out, _ = hf.speculative_greedy(self.model, self.drafter, [0], 20, 8, {3}, gap=0)
        self.assertEqual(out, [1, 2, 3])
        self.model.accept.assert_called_once_with(2)

    def test_invalid_draft_or_verify_logits_rejected_before_commit(self):
        self.model.call.return_value[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "verify"):
            hf.speculative_greedy(self.model, self.drafter, [0], 20, 4, {19})
        self.model.accept.assert_not_called()
        self.model.call.reset_mock()
        self.drafter.propose.return_value[1]["logits"][0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "drafter"):
            hf.speculative_greedy(self.model, self.drafter, [0], 20, 4, {19})
        self.model.call.assert_not_called()

    def test_no_draft_call_cannot_claim_speculative_pass(self):
        self.model.feed.return_value = self.row(19)
        with self.assertRaisesRegex(ValueError, "not exercised"):
            hf.speculative_greedy(self.model, self.drafter, [0], 20, 4, {19})
        self.drafter.propose.assert_not_called()

    def test_missing_target_taps_cannot_enter_drafter_or_committed_state(self):
        self.model.features.side_effect = None
        self.model.features.return_value = np.zeros((3, 5120), dtype=np.float16)
        with self.assertRaisesRegex(ValueError, "all five taps"):
            hf.speculative_greedy(self.model, self.drafter, [0], 20, 4, {19})
        self.model.accept.assert_not_called()
        self.drafter.add_context.assert_not_called()

    def test_multicycle_greedy_stream_matches_target_and_updates_position(self):
        self.drafter.propose.side_effect = lambda anchor, position: (list(range(anchor + 1, anchor + 8)), dict(
            logits=np.zeros((7, 20)), hp=np.zeros((7, 256)), hidden=np.zeros((8, 5120))))
        self.model.call.side_effect = lambda ids: np.stack([self.row(i + 1) for i in ids])

        def accept(n):
            self.model.pos += n
        self.model.accept.side_effect = accept
        out, stats = hf.speculative_greedy(self.model, self.drafter, [0], 20, 13, {19}, gap=0)
        self.assertEqual(out, list(range(1, 14)))
        self.assertEqual([c.args[0] for c in self.model.accept.call_args_list], [8, 4])
        self.assertEqual([c.args for c in self.drafter.propose.call_args_list], [(1, 10), (9, 18)])
        self.assertEqual(stats["verify_calls"], 2)
        self.assertEqual(self.model.pos, 22)


if __name__ == "__main__":
    unittest.main()
