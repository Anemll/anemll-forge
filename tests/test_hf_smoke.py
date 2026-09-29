"""CPU-only smoke orchestration tests; no runtime packages or models are loaded."""
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import hf_release

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "smoke orchestration tests require numpy")
class SmokeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        (self.root / "model").mkdir()
        (self.root / "model/config.json").write_text(json.dumps({"text_config": {
            "num_hidden_layers": 64, "hidden_size": 5120, "vocab_size": 6}}))
        (self.root / "coreai").mkdir()
        (self.root / "coreai/manifest.json").write_text(json.dumps({"kv_len": {"64": 64}}))
        self.manifest = {"model_path": "model", "runtimes": {
            "coreml": {"path": "coreml", "contexts": [64]},
            "coreai": {"path": "coreai", "contexts": [64]}}}
        self.tok = Mock()
        self.tok.encode.return_value = types.SimpleNamespace(ids=[0, 1])
        self.tok.token_to_id.side_effect = {"<|im_end|>": 5, "<|endoftext|>": None}.get
        self.tok.decode.return_value = "visible output"
        tokenizers = types.ModuleType("tokenizers")
        tokenizers.Tokenizer = Mock()
        tokenizers.Tokenizer.from_file.return_value = self.tok
        self.model = Mock()
        self.loader = Mock(return_value=self.model)
        coreml = types.ModuleType("qwen38_ane_model")
        coreml.load_model = self.loader
        coreai = types.ModuleType("qwen38_coreai_model")
        coreai.CoreAIQwen = Mock(return_value=self.model)
        self.coreai_constructor = coreai.CoreAIQwen
        for patcher in (
            patch.dict(sys.modules, {"tokenizers": tokenizers,
                                    "qwen38_ane_model": coreml,
                                    "qwen38_coreai_model": coreai}),
            patch.object(hf_release.sys, "platform", "darwin"),
            patch.dict(os.environ, {}, clear=False),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def logits(self, winner):
        values = np.zeros(6, dtype=np.float32)
        values[winner] = 1
        return values

    def smoke(self, *, runtime="coreml", ctx=64, tokens=3):
        return hf_release.smoke(self.root, self.manifest, runtime, ctx, "short prompt", tokens)

    def test_greedy_generation_is_bounded_and_uses_last_prompt_logits(self):
        self.model.feed.return_value = self.logits(2)
        self.model.step.side_effect = [self.logits(3), self.logits(4)]
        result = self.smoke()
        self.loader.assert_called_once_with()
        self.tok.encode.assert_called_once_with("short prompt", add_special_tokens=False)
        self.model.feed.assert_called_once_with([0, 1])
        self.assertEqual([c.args[0] for c in self.model.step.call_args_list], [2, 3])
        self.tok.decode.assert_called_once_with([2, 3, 4], skip_special_tokens=True)
        self.assertEqual(result["prompt_tokens"], 2)
        self.assertEqual(result["generated_tokens"], 3)

    def test_stop_token_ends_generation_without_feeding_stop(self):
        self.model.feed.return_value = self.logits(2)
        self.model.step.return_value = self.logits(5)
        result = self.smoke(tokens=12)
        self.model.step.assert_called_once_with(2)
        self.tok.decode.assert_called_once_with([2, 5], skip_special_tokens=True)
        self.assertEqual(result["generated_tokens"], 2)

    def test_nonfinite_logits_rejected_at_prompt_or_decode(self):
        for bad in (np.nan, np.inf, -np.inf):
            for step in (0, 1):
                with self.subTest(value=bad, step=step):
                    self.model.reset_mock()
                    self.tok.decode.reset_mock()
                    values = self.logits(2)
                    values[0] = bad
                    self.model.feed.return_value = values if step == 0 else self.logits(2)
                    self.model.step.return_value = values
                    with self.assertRaisesRegex(ValueError, f"Invalid logits at step {step}"):
                        self.smoke()
                    self.assertEqual(self.model.step.call_count, step)
                    self.tok.decode.assert_not_called()

    def test_wrong_logit_shape_rejected(self):
        self.model.feed.return_value = np.zeros((1, 6), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "Invalid logits at step 0"):
            self.smoke()
        self.model.step.assert_not_called()

    def test_empty_decoded_output_is_not_success(self):
        self.model.feed.return_value = self.logits(5)
        self.tok.decode.return_value = "  \n"
        with self.assertRaisesRegex(ValueError, "no visible text"):
            self.smoke()
        self.model.step.assert_not_called()

    def test_context_capacity_rejected_before_loading(self):
        for runtime in ("coreml", "coreai"):
            with self.subTest(runtime=runtime):
                if runtime == "coreai":
                    (self.root / "coreai/manifest.json").write_text(json.dumps({"kv_len": {"64": 12}}))
                with self.assertRaisesRegex(ValueError, "context capacity"):
                    self.smoke(runtime=runtime, ctx=12 if runtime == "coreml" else 64)
        self.loader.assert_not_called()
        self.coreai_constructor.assert_not_called()

    def test_empty_or_invalid_token_ids_rejected_before_loading(self):
        for ids, message in (([], "context capacity"), ([6], "outside"), ([-1], "outside")):
            with self.subTest(ids=ids):
                self.tok.encode.return_value = types.SimpleNamespace(ids=ids)
                with self.assertRaisesRegex(ValueError, message):
                    self.smoke()
        self.loader.assert_not_called()

    def test_coreai_pins_context_and_checked_bridge(self):
        self.model.feed.return_value = self.logits(2)
        os.environ.update(COREAI_BRIDGE_DIR="/wrong/directory", COREAI_BRIDGE_LIB="/wrong/library")
        bridge = Path(hf_release.__file__).resolve().parents[1] / "coreai/swift_bridge/libcoreai_bridge.dylib"
        original_is_file = Path.is_file

        def fake_is_file(path):
            return True if path == bridge else original_is_file(path)

        with patch.object(Path, "is_file", fake_is_file):
            result = self.smoke(runtime="coreai", tokens=1)
        self.coreai_constructor.assert_called_once_with(ctx=64, ladder=[64])
        self.loader.assert_not_called()
        self.assertEqual(os.environ["COREAI_BRIDGE"], "1")
        self.assertEqual(os.environ["COREAI_BRIDGE_DIR"], str(bridge.parent))
        self.assertEqual(os.environ["COREAI_BRIDGE_LIB"], str(bridge))
        self.assertEqual(result["generated_tokens"], 1)
        self.model.step.assert_not_called()


if __name__ == "__main__":
    unittest.main()
