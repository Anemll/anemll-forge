import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import forge


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.model = self.root / "model with spaces"
        self.model.mkdir()
        (self.model / "config.json").write_text(json.dumps({"text_config": {
            "hidden_size": 5120, "num_hidden_layers": 64}}))

    def args(self, *args):
        return forge.parser().parse_args([*args, "--model", str(self.model)])

    def test_build_recipe_and_paths(self):
        export = self.root / "export with spaces"
        export.mkdir()
        a = self.args("convert", "--export", str(export), "--output", str(self.root / "builds"))
        command, env = forge.prepare(a)
        self.assertEqual(command[-1], "build_v3")
        self.assertEqual(env["EXPORT_DIR"], str(export))
        self.assertEqual(env["MLP_SILU"], "tanh")
        self.assertEqual(env["MLP_DS"], "1")
        self.assertEqual(len(env["CHUNK_PLAN"].split(",")), 16)
        self.assertFalse((self.root / "builds").exists())

    def test_refuses_existing_build_destination(self):
        export = self.root / "quantized"
        export.mkdir()
        out = self.root / "builds"
        (out / export.name).mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "already exists"):
            forge.prepare(self.args("convert", "--export", str(export), "--output", str(out)))

    def test_model_shape_rejected(self):
        (self.model / "config.json").write_text('{"text_config": {"hidden_size": 1024}}')
        with self.assertRaisesRegex(ValueError, "64 layers"):
            forge.prepare(self.args("convert", "--export", str(self.root), "--output", str(self.root / "out")))

    def test_server_uses_loopback_and_checkpoint_specific_cache(self):
        build = self.root / "build"
        build.mkdir()
        (build / "manifest_ctx16384_v4.json").write_text('{}')
        cmd, env = forge.prepare(self.args("serve", "--build", str(build)))
        self.assertEqual(cmd[cmd.index("--host")+1], "127.0.0.1")
        self.assertEqual(env["EMBED_NPY"], str(self.model / ".anemll-forge/embed_tokens_fp16.npy"))

    def test_dry_run_does_not_launch_or_write(self):
        export = self.root / "export"
        export.mkdir()
        with patch("forge.subprocess.call") as run, contextlib.redirect_stdout(io.StringIO()) as out:
            result = forge.main(["convert", "--model", str(self.model), "--export", str(export),
                                 "--output", str(self.root / "builds"), "--dry-run"])
        self.assertEqual(result, 0)
        run.assert_not_called()
        self.assertEqual(json.loads(out.getvalue())["environment"]["GDN_SV"], "64")
        self.assertFalse((self.root / "builds").exists())

    def test_coreai_rejects_contexts_outside_manifest(self):
        build = self.root / "coreai"
        build.mkdir()
        (build / "manifest.json").write_text('{"ctxs": [8192, 16384]}')
        for ctx in (2048, 12000, 32768):
            with self.subTest(ctx=ctx), self.assertRaisesRegex(ValueError, "Unsupported Core AI"):
                forge.prepare(self.args("serve", "--runtime", "coreai", "--build", str(build), "--ctx", str(ctx)))
        _, env = forge.prepare(self.args("serve", "--runtime", "coreai", "--build", str(build), "--ctx", "8192"))
        self.assertEqual(env["RUNTIME"], "coreai")

    def test_coreml_rejects_competing_v5_manifest(self):
        build = self.root / "build"
        build.mkdir()
        for v in (4, 5):
            (build / f"manifest_ctx16384_v{v}.json").write_text('{}')
        with self.assertRaisesRegex(ValueError, "competing v5"):
            forge.prepare(self.args("serve", "--build", str(build)))

    def test_quantizer_requires_dataset_and_safe_tag(self):
        wiki = self.root / "wiki"
        wiki.mkdir()
        args = ["quantize", "--wiki", str(wiki), "--output", str(self.root / "runs")]
        with self.assertRaisesRegex(ValueError, "wiki2_train"):
            forge.prepare(self.args(*args))
        for s in ("train", "test"):
            (wiki / f"wiki2_{s}.txt").write_text("fixture")
        with self.assertRaisesRegex(ValueError, "single directory"):
            forge.prepare(self.args(*args, "--tag", "../escape"))
        _, env = forge.prepare(self.args(*args))
        self.assertEqual(env["HEAD"], "LUT4 per-tensor + pcs")


if __name__ == "__main__":
    unittest.main()
