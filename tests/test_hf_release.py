"""Portable release checks; no third-party libraries, real downloads, or model loading."""
import argparse
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

SPEC = importlib.util.spec_from_file_location(
    "hf_release_under_test", Path(__file__).resolve().parents[1] / "scripts/hf_release.py")
hf = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hf)


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "bundle"
        self.root.mkdir()
        for name in hf.ROOT_DOCUMENTS:
            (self.root / name).write_text("Release attribution fixture\n")
        model = self.root / "model"
        model.mkdir()
        self.write_json(model / "config.json", {"text_config": dict(
            num_hidden_layers=64, hidden_size=5120, vocab_size=1)})
        self.write_json(model / "tokenizer.json", {})
        self.write_json(model / "tokenizer_config.json", {})
        self.embedding()
        for runtime in ("coreai", "coreml"):
            build = self.root / runtime
            build.mkdir()
            chunks = []
            for start in range(0, 64, 4):
                name = f"chunk_{start}.aimodel" if runtime == "coreai" else f"chunk_{start}.mlmodelc"
                (build / name).mkdir()
                (build / name / "data.bin").write_bytes(b"test asset")
                chunks.append(dict(file=name, layers=[start, start + 3], entries=["v8_8k", "p64_8k"]))
            head = "head.aimodel" if runtime == "coreai" else "head.mlmodelc"
            (build / head).mkdir()
            (build / head / "data.bin").write_bytes(b"test head")
            if runtime == "coreai":
                man = dict(version=1, T=8, TP=64, ctxs=[8192], chunks=chunks, head={"file": head})
                name = "manifest.json"
            else:
                man = dict(version=4, T=8, ctx=8192, chunks=chunks, head=head)
                name = "manifest_ctx8192_v4.json"
            self.write_json(build / name, man)
        (self.root / "export").mkdir()
        (self.root / "export/weights.bin").write_bytes(b"optional conversion export")

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value))

    def embedding(self, descr="<f2", shape=(1, 5120), fortran=False, truncate=False):
        header = repr(dict(descr=descr, fortran_order=fortran, shape=shape)).encode("latin1")
        header += b" " * ((64 - (10 + len(header) + 1) % 64) % 64) + b"\n"
        (self.root / "model/embed_tokens_fp16.npy").write_bytes(
            b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header
            + b"\0" * (shape[0] * shape[1] * 2 - int(truncate)))

    def manifest(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(hf.make_manifest(argparse.Namespace(
                bundle=self.root, model_id="example/model", model_revision="f" * 40, plain=True)), 0)
        return hf.load_manifest(self.root)

    def test_default_repository(self):
        parser = argparse.ArgumentParser()
        hf.add_commands(parser.add_subparsers(dest="command", required=True))
        args = parser.parse_args(["download", "--output", str(self.base / "download")])
        self.assertEqual(args.repo, "anemll/anemll-forge-qwen3.8-27B")

    def test_required_documentation_and_modification_notices(self):
        m = self.manifest()
        docs = [f for f in m["files"] if f["component"] == "documentation"]
        self.assertEqual({f["path"] for f in docs}, hf.ROOT_DOCUMENTS)
        for f in m["files"]:
            if hf.requires_modification_notice(f["path"], f["component"]):
                self.assertIn("ANEMLL", f["modification_notice"])
                self.assertIn("MODIFICATIONS.md", f["modification_notice"])
        for name in hf.ROOT_DOCUMENTS:
            bad = copy.deepcopy(m)
            bad["files"] = [f for f in bad["files"] if f["path"] != name]
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "documentation"):
                hf.validate_manifest(bad)
        bad = copy.deepcopy(m)
        next(f for f in bad["files"] if f["path"] == "model/embed_tokens_fp16.npy").pop("modification_notice")
        with self.assertRaisesRegex(ValueError, "modification notice"):
            hf.validate_manifest(bad)

    def test_root_documentation_corruption_is_detected(self):
        m = self.manifest()
        p = self.root / "NOTICE"
        p.write_bytes(b"X" * p.stat().st_size)
        for runtime in ("coreai", "coreml"):
            with self.subTest(runtime=runtime), self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                hf.verify(self.root, m, runtime, plain=True)

    def test_documentation_symlink_rejected(self):
        p = self.root / "LICENSE"
        p.unlink()
        outside = self.base / "license.txt"
        outside.write_text("license fixture")
        p.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.manifest()

    def test_missing_root_documentation_rejected(self):
        (self.root / "QWEN_SOURCE.json").unlink()
        with self.assertRaisesRegex(ValueError, "documentation"):
            self.manifest()

    def test_documentation_component_rejects_arbitrary_root_file(self):
        m = self.manifest()
        bad = copy.deepcopy(m)
        next(f for f in bad["files"] if f["path"] == "README.md")["path"] = "private.txt"
        with self.assertRaisesRegex(ValueError, "documentation"):
            hf.validate_manifest(bad)

    def test_manifest_roundtrip_and_both_runtime_layouts(self):
        m = self.manifest()
        self.assertEqual(m["upstream_model"]["id"], "example/model")
        self.assertEqual(set(m["runtimes"]), {"coreai", "coreml"})
        self.assertEqual(json.loads((self.root / "release.json").read_text()), m)
        for runtime in m["runtimes"]:
            result = hf.verify(self.root, m, runtime, plain=True)
            selected = [f for f in m["files"] if f["component"] in ("documentation", "model", runtime)]
            self.assertEqual(result["verified_files"], len(selected))
            self.assertEqual(result["verified_bytes"], sum(f["bytes"] for f in selected))

    def test_integrity_detects_same_size_corruption(self):
        m = self.manifest()
        (self.root / "coreai/head.aimodel/data.bin").write_bytes(b"bad! head")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            hf.verify(self.root, m, "coreai", plain=True)

    def test_missing_runtime_asset_rejected_before_manifest_written(self):
        shutil.rmtree(self.root / "coreai/head.aimodel")
        with self.assertRaisesRegex(ValueError, "asset missing"):
            self.manifest()
        self.assertFalse((self.root / "release.json").exists())

    def test_invalid_embeddings_rejected(self):
        for kwargs in ({"descr": "<f4"}, {"shape": (1, 5119)}, {"fortran": True}, {"truncate": True}):
            with self.subTest(**kwargs):
                self.embedding(**kwargs)
                with self.assertRaises(ValueError):
                    self.manifest()
        (self.root / "model/embed_tokens_fp16.npy").write_bytes(b"not numpy")
        with self.assertRaisesRegex(ValueError, "NumPy"):
            self.manifest()
        (self.root / "model/embed_tokens_fp16.npy").write_bytes(b"\x93NUMPY\x01\x00\x01")
        with self.assertRaisesRegex(ValueError, "Truncated"):
            self.manifest()

    def test_uninventoried_compiled_override_rejected(self):
        m = self.manifest()
        extra = self.root / "coreai/chunk_0.aimodelc"
        extra.mkdir()
        (extra / "data.bin").write_bytes(b"unreviewed compiled variant")
        with self.assertRaisesRegex(ValueError, "not inventoried"):
            hf.verify(self.root, m, "coreai", plain=True)
        # Regenerating the release binds both source and compiled variants to hashes.
        m = self.manifest()
        hf.verify(self.root, m, "coreai", plain=True)
        (extra / "data.bin").write_bytes(b"changed compiled variant")
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            hf.verify(self.root, m, "coreai", plain=True)

    def test_explicit_compiled_reference_must_be_relative_and_present(self):
        p = self.root / "coreai/manifest.json"
        man = json.loads(p.read_text())
        for name in ("../outside.aimodelc", "missing.aimodelc"):
            with self.subTest(name=name):
                man["chunks"][0]["compiled"] = name
                self.write_json(p, man)
                with self.assertRaises(ValueError):
                    self.manifest()

    def test_hostile_and_noncanonical_paths_rejected(self):
        for name in (".", "", "/tmp/a", "../a", "a/../b", "a//b", "a/./b", "a/", "a\\b", "a*", "a?", "a[0]"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                hf.relative(name)
        self.assertEqual(str(hf.relative("coreai/chunk.aimodel")), "coreai/chunk.aimodel")

    def test_symlink_escape_rejected(self):
        (self.root / "escape").symlink_to(self.base, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "escapes"):
            hf.local(self.root, "escape/outside")

    def test_component_roots_must_be_distinct_and_nonoverlapping(self):
        m = self.manifest()
        for root in ("model", "model/coreai"):
            bad = copy.deepcopy(m)
            bad["runtimes"]["coreai"]["path"] = root
            for f in bad["files"]:
                if f["component"] == "coreai":
                    f["path"] = root + "/" + f["path"].split("/", 1)[1]
            with self.subTest(root=root), self.assertRaises(ValueError):
                hf.validate_manifest(bad)

    def test_chunk_coverage_must_be_complete_ordered_and_nonoverlapping(self):
        p = self.root / "coreai/manifest.json"
        original = json.loads(p.read_text())
        for chunks in (original["chunks"][:-1], [original["chunks"][0]] + original["chunks"], list(reversed(original["chunks"]))):
            with self.subTest(chunks=chunks):
                bad = copy.deepcopy(original)
                bad["chunks"] = chunks
                self.write_json(p, bad)
                with self.assertRaises(ValueError):
                    self.manifest()

    def test_verify_batch_and_context_entry_required(self):
        p = self.root / "coreai/manifest.json"
        original = json.loads(p.read_text())
        bad = copy.deepcopy(original)
        bad["T"] = 1
        self.write_json(p, bad)
        with self.assertRaises(ValueError):
            self.manifest()
        bad = copy.deepcopy(original)
        bad["chunks"][0]["entries"] = ["v8_2k"]
        self.write_json(p, bad)
        with self.assertRaises(ValueError):
            self.manifest()

    def mock_hub(self):
        module = types.ModuleType("huggingface_hub")
        sha = "a" * 40
        api = Mock()
        api.repo_info.return_value = types.SimpleNamespace(sha=sha)
        module.HfApi = Mock(return_value=api)
        module.hf_hub_download = Mock(return_value=str(self.root / "release.json"))
        def snapshot(**kw):
            dest = Path(kw["local_dir"])
            for name in kw["allow_patterns"]:
                p = dest / name
                p.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.root / name, p)
            return str(dest)
        module.snapshot_download = Mock(side_effect=snapshot)
        return module, api, sha

    def test_download_pins_revision_and_selects_requested_components(self):
        m = self.manifest()
        for runtime, include_export in (("coreai", False), ("coreml", True)):
            with self.subTest(runtime=runtime):
                hub, api, sha = self.mock_hub()
                output = self.base / (runtime + "-download")
                args = argparse.Namespace(repo="owner/release", revision="v1", runtime=runtime, include_export=include_export, output=output, plain=True)
                with patch.dict(sys.modules, {"huggingface_hub": hub}), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(hf.download(args), 0)
                api.repo_info.assert_called_once_with(repo_id="owner/release", repo_type="model", revision="v1")
                self.assertEqual(hub.hf_hub_download.call_args.kwargs["revision"], sha)
                call = hub.snapshot_download.call_args.kwargs
                self.assertEqual(call["revision"], sha)
                groups = {"documentation", "model", runtime} | ({"export"} if include_export else set())
                expected = {"release.json"} | {f["path"] for f in m["files"] if f["component"] in groups}
                self.assertEqual(set(call["allow_patterns"]), expected)
                self.assertFalse((output / ("coreml" if runtime == "coreai" else "coreai")).exists())
                self.assertEqual((output / "export").exists(), include_export)

    def test_export_request_rejected_when_bundle_has_none(self):
        shutil.rmtree(self.root / "export")
        self.manifest()
        hub, _, _ = self.mock_hub()
        args = argparse.Namespace(repo="owner/release", revision="main", runtime="coreai", include_export=True, output=self.base / "download", plain=True)
        with patch.dict(sys.modules, {"huggingface_hub": hub}), self.assertRaisesRegex(ValueError, "no optional export"):
            hf.download(args)
        hub.snapshot_download.assert_not_called()

    def quick_args(self, report=None):
        return argparse.Namespace(command="quick-test", bundle=self.root, runtime="coreai", ctx=None, check_only=True, prompt="Example", tokens=4, report=report, plain=True)

    def test_check_only_never_calls_inference(self):
        self.manifest()
        report = self.base / "report.json"
        with patch.object(hf, "smoke", side_effect=AssertionError("must not load model")) as smoke, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(hf.run(self.quick_args(report)), 0)
        smoke.assert_not_called()
        result = json.loads(report.read_text())
        self.assertFalse(result["inference_run"])
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["ctx"], 8192)

    def test_report_cannot_overwrite_bundle_file(self):
        self.manifest()
        report = self.root / "model/config.json"
        before = report.read_bytes()
        with self.assertRaises(ValueError):
            hf.run(self.quick_args(report))
        self.assertEqual(report.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
