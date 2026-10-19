"""Context-capped builds (scripts/ctx_variant.py): the capped manifest, the cap chosen for a server context, deriving
once with `forge.py compile --ctx` and reusing it at runtime; the Core AI strip is replaced by a stub."""
import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import ctx_variant  # noqa: E402
import forge  # noqa: E402

K = 1024


def manifest():
    ctxs = [8 * K, 16 * K, 32 * K, 64 * K, 96 * K, 128 * K, 192 * K]
    ptp = {str(c): (32 if c > 96 * K else 64) for c in ctxs}
    entries = [f"v8_{c // K}k" for c in ctxs] + [f"p{ptp[str(c)]}_{c // K}k" for c in ctxs]
    chunk = lambda i: {"file": f"chunk_{i}.aimodel", "entries": list(entries), "numerics": {"KV_KEYS_T": True}}
    return {"ctxs": ctxs, "pctxs": list(ctxs), "TP": 64, "ptp": ptp,
            "kv_len": {str(c): (65472 if c == 64 * K else c) for c in ctxs},
            "pkv_len": {str(c): (65472 if c == 64 * K else c) for c in ctxs},
            "chunks": [chunk(0), chunk(1)], "head": {"file": "head_T8.aimodel"}}


class CappedManifest(unittest.TestCase):
    def test_entries_and_metadata_up_to_the_cap(self):
        m = ctx_variant.capped_manifest(manifest(), 16 * K, Path("/b"))
        self.assertEqual(m["ctxs"], [8 * K, 16 * K])
        self.assertEqual(m["pctxs"], [8 * K, 16 * K])
        self.assertEqual(m["chunks"][0]["entries"], ["v8_8k", "v8_16k", "p64_8k", "p64_16k"])
        self.assertEqual(set(m["kv_len"]), {"8192", "16384"})
        self.assertEqual(m["ptp"], {"8192": 64, "16384": 64})
        self.assertEqual(m["TP"], 64)
        self.assertEqual(m["derived_ctx"], {"cap": 16 * K, "from": "/b"})
        # a cap among the 32-row entries keeps both row counts; TP is the largest
        m = ctx_variant.capped_manifest(manifest(), 128 * K)
        self.assertIn("p32_128k", m["chunks"][0]["entries"])
        self.assertNotIn("p32_192k", m["chunks"][0]["entries"])
        self.assertEqual(m["TP"], 64)
        with self.assertRaises(ValueError):
            ctx_variant.capped_manifest(manifest(), 4 * K)

    def test_parse_and_cap_for_a_server_context(self):
        self.assertEqual(ctx_variant.parse_cap("16K"), 16 * K)
        self.assertEqual(ctx_variant.parse_cap("16384"), 16 * K)
        self.assertIsNone(ctx_variant.parse_cap(None))
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "manifest.json").write_text(json.dumps(manifest()))
            self.assertEqual(ctx_variant.capped_for(Path(d), 16 * K), 16 * K)
            self.assertEqual(ctx_variant.capped_for(Path(d), 20000), 16 * K)  # between entries: the one below
            self.assertIsNone(ctx_variant.capped_for(Path(d), 192 * K))       # the whole build
            self.assertIsNone(ctx_variant.capped_for(Path(d), 300000))
            self.assertIsNone(ctx_variant.capped_for(Path(d), None))


class Prepare(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def strip(src, dst, keep):
            self.calls.append((src.name, sorted(keep)))
            shutil.copytree(src, dst)
            return list(keep), []
        stub = types.ModuleType("strip_functions")
        stub.strip, stub.require = strip, lambda: None
        self.addCleanup(sys.modules.pop, "strip_functions", None)
        sys.modules["strip_functions"] = stub
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root, self.state = Path(tmp.name) / "build", Path(tmp.name) / "state"
        self.root.mkdir()
        for name in ("chunk_0.aimodel", "chunk_1.aimodel", "head_T8.aimodel"):
            (self.root / name).mkdir()
            (self.root / name / "main.mlirb").write_bytes(b"x" * 100)
        (self.root / "manifest.json").write_text(json.dumps(manifest()))
        (self.root / "basis.json").write_text('{"basis_id": "test"}')
        self.log = []

    def prepare(self, ctx, create=True):
        return ctx_variant.prepare(self.root, ctx, log=self.log.append, state=self.state, create=create)

    def test_full_context_and_runtime_without_a_derived_build_use_the_download(self):
        self.assertEqual(self.prepare("192K"), self.root)
        self.assertEqual(self.prepare(None), self.root)
        self.assertEqual(self.prepare("16K", create=False), self.root)  # the runtime never derives
        self.assertEqual(self.calls, [])

    def test_compile_derives_once_and_the_runtime_reuses_it(self):
        out = self.prepare("16K")
        self.assertNotEqual(out, self.root)
        self.assertEqual(out.parent, self.state / "builds")
        self.assertEqual(self.calls, [("chunk_0.aimodel", ["p64_16k", "p64_8k", "v8_16k", "v8_8k"]),
                                      ("chunk_1.aimodel", ["p64_16k", "p64_8k", "v8_16k", "v8_8k"])])
        m = json.loads((out / "manifest.json").read_text())
        self.assertEqual(m["ctxs"], [8 * K, 16 * K])
        self.assertTrue((out / "head_T8.aimodel").is_symlink())
        self.assertTrue((out / "basis.json").is_file())
        self.assertEqual(json.loads((self.root / "manifest.json").read_text()), manifest())  # download untouched
        self.calls.clear()
        self.assertEqual(self.prepare("16K"), out)                 # derived once
        self.assertEqual(self.prepare("16K", create=False), out)   # the server at CTX=16K
        self.assertEqual(self.prepare(8 * K, create=False), out)   # a smaller cap fits in it too
        self.assertEqual(self.prepare("64K", create=False), self.root)  # a larger one needs the full build
        self.assertEqual(self.calls, [])

    def test_partial_chunks_are_redone(self):
        out = ctx_variant.path(self.root, 16 * K, self.state)
        (out / "chunk_0.partial.aimodel").mkdir(parents=True)
        self.prepare("16K")
        self.assertFalse((out / "chunk_0.partial.aimodel").exists())
        self.assertTrue((out / "chunk_0.aimodel").is_dir())


class ForgeCompileCtx(unittest.TestCase):
    def test_ctx_reaches_the_compile_child(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "manifest.json").write_text(json.dumps(manifest()))
            import contextlib
            import io
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = forge.main(["compile", "--build", d, "--ctx", "16K", "--dry-run"])
            self.assertEqual(rc, 0)
            argv = json.loads(out.getvalue())["argv"]
            self.assertEqual(argv[argv.index("--ctx") + 1], "16K")


if __name__ == "__main__":
    unittest.main()
