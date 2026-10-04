"""First-load compile guide: estimates, hints, cache detection and the guided load loop (no ANE, no models)."""
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import coreai_compile_guide as G

CTXS = [8192, 16384, 32768, 49152, 65472]


def settings(vb, pb, formats):
    return {"gdn_fast": True, "att_block": vb, "att_block_prefill": pb, "formats": formats, "ctxs": CTXS, "pctxs": CTXS}


class EstimateTests(unittest.TestCase):
    def test_fit_tracks_measured_chunks(self):
        measured = {(16384, 16384, 2): 82.9, (16384, 16384, 1): 45.9, (2048, 2048, 2): 407.6, (2048, 4096, 2): 135.9,
                    (2048, 2048, 1): 243.8, (2048, 4096, 1): 90.3, (4096, 4096, 1): 80.7}
        for (vb, pb, f), sec in measured.items():
            est = G.estimate_chunk_s(settings(vb, pb, f))
            self.assertLess(abs(est / sec - 1), 0.3, (vb, pb, f, est, sec))

    def test_manifest_settings_and_release_defaults(self):
        man = {"kv_cache": {"format": "selectable"}, "kv_len": {"8192": 8192}, "pkv_len": {"8192": 8192},
               "chunks": [{"numerics": {"GDN_FAST": True, "ATT_BLOCK": 2048}}]}
        s = G.build_settings(man)
        self.assertEqual((s["att_block"], s["att_block_prefill"], s["formats"]), (2048, 2048, 2))
        old = G.build_settings({"kv_len": {"8192": 8192}, "chunks": [{"numerics": {"SILU": "tanh"}}]})
        self.assertEqual((old["gdn_fast"], old["att_block"], old["formats"]), (False, 16384, 1))

    def test_hints_name_only_what_helps(self):
        slow = G.hints(settings(2048, 2048, 2), 3600)
        self.assertTrue(any("ATT_BLOCK_PREFILL=4096" in h for h in slow))
        self.assertTrue(any("--kv-cache-dtype v8" in h for h in slow))
        self.assertTrue(any("fewer contexts" in h for h in slow))
        default = G.hints(settings(2048, 4096, 1), 1400)   # the builder default: nothing to change in its options
        self.assertFalse(any("ATT_BLOCK_PREFILL" in h or "--kv-cache-dtype" in h for h in default))
        self.assertTrue(any("fewer contexts" in h for h in default))    # still over 15 minutes for a quick test
        self.assertTrue(any("forge.py compile" in h for h in default))
        self.assertFalse(any("fewer contexts" in h for h in G.hints(settings(2048, 4096, 1), 600)))


class CacheTests(unittest.TestCase):
    def test_cached_only_in_requested_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pkg = tmp / "chunk.aimodel"
            pkg.mkdir()
            (pkg / "main.hash").write_bytes(b"\x01\x02")
            with unittest.mock.patch.object(G, "CACHE", tmp / "cache"), \
                    unittest.mock.patch.object(G, "os_build", return_value="26A434"):
                self.assertFalse(G.is_cached(pkg, 2))
                d = G.cache_dir(pkg) / "spec" / "model.aimodelx" / "x.mpsgraphpackage"
                d.mkdir(parents=True)
                (d / "manifest.plist").write_bytes(b'{\\"aneBondedCompileMode\\" : 2,\\n}')
                self.assertTrue(G.is_cached(pkg, 2))
                self.assertFalse(G.is_cached(pkg, 0))
            self.assertTrue(G.is_cached(tmp / "x.aimodelc"))


class GuideTests(unittest.TestCase):
    def guide(self, cold, lines):
        items = [("a.aimodel", Path("/a"), 10.0), ("b.aimodel", Path("/b"), 10.0), ("c.aimodel", Path("/c"), 20.0)]
        with unittest.mock.patch.object(G, "is_cached", side_effect=lambda p, mode=None: p.name not in cold):
            return G.CompileGuide(items, log=lines.append, settings="test", hint_lines=["hint"])

    def test_announce_and_eta_rescale(self):
        lines = []
        g = self.guide({"a", "c"}, lines)
        g.announce()
        self.assertIn("2 of 3 packages are not compiled", lines[0])
        self.assertTrue(any("hint" in l for l in lines))
        self.assertEqual(g.remaining_s(), 30.0)
        g.done["a.aimodel"] = 20.0          # took twice the estimate
        self.assertEqual(g.remaining_s(), 40.0)

    def test_warm_load_is_direct_and_cold_load_reports(self):
        lines = []
        g = self.guide({"a"}, lines)
        self.assertEqual(g.load("b.aimodel", lambda: "warm"), "warm")
        self.assertEqual(lines, [])
        with unittest.mock.patch.object(G, "HEARTBEAT_S", 0.01):
            self.assertEqual(g.load("a.aimodel", lambda: time.sleep(1.2) or "cold"), "cold")
        self.assertTrue(lines[0].startswith(G.TAG + " compiling a.aimodel (1/1)"))
        self.assertTrue(any("so far" in l for l in lines))
        self.assertIn("compiled a.aimodel", lines[-1])

    def test_errors_reach_the_caller(self):
        g = self.guide({"a"}, [])

        def boom():
            raise RuntimeError("compile failed")
        with self.assertRaisesRegex(RuntimeError, "compile failed"):
            g.load("a.aimodel", boom)

    def test_ctrl_c_during_a_compile_exits_at_once_with_resume_hint(self):
        script = (
            "import sys, time, unittest.mock\n"
            f"sys.path.insert(0, {str(Path(G.__file__).parent)!r})\n"
            "import coreai_compile_guide as G\n"
            "with unittest.mock.patch.object(G, 'is_cached', return_value=False):\n"
            "    g = G.CompileGuide([('slow.aimodel', '/x', 600)], log=lambda m: print(m, flush=True))\n"
            "print('ready', flush=True)\n"
            "g.load('slow.aimodel', lambda: time.sleep(600))\n")
        p = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
        self.assertEqual(p.stdout.readline().strip(), "ready")
        time.sleep(0.5)
        t = time.time()
        p.send_signal(signal.SIGINT)
        out = p.stdout.read()
        p.stdout.close()
        self.assertEqual(p.wait(timeout=10), 130)
        self.assertLess(time.time() - t, 5)
        self.assertIn("interrupted while compiling slow.aimodel", out)
        self.assertIn("run the same command again to resume", out)

    def test_all_cached_is_one_line(self):
        lines = []
        self.guide(set(), lines).announce()
        self.assertEqual(len(lines), 1)
        self.assertIn("already compiled", lines[0])


if __name__ == "__main__":
    unittest.main()
