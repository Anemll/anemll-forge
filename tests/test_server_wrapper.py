"""Shell lifecycle checks in a temporary checkout with synthetic server processes."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest

try:
    from . import test_drafter_release as drafter_fixture
except ImportError:
    import test_drafter_release as drafter_fixture


class ServerWrapperTests(unittest.TestCase):
    def setUp(self):
        self.fixture = drafter_fixture.DrafterReleaseTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "checkout with spaces"
        self.root.mkdir()
        scripts = self.root / "scripts"
        scripts.mkdir()
        source = Path(__file__).resolve().parents[1]
        shutil.copy2(source / "forge.py", self.root / "forge.py")
        for name in ("qwen38_server.sh", "qwen38_server_process.py", "hf_release.py", "qwen38_kv_cache.py",
                     "qwen38_hardware_profile.py"):
            shutil.copy2(source / "scripts" / name, scripts / name)
        bridge = self.root / "coreai/swift_bridge"
        bridge.mkdir(parents=True)
        (bridge / "libcoreai_bridge.dylib").write_bytes(b"synthetic bridge fixture")
        (bridge / "coreai_bridge.py").write_text("def lib(): return object()\n")
        (scripts / "qwen38_server.py").write_text(
            "import time\n"
            "for index in range(16):\n"
            " print(f'[00:00:00] loaded chunk_L{index*4:02}-{index*4+3:02}.aimodel (2 entries, 0s, bridge)', flush=True)\n"
            " if index == 0: time.sleep(1.2)\n"
            "time.sleep(1.2)\n"
            "print('[qwen38] serving OpenAI API', flush=True)\n"
            "time.sleep(60)\n")
        shim = self.root / "python-shim"
        shim.write_text(f"#!{sys.executable}\nimport os, sys\n"
                        "if sys.argv[1:] == ['-c', 'import numpy']: sys.exit(0)\n"
                        "os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n")
        shim.chmod(0o755)
        self.pidfile = self.root / "server.pid"
        self.env = dict(os.environ, PY=str(shim), MODEL=str(self.fixture.root / "model"),
                        BUILD=str(self.fixture.build), CTX="8K", PORT="18765", DRAFT="on", PI_SYNC="0",
                        ANEMLL_FORGE_STATE=str(self.root), LOG=str(self.root / "server.log"),
                        PIDFILE=str(self.pidfile), COREAI_BRIDGE_DIR=str(bridge),
                        COREAI_BRIDGE_LIB=str(bridge / "libcoreai_bridge.dylib"))
        for key in ("DRAFTER", "EXTRA_ARGS", "KV_CACHE_DTYPE"):
            self.env.pop(key, None)

    def run_wrapper(self, action, **overrides):
        return subprocess.run(["bash", str(self.root / "scripts/qwen38_server.sh"), action],
                              env={**self.env, **overrides}, text=True, capture_output=True, timeout=15)

    def synthetic_server(self, port=18765):
        server = subprocess.Popen([sys.executable, str(self.root / "scripts/qwen38_server.py"),
                                   "--port", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        threading.Thread(target=server.wait, daemon=True).start()

        def cleanup():
            if server.poll() is None:
                server.terminate()
            server.wait(timeout=10)

        self.addCleanup(cleanup)
        return server

    def test_invalid_drafter_restart_preserves_running_server(self):
        server = self.synthetic_server()
        self.pidfile.write_text(f"{server.pid}\n")
        result = self.run_wrapper("restart", DRAFT=str(self.root / "missing.aimodel"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing Core AI DFlash2 drafter", result.stderr)
        self.assertIsNone(server.poll())
        self.assertEqual(self.pidfile.read_text(), f"{server.pid}\n")

    def test_v8_mismatch_restart_preserves_running_server(self):
        server = self.synthetic_server()
        self.pidfile.write_text(f"{server.pid}\n")
        result = self.run_wrapper("restart", KV_CACHE_DTYPE="v8")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("matching --build", result.stderr)
        self.assertIsNone(server.poll())
        self.assertEqual(self.pidfile.read_text(), f"{server.pid}\n")

    def test_missing_bridge_restart_preserves_running_server(self):
        server = self.synthetic_server()
        self.pidfile.write_text(f"{server.pid}\n")
        result = self.run_wrapper("restart", COREAI_BRIDGE_LIB=str(self.root / "missing.dylib"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Build the Swift bridge first", result.stderr)
        self.assertIsNone(server.poll())

    def test_extra_arguments_cannot_override_managed_port_or_context(self):
        server = self.synthetic_server()
        self.pidfile.write_text(f"{server.pid}\n")
        result = self.run_wrapper("restart", EXTRA_ARGS="--port 18766 --ctx 65536")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("EXTRA_ARGS is not supported", result.stderr)
        self.assertIsNone(server.poll())

    def test_stop_leaves_other_port_running(self):
        first, other = self.synthetic_server(), self.synthetic_server(18766)
        self.pidfile.write_text(f"{first.pid}\n")
        result = self.run_wrapper("stop")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIsNotNone(first.poll())
        self.assertIsNone(other.poll())

    def test_stale_pid_does_not_signal_unrelated_process(self):
        self.pidfile.write_text(f"{os.getpid()}\n")
        result = self.run_wrapper("stop")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("no verified server", result.stdout)

    @unittest.skipUnless(sys.platform == "darwin", "Core AI launcher startup requires macOS")
    def test_start_reports_progress_and_manages_child(self):
        self.addCleanup(self.run_wrapper, "stop")
        result = self.run_wrapper("start")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("1/16 (6%)", result.stdout)
        self.assertIn("target chunks loaded: 16/16", result.stdout)
        self.assertIn("serving OpenAI API", result.stdout)


if __name__ == "__main__":
    unittest.main()
