"""Process ownership checks with synthetic argv and process trees; no models."""
import os
from pathlib import Path
import signal
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import qwen38_server_process as process


class ServerProcessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "checkout with spaces"
        self.root.mkdir()
        self.pidfile = self.root / "server.pid"
        self.launcher = ["python", "-u", str(self.root / "forge.py"), "serve", "--port", "8765"]
        self.server = ["python", str(self.root / "scripts/qwen38_server.py"), "--port", "8765"]

    def test_kernel_argv_excludes_environment_and_preserves_spaces(self):
        raw = struct.pack("i", len(self.launcher)) + b"/python\0\0\0"
        raw += b"\0".join(arg.encode() for arg in self.launcher) + b"\0ENV=synthetic\0"
        self.assertEqual(process.darwin_argv(raw), self.launcher)

    def test_matches_script_and_effective_port(self):
        self.assertTrue(process.matches(self.launcher, self.root, 8765))
        self.assertTrue(process.matches(self.server, self.root, 8765))
        self.assertFalse(process.matches(self.server, self.root, 8766))
        self.assertFalse(process.matches(self.launcher + ["--port", "8766"], self.root, 8765))
        self.assertFalse(process.matches(["python", "-c", str(self.root / "forge.py"), "serve", "--port", "8765"], self.root, 8765))
        self.assertFalse(process.matches(["vim", *self.launcher[2:]], self.root, 8765))

    def test_only_recorded_instance_and_matching_children_are_selected(self):
        self.pidfile.write_text("101\n")
        argv = {101: self.launcher, 102: self.server, 103: ["python", "helper.py"],
                201: self.launcher, 202: self.server}
        rows = "101 1\n102 101\n103 102\n201 1\n202 201\n"
        with patch.object(process, "process_argv", side_effect=lambda pid: argv.get(pid, [])), \
                patch.object(process.subprocess, "check_output", return_value=rows):
            self.assertEqual(process.managed_pids(self.root, 8765, self.pidfile), [101, 102])

    def test_stale_pid_is_not_signalled(self):
        self.pidfile.write_text("101\n")
        with patch.object(process, "process_argv", return_value=["python", "unrelated.py"]), \
                patch.object(process.os, "kill") as kill:
            self.assertEqual(process.stop(self.root, 8765, self.pidfile), 0)
        kill.assert_not_called()

    def test_stop_signals_only_verified_launcher_and_server(self):
        self.pidfile.write_text("101\n")
        argv = {101: self.launcher, 102: self.server, 201: self.launcher}
        rows = "101 1\n102 101\n201 1\n"
        sent = []

        def kill(pid, sig):
            sent.append((pid, sig))
            argv.pop(pid, None)

        with patch.object(process, "process_argv", side_effect=lambda pid: argv.get(pid, [])), \
                patch.object(process.subprocess, "check_output", return_value=rows), \
                patch.object(process.os, "kill", side_effect=kill):
            self.assertEqual(process.stop(self.root, 8765, self.pidfile), 0)
        self.assertEqual(sent, [(102, signal.SIGTERM), (101, signal.SIGTERM)])
        self.assertIn(201, argv)
        self.assertFalse(self.pidfile.exists())

    def test_actual_python_process_argv_can_be_read(self):
        argv = process.process_argv(os.getpid())
        self.assertTrue(argv)
        self.assertFalse(process.matches(argv, self.root, 8765))


if __name__ == "__main__":
    unittest.main()
