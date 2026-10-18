"""forge.py never leaves its child running alone: SIGTERM / SIGHUP are passed on (call_child), and a child that
watches its parent (coreai_compile_guide.watch_parent) stops when forge.py dies by kill -9. Before, `kill <forge.py>`
left a compile running on its own, still compiling and writing compile caches (17 Oct 2026)."""
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
# a stand-in for coreai_compile.py / the server: watch the parent if asked, write the pid, then wait
CHILD = ("import os, sys, time; sys.path.insert(0, {scripts!r}); import coreai_compile_guide as G; "
         "w = G.watch_parent(0.1) if {watch} else False; "
         "open({pidf!r}, 'w').write(f'{{os.getpid()}} {{int(w)}} {{os.environ.get(G.PARENT_ENV, \"\")}}'); time.sleep(60)")


def gone(pid: int, timeout: float = 5.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def read_pid(pidf: Path, timeout: float = 30.0) -> list[str]:
    end = time.time() + timeout
    while time.time() < end:
        if pidf.exists() and pidf.read_text().strip():
            return pidf.read_text().split(" ")
        time.sleep(0.05)
    raise AssertionError("child did not start")


class ForgeChildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.pidf = self.tmp / "child.pid"

    def parent(self, watch: bool) -> subprocess.Popen:
        child = CHILD.format(scripts=str(ROOT / "scripts"), watch=watch, pidf=str(self.pidf))
        code = (f"import os, sys; sys.path.insert(0, {str(ROOT)!r}); import forge; "
                f"sys.exit(forge.call_child([{PY!r}, '-c', {child!r}], dict(os.environ)) & 0xff)")
        return subprocess.Popen([PY, "-c", code], cwd=ROOT, start_new_session=True)

    def test_term_and_hup_are_passed_on(self):
        for sig in (signal.SIGTERM, signal.SIGHUP):
            with self.subTest(sig=sig.name):
                self.pidf.unlink(missing_ok=True)
                p = self.parent(watch=False)
                pid = int(read_pid(self.pidf)[0])
                p.send_signal(sig)
                p.wait(10)
                self.assertTrue(gone(pid), f"child still running after {sig.name} to forge.py")

    def test_child_stops_after_kill_9(self):
        p = self.parent(watch=True)
        pid, watching, inherited = (read_pid(self.pidf) + [""])[:3]
        self.assertEqual(watching, "1")
        self.assertEqual(inherited, "")   # removed, so the child's own processes do not watch
        p.kill()
        p.wait(10)
        self.assertTrue(gone(int(pid)), "child still running after kill -9 of forge.py")

    def test_without_forge_nothing_is_watched(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import coreai_compile_guide as G
        os.environ[G.PARENT_ENV] = str(os.getpid())   # not our parent's pid: e.g. inherited by a grandchild
        self.assertFalse(G.watch_parent())
        self.assertNotIn(G.PARENT_ENV, os.environ)
        self.assertFalse(G.watch_parent())          # unset: run directly, not by forge.py


if __name__ == "__main__":
    unittest.main()
