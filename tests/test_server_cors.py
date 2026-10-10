"""Cross-origin access to the server: off by default; CORS_ORIGINS opens only the read-only status endpoints."""
import http.client
import sys
import threading
import unittest
from unittest.mock import Mock
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import qwen38_server as S  # noqa: E402


def serve(origins):
    engine = SimpleNamespace(a=SimpleNamespace(cors_origin=origins, think=False, summary_think=False),
                             ctx=8192, model=SimpleNamespace(pos=0))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(engine))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def request(srv, method, path, headers, body=None):
    c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
    c.request(method, path, body=body, headers=headers)
    r = c.getresponse(); r.read(); c.close()
    return r.status, r.getheader("Access-Control-Allow-Origin"), r.getheader("Access-Control-Allow-Methods")


class CorsTests(unittest.TestCase):
    ORIGIN = {"Origin": "http://localhost:5173"}

    def test_live_decode_cleared_on_success_and_failure(self):
        engine = S.Engine.__new__(S.Engine)
        engine.decode_live = {"active": True}
        engine._generate = Mock(return_value="done")
        self.assertEqual(engine.generate(), "done")
        self.assertIsNone(engine.decode_live)
        engine.decode_live = {"active": True}
        engine._generate = Mock(side_effect=RuntimeError("test failure"))
        with self.assertRaises(RuntimeError):
            engine.generate()
        self.assertIsNone(engine.decode_live)

    def test_off_by_default(self):
        srv = serve("")
        try:
            self.assertEqual(request(srv, "GET", "/health", self.ORIGIN)[1], None)
            self.assertEqual(request(srv, "OPTIONS", "/health", {**self.ORIGIN, "Access-Control-Request-Method": "GET"})[1], None)
        finally:
            srv.shutdown()

    def test_allowed_origin_reads_status_only(self):
        srv = serve("http://localhost:5173")
        try:
            self.assertEqual(request(srv, "GET", "/health", self.ORIGIN)[:2], (200, "http://localhost:5173"))
            for path in ("/health/?t=1", "/v1/models?t=1"):
                self.assertEqual(request(srv, "GET", path, self.ORIGIN)[:2], (200, "http://localhost:5173"))
            self.assertEqual(request(srv, "GET", "/health", {"Origin": "https://evil.example"})[1], None)
            post = {**self.ORIGIN, "Content-Type": "application/json"}
            self.assertEqual(request(srv, "POST", "/v1/chat/completions", post, b"{bad")[1], None)  # never on completions
            pre = request(srv, "OPTIONS", "/v1/chat/completions", {**self.ORIGIN, "Access-Control-Request-Method": "POST"})
            self.assertEqual(pre[:2], (403, None))
            pre = request(srv, "OPTIONS", "/health", {**self.ORIGIN, "Access-Control-Request-Method": "GET"})
            self.assertEqual(pre, (204, "http://localhost:5173", "GET"))
        finally:
            srv.shutdown()


if __name__ == "__main__":
    unittest.main()
