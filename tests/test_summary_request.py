"""Context-summary requests (Pi's compaction) are recognized by their system prompt (qwen38_server --summary-no-think)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    from qwen38_server import is_summary_request
except Exception as exc:  # pragma: no cover - environment without the server's dependencies
    is_summary_request, IMPORT_ERROR = None, exc


@unittest.skipIf(is_summary_request is None, "qwen38_server dependencies not importable")
class SummaryRequestTests(unittest.TestCase):
    def test_pi_summarizer_prompt(self):
        pi = "You are a context summarization assistant. Your task is to read a conversation between a user and an AI"
        self.assertTrue(is_summary_request([{"role": "system", "content": pi}, {"role": "user", "content": "x"}]))
        self.assertTrue(is_summary_request([{"role": "developer", "content": [{"type": "text", "text": pi}]}]))

    def test_other_requests(self):
        self.assertFalse(is_summary_request([{"role": "system", "content": "You are Pi, a coding agent."}]))
        self.assertFalse(is_summary_request([{"role": "user", "content": "You are a context summarization assistant"}]))
        self.assertFalse(is_summary_request([]))
        self.assertFalse(is_summary_request(None))


if __name__ == "__main__":
    unittest.main()
