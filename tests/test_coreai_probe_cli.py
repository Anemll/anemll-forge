import contextlib
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("greedy", Path(__file__).resolve().parents[1] / "coreai/qwen38_coreai_greedy.py")
greedy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(greedy)


class ProbeCLITests(unittest.TestCase):
    def run_probe(self, result):
        with patch.object(greedy.sys, "argv", ["probe.py", "probe", "chunk.aimodel"]), \
             patch.object(greedy, "probe", return_value=result), \
             contextlib.redirect_stdout(io.StringIO()):
            return greedy.main()

    def test_gpu_fallback_returns_failure_even_without_runtime_error(self):
        self.assertEqual(self.run_probe(dict(error=None, placement="gpu_regions_present")), 1)

    def test_unknown_placement_and_errors_return_failure(self):
        self.assertEqual(self.run_probe(dict(error=None, placement="unknown")), 1)
        self.assertEqual(self.run_probe(dict(error="load failed", placement="fully_ane")), 1)

    def test_cached_fully_ane_does_not_require_new_compiler_stats(self):
        self.assertEqual(self.run_probe(dict(error=None, placement="fully_ane", modelSize=None)), 0)


if __name__ == "__main__":
    unittest.main()
