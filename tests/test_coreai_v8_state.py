"""Exercise production accept/growth/restore using host buffer doubles."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import qwen38_coreai_model as runtime
from qwen38_kv_cache import quantize_values


class Buffer:
    def __init__(self, shape, dtype):
        self.shape, self.dtype = shape, np.dtype(dtype)
        self.np = np.zeros(shape, dtype)


class Function:
    def __init__(self, capacity, wrong_dtype=False):
        self.capacity, self.wrong_dtype = capacity, wrong_dtype
    def buffer(self, kind, name):
        shape = (1, self.capacity) if name == "mask" else (4, self.capacity) if name.startswith("vs") else (4, self.capacity, 256)
        dtype = np.int8 if name.startswith("v") and not name.startswith("vs") and not self.wrong_dtype else np.float16
        return Buffer(shape, dtype)


class StateTests(unittest.TestCase):
    def model(self):
        m = runtime.CoreAIQwenBridge.__new__(runtime.CoreAIQwenBridge)
        m.kv_cache_dtype, m.nkv, m.hd = "v8", 4, 256
        m.kvlen, m.ladder = {2048: 128, 8192: 512}, [2048, 8192]
        m.pos = m.pending = m.hi = 0
        m.stats, m.log, m._plans = {"resize": []}, lambda _: None, {}
        rng = np.random.default_rng(42)
        output = {n: rng.normal(size=(4, 8, 256)).astype(np.float16) for n in ("k3_new", "v3_new")}
        m.chunks = [{"att_j": [3], "fns": {"v8_2k": Function(128), "v8_8k": Function(512)},
                     "ow": {"v8_2k": output}, "curw": {"rec0": np.zeros((2, 2), np.float16)}}]
        m._last, m._n = "v8_2k", 8
        m._alloc_kv(2048, 0)
        return m, output

    def test_rejected_speculative_rows_are_never_written(self):
        for count in (0, 1, 7, 8):
            with self.subTest(count=count):
                m, output = self.model()
                m.accept(count)
                np.testing.assert_array_equal(m.kv[0]["k3"][1][:, :count], output["k3_new"][:, :count])
                self.assertTrue(np.all(m.kv[0]["k3"][1][:, count:] == 0))
                self.assertTrue(np.all(m.kv[0]["v3"][1][:, count:] == 0))
                self.assertTrue(np.all(m.kv[0]["vs3"][1][:, count:] == 1))
                self.assertEqual((m.pos, m.hi, m.pending), (count, count, count))

    def test_growth_and_snapshot_restore_copy_codes_scales_and_keys(self):
        m, output = self.model()
        m.accept(7)
        m.chunks[0]["curw"]["rec0"][:] = 3
        snap = m.snapshot()
        expected = {n: v[1][:, :7].copy() for n, v in m.kv[0].items()}
        codes, scales = quantize_values(output["v3_new"][:, :7])
        np.testing.assert_array_equal(expected["v3"], codes)
        np.testing.assert_array_equal(expected["vs3"], scales)
        m._plans = {"old context": object()}
        m.resize(8192)
        self.assertEqual(m._plans, {})
        for n, rows in expected.items(): np.testing.assert_array_equal(m.kv[0][n][1][:, :7], rows)
        m.chunks[0]["curw"]["rec0"][:] = 9
        m.pos, m.pending = 8, 1
        m.restore(snap)
        self.assertEqual((m.ctx, m.pos, m.pending), (2048, 7, 7))
        self.assertTrue(np.all(m.chunks[0]["curw"]["rec0"] == 3))
        for n, rows in expected.items(): np.testing.assert_array_equal(m.kv[0][n][1][:, :7], rows)
        self.assertEqual(m.kv[0]["v3"][0].dtype, np.dtype(np.int8))

    def test_manifest_dtype_mismatch_is_a_hard_error(self):
        m, _ = self.model()
        m.chunks[0]["fns"]["v8_2k"] = Function(128, wrong_dtype=True)
        with self.assertRaisesRegex(ValueError, "metadata/layout mismatch"):
            m._alloc_kv(2048, 0)

    def test_feed_falls_back_after_growth_without_batched_prefill(self):
        m = runtime.CoreAIQwen.__new__(runtime.CoreAIQwen)
        m.ctx, m.pos, m.T, m.TP = 24576, 24560, 8, 64
        m.pkvlen = {24576: 24576}
        m.chunks = [{"fns": {"p64_24k": object(), "v8_31k": object()}}]
        m.cap = lambda ctx: ctx
        calls = []
        def fit(need):
            if need > m.ctx:
                m.ctx = 31744
            return need <= m.ctx
        def call(ids):
            calls.append(len(ids))
            return np.zeros((len(ids), 2))
        def accept(n):
            m.pos += n
        m.fit, m.call, m.accept = fit, call, accept
        m.prefill_block = lambda ids: self.fail("No p64 entry exists after growth")
        m.feed(list(range(96)))
        self.assertEqual((m.ctx, m.pos), (31744, 24656))
        self.assertEqual(calls, [8] * 12)


if __name__ == "__main__": unittest.main()
