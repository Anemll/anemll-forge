"""Prefill rows per context entry (builder TPS / TPS_LONG): manifest helpers, entry maps, and the runtime's block
sizes when a prompt crosses from a 64-row entry into a 32-row one (no ANE, no models)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import qwen38_coreai_model as runtime
import coreai_compile_guide as G
from qwen38_kv_cache import cache_entries, prefill_groups, prefill_rows

LADDER = [8192, 65536, 98304, 196608]
PTP = {"8192": 64, "65536": 64, "98304": 32, "196608": 32}
KVLEN = {8192: 8192, 65536: 65472, 98304: 98304, 196608: 196608}


class ManifestTests(unittest.TestCase):
    def test_rows_per_context_and_older_manifests(self):
        self.assertEqual(prefill_rows({"TP": 64, "ptp": PTP, "pctxs": LADDER}),
                         {8192: 64, 65536: 64, 98304: 32, 196608: 32})
        self.assertEqual(prefill_rows({"TP": 32, "pctxs": [8192, 16384]}), {8192: 32, 16384: 32})  # before ptp
        self.assertEqual(prefill_rows({"TP": 0, "pctxs": []}), {})
        self.assertEqual(prefill_groups({"TP": 64, "ptp": PTP}), "prefill 64-row 8K-64K, 32-row 96K-192K")
        self.assertEqual(prefill_groups({"TP": 0}), "no prefill entries")

    def test_selectable_entry_map_needs_each_contexts_rows(self):
        man = {"ctxs": [8192, 98304], "pctxs": [8192, 98304], "TP": 64, "ptp": {"8192": 64, "98304": 32},
               "kv_cache": {"format": "selectable"}}
        names = {"v8_8k", "v8_96k", "p64_8k", "p32_96k"}
        chunk = {"file": "c.aimodel", "entries": sorted(names), "entries_by_kv": {"v8": {n: n for n in names}}}
        self.assertEqual(set(cache_entries(man, chunk, "v8")), names)
        del chunk["entries_by_kv"]["v8"]["p32_96k"]
        with self.assertRaisesRegex(ValueError, "Incomplete v8"):
            cache_entries(man, chunk, "v8")

    def test_graph_line_names_the_split(self):
        nums = {"GDN_FAST": True, "ATT_BLOCK": 2048, "ATT_BLOCK_PREFILL": 4096, "ATT_BLOCK_PREFILL_LONG": 8192,
                "PREFILL_LONG_FROM": 65536, "TPS": [64], "TPS_LONG": [32]}
        man = {"TP": 64, "ptp": PTP, "pctxs": LADDER, "chunks": [{"numerics": nums}]}
        g = runtime.target_graph(man)
        self.assertEqual((g["att_block_prefill_long"], g["prefill_long_from"]), (8192, 65536))
        line = runtime.graph_line(man, Path("/b"))
        self.assertIn("ATT_BLOCK_PREFILL=4096 ATT_BLOCK_PREFILL_LONG=8192 above 64K", line)
        self.assertIn("prefill 64-row 8K-64K, 32-row 96K-192K", line)

    def test_compile_estimate_counts_long_tiles(self):
        man = {"kv_len": {str(c): KVLEN[c] for c in LADDER}, "pkv_len": {str(c): KVLEN[c] for c in LADDER},
               "chunks": [{"numerics": {"GDN_FAST": True, "ATT_BLOCK": 2048, "ATT_BLOCK_PREFILL": 4096,
                                        "ATT_BLOCK_PREFILL_LONG": 8192, "PREFILL_LONG_FROM": 65536}}]}
        s = G.build_settings(man)
        narrow = dict(s, att_block_prefill_long=4096)   # every prefill entry with 4096-wide tiles
        self.assertLess(G.estimate_chunk_s(s), G.estimate_chunk_s(narrow))


class FeedTests(unittest.TestCase):
    def model(self, ctx, pos):
        m = runtime.CoreAIQwenBridge.__new__(runtime.CoreAIQwenBridge)
        m.T, m.TP, m.ptp = 8, 64, prefill_rows({"TP": 64, "ptp": PTP})
        m.ladder, m.kvlen, m.pkvlen, m.ctx, m.pos = LADDER, dict(KVLEN), dict(KVLEN), ctx, pos
        m.chunks = [{"fns": {f"p{r}_{c // 1024}k": None for c, r in m.ptp.items()}}]
        m.calls = []

        def resize(c):
            m.ctx = c

        def prefill_block(ids):
            assert m.fit(m.pos + len(ids)) and len(ids) <= m.tp(), (len(ids), m.ctx)
            m.calls.append(("prefill", len(ids), m.ctx))
            m.pos += len(ids)
            return None

        def call(ids):
            m.calls.append(("verify", len(ids), m.ctx))
            return [None] * len(ids)

        def accept(k):
            m.pos += k
        m.resize, m.prefill_block, m.call, m.accept = resize, prefill_block, call, accept
        return m

    def test_block_crossing_into_the_32_row_entry_shrinks(self):
        m = self.model(65536, 65400)                     # 72 rows left in the 64K entry (65472)
        m.feed(list(range(200)))
        self.assertEqual(m.calls, [("prefill", 64, 65536)] + [("prefill", 32, 98304)] * 4 + [("verify", 8, 98304)])
        self.assertEqual(m.pos, 65600)

    def test_short_and_long_entries_keep_their_rows(self):
        m = self.model(8192, 0)
        m.feed(list(range(130)))
        self.assertEqual(m.calls, [("prefill", 64, 8192)] * 2 + [("verify", 2, 8192)])
        m = self.model(196608, 150000)
        m.feed(list(range(70)))
        self.assertEqual(m.calls, [("prefill", 32, 196608)] * 2 + [("verify", 6, 196608)])

    def test_has_prefill_follows_the_entry(self):
        m = self.model(65536, 0)
        self.assertEqual((m.tp(), m.has_prefill()), (64, True))
        m.ctx = 98304
        self.assertEqual((m.tp(), m.has_prefill()), (32, True))
        m.pkvlen[98304] = 98000                           # prefill KV history must match the verify buffers
        self.assertFalse(m.has_prefill())


if __name__ == "__main__":
    unittest.main()
