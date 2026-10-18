"""Mapped model files: tables on the home volume are mapped, tables on other volumes are read by rows (NpyRows) or
copied (the selector), so a volume that drops cannot kill the server with SIGBUS (17 Oct 2026)."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mapped_files as MF  # noqa: E402


class MappedFilesTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(dir=Path.home() / ".cache" if (Path.home() / ".cache").is_dir() else None))
        self.table = np.random.default_rng(0).normal(size=(1000, 64)).astype(np.float16)
        self.path = self.dir / "emb.npy"
        np.save(self.path, self.table)

    def test_rows_match_the_table(self):
        rows = MF.NpyRows(self.path)
        self.assertEqual(rows.shape, self.table.shape)
        self.assertEqual(rows.dtype, np.float16)
        self.assertTrue(rows.flags.c_contiguous)
        np.testing.assert_array_equal(rows[7], self.table[7])
        np.testing.assert_array_equal(rows[-1], self.table[-1])
        ids = [5, 999, 0, 5]
        np.testing.assert_array_equal(rows[ids], self.table[ids])
        np.testing.assert_array_equal(rows[np.array(ids)].T, self.table[ids].T)   # as the runtime uses it
        with self.assertRaises(IndexError):
            rows[1000]

    def test_load_rows_maps_only_on_the_home_volume(self):
        with mock.patch.object(MF, "mappable", return_value=True):
            self.assertIsInstance(MF.load_rows(self.path), np.memmap)
        with mock.patch.object(MF, "mappable", return_value=False):
            self.assertIsInstance(MF.load_rows(self.path), MF.NpyRows)

    def test_volume_check(self):
        if Path.home().stat().st_dev == self.dir.stat().st_dev:
            self.assertTrue(MF.mappable(self.path))
        ext = [v for v in Path("/Volumes").iterdir() if v.is_dir() and v.stat().st_dev != Path.home().stat().st_dev] \
            if Path("/Volumes").is_dir() else []
        if ext:
            self.assertFalse(MF.mappable(ext[0]))
            with mock.patch.dict("os.environ", {"ANEMLL_MAP_EXTERNAL": "1"}):
                self.assertTrue(MF.mappable(ext[0]))
        self.assertFalse(MF.mappable(self.dir / "missing.npy"))

    def test_selector_copied_off_the_home_volume(self):
        import torch
        from safetensors.torch import save_file
        import dflash2_ane_drafter as D
        t = {"candidate_selector.predecessor_codebook": torch.randn(16, 8, dtype=torch.bfloat16),
             "candidate_selector.successor_codebook": torch.randn(32, 8, dtype=torch.bfloat16)}
        save_file(t, str(self.dir / "selector.safetensors"))
        for mapped in (True, False):
            with self.subTest(mapped=mapped), mock.patch.object(MF, "mappable", return_value=mapped):
                got = D.load_codebooks(self.dir)
                for k, v in t.items():
                    self.assertTrue(torch.equal(got[k], v))


if __name__ == "__main__":
    unittest.main()
