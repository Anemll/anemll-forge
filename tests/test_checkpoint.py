"""Small embedding-only inference regression tests; run in the compatible ML environment."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import qwen38_ane_model as model
        self.np, self.model = np, model
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.embedding = self.root / "embed_tokens_fp16.npy"
        (self.root / "config.json").write_text(json.dumps({"text_config": {
            "vocab_size": 3, "hidden_size": 5120}}))

    def load(self):
        with patch.object(self.model, "MODEL", self.root), patch.dict(os.environ, EMBED_NPY=str(self.embedding)):
            ck = self.model.Checkpoint()
            with patch.object(ck, "get", side_effect=AssertionError("Original tensors must not be opened")):
                table = ck.embed_table()
            self.assertIsNone(ck._wmap)
            return table

    def test_prepared_embedding_needs_neither_shards_nor_index(self):
        original = self.np.arange(3 * 5120).reshape(3, 5120).astype(self.np.float16)
        self.np.save(self.embedding, original)
        loaded = self.load()
        self.assertIsInstance(loaded, self.np.memmap)
        self.assertFalse(loaded.flags.writeable)
        self.np.testing.assert_array_equal(loaded, original)
        self.assertFalse((self.root / "model.safetensors.index.json").exists())

    def test_wrong_dtype_shape_or_storage_order_rejected(self):
        for array in (self.np.zeros((3, 5120), dtype=self.np.float32),
                      self.np.zeros((2, 5120), dtype=self.np.float16),
                      self.np.asfortranarray(self.np.zeros((3, 5120), dtype=self.np.float16))):
            with self.subTest(dtype=array.dtype, shape=array.shape, contiguous=array.flags.c_contiguous):
                self.np.save(self.embedding, array)
                with self.assertRaisesRegex(ValueError, "C-contiguous float16"):
                    self.load()


if __name__ == "__main__":
    unittest.main()
