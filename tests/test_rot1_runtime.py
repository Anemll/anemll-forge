"""rot1 R0.6 / R0.8 (next/rot1/RUNBOOK.md): converter rotation switches, mixer rotation metadata in the export loader,
build / embedding basis checks and the drafter's fix-B un-rotation. CPU only."""
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "coreai"))

import rot1_fold as F  # noqa: E402
import rot1_runtime as RT  # noqa: E402


def write_basis(d: Path, kind: str, seed: int = 1, name: str = "u15r1h") -> dict:
    d.mkdir(parents=True, exist_ok=True)
    r = F.make_rotation(kind, seed)
    r32 = None if r is None else np.ascontiguousarray(r, np.float32)
    if r32 is not None:
        np.save(d / "R.npy", r32)
    b = {"name": name, "r_kind": kind, "r_seed": seed, "n": F.HIDDEN, "fold_version": F.FOLD_VERSION,
         "fold_norms": True, "r_sha256": F.r_sha256(r32), "probe_seed": F.PROBE_SEED, "fingerprint": {}}
    b["basis_id"] = F.basis_id(b)
    (d / "basis.json").write_text(json.dumps(b))
    return b


class UnrotateTests(unittest.TestCase):
    def test_had20x256_fast_path_inverts_the_fold(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            write_basis(d, "had20x256", seed=7)
            r = np.load(d / "R.npy").astype(np.float64)
            x = np.random.default_rng(1).standard_normal((4, F.HIDDEN))
            xr = x @ r.T  # the folded basis: rows h' = h R^T
            u = RT.Unrotate(d)
            self.assertIsNotNone(u.fast)
            np.testing.assert_allclose(u(xr.astype(np.float32)), x, rtol=0, atol=1e-4)
            taps = np.concatenate([xr[0], xr[1], xr[2], xr[3], xr[0]]).astype(np.float32)  # 5 taps of 5120
            back = u(taps.reshape(-1, F.HIDDEN)).reshape(-1)
            np.testing.assert_allclose(back[:F.HIDDEN], x[0], atol=1e-4)

    def test_dense_path_and_sha_check(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            write_basis(d, "block1024", seed=3, name="u15b")
            r = np.load(d / "R.npy").astype(np.float64)
            x = np.random.default_rng(2).standard_normal((2, F.HIDDEN))
            u = RT.Unrotate(d)
            self.assertIsNone(u.fast)
            np.testing.assert_allclose(u((x @ r.T).astype(np.float32)), x, atol=1e-4)
            np.save(d / "R.npy", np.eye(F.HIDDEN, dtype=np.float32))  # tampered R
            with self.assertRaises(RuntimeError):
                RT.Unrotate(d)

    def test_unrotate_for_plain_and_identity_builds(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            (d / "manifest.json").write_text(json.dumps({"chunks": []}))
            self.assertIsNone(RT.unrotate_for(d))
            b = write_basis(d, "identity", name="u15f")
            (d / "manifest.json").write_text(json.dumps({"chunks": [], "rot1_basis": {k: b[k] for k in (
                "basis_id", "name", "r_kind", "r_seed", "fold_norms")}}))
            self.assertIsNone(RT.unrotate_for(d))  # identity R: residual basis unchanged


class CheckBuildTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.root = Path(self.t.name)
        os.environ.pop("ALLOW_TIMING_ONLY", None)

    def tearDown(self):
        self.t.cleanup()
        os.environ.pop("ALLOW_TIMING_ONLY", None)

    def emb(self, sub, basis=None):
        d = self.root / sub
        d.mkdir(parents=True, exist_ok=True)
        if basis is not None:
            (d / "basis.json").write_text(json.dumps(basis))
        return d / "embed_tokens_fp16.npy"

    def test_basis_pairing(self):
        b = write_basis(self.root / "basis", "had20x256", seed=1)
        rec = {k: b[k] for k in ("basis_id", "name", "r_kind", "r_seed", "fold_norms")}
        plain, rotated = {"chunks": []}, {"chunks": [], "rot1_basis": rec}
        RT.check_build(plain, self.emb("orig"))                       # plain build, plain table
        RT.check_build(rotated, self.emb("rot", b))                   # matching basis
        with self.assertRaises(RuntimeError):
            RT.check_build(plain, self.emb("rot2", b))                # rotated table under a plain build
        with self.assertRaises(RuntimeError):
            RT.check_build(rotated, self.emb("orig2"))                # plain table under a rotated build
        other = write_basis(self.root / "other", "had20x256", seed=2, name="u15r1h_s2")
        with self.assertRaises(RuntimeError):
            RT.check_build(rotated, self.emb("rot3", other))           # another basis

    def test_timing_only_refused(self):
        man = {"chunks": [{"numerics": {"TIMING_ONLY": True}}]}
        with self.assertRaises(RuntimeError):
            RT.check_build(man, self.emb("orig"))
        os.environ["ALLOW_TIMING_ONLY"] = "1"
        RT.check_build(man, self.emb("orig"))


class ConverterSwitchTests(unittest.TestCase):
    """mlp_rotation / mixer_rotation / rot_numerics / rot1_basis_check in coreai/qwen38_coreai_build.py."""

    @classmethod
    def setUpClass(cls):
        # These CPU tests need architecture dimensions, not a downloaded checkpoint.
        config = dict(linear_num_key_heads=16, linear_num_value_heads=48,
                      linear_key_head_dim=128, linear_value_head_dim=128,
                      num_attention_heads=24, num_key_value_heads=4, head_dim=256,
                      hidden_size=5120, intermediate_size=17408, rms_norm_eps=1e-6,
                      rope_parameters={"partial_rotary_factor": 0.25})
        with patch("qwen38_ane_model.cfg", return_value=config):
            import qwen38_coreai_build as B
        cls.B = B

    def setUp(self):
        B = self.B
        self.saved = (B.ROT_IN, B.ROT_MID, B.ROT_ALL_OFF, B.MIX_ROT)
        B._TIMING_ONLY.clear()

    def tearDown(self):
        B = self.B
        B.ROT_IN, B.ROT_MID, B.ROT_ALL_OFF, B.MIX_ROT = self.saved
        B._TIMING_ONLY.clear()

    def test_defaults_follow_the_export(self):
        B = self.B
        W = {"5/mlp.rotation": np.array([1005, 2005]), "6/mlp.rotation": np.array([-1, 2006]),
             "6/mix.rotation": np.array([3006, -1])}
        self.assertEqual(B.mlp_rotation(W, 5), (1005, 2005))
        self.assertEqual(B.mlp_rotation(W, 6), (None, 2006))        # folded-R1 export without rin
        self.assertEqual(B.mixer_rotation(W, 5), (None, None))
        self.assertEqual(B.mixer_rotation(W, 6), (3006, None))
        self.assertEqual(B.rot_numerics([5, 6]), {})                  # default build of these exports: nothing recorded

    def test_switches_mark_timing_only(self):
        B = self.B
        W = {"5/mlp.rotation": np.array([1005, 2005])}
        B.ROT_IN = False
        self.assertEqual(B.mlp_rotation(W, 5), (None, 2005))
        B.MIX_ROT = "readers+writers"
        self.assertEqual(B.mixer_rotation(W, 5), (3005, 4005))      # forced on a plain-mixer export
        n = B.rot_numerics([4, 5, 6, 7])
        self.assertEqual(n["MIX_ROT"], "readers+writers")
        self.assertFalse(n["ROT_IN"])
        self.assertTrue(n["TIMING_ONLY"])
        B._TIMING_ONLY.clear()
        B.ROT_IN, B.MIX_ROT, B.ROT_ALL_OFF = True, "", True
        self.assertEqual((B.mlp_rotation(W, 5), B.mixer_rotation(W, 5)), ((None, None), (None, None)))
        self.assertTrue(B.rot_numerics([5])["TIMING_ONLY"])

    def test_hadamard_module_matches_the_export_rotation(self):
        B = self.B
        n, seed = 2048, 3007
        m = B.Hadamard(n, seed).float()
        x = torch.randn(1, n, 1, 3)
        y = m(x)[0, :, 0, :].T.numpy()  # (T, n) rows
        xr = x[0, :, 0, :].T.numpy()
        h = F.hadamard_matrix(1024) / 32.0
        s = np.random.default_rng(seed).choice([-1.0, 1.0], n)
        want = np.concatenate([(xr[:, b * 1024:(b + 1) * 1024] * s[b * 1024:(b + 1) * 1024]) @ h for b in range(2)], 1)
        np.testing.assert_allclose(y, want, atol=2e-3)

    def test_basis_check(self):
        B = self.B
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            ex, mo, plain = t / "export", t / "model", t / "orig"
            for d in (ex, plain):
                d.mkdir()
            self.assertIsNone(B.rot1_basis_check(plain, plain))
            b = write_basis(mo, "had20x256", seed=1)
            (ex / "basis.json").write_text(json.dumps(b))
            self.assertEqual(B.rot1_basis_check(ex, mo)["basis_id"], b["basis_id"])
            with self.assertRaises(SystemExit):
                B.rot1_basis_check(ex, plain)                          # folded export, original MODEL
            with self.assertRaises(SystemExit):
                B.rot1_basis_check(plain, mo)                          # plain export, rotated MODEL
            (mo / "R.npy").unlink()
            with self.assertRaises(SystemExit):
                B.rot1_basis_check(ex, mo)                             # R.npy needed for fix B


class LoaderTests(unittest.TestCase):
    """qwen38_ane_model.layer_quant reads MLP / mixer rotation seeds and moves the factors' b into each basis."""

    def test_mixer_rotation_metadata(self):
        from safetensors.torch import save_file
        import qwen38_ane_model as M
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            g = torch.Generator().manual_seed(0)
            mlp = {f"{m}.weight": torch.randn(8, 1024, generator=g) for m in ("gate", "up", "down")}
            save_file(mlp, t / "layer_00.safetensors", metadata={"basis": "online", "block": "1024", "seed_mid": "2000"})
            b_qkv = torch.randn(4, 1024, generator=g)
            mix = {"linear_attn.in_proj_qkv.weight": torch.randn(8, 1024, generator=g),
                   "linear_attn.in_proj_qkv.lr_a": torch.randn(8, 4, generator=g),
                   "linear_attn.in_proj_qkv.lr_b": b_qkv,
                   "linear_attn.out_proj.weight": torch.randn(8, 1024, generator=g)}
            save_file(mix, t / "layer_00_mixer.safetensors",
                      metadata={"basis": "online", "block": "1024", "seed_in": "3000", "seed_out": "4000"})
            saved = M.EXPORT_DIR
            M.EXPORT_DIR = t
            try:
                q = M.layer_quant(None, 0, {})
            finally:
                M.EXPORT_DIR = saved
            self.assertEqual(q["mlp.rotation"], (None, 2000))
            self.assertEqual(q["mix.rotation"], (3000, 4000))
            # dense weights carry no factors; the qkv factor b moved to the seed_in basis (b @ M)
            self.assertEqual(q["linear_attn.in_proj_qkv.weight"][0], "dense")
            h = F.hadamard_matrix(1024) / 32.0
            s = np.random.default_rng(3000).choice([-1.0, 1.0], 1024)
            # a dense matrix keeps no factors; check the factor rotation on a LUT matrix
            lr = M.as_quant({"x.lut": torch.zeros(4, 1), "x.idx": torch.zeros(8, 1024, dtype=torch.uint8),
                             "x.lr_a": torch.ones(8, 4), "x.lr_b": b_qkv}, "x", 3000)[-1]
            np.testing.assert_allclose(lr[1], (b_qkv.numpy() * s) @ h, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
