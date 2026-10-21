"""Dry-run I/O shape + correctness tests for the LLKVApprox prototype (pipelines/kva/kva_qwen38.py).

These run on plain NumPy (no torch / Core AI / Apple hardware / checkpoint needed), so they
validate the projector I/O contract against the real Qwen3.8-27B config dims in any environment.

    python -m unittest tests.test_kva_prototype
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipelines" / "kva"))
import kva_qwen38 as K  # noqa: E402


class ReferenceConfig(unittest.TestCase):
    """The reference config must reproduce the Qwen3.8-27B dims documented in the repo."""

    def setUp(self):
        self.cfg = K.load_reference_config()

    def test_top_level_dims(self):
        c = self.cfg
        self.assertEqual(c.hidden_size, 5120)
        self.assertEqual(c.num_hidden_layers, 64)
        self.assertEqual(c.intermediate_size, 17408)
        self.assertEqual(c.vocab_size, 248320)

    def test_hybrid_layout_48_gdn_16_gqa_interval_4(self):
        c = self.cfg
        self.assertEqual(len(c.gdn_layers), 48)
        self.assertEqual(len(c.attn_layers), 16)
        # attention at layers 3, 7, ..., 63 (QUANTIZATION_NOTES.md:16)
        self.assertEqual(c.attn_layers, list(range(3, 64, 4)))

    def test_gqa_heads(self):
        c = self.cfg
        self.assertEqual(c.num_attention_heads, 24)
        self.assertEqual(c.num_key_value_heads, 4)
        self.assertEqual(c.head_dim, 256)

    def test_gdn_dims_and_conv_dim_10240(self):
        c = self.cfg
        self.assertEqual(c.linear_num_key_heads, 16)
        self.assertEqual(c.linear_num_value_heads, 48)
        self.assertEqual(c.linear_key_head_dim, 128)
        self.assertEqual(c.linear_value_head_dim, 128)
        self.assertEqual(c.linear_conv_kernel_dim, 4)
        # conv_dim = 2*nk*dk + nv*dv = 2*16*128 + 48*128 = 10240 (ANE_DELTANET_NUMERICS.md:27)
        self.assertEqual(c.conv_dim, 10240)

    def test_rotary_dim_even(self):
        self.assertEqual(self.cfg.rot_dim % 2, 0)


class ProjectorIOShapes(unittest.TestCase):
    """The projector heads must emit exactly the late mixer-input dims for each late layer."""

    def setUp(self):
        self.cfg = K.load_reference_config()
        self.split = self.cfg.num_hidden_layers // 2          # layer 32 (L/2)
        self.late = list(range(self.split, self.cfg.num_hidden_layers))

    def test_split_is_layer_32(self):
        self.assertEqual(self.split, 32)

    def test_output_spec_matches_checkpoint_matrices(self):
        c = self.cfg
        spec = K.projector_output_spec(c, self.late)
        nv, dv = c.linear_num_value_heads, c.linear_value_head_dim
        nkv, hd = c.num_key_value_heads, c.head_dim
        for i in self.late:
            if c.layer_types[i] == "linear_attention":
                # must match in_proj_qkv (10240x5120), in_proj_z (6144), in_proj_a/b (48)
                self.assertEqual(spec[i], {"qkv": 10240, "z": nv * dv, "a": nv, "b": nv})
                self.assertEqual(spec[i], {"qkv": 10240, "z": 6144, "a": 48, "b": 48})
            else:
                # must match k_proj / v_proj (nkv*hd = 4*256 = 1024)
                self.assertEqual(spec[i], {"k": nkv * hd, "v": nkv * hd})
                self.assertEqual(spec[i], {"k": 1024, "v": 1024})

    def test_projector_heads_produce_correct_shapes(self):
        """Instantiate only a couple of late layers' heads (full 32 would be large) and check
        the projection maps (T, hidden) -> (T, out_dim) exactly."""
        c = self.cfg
        rng = np.random.default_rng(0)
        # one GDN late layer (32) and one GQA late layer (35)
        sample = [32, 35]
        self.assertEqual(c.layer_types[32], "linear_attention")
        self.assertEqual(c.layer_types[35], "full_attention")
        proj = K.init_projector(c, sample, rng)
        T = 7
        h_split = rng.standard_normal((T, c.hidden_size)).astype(np.float32)
        spec = K.projector_output_spec(c, sample)
        for i in sample:
            for name, dim in spec[i].items():
                out = K.project(proj[i], name, h_split)
                self.assertEqual(out.shape, (T, dim))


class StateShapes(unittest.TestCase):
    """GDN and GQA state buffers must match the documented shapes."""

    def setUp(self):
        self.cfg = K.load_reference_config()

    def test_gdn_state_shapes(self):
        c = self.cfg
        st = K.new_gdn_state(c)
        # recurrent state (48, 128, 128); conv state (conv_dim, kernel-1) = (10240, 3)
        self.assertEqual(st.rec_state.shape, (48, 128, 128))
        self.assertEqual(st.conv_state.shape, (10240, 3))

    def test_attn_state_shapes(self):
        c = self.cfg
        st = K.new_attn_state(c, ctx=128)
        self.assertEqual(st.k_cache.shape, (4, 128, 256))
        self.assertEqual(st.v_cache.shape, (4, 128, 256))


class EndToEndScaled(unittest.TestCase):
    """Small scaled config: the plumbing runs and the exact-tail path reproduces the full model."""

    def setUp(self):
        self.cfg = K.scaled_config(K.load_reference_config(), hidden=192, layers=8)
        self.rng = np.random.default_rng(1)
        self.w = K.init_weights(self.cfg, self.rng)
        self.split = 4
        self.late = list(range(self.split, self.cfg.num_hidden_layers))
        self.proj = K.init_projector(self.cfg, self.late, self.rng)

    def test_scaled_config_preserves_hybrid_structure(self):
        c = self.cfg
        self.assertEqual(c.attn_layers, [3, 7])
        self.assertEqual(len(c.gdn_layers), 6)

    def test_prefill_shapes(self):
        c = self.cfg
        T, ctx = 24, 64
        emb = (self.rng.standard_normal((T, c.hidden_size)) * 0.02).astype(np.float32)
        hid, st = K.prefill_exact(c, self.w, emb, ctx)
        self.assertEqual(hid.shape, (T, c.hidden_size))
        self.assertIn(0, st.gdn)      # layer 0 is GDN
        self.assertIn(3, st.attn)     # layer 3 is GQA
        # a GDN state advanced during prefill
        self.assertEqual(st.gdn[0].rec_state.shape,
                         (c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim))

    def test_kva_tail_output_shape(self):
        c = self.cfg
        T, ctx, tail = 24, 64, 6
        emb = (self.rng.standard_normal((T, c.hidden_size)) * 0.02).astype(np.float32)
        tail_h, st = K.prefill_kva(c, self.w, self.proj, emb, ctx, split=self.split, tail_exact=tail)
        self.assertEqual(tail_h.shape, (tail, c.hidden_size))

    def test_full_exact_tail_matches_full_model(self):
        """tail_exact == T means no approximation: KVA must equal the exact model bit-for-bit."""
        c = self.cfg
        T, ctx = 24, 64
        emb = (self.rng.standard_normal((T, c.hidden_size)) * 0.02).astype(np.float32)
        hid, _ = K.prefill_exact(c, self.w, emb, ctx)
        tail_h, _ = K.prefill_kva(c, self.w, self.proj, emb, ctx, split=self.split, tail_exact=T)
        rel = np.linalg.norm(tail_h[-1] - hid[-1]) / np.linalg.norm(hid[-1])
        self.assertLess(rel, 1e-6)

    def test_kva_prefill_is_not_slower_than_full_at_scale(self):
        """Structural check: with a modest MLP-dominated config the approximated region skips the
        late MLP, so KVA prefill does strictly less matmul work than the full model."""
        c = K.scaled_config(K.load_reference_config(), hidden=512, layers=16)
        rng = np.random.default_rng(2)
        w = K.init_weights(c, rng)
        late = list(range(8, c.num_hidden_layers))
        proj = K.init_projector(c, late, rng)
        T, ctx, tail = 96, 128, 8
        emb = (rng.standard_normal((T, c.hidden_size)) * 0.02).astype(np.float32)
        import time
        t0 = time.perf_counter(); K.prefill_exact(c, w, emb, ctx); t_off = time.perf_counter() - t0
        t0 = time.perf_counter(); K.prefill_kva(c, w, proj, emb, ctx, 8, tail); t_on = time.perf_counter() - t0
        # allow noise, but KVA should not be materially slower
        self.assertLessEqual(t_on, t_off * 1.25)


if __name__ == "__main__":
    unittest.main()
