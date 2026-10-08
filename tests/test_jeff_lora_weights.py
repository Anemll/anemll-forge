"""LoRA factor algebra: streamed (x @ A) @ sB matches the merged weight delta."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coreai"))
from jeff_lora_weights import (bytes_for, factor_shapes, graph_key, merged_matrix, project_factors,
                               weight_delta)


def test_graph_key():
    assert graph_key("language_model.layers.3.self_attn.q_proj") == "3/self_attn.q_proj.weight"
    assert graph_key("language_model.layers.0.linear_attn.in_proj_qkv") == "0/linear_attn.in_proj_qkv.weight"


def test_stream_matches_merged_weight():
    rng = np.random.default_rng(0)
    in_f, out_f, rank = 8, 5, 2
    weight = rng.normal(size=(out_f, in_f)).astype(np.float32)
    a_mm = rng.normal(size=(in_f, rank)).astype(np.float32)   # A
    sb = rng.normal(size=(rank, out_f)).astype(np.float32)    # sB
    x = rng.normal(size=(4, in_f)).astype(np.float32)
    delta = weight_delta(a_mm, sb)
    np.testing.assert_allclose(delta, sb.T @ a_mm.T, atol=1e-6)
    y_stream = x @ weight.T + (x @ a_mm) @ sb
    y_merged = x @ (weight + delta).T
    np.testing.assert_allclose(y_stream, y_merged, atol=1e-5)
    stored = merged_matrix(weight, a_mm, sb)
    np.testing.assert_array_equal(stored, (weight + delta).astype(np.float16))


def test_project_layouts_and_rank_pad():
    a_mm = np.arange(4 * 2, dtype=np.float32).reshape(4, 2)
    sb = np.arange(2 * 3, dtype=np.float32).reshape(2, 3)
    conv_a, conv_b = project_factors(a_mm, sb, "conv", 2)
    assert conv_a.shape == (2, 4, 1, 1)
    assert conv_b.shape == (3, 2, 1, 1)
    np.testing.assert_array_equal(conv_a.reshape(2, 4), a_mm.T.astype(np.float16))
    mm_a, mm_b = project_factors(a_mm, sb, "matmul", 4)
    assert mm_a.shape == (4, 4) and mm_b.shape == (4, 3)
    assert np.all(mm_a[:, 2:] == 0) and np.all(mm_b[2:] == 0)
    np.testing.assert_array_equal(mm_a[:, :2], a_mm.astype(np.float16))
    n_a, n_b = project_factors(a_mm, sb, "nchw", 2)
    assert n_a.shape == (1, 4, 1, 2) and n_b.shape == (1, 2, 1, 3)
    assert factor_shapes(1024, 3584, 16, "conv") == ((16, 1024, 1, 1), (3584, 16, 1, 1))


def test_bytes_count_fp16():
    factors = {"0/mlp.gate_proj.weight": (np.zeros((4, 2), np.float32), np.zeros((2, 6), np.float32))}
    assert bytes_for(factors) == (4 * 2 + 2 * 6) * 2
    assert bytes_for(factors, [1]) == 0
