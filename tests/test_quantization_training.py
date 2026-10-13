"""CPU fixtures for the recovered GPTQ/QAT/export path; no checkpoint or GPU needed."""
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
with patch.dict(os.environ, {"DEVICE": "cpu"}):
    import qwen38_gptq_27b as G
    import qwen38_qat_kl as Q
    import qwen38_kl as K


def lut_tensors(key, cout=4, cin=32, cd=2):
    return {f"{key}.lut": torch.randn(16, cd).half(),
            f"{key}.idx": torch.randint(16, (cout // cd, cin), dtype=torch.uint8),
            f"{key}.scale": torch.linspace(.1, 1, cout).half()}


class TrainingNumerics(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        torch.set_num_threads(1)

    def test_centroid_gradient_matches_native_gather(self):
        with torch.enable_grad():
            lut = torch.randn(16, 2, requires_grad=True)
            idx = torch.randint(16, (7, 11), dtype=torch.uint8)
            grad = torch.randn(7, 11, 2)
            (Q.LUTGather.apply(lut, idx) * grad).sum().backward()
            actual = lut.grad.clone()
            lut.grad = None
            (lut[idx.long()] * grad).sum().backward()
            torch.testing.assert_close(actual, lut.grad)

    def test_optimizer_updates_all_scale_groups_without_changing_codes(self):
        keys = ["gate", "linear_attn.in_proj_qkv", "self_attn.k_proj", "lm_head"]
        files = {}
        for key in keys:
            t = lut_tensors(key)
            if key == "self_attn.k_proj":
                t = {f"{key}.int8": torch.randint(-127, 128, (4, 32), dtype=torch.int8),
                     f"{key}.scale": torch.linspace(.1, 1, 4).half()}
            if key == "linear_attn.in_proj_qkv":
                t.update({f"{key}.lr_a": torch.randn(4, 2).half(),
                          f"{key}.lr_b": torch.randn(2, 32).half()})
            files[key] = [Q.QLinear(t, key).requires_grad_(False)]
        with patch.object(Q, "PARTS", {"lut", "scale", "head", "lr"}), torch.enable_grad():
            groups = Q.params(files)
            self.assertEqual({g["name"]: len(g["params"]) for g in groups},
                             {"lut": 2, "scale": 3, "head": 2, "lr": 2})
            opt = torch.optim.Adam(groups)
            codes = {k: (v[0].idx if v[0].kind == "lut" else v[0].codes).clone()
                     for k, v in files.items()}
            loss = sum(q(torch.randn(3, 32)).square().mean() for [q] in files.values())
            loss.backward()
            for [q] in files.values():
                self.assertTrue(torch.isfinite(q.log_s.grad).all())
                self.assertTrue((q.log_s.grad != 0).all())
            opt.step()
            for key, [q] in files.items():
                self.assertTrue((q.log_s != 0).all())
                self.assertNotIn("idx", dict(q.named_parameters()))
                self.assertNotIn("codes", dict(q.named_parameters()))
                torch.testing.assert_close(codes[key], q.idx if q.kind == "lut" else q.codes)
                self.assertEqual(q.export()[f"{key}.scale"].dtype, torch.float16)

    def test_rotated_qat_projection_matches_evaluator_with_original_basis_factors(self):
        key = "self_attn.q_proj"
        t = lut_tensors(key, cin=1024)
        t.update({f"{key}.lr_a": torch.randn(4, 2).half(),
                  f"{key}.lr_b": torch.randn(2, 1024).half()})
        seed = 3000
        q = Q.QLinear(t, key, Q.rotation_signs(1024, seed, "cpu"))
        w = K.dequant(t, key) @ K.rotation(1024, seed).T
        w = K.exported_lowrank(t, key, w)
        x = torch.randn(3, 1024)
        torch.testing.assert_close(q(x), x @ w.T, rtol=2e-5, atol=2e-4)
        self.assertEqual(Q.mixer_seed({"basis": "online", "seed_in": "3000"}, key), 3000)
        self.assertIsNone(Q.mixer_seed({"basis": "online", "seed_in": "3000"}, "self_attn.o_proj"))
        self.assertEqual(Q.mixer_seed({"basis": "online", "seed_in": "3000", "seed_out": "4000"},
                                     "self_attn.o_proj"), 4000)

    def test_gptq_lowrank_export_matches_propagated_projection(self):
        # Dense orthogonal fixture exercises the same basis contract with small matrices.
        r, _ = torch.linalg.qr(torch.randn(32, 32))
        mod = torch.nn.Linear(32, 32, bias=False)
        xs = [torch.randn(48, 32), torch.randn(48, 32), torch.randn(4, 32)]
        with patch.multiple(G, DEVICE=torch.device("cpu"), EXPORT=True, METHOD="gptq",
                            AW=True, FIXED_CB=None, LR_ITERS=1, KMEANS_SEED=1):
            q, encoding, _, ab = G.quantize_linear(mod, xs, "vector 2x16 + pcs", lr=4, rot=r)
            t = G.pack("projection", encoding, q, ab)
        self.assertEqual(t["projection.lr_a"].shape, (32, 4))
        self.assertEqual(t["projection.lr_b"].shape, (4, 32))
        w = K.dequant(t, "projection") @ r.T
        w = K.exported_lowrank(t, "projection", w)
        torch.testing.assert_close(mod(xs[-1]), xs[-1] @ w.T, rtol=1e-5, atol=1e-6)

    def test_qat_refuses_existing_output_before_model_load(self):
        with tempfile.TemporaryDirectory() as tmp, patch.multiple(Q, OUT_DIR=Path(tmp), EXPORT_DIR=Path(tmp)):
            with patch.object(Q, "build") as build:
                with self.assertRaises(FileExistsError):
                    Q.train()
                build.assert_not_called()


class ExportAndCorpus(unittest.TestCase):
    def test_evaluator_overlay_matches_materialized_weights_and_resets_factor_bytes(self):
        def model():
            m = torch.nn.Module()
            m.model = torch.nn.Module()
            m.model.language_model = torch.nn.Module()
            layer = torch.nn.Module()
            layer.self_attn = torch.nn.Module()
            layer.self_attn.q_proj = torch.nn.Linear(1024, 4, bias=False)
            m.model.language_model.layers = torch.nn.ModuleList([layer])
            m.lm_head = torch.nn.Linear(1024, 4, bias=False)
            return m

        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "overlay", Path(tmp) / "full"
            src.mkdir()
            key, name = "self_attn.q_proj", "layer_00_mixer.safetensors"
            t = lut_tensors(key, cin=1024)
            t.update({f"{key}.lr_a": torch.randn(4, 2).half(),
                      f"{key}.lr_b": torch.randn(2, 1024).half()})
            save_file(t, src / name, metadata={"basis": "online", "seed_in": "3000"})
            scale = t[f"{key}.scale"] * 2
            save_file({f"{name}::{key}.scale": scale}, src / "overlay.qat")
            subprocess.run([sys.executable, str(ROOT / "scripts/qwen38_overlay_materialize.py"),
                            str(src), str(dst)], check=True, capture_output=True, text=True)
            m = model()
            with patch.multiple(K, EXPORT_DIR=src, DEVICE=torch.device("cpu"), LR_RANK=0,
                                QLAYERS=(0, 100), PARTS={"attn"}):
                nbytes = K.apply_export(m)
                factor_bytes = K.LR_BYTES
                expected = m.model.language_model.layers[0].self_attn.q_proj.weight.clone()
                self.assertGreater(factor_bytes, 0)
                K.apply_export(m)
                self.assertEqual(K.LR_BYTES, factor_bytes)
            with patch.multiple(K, EXPORT_DIR=dst, DEVICE=torch.device("cpu"), LR_RANK=0,
                                QLAYERS=(0, 100), PARTS={"attn"}):
                self.assertEqual(K.apply_export(m), nbytes)
                self.assertEqual(K.LR_BYTES, factor_bytes)
                torch.testing.assert_close(m.model.language_model.layers[0].self_attn.q_proj.weight,
                                           expected, rtol=0, atol=0)
            t[f"{key}.scale"] = scale
            q = Q.QLinear(t, key, Q.rotation_signs(1024, 3000, "cpu"))
            reference = K.dequant(t, key) @ K.rotation(1024, 3000).T
            reference = K.exported_lowrank(t, key, reference)
            torch.testing.assert_close(expected, reference.bfloat16(), rtol=0, atol=0)
            x = torch.randn(3, 1024)
            torch.testing.assert_close(q(x), x @ reference.T, rtol=2e-5, atol=2e-4)

    def test_completed_partial_corpus_does_not_repeat_last_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, prompts = Path(tmp) / "corpus.npy", Path(tmp) / "prompts.json"
            prompts.write_text(json.dumps([{"p": "a", "tools": False, "cat": "test"},
                                          {"p": "b", "tools": False, "cat": "test"}]))
            np.savez(out.with_suffix(".partial.npz"), ids=np.arange(8), lengths=[4, 4], plens=[1, 1])
            spec = importlib.util.spec_from_file_location("corpus_fixture", ROOT / "scripts/qwen38_corpus_gen.py")
            mod = importlib.util.module_from_spec(spec)
            with patch.dict(os.environ, {"CAL_OUT": str(out), "CORPUS_PROMPTS": str(prompts), "SEQ": "4"}):
                spec.loader.exec_module(mod)
            tok = SimpleNamespace(eos_token_id=2, convert_tokens_to_ids=lambda s: 1)
            with patch.object(mod.K, "load_model", return_value=(tok, object())):
                mod.main()  # No tokenizer/model call is possible: all sequences are already cached.
            self.assertEqual(json.loads(out.with_suffix(".json").read_text())["sequences"], 2)
            self.assertEqual(np.load(out).shape, (2, 4))

    def test_materialized_overlay_preserves_codes_scales_factors_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "overlay", Path(tmp) / "full"
            src.mkdir()
            key = "self_attn.q_proj"
            base = lut_tensors(key)
            base.update({f"{key}.lr_a": torch.randn(4, 2).half(),
                         f"{key}.lr_b": torch.randn(2, 32).half()})
            name = "layer_00_mixer.safetensors"
            meta = {"basis": "online", "block": "1024", "seed_in": "3000", "seed_out": "4000"}
            save_file(base, src / name, metadata=meta)
            replacement = {k: v + .125 for k, v in base.items() if not k.endswith(".idx")}
            save_file({f"{name}::{k}": v for k, v in replacement.items()}, src / "overlay.qat")
            command = [sys.executable, str(ROOT / "scripts/qwen38_overlay_materialize.py"), str(src), str(dst)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            got = load_file(dst / name)
            for k, v in base.items():
                torch.testing.assert_close(got[k], replacement.get(k, v), rtol=0, atol=0)
            with safe_open(dst / name, framework="pt") as f:
                self.assertEqual(f.metadata(), meta)
            self.assertFalse((dst / name).is_symlink())
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("new directory", result.stderr)

    def test_prompt_builder_candidates_contain_checked_release_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "candidates.json"
            subprocess.run([sys.executable, str(ROOT / "configs/kl/qat_corpus_build_prompts.py"),
                            "--output", str(out)], check=True, capture_output=True, text=True, cwd=tmp)
            candidates = json.loads(out.read_text())
            release = json.loads((ROOT / "configs/kl/qat_corpus_prompts.json").read_text())
            self.assertEqual(len(candidates), 560)
            self.assertEqual(len(release), 527)
            self.assertTrue(all(p in candidates for p in release))

    def test_corpus_metadata_separates_train_and_hold_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "rows.npy"
            rows = np.arange(32).reshape(8, 4)
            np.save(p, rows)
            p.with_suffix(".json").write_text(json.dumps({"n_train": 5}))
            with patch.multiple(Q, TRAIN=f"{p}:meta", WIKI_ROWS=0, HOLD=2, SEQ=4):
                train, hold = Q.rows()
            np.testing.assert_array_equal(train, rows[:5])
            np.testing.assert_array_equal(hold, rows[5:7])


if __name__ == "__main__":
    unittest.main()
