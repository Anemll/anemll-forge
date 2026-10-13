"""CPU fixtures for the recovered GPTQ/QAT/export path; no checkpoint or GPU needed."""
import json
import importlib.util
import io
import os
from contextlib import redirect_stdout
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

    def test_fp32_teacher_cache_preserves_identical_distribution_kl(self):
        logits = torch.randn(32, 1024) * 4
        lp, ids = logits.log_softmax(-1).topk(256, -1)
        with patch.object(Q, "TEACHER_DTYPE", "fp32"):
            cached = Q.cache_log_probs(lp)
        self.assertEqual(cached.dtype, np.float32)
        torch.testing.assert_close(torch.from_numpy(cached), lp, rtol=0, atol=0)
        loss = Q.kl_topk(logits, ids, torch.from_numpy(cached))
        torch.testing.assert_close(loss, torch.zeros_like(loss), rtol=0, atol=1e-7)
        with patch.object(Q, "TEACHER_DTYPE", "fp16"):
            historical = Q.cache_log_probs(lp)
        np.testing.assert_array_equal(historical, lp.half().numpy())

    def test_teacher_writes_precision_metadata_and_matching_arrays(self):
        rows = np.arange(8).reshape(2, 4)
        model = lambda input_ids, **kw: SimpleNamespace(logits=torch.arange(12).float().repeat(len(input_ids), 4, 1))
        for precision, dtype in (("fp16", np.float16), ("fp32", np.float32)):
            with self.subTest(precision=precision), tempfile.TemporaryDirectory() as tmp:
                with patch.multiple(Q, TEACHER=Path(tmp), TEACHER_DTYPE=precision, TOPK=3, BS=1,
                                    DEVICE=torch.device("cpu")), patch.object(Q, "rows", return_value=(rows, rows)), \
                        patch.object(Q, "load_teacher_model", return_value=model):
                    Q.teacher()
                metadata = json.loads((Path(tmp) / "teacher.json").read_text())
                self.assertEqual(metadata["log_probs_dtype"], precision)
                for name in ("train", "hold"):
                    with np.load(Path(tmp) / f"{name}.npz") as cache:
                        self.assertEqual(cache["lp"].dtype, dtype)
                        self.assertEqual(cache["lp"].shape, (2, 3, 3))

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

    def test_qat_rejects_tail_and_long_rows_before_model_load(self):
        with tempfile.TemporaryDirectory() as tmp, patch.multiple(
                Q, OUT_DIR=Path(tmp) / "out", EXPORT_DIR=Path(tmp), LONG_EVERY=2,
                TAIL=Q.parse_tail("topfrac:0.05,w:4")), patch.object(Q, "build") as build:
            with self.assertRaisesRegex(ValueError, "LOSS_TAIL and LONG_EVERY"):
                Q.train()
            build.assert_not_called()

    def test_memory_logging_uses_only_selected_cuda_device(self):
        with patch.object(torch.cuda, "max_memory_allocated", return_value=2**30) as peak:
            for device in ("cpu", "mps"):
                with patch.object(Q, "DEVICE", torch.device(device)):
                    self.assertEqual(Q.memory_log_suffix(), "")
            peak.assert_not_called()
            with patch.object(Q, "DEVICE", torch.device("cuda:1")):
                self.assertIn("1.0 GiB", Q.memory_log_suffix())
            peak.assert_called_once_with(torch.device("cuda:1"))

    def test_cpu_training_steps_export_and_log_distinct_selection_metric(self):
        class TinyStudent(torch.nn.Module):
            def __init__(self, tensors):
                super().__init__()
                self.q = Q.QLinear(tensors, "gate")

            def forward(self, input_ids, **kwargs):
                x = torch.nn.functional.one_hot(input_ids, num_classes=4).float()
                return SimpleNamespace(logits=self.q(x))

        # Exercise both training branches, real KL/backward/Adam and final overlay export.
        for long_every in (0, 1):
            with self.subTest(long_every=long_every), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source, teacher, output = root / "source", root / "teacher", root / "output"
                source.mkdir(); teacher.mkdir()
                tensors = lut_tensors("gate", cin=4)
                name = "layer_00.safetensors"
                save_file(tensors, source / name)
                model = TinyStudent(tensors)
                initial = model.q.lut.detach().clone()
                rows = np.array([[0, 1, 2, 3], [3, 2, 1, 0]])
                lp, ids = torch.tensor([2., 1., 0., -1.]).log_softmax(-1).topk(2)
                cache = {"rows": rows, "ids": ids.repeat(2, 3, 1).numpy(),
                         "lp": lp.repeat(2, 3, 1).numpy()}
                for split in ("train", "hold"):
                    np.savez(teacher / f"{split}.npz", **cache)
                long_path = root / "long_teacher.npz"
                np.savez(long_path, ids=ids.repeat(5, 1).numpy(), lp=lp.repeat(5, 1).numpy())

                def long_sum(student, tokens, head, teacher_ids, teacher_lp):
                    logits = student(torch.from_numpy(tokens)[None]).logits[0, :-1]
                    losses = Q.kl_topk(logits, teacher_ids, teacher_lp)
                    return losses[head - 1:].sum(), len(tokens) - head

                console = io.StringIO()
                with patch.multiple(Q, DEVICE=torch.device("cpu"), OUT_DIR=output, EXPORT_DIR=source,
                                    TEACHER=teacher, BS=2, STEPS=2, EVAL_EVERY=2, WARMUP=1,
                                    PARTS={"lut", "scale"}, MODEL_MASTERS={}, TAIL={}, DEV_TRACE=None,
                                    SEL_TRACE=root / "selection", LONG_EVERY=long_every, LONG_SMOKE=False,
                                    LONG=root / "long.npz", LONG_TEACHER=long_path), \
                        patch.object(Q, "build", return_value=(model, {name: [model.q]})), \
                        patch.object(Q, "trace_kl", side_effect=[np.array([8.]), np.array([7.])]), \
                        patch.object(Q, "long_rows", return_value=([np.array([0, 1, 2, 3, 0, 1])], [1])), \
                        patch.object(Q, "long_kl_sum", side_effect=long_sum), \
                        patch.object(torch.cuda, "max_memory_allocated", side_effect=AssertionError("CPU run")), \
                        patch.dict(os.environ, {"MATERIALIZE": "0"}), redirect_stdout(console), torch.enable_grad():
                    expected_hold = Q.evaluate(model, cache)
                    Q.train()
                record = json.loads((output / "qat_kl.json").read_text())
                self.assertEqual(record["best_step"], 2)
                self.assertEqual(record["best_crit"], 7.)
                self.assertEqual(record["log"][0]["hold_kl"], expected_hold)
                self.assertIn(f"step 0 hold-out KL {expected_hold:.5f}", console.getvalue())
                self.assertIn("step 0 KL 8.00000", console.getvalue())
                self.assertNotIn("CUDA mem", console.getvalue())
                self.assertEqual(record["long_tokens_seen"], 10 if long_every else 0)
                self.assertFalse(torch.equal(initial, model.q.lut))
                self.assertIn(f"{name}::gate.lut", load_file(output / "overlay.qat"))

    def test_pre_gain_capture_records_only_one_forward_and_removes_hooks(self):
        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.input_layernorm = torch.nn.LayerNorm(4)
                self.post_attention_layernorm = torch.nn.LayerNorm(4)

            def forward(self, x):
                self.input_layernorm(x)
                return self.post_attention_layernorm(x * torch.tensor([1., 2., 3., 4.]))

        layer = Layer()
        hs = [torch.randn(2, 3, 4), torch.randn(1, 3, 4)]
        forward = lambda text, i, layer, batches: [layer(x) for x in batches]
        with patch.object(G, "run_layer", side_effect=forward):
            acc = G.capture_pre_gain_hessian(None, 0, layer, hs)
        for name, mult in ((G.rf.NORM_IN, 1.), (G.rf.NORM_POST, torch.tensor([1., 2., 3., 4.]))):
            x = torch.cat([b.reshape(-1, 4) for b in hs]) * mult
            u = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + layer.get_submodule(name).eps)
            self.assertEqual(acc[f"{name}_n"], 9)
            torch.testing.assert_close(acc[name] / 9, u.T @ u / 9)
            self.assertFalse(layer.get_submodule(name)._forward_pre_hooks)
        before = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in acc.items()}
        forward(None, 0, layer, [h * 3 for h in hs])
        for key in acc:
            torch.testing.assert_close(acc[key], before[key])
        with tempfile.TemporaryDirectory() as tmp, patch.multiple(G, OUT=Path(tmp), TAG="fixture", BASIS_ID=None):
            G.save_hu(0, acc)
            with safe_open(Path(tmp) / "hu/fixture/layer_00.safetensors", framework="pt") as f:
                self.assertEqual(f.metadata()["tokens"], "9")
                self.assertEqual(f.metadata()["stream"], "before_layer_quantization")
        with patch.object(G, "run_layer", side_effect=RuntimeError("fixture")):
            with self.assertRaisesRegex(RuntimeError, "fixture"):
                G.capture_pre_gain_hessian(None, 0, layer, hs)
        self.assertFalse(layer.input_layernorm._forward_pre_hooks)
        self.assertFalse(layer.post_attention_layernorm._forward_pre_hooks)


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

    def test_teacher_recipe_holds_out_four_rows_from_each_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            counts = {"corpus": 637, "pi": 44, "chat": 32}
            specs, expected_hold = [], []
            for offset, (name, count) in enumerate(counts.items()):
                p = root / f"{name}.npy"
                data = np.arange((count + 4) * 4).reshape(-1, 4) + offset * 10000
                np.save(p, data)
                p.with_suffix(".json").write_text(json.dumps({"n_train": count}))
                specs.append(f"{p}:" + ("meta" if name == "corpus" else str(count)))
                expected_hold.append(data[count:])
            wiki = np.arange(48 * 4).reshape(-1, 4) + 30000
            np.save(root / "qwen38_train_ids.npy", wiki.reshape(-1))
            expected_hold.append(wiki[44:])
            with patch.multiple(Q, TRAIN=",".join(specs), WIKI=root, WIKI_ROWS=44, HOLD=4, SEQ=4):
                train, hold = Q.rows()
            self.assertEqual(train.shape, (757, 4))
            self.assertEqual(hold.shape, (16, 4))
            np.testing.assert_array_equal(hold, np.concatenate(expected_hold))


if __name__ == "__main__":
    unittest.main()
