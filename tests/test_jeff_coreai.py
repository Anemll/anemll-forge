"""Jeff decision convert/smoke: config, readout, plan, and launcher gate bypass."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "coreai"))
import forge
from jeff_coreai import (JEFF_DEFAULT, SYSTEM_PROMPT, chunk_plan, convert_plan, decision_messages, int8_per_channel,
                         is_hybrid_qwen35, is_jeff_decision_checkpoint, layer_arrays, load_decision_config,
                         load_text_config, question_options, readout_probs)

JEFF_SRC = Path(os.environ.get("JEFF_SRC", "/Users/anemll/Models/jeff/jeff-src/src"))
# Jeff v1.3's codes: A..Z, then the two-letter pairs that are one token ("BQ" is not, so index 68 is "BR")
CODES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["AA", "AB"]
SUPPORT_ROW = {
    "state": {"service": "Customer support chat of an online shop",
              "message": "I sent the jacket back two weeks ago and still have not seen the money."},
    "question": {"type": "choice",
                 "instructions": "What does the customer want?",
                 "criteria": {"track_refund": "Check the status of a refund they are expecting.",
                              "get_refund": "Get their money back for a purchase.",
                              "other": None}},
}


def tiny_cfg(**overrides):
    cfg = {
        "num_hidden_layers": 4,
        "hidden_size": 32,
        "intermediate_size": 64,
        "vocab_size": 32,
        "rms_norm_eps": 1e-6,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 4,
        "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.25},
    }
    cfg.update(overrides)
    return cfg


def write_tiny_checkpoint(root: Path, cfg=None, n_codes=8):
    from safetensors.numpy import save_file
    cfg = cfg or tiny_cfg()
    hid = cfg["hidden_size"]
    vocab = cfg["vocab_size"]
    nk, nv, dk, dv = (cfg[k] for k in ("linear_num_key_heads", "linear_num_value_heads",
                                        "linear_key_head_dim", "linear_value_head_dim"))
    nh, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    mid = cfg["intermediate_size"]
    cdim = 2 * nk * dk + nv * dv
    rng = np.random.default_rng(0)
    w = {f"model.language_model.embed_tokens.weight": rng.standard_normal((vocab, hid)).astype(np.float16) * 0.02,
         "model.language_model.norm.weight": np.zeros(hid, np.float16)}
    for i, kind in enumerate(cfg["layer_types"]):
        p = f"model.language_model.layers.{i}."
        w[p + "input_layernorm.weight"] = np.zeros(hid, np.float16)
        w[p + "post_attention_layernorm.weight"] = np.zeros(hid, np.float16)
        w[p + "mlp.gate_proj.weight"] = rng.standard_normal((mid, hid)).astype(np.float16) * 0.02
        w[p + "mlp.up_proj.weight"] = rng.standard_normal((mid, hid)).astype(np.float16) * 0.02
        w[p + "mlp.down_proj.weight"] = rng.standard_normal((hid, mid)).astype(np.float16) * 0.02
        if kind == "linear_attention":
            w[p + "linear_attn.in_proj_qkv.weight"] = rng.standard_normal((cdim, hid)).astype(np.float16) * 0.02
            w[p + "linear_attn.in_proj_z.weight"] = rng.standard_normal((nv * dv, hid)).astype(np.float16) * 0.02
            w[p + "linear_attn.out_proj.weight"] = rng.standard_normal((hid, nv * dv)).astype(np.float16) * 0.02
            w[p + "linear_attn.in_proj_a.weight"] = rng.standard_normal((nv, hid)).astype(np.float16) * 0.02
            w[p + "linear_attn.in_proj_b.weight"] = rng.standard_normal((nv, hid)).astype(np.float16) * 0.02
            w[p + "linear_attn.conv1d.weight"] = rng.standard_normal((cdim, 1, 4)).astype(np.float16) * 0.02
            w[p + "linear_attn.A_log"] = np.zeros(nv, np.float16)
            w[p + "linear_attn.dt_bias"] = np.zeros(nv, np.float16)
            w[p + "linear_attn.norm.weight"] = np.zeros((nv, dv), np.float16)
        else:
            w[p + "self_attn.q_proj.weight"] = rng.standard_normal((2 * nh * hd, hid)).astype(np.float16) * 0.02
            w[p + "self_attn.k_proj.weight"] = rng.standard_normal((nkv * hd, hid)).astype(np.float16) * 0.02
            w[p + "self_attn.v_proj.weight"] = rng.standard_normal((nkv * hd, hid)).astype(np.float16) * 0.02
            w[p + "self_attn.o_proj.weight"] = rng.standard_normal((hid, nh * hd)).astype(np.float16) * 0.02
            w[p + "self_attn.q_norm.weight"] = np.zeros(hd, np.float16)
            w[p + "self_attn.k_norm.weight"] = np.zeros(hd, np.float16)
    (root / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "text_config": cfg}))
    save_file(w, root / "model.safetensors")
    save_file({"weight": rng.standard_normal((n_codes, hid)).astype(np.float16) * 0.05},
              root / "readout.safetensors")
    (root / "decision_config.json").write_text(json.dumps({
        "codes": CODES[:n_codes], "temperature": 1.25, "prompt_layout": "live-last"}))
    return cfg


class JeffPromptTests(unittest.TestCase):
    def test_live_last_dict_state(self):
        system, user = decision_messages(SUPPORT_ROW, CODES, "live-last")
        self.assertEqual(system, {"role": "system", "content": SYSTEM_PROMPT})
        self.assertEqual(user["content"], [{"type": "text", "text": (
            "Question:\nWhat does the customer want?\n\n"
            'State:\n{"service": "Customer support chat of an online shop"}\n\n'
            "Options:\n"
            "A: track_refund: Check the status of a refund they are expecting.\n"
            "B: get_refund: Get their money back for a purchase.\n"
            "C: other\n\n"
            'Latest:\n{"message": "I sent the jacket back two weeks ago and still have not seen the money."}\n\n'
            "Return only the letter code of the best option.")}])

    def test_state_first_and_plain_text_state(self):
        row = {"state": "disk at 97%", "question": {"type": "noul"}}
        for layout in ("state-first", "live-last"):   # live-last keeps a plain-text state first
            text = decision_messages(row, CODES, layout)[1]["content"][0]["text"]
            self.assertEqual(text, "State:\ndisk at 97%\n\nQuestion:\nChoose the best matching option.\n\n"
                                   "Options:\nA: No / false\nB: Yes / true\n\n"
                                   "Return only the letter code of the best option.")

    def test_question_options(self):
        self.assertEqual(question_options({"type": "score", "criteria": ["low", "high"]}), (["0", "1"], ["low", "high"]))
        self.assertEqual(question_options({"type": "noul", "true_first": True})[0], ["true", "false"])
        with self.assertRaises(ValueError):
            decision_messages(SUPPORT_ROW, CODES[:2], "live-last")
        with self.assertRaises(ValueError):
            decision_messages(SUPPORT_ROW, CODES, "options-first")

    @unittest.skipUnless((JEFF_SRC / "jeff" / "model.py").is_file() and (JEFF_DEFAULT / "tokenizer.json").is_file(),
                         "needs a firelex/jeff checkout (JEFF_SRC) and the local jeff-base checkpoint")
    def test_prompt_ids_match_upstream_jeff(self):
        try:
            from transformers import AutoTokenizer
            sys.path.insert(0, str(ROOT / "scripts"))
            from jeff_reference import support_row, upstream_messages
        except ImportError as e:
            self.skipTest(f"needs transformers + torch: {e}")
        from jeff_coreai import prompt_ids
        decision = load_decision_config(JEFF_DEFAULT)
        self.assertEqual(decision["codes"][68], "BR")
        tok = AutoTokenizer.from_pretrained(str(JEFF_DEFAULT))
        upstream = upstream_messages(JEFF_SRC)
        for row in (SUPPORT_ROW, support_row(3, 100), {"state": "plain text", "question": {"type": "noul"}}):
            msgs = upstream(row, decision["codes"], decision["prompt_layout"])
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            self.assertEqual(prompt_ids(JEFF_DEFAULT, row, decision, tok),
                             tok(text, add_special_tokens=False)["input_ids"])


class JeffCoreAITests(unittest.TestCase):
    def test_readout_softmax_and_temperature(self):
        hidden = np.array([1.0, 0.0, 0.0], np.float32)
        readout = np.array([[1.0, 0, 0], [0, 0, 0], [0.5, 0, 0]], np.float32)
        p = readout_probs(hidden, readout, 3, 1.0)
        self.assertAlmostEqual(p.sum(), 1.0)
        self.assertEqual(int(np.argmax(p)), 0)
        hotter = readout_probs(hidden, readout, 3, 10.0)
        self.assertGreater(hotter[1], p[1])

    def test_softmax_rejects_empty_n_options_via_readout(self):
        with self.assertRaises(ValueError):
            readout_probs(np.ones(2), np.ones((2, 2)), 0, 1.0)

    def test_hybrid_detection_and_text_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps({"text_config": tiny_cfg()}))
            cfg = load_text_config(root)
            self.assertTrue(is_hybrid_qwen35(cfg))
            self.assertFalse(is_jeff_decision_checkpoint(root, cfg))
            (root / "readout.safetensors").write_bytes(b"x")
            self.assertTrue(is_jeff_decision_checkpoint(root, cfg))

    def test_flattened_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps(tiny_cfg()))
            self.assertEqual(load_text_config(root)["hidden_size"], 32)

    def test_chunk_plan_and_convert_plan(self):
        self.assertEqual(chunk_plan(24, 4)[0], list(range(4)))
        self.assertEqual(chunk_plan(24, 4)[-1], list(range(20, 24)))
        ck = type("CK", (), {})()
        ck.model = Path("/Users/anemll/Models/jeff/jeff-base-v1.3")
        ck.cfg = tiny_cfg(num_hidden_layers=8, layer_types=["linear_attention", "full_attention"] * 4)
        ck.readout = np.zeros((255, 32), np.float16)
        ck.decision = {"temperature": 0.9, "prompt_layout": "live-last"}
        plan = convert_plan(ck, ctx=256, prefill=16, quant="fp16", chunk=4)
        self.assertEqual(plan["kind"], "jeff-decision")
        self.assertFalse(plan["dflash2"])
        self.assertFalse(plan["vq_gptq"])
        self.assertEqual(plan["head"]["shape"], [255, 32])
        self.assertEqual(len(plan["chunk_plan"]), 2)
        with self.assertRaises(ValueError):
            convert_plan(ck, ctx=256, prefill=7, quant="fp16")
        both = convert_plan(ck, ctx=256, prefill=64, quant="fp16", prefills=(32, 64))
        self.assertEqual(both["prefills"], [32, 64])
        self.assertEqual(both["prefill_rows"], 64)

    def test_int8_and_layer_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_tiny_checkpoint(root)
            from jeff_coreai import JeffCheckpoint
            ck = JeffCheckpoint(root)
            self.assertEqual(ck.readout.shape, (8, 32))
            self.assertEqual(ck.decision["temperature"], 1.25)
            arrs = layer_arrays(ck, 0, "fp16")
            self.assertIn("0/linear_attn.in_proj_qkv.weight/dense", arrs)
            self.assertIn("0/linear_attn.conv1d.weight", arrs)
            arrs8 = layer_arrays(ck, 0, "int8")
            self.assertIn("0/linear_attn.in_proj_qkv.weight/int8", arrs8)
            self.assertIn("0/linear_attn.in_proj_qkv.weight/scale", arrs8)
            self.assertNotIn("0/linear_attn.in_proj_qkv.weight/act_amax", arrs8)
            qkv = ck.layer(0)["linear_attn.in_proj_qkv.weight"]
            scales = {"0/linear_attn.in_proj_qkv.weight": {"in": 0.02, "out": 0.05}}
            for name in ("linear_attn.in_proj_z.weight", "linear_attn.out_proj.weight",
                         "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"):
                scales[f"0/{name}"] = {"in": 0.02, "out": 0.05}
            arrs_w = layer_arrays(ck, 0, "w8a8", scales)
            self.assertIn("0/linear_attn.in_proj_qkv.weight/int8", arrs_w)
            amax = arrs_w["0/linear_attn.in_proj_qkv.weight/act_amax"]
            self.assertEqual(amax.shape, (qkv.shape[1],))
            self.assertTrue(np.allclose(amax, np.float16(0.02)))
            down = ck.layer(0)["mlp.down_proj.weight"]
            self.assertEqual(arrs_w["0/mlp.down_proj.weight/out_amax"].shape, (down.shape[0],))
            # A uniform input amax scales every weight in a row by the same amount, so the INT8 codes match the
            # unfolded matrix. The per-channel case is what the calibration file actually stores.
            codes, _scale = int8_per_channel(np.asarray(qkv, np.float32))
            np.testing.assert_array_equal(arrs_w["0/linear_attn.in_proj_qkv.weight/int8"], codes)
            attn_scales = {}
            for name in ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                         "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
                         "mlp.down_proj.weight"):
                attn_scales[f"3/{name}"] = {"in": 0.03, "out": 0.04}
            arrs_a = layer_arrays(ck, 3, "w8a8", attn_scales)
            self.assertIn("3/self_attn.q_proj.weight/act_amax", arrs_a)
            self.assertNotIn("3/self_attn.q_proj.weight/out_amax", arrs_a)
            self.assertNotIn("3/self_attn.k_proj.weight/out_amax", arrs_a)
            self.assertIn("3/self_attn.v_proj.weight/out_amax", arrs_a)
            self.assertIn("3/self_attn.o_proj.weight/out_amax", arrs_a)
            w = ck.layer(3)["self_attn.q_proj.weight"]
            codes, scale = int8_per_channel(w)
            recon = codes.astype(np.float32) * scale[:, None].astype(np.float32)
            err = np.max(np.abs(recon - np.asarray(w, np.float32)))
            self.assertLess(err, np.max(np.abs(w)) / 50 + 1e-3)
            attn = layer_arrays(ck, 3, "fp16")
            self.assertIn("3/self_attn.q_proj.weight/dense", attn)

    def test_bind_builder_cfg_retargets_widths(self):
        from jeff_coreai_build import bind_builder_cfg
        B = type("B", (), {})()
        B.KNOWN_LUTS = {}
        cfg = tiny_cfg()
        bind_builder_cfg(B, cfg)
        self.assertEqual(B.hid, 32)
        self.assertEqual(B.cdim, 2 * 2 * 8 + 2 * 8)
        self.assertEqual(B.rot, 4)
        self.assertEqual(B.TAPS, [])
        self.assertEqual(B.KV_CACHE_DTYPE, "fp16")

    def test_decision_config_requires_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                load_decision_config(root)
            (root / "decision_config.json").write_text(json.dumps({"temperature": 0.7}))
            with self.assertRaisesRegex(ValueError, "codes"):
                load_decision_config(root)
            (root / "decision_config.json").write_text(json.dumps({"temperature": 0.7, "codes": ["A", "B"]}))
            d = load_decision_config(root)
            self.assertEqual((d["temperature"], d["prompt_layout"]), (0.7, "state-first"))


class JeffLauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.model = self.root / "jeff"
        self.model.mkdir()
        (self.model / "config.json").write_text(json.dumps({"text_config": tiny_cfg()}))

    def test_jeff_convert_bypasses_64x5120_gate(self):
        out = self.root / "build"
        a = forge.parser().parse_args([
            "jeff-convert", "--model", str(self.model), "--output", str(out), "--dry-run"])
        command, env = forge.prepare(a)
        self.assertIn("jeff_coreai_convert.py", command[1])
        self.assertEqual(env["MODEL"], str(self.model))
        self.assertIn("--dry-run", command)

    def test_regular_convert_still_rejects_jeff_shape(self):
        (self.model / "config.json").write_text(json.dumps({"text_config": {
            "hidden_size": 1024, "num_hidden_layers": 24,
            "layer_types": ["linear_attention", "full_attention"]}}))
        with self.assertRaisesRegex(ValueError, "64 layers"):
            forge.prepare(forge.parser().parse_args([
                "convert", "--model", str(self.model), "--export", str(self.root),
                "--output", str(self.root / "out")]))

    def test_jeff_convert_rejects_dense_config(self):
        (self.model / "config.json").write_text(json.dumps({"text_config": {
            "hidden_size": 1024, "num_hidden_layers": 24}}))
        with self.assertRaisesRegex(ValueError, "hybrid"):
            forge.prepare(forge.parser().parse_args([
                "jeff-convert", "--model", str(self.model), "--output", str(self.root / "out")]))

    def test_jeff_prefill_multiple_of_eight(self):
        with self.assertRaisesRegex(ValueError, "multiple of 8"):
            forge.prepare(forge.parser().parse_args([
                "jeff-convert", "--model", str(self.model), "--output", str(self.root / "out"),
                "--prefill", "250"]))
        with self.assertRaisesRegex(ValueError, "multiple of 8"):
            forge.prepare(forge.parser().parse_args([
                "jeff-convert", "--model", str(self.model), "--output", str(self.root / "out"),
                "--prefill-extra", "12"]))

    def test_jeff_prefill_extra_argv(self):
        cmd, _ = forge.prepare(forge.parser().parse_args([
            "jeff-convert", "--model", str(self.model), "--output", str(self.root / "out"),
            "--prefill", "256", "--prefill-extra", "32,64", "--dry-run"]))
        self.assertEqual(cmd[cmd.index("--prefill-extra") + 1], "32,64")

    def test_jeff_smoke_argv(self):
        cmd, _ = forge.prepare(forge.parser().parse_args([
            "jeff-smoke", "--model", str(self.model), "--ids", "1,2,3"]))
        self.assertIn("jeff_coreai_smoke.py", cmd[1])
        self.assertEqual(cmd[cmd.index("--ids") + 1], "1,2,3")
        self.assertEqual(cmd[0], sys.executable)

    def test_jeff_coreai_python_override(self):
        cases = self.root / "cases.json"
        with patch.dict(os.environ, {"COREAI_PYTHON": "/sdk/bin/python"}):
            cmd, _ = forge.prepare(forge.parser().parse_args([
                "jeff-smoke", "--model", str(self.model), "--build", str(self.root), "--cases", str(cases),
                "--bench", "3"]))
            self.assertEqual(cmd[0], "/sdk/bin/python")
            self.assertEqual(cmd[cmd.index("--cases") + 1], str(cases))
            self.assertEqual(cmd[cmd.index("--bench") + 1], "3")
            cmd, _ = forge.prepare(forge.parser().parse_args([
                "jeff-convert", "--model", str(self.model), "--output", str(self.root / "o")]))
            self.assertEqual(cmd[0], "/sdk/bin/python")

    def test_compile_uses_coreai_python_for_jeff_builds_only(self):
        build = self.root / "coreai"
        build.mkdir()
        for kind, expected in (("jeff-decision", "/sdk/bin/python"), (None, sys.executable)):
            (build / "manifest.json").write_text(json.dumps({"kind": kind} if kind else {"chunks": []}))
            with patch.dict(os.environ, {"COREAI_PYTHON": "/sdk/bin/python"}), \
                    patch("builtins.print") as out:
                self.assertEqual(forge.main(["compile", "--build", str(build), "--dry-run"]), 0)
            self.assertEqual(json.loads(out.call_args[0][0])["argv"][0], expected)

    def test_jeff_convert_dry_run_does_not_write(self):
        out = self.root / "builds"
        with patch("forge.subprocess.call") as run:
            rc = forge.main(["jeff-convert", "--model", str(self.model), "--output", str(out), "--dry-run"])
        self.assertEqual(rc, 0)
        run.assert_not_called()
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
