"""Tiny random JetSpec++ artifacts in the release's exact layout (queue 7, 19 Nov 2026), for a plumbing test of
jsp_replay.py while the served forge is up (a few MB, no real weights loaded). Every loader, key check and sha256 pin
of the reference runs unchanged; only the widths shrink (hidden 64, vocab 512; 64 target layers, taps 5 / 19 / 33 /
47 / 61, the GDN / full tail at 62 / 63, B16 and five draft layers as in the release).
    PY make_tiny.py OUT_DIR
-> OUT_DIR/target (config, index, one shard: embed, layers 62-63, norm, lm_head), OUT_DIR/release (config.json,
dflash.py, model.safetensors, adapter/, calibration/), OUT_DIR/traces/t0, t1 (random token streams, random taps)."""
import copy, hashlib, importlib.util, json, shutil, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jsp_replay as R  # noqa: E402  (puts the JetSpec source on sys.path)
from jetspec.engines.jetspec_plus_plus.draft.method.adapter import CtxKVC1Head, HybridC1RMSNorm  # noqa: E402
from jetspec.engines.jetspec_plus_plus.draft.reference.models import hybrid as H  # noqa: E402
from jetspec.engines.jetspec_plus_plus.draft.reference.models.drafter import load_qwen3_config_with_rope_compat  # noqa: E402

W, V, I = 64, 512, 128
MASK, EOS = 500, 511


def rescale(module, std=0.02, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            if p.ndim >= 2:
                p.copy_(torch.randn(p.shape, generator=g) * std)
    return module


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out = Path(sys.argv[1])
    if out.exists():
        shutil.rmtree(out)
    rel = R.RELEASE
    # ---- target
    tgt = out / "target"; tgt.mkdir(parents=True)
    real = json.loads((R.TARGET / "config.json").read_text())
    tc = dict(real["text_config"], hidden_size=W, intermediate_size=I, num_attention_heads=4, num_key_value_heads=2,
              head_dim=32, linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2,
              linear_num_value_heads=4, vocab_size=V, bos_token_id=EOS, eos_token_id=EOS)
    cfg_raw = {"architectures": real["architectures"], "model_type": real["model_type"], "text_config": tc,
               "tie_word_embeddings": False}
    (tgt / "config.json").write_text(json.dumps(cfg_raw, indent=1))
    cfg = H.text_config(cfg_raw)
    torch.manual_seed(0)
    state = {"model.language_model.embed_tokens.weight": torch.randn(V, W) * 0.5,
             "model.language_model.norm.weight": torch.zeros(W) if True else None,
             "lm_head.weight": torch.randn(V, W) * 0.05}
    for i in (62, 63):
        layer = rescale(H.HybridLayer(cfg, cfg.layer_types[i]), seed=i)
        for k, v in layer.state_dict().items():
            state[f"model.language_model.layers.{i}.{k}"] = v.clone()
    norm = HybridC1RMSNorm(W, cfg.rms_norm_eps)
    state["model.language_model.norm.weight"] = norm.state_dict()["weight"].clone()
    state = {k: v.to(torch.bfloat16).contiguous() for k, v in state.items()}
    from safetensors.torch import save_file
    save_file(state, str(tgt / "model-00001-of-00001.safetensors"), metadata={"format": "pt"})
    (tgt / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {}, "weight_map": {k: "model-00001-of-00001.safetensors" for k in state}}, indent=1))
    # ---- drafter
    rd = out / "release"; rd.mkdir()
    dc = json.loads((rel / "config.json").read_text())
    dc.update(hidden_size=W, intermediate_size=I, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
              vocab_size=V, bos_token_id=EOS, eos_token_id=EOS)
    dc["dflash_config"] = dict(dc["dflash_config"], mask_token_id=MASK)
    (rd / "config.json").write_text(json.dumps(dc, indent=1))
    shutil.copy(rel / "dflash.py", rd / "dflash.py")
    spec = importlib.util.spec_from_file_location("tiny_dflash", rd / "dflash.py")
    mod = importlib.util.module_from_spec(spec); sys.modules["tiny_dflash"] = mod; spec.loader.exec_module(mod)
    config = load_qwen3_config_with_rope_compat(str(rd), local_files_only=True)
    config._attn_implementation = "sdpa"
    head = rescale(mod.DFlashDraftModel(config), seed=1)
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in head.state_dict().items()},
              str(rd / "model.safetensors"), metadata={"format": "pt"})
    # ---- adapter
    ad = rd / "adapter"; ad.mkdir()
    ac = json.loads((rel / "adapter" / "config.json").read_text())
    ac.update(hidden_size=W, intermediate_size=I, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
              vocab_size=V, bos_token_id=EOS, eos_token_id=EOS)
    ck = copy.deepcopy(ac["dflash_config"]["ctxkv_head"])
    ck["target_text_config"].update(hidden_size=W, intermediate_size=I, num_attention_heads=4, num_key_value_heads=2,
                                    head_dim=32, vocab_size=V)
    ac["dflash_config"] = dict(ac["dflash_config"], mask_token_id=MASK, ctxkv_head=ck)
    (ad / "config.json").write_text(json.dumps(ac, indent=1))
    from transformers import Qwen3Config
    acfg = Qwen3Config.from_dict(ac); acfg.rope_theta = float(ac["rope_theta"]); acfg._attn_implementation = "sdpa"
    ahead = rescale(CtxKVC1Head(acfg), seed=2)
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in ahead.state_dict().items()},
              str(ad / "head.safetensors"), metadata={"FFN_SOURCE": "target_init"})
    save_file({"fc.weight": (torch.randn(W, 5 * W) * 0.02).to(torch.bfloat16),
               "hidden_norm.weight": torch.ones(W, dtype=torch.bfloat16)},
              str(ad / "converter.safetensors"), metadata={"SOURCE": "tiny", "FROZEN": "true"})
    # ---- calibration (the real thresholds; pins of the tiny target)
    cd = rd / "calibration"; (cd / "decoder").mkdir(parents=True)
    shutil.copy(rel / "calibration" / "thresholds.json", cd / "thresholds.json")
    (cd / "decoder" / "manifest.json").write_text(json.dumps({"source": "target_slice", "target_files": {
        n: sha(tgt / n) for n in ("config.json", "model.safetensors.index.json", "model-00001-of-00001.safetensors")}},
        indent=1))
    # ---- traces: random streams (tau about 1 with random weights; this tests the plumbing, not acceptance)
    rng = np.random.default_rng(0)
    for t, (P, G) in enumerate(((48, 120), (130, 90))):
        d = out / "traces" / f"t{t}"; d.mkdir(parents=True)
        toks = rng.integers(0, MASK, P + G)
        np.save(d / "tokens.npy", toks.astype(np.int32))
        np.save(d / "taps.npy", (rng.standard_normal((P + G, 5 * W)) * 0.5).astype(np.float16))
        np.save(d / "filled.npy", np.ones(P + G, bool))
        (d / "meta.json").write_text(json.dumps({"prompt_len": P, "index": t, "finish": "length", "tiny": True}))
    print("tiny artifacts in", out)


if __name__ == "__main__":
    main()
