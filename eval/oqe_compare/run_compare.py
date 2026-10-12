#!/usr/bin/env python3
"""Apples-to-apples KL: oQe 4-bit MLX vs Anemll ANE on the oQe model-card corpora.

Both models are scored on the exact Code (Python) and Novel (English) windows
the card describes, against the same BF16 next-token logits, with the same
tokenizer and the same full-vocabulary KL. This is not the Forge Tier A
KL-512 chat-trace gate.

Does not invent results. A real run needs Apple Silicon and ~60 GiB for BF16.

    eval/oqe_compare/run_m5max.sh
    python eval/oqe_compare/run_compare.py --preset m5max
    python eval/oqe_compare/run_compare.py --smoke --out /tmp/oqe-smoke
    python eval/oqe_compare/run_compare.py --dry-run --preset m5max
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from eval.oqe_compare import artifacts, backends, corpora, kl_metrics, size_report
else:
    from . import artifacts, backends, corpora, kl_metrics, size_report

M5_BF16 = Path("/Users/anemll/Models/Qwen3.8-27B")
M5_ANE = Path("/Users/anemll/Models/anemll-forge/qwen38-27b-tierA")
M5_OQE = Path("/Users/anemll/Models/Qwen3.8-27B-oQ4e-mtp")
OQE_HF_ID = "Jundot/Qwen3.8-27B-oQ4e-mtp"
OQE_TWEET = "https://x.com/jundotkim/status/2109557358028337547"

# Card-published mean next-token KL (updated oQ4e). Not our measurements.
CARD_PUBLISHED = {
    "Code (Python)": {"previous_oqe": 0.0143, "updated_oqe": 0.0117, "change": "-18.1%"},
    "Novel (English)": {"previous_oqe": 0.0696, "updated_oqe": 0.0600, "change": "-13.8%"},
}


def log(msg):
    print(msg, flush=True)


def apply_preset(a):
    if a.preset != "m5max":
        return a
    a.bf16 = a.bf16 or M5_BF16
    a.ane = a.ane or M5_ANE
    if a.oqe is None:
        a.oqe = M5_OQE if M5_OQE.exists() else Path(OQE_HF_ID)
    if a.tokenizer is None:
        a.tokenizer = a.bf16
    if a.out is None:
        a.out = Path("/Users/anemll/Models/anemll-forge/oqe_compare")
    return a


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=("m5max",), help="M5 Max paths: BF16, ANE Tier A, oQe 4-bit")
    p.add_argument("--bf16", type=Path, help="BF16 Qwen3.8-27B directory (reference + tokenizer)")
    p.add_argument("--oqe", type=Path, help="oQe 4-bit MLX directory or Hugging Face id")
    p.add_argument("--ane", type=Path, help="ANE bundle root or Core AI build directory")
    p.add_argument("--tokenizer", type=Path, help="Tokenizer path (defaults to --bf16)")
    p.add_argument("--out", type=Path, help="Output directory")
    p.add_argument("--stage", choices=("prepare", "reference", "eval-oqe", "eval-ane", "summarize", "all"),
                   default="all")
    p.add_argument("--fetch-corpora", action="store_true", default=True)
    p.add_argument("--no-fetch-corpora", action="store_false", dest="fetch_corpora")
    p.add_argument("--window", type=int, default=corpora.WINDOW)
    p.add_argument("--n-windows", type=int, default=corpora.N_WINDOWS)
    p.add_argument("--ref-dtype", choices=("f16", "f32"), default="f16",
                   help="BF16 log-softmax cache dtype (f16 ~16 GiB at 27B vocab)")
    p.add_argument("--ane-format", default="v8", choices=("fp16", "v8", "kv8"))
    p.add_argument("--ane-ctx", type=int, default=8192, help="Core AI entry; must be >= window")
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--embed-npy", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="CPU dummy LM over synthetic text; no 27B weights")
    p.add_argument("--skip-oqe", action="store_true")
    p.add_argument("--skip-ane", action="store_true")
    return apply_preset(p.parse_args(argv))


def tokenizer_for(a):
    if a.smoke:
        return backends.DummyTokenizer().encode
    path = a.tokenizer or a.bf16
    if path is None:
        raise SystemExit("--tokenizer or --bf16 is required (or --smoke / --preset m5max)")
    return backends.load_hf_tokenizer(path)


def prepare(a, out: Path) -> dict:
    encode = tokenizer_for(a)
    plan = corpora.plan_windows(
        encode, out / "corpora", fetch=a.fetch_corpora and not a.smoke,
        smoke=a.smoke, n_windows=a.n_windows, window=a.window,
    )
    artifacts.write_trace(out, plan["sequences"])
    corpora.dump_plan(plan, out / "windows.json")
    meta = {
        "protocol": "oqe-card",
        "metric": "full-vocab next-token KL (nats) vs shared BF16 logits",
        "card_url": corpora.CARD_URL,
        "card_readme_url": corpora.CARD_README_URL,
        "tweet": OQE_TWEET,
        "tokenizer": getattr(encode, "name", None) or ("dummy-utf8" if a.smoke else str(a.tokenizer or a.bf16)),
        "window": a.window,
        "n_windows": a.n_windows,
        "smoke": a.smoke,
        "n_sequences": plan["n_sequences"],
        "input_tokens_per_corpus": a.n_windows * a.window,
        "scored_positions_per_corpus": a.n_windows * (a.window - 1),
        "positions": plan["positions_if_teacher_forced"],
        "categories": [s.label for s in corpora.CORPORA],
        "sources": plan["sources"],
        "card_published": CARD_PUBLISHED,
        "not_the_tier_a_kl512_gate": True,
    }
    artifacts.write_json(out / "prepare.json", meta)
    log(json.dumps({k: meta[k] for k in ("protocol", "n_sequences", "positions", "categories")}, indent=2))
    return meta


def run_reference(a, out: Path) -> dict:
    seqs, _, _ = artifacts.load_trace(out)
    dtype = "f32" if a.smoke else a.ref_dtype
    if a.smoke:
        lm = backends.DummyLM(bias=0.0, name="dummy-bf16")
        meta, nll = backends.reference_pass(seqs, lm.next_logits, out, dtype_name=dtype)
    else:
        if a.bf16 is None:
            raise SystemExit("--bf16 is required for the reference pass")
        model, device = backends.load_bf16_model(a.bf16)

        def next_logits(seq, _m=model, _d=device):
            return backends.torch_next_logits(_m, seq, _d)

        t0 = time.time()
        meta, nll = backends.reference_pass(
            seqs, next_logits, out, dtype_name=dtype,
            progress=lambda i, n, t: log(f"bf16 {i + 1}/{n} ({t} tokens)") if i % 4 == 0 else None,
        )
        log(f"bf16 reference done in {time.time() - t0:.0f}s")
        del model
    summary = {
        "backend": "dummy-bf16" if a.smoke else "torch-bf16",
        "positions": int(len(nll)),
        "vocab": meta["vocab"],
        "ref_dtype": meta["dtype"],
        "ref_ppl": float(np.exp(nll.astype(np.float64).mean())),
        "metric": "full-vocab next-token log-softmax cache",
    }
    artifacts.write_json(out / "reference.json", summary)
    log(json.dumps(summary, indent=2))
    return summary


def _save_eval(out: Path, tag: str, result: dict) -> dict:
    slim = {k: v for k, v in result.items() if k not in ("kl", "agreement", "nll")}
    artifacts.write_json(out / f"kl_{tag}.json", slim)
    if "kl" in result:
        np.save(out / f"klpos_{tag}.npy", result["kl"].astype(np.float32))
    log(json.dumps({"tag": tag, **slim}, indent=2))
    return slim


def run_oqe(a, out: Path) -> dict:
    seqs, _, _ = artifacts.load_trace(out)
    ref_lp, _ = artifacts.load_ref_lp(out)
    ref_nll = artifacts.load_ref_nll(out)
    if a.smoke:
        lm = backends.DummyLM(bias=0.35, name="dummy-oqe")
        result = backends.eval_pass(seqs, lm.next_logits, ref_lp, ref_nll)
        result["backend"] = "dummy-oqe"
    else:
        if a.oqe is None:
            raise SystemExit("--oqe is required")
        model, _tok = backends.load_mlx_model(a.oqe)

        def next_logits(seq, _m=model):
            return backends.mlx_next_logits(_m, seq)

        t0 = time.time()
        result = backends.eval_pass(
            seqs, next_logits, ref_lp, ref_nll,
            progress=lambda i, n, t: log(f"oqe {i + 1}/{n} ({t} tokens)") if i % 4 == 0 else None,
        )
        result["backend"] = "mlx-oqe"
        result["eval_s"] = time.time() - t0
    oqe_path = a.oqe if a.oqe and Path(str(a.oqe)).exists() else None
    result["size"] = size_report.measure_dir(oqe_path, published=size_report.published_oqe_4bit())
    return _save_eval(out, "oqe", result)


def run_ane(a, out: Path) -> dict:
    if a.smoke:
        seqs, _, _ = artifacts.load_trace(out)
        ref_lp, _ = artifacts.load_ref_lp(out)
        ref_nll = artifacts.load_ref_nll(out)
        lm = backends.DummyLM(bias=0.55, name="dummy-ane")
        result = backends.eval_pass(seqs, lm.next_logits, ref_lp, ref_nll)
        result["backend"] = "dummy-ane"
        result["size"] = size_report.measure_dir(a.ane, published=size_report.published_anemll_u48())
        return _save_eval(out, "ane", result)
    if a.ane is None:
        raise SystemExit("--ane is required")
    build = backends.resolve_ane_build(a.ane)
    bundle = a.ane.expanduser().resolve()
    model_dir = a.model_dir
    if model_dir is None and (bundle / "model").exists():
        model_dir = bundle / "model"
    embed = backends.resolve_embed(bundle, a.embed_npy)
    extra = {}
    if model_dir is not None:
        extra["MODEL"] = model_dir
        extra["EMBED_NPY"] = embed or (model_dir / "embed_tokens_fp16.npy")
    elif embed is not None:
        extra["EMBED_NPY"] = embed
    return backends.run_ane_card(
        build, out, out / "ane_run", fmt=a.ane_format, ctx=a.ane_ctx, extra_env=extra,
    )


def by_corpus(out: Path, tag: str):
    plan_path = out / "windows.json"
    pos_path = out / f"klpos_{tag}.npy"
    if not plan_path.exists() or not pos_path.exists():
        return None
    plan = json.loads(plan_path.read_text())
    kl = np.load(pos_path)
    off = 0
    by = {}
    for w in plan["windows"]:
        n = int(w["length"]) - 1
        by.setdefault(w["label"], []).append(kl[off:off + n])
        off += n
    return {
        label: {
            "positions": int(np.concatenate(chunks).size),
            "input_tokens": int(sum(1 for w in plan["windows"] if w["label"] == label)
                                * plan["windows"][0]["length"]),
            "mean_kl": float(np.concatenate(chunks).mean()),
        }
        for label, chunks in by.items()
    }


def comparison_rows(out: Path) -> list[dict]:
    splits = {tag: by_corpus(out, tag) for tag in ("oqe", "ane")}
    rows = []
    for label, pub in CARD_PUBLISHED.items():
        oqe = (splits["oqe"] or {}).get(label, {})
        ane = (splits["ane"] or {}).get(label, {})
        rows.append({
            "corpus": label,
            "input_tokens": oqe.get("input_tokens") or ane.get("input_tokens"),
            "scored_positions": oqe.get("positions") or ane.get("positions"),
            "oqe_published_updated": pub["updated_oqe"],
            "oqe_published_previous": pub["previous_oqe"],
            "oqe_this_run": oqe.get("mean_kl"),
            "ane_this_run": ane.get("mean_kl"),
        })
    return rows


def summarize_run(a, out: Path) -> dict:
    rows = comparison_rows(out)
    table = {
        "metric": "mean next-token KL vs BF16 (full vocabulary, nats)",
        "protocol": "oqe-card",
        "card_url": corpora.CARD_URL,
        "categories": list(CARD_PUBLISHED),
        "comparison": rows,
        "sizes": {
            "oqe_published": size_report.published_oqe_4bit(),
            "ane_u48_published": size_report.published_anemll_u48(),
            "oqe_measured": size_report.measure_dir(
                a.oqe if a.oqe and Path(str(a.oqe)).exists() else None,
                published=size_report.published_oqe_4bit(),
            ),
            "ane_measured": size_report.measure_dir(a.ane, published=size_report.published_anemll_u48()),
        },
        "smoke": a.smoke,
        "empty_cells_mean_not_run": True,
        "invented_results": False,
        "not_the_tier_a_kl512_gate": True,
    }
    artifacts.write_json(out / "summary.json", table)
    log(json.dumps(table, indent=2))
    return table


def dry_run(a) -> dict:
    out = a.out or Path("eval/oqe_compare/results")
    plan = {
        "dry_run": True,
        "smoke": a.smoke,
        "preset": a.preset,
        "paths": {
            "bf16": None if a.bf16 is None else str(a.bf16),
            "oqe": None if a.oqe is None else str(a.oqe),
            "ane": None if a.ane is None else str(a.ane),
            "out": str(out),
        },
        "card": {
            "url": corpora.CARD_URL,
            "readme": corpora.CARD_README_URL,
            "tweet": OQE_TWEET,
            "methodology": (
                "Mean next-token KL divergence from the original BF16 model on the "
                "oMLX benchmark corpora. Each corpus is scored on 16 windows of "
                "2,048 tokens spread across the text (32,768 tokens)."
            ),
            "categories": list(CARD_PUBLISHED),
            "published": CARD_PUBLISHED,
        },
        "comparison_will_be": [
            {"corpus": label, "oqe_published_updated": pub["updated_oqe"],
             "oqe_this_run": None, "ane_this_run": None}
            for label, pub in CARD_PUBLISHED.items()
        ],
        "commands": [
            f"eval/oqe_compare/run_m5max.sh",
            f"python eval/oqe_compare/run_compare.py --preset m5max --out {out}",
        ],
        "note": "No models loaded. No KL results computed. Tier A KL-512 is not a comparison column.",
        "invented_results": False,
    }
    out.mkdir(parents=True, exist_ok=True)
    artifacts.write_json(out / "dry_run.json", plan)
    log(json.dumps(plan, indent=2))
    return plan


def run_stages(a):
    if a.out is None:
        raise SystemExit("--out is required (or --preset m5max / --smoke)")
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    stages = ("prepare", "reference", "eval-oqe", "eval-ane", "summarize")
    wanted = list(stages) if a.stage == "all" else [a.stage]
    for stage in wanted:
        if stage == "prepare":
            prepare(a, out)
        elif stage == "reference":
            run_reference(a, out)
        elif stage == "eval-oqe":
            if a.skip_oqe:
                log("skip oqe")
            else:
                run_oqe(a, out)
        elif stage == "eval-ane":
            if a.skip_ane:
                log("skip ane")
            else:
                run_ane(a, out)
        elif stage == "summarize":
            summarize_run(a, out)
        else:
            raise SystemExit(f"unknown stage {stage}")


def main(argv=None):
    kl_metrics.self_check()
    a = parse_args(argv)
    if a.smoke and a.out is None:
        a.out = Path("/tmp/oqe-compare-smoke")
    if a.dry_run:
        dry_run(a)
        return 0
    if a.out is None:
        raise SystemExit("--out is required (or --preset m5max / --smoke)")
    artifacts.write_json(a.out / "run.json", {
        "preset": a.preset, "stage": a.stage, "smoke": a.smoke,
        "bf16": None if a.bf16 is None else str(a.bf16),
        "oqe": None if a.oqe is None else str(a.oqe),
        "ane": None if a.ane is None else str(a.ane),
        "not_the_tier_a_kl512_gate": True,
    })
    run_stages(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
