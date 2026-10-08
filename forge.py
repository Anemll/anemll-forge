#!/usr/bin/env python3
"""Explicit-path entry points for the first ANEMLL Forge source port."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
import hf_release
import ane_compile_mode as SOC
import coreai_compile_guide as G
from qwen38_kv_cache import cache_format
NUMERICS = dict(SILU="tanh", MLP_SILU="tanh", GDN_SQ="16", GDN_SV="64",
                MLP_DS="1", MLP_DS_DYN="0", V3_KV_IN="1", V3_PREFILL="0")


def path(value):
    return Path(value).expanduser().resolve()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="report environment without loading a model")
    c = sub.add_parser("compile", help="compile a Core AI build for this Mac's ANE ahead of serving, with progress and ETA")
    c.add_argument("--build", type=path, required=True, help="Core AI target build directory")
    c.add_argument("--draft", type=path, help="DFlash2 drafter package; default: bundle/drafter/dflash2_lut4_gptq.aimodel "
                   "next to the build, when present")
    c.add_argument("--force", action="store_true", help="drop this Python's cached ANE specializations of the build "
                   "(and drafter) first, so every package recompiles")
    c.add_argument("--follow", action="store_true", help="run alongside a build: compile each chunk as soon as the "
                   "builder lists it, then the head and drafter when it finishes")
    c.add_argument("--dry-run", action="store_true", help="print command only")
    hf_release.add_commands(sub)
    jc = sub.add_parser("jeff-convert", help="prefill-only Core AI export of a Jeff decision checkpoint (no 27B VQ/DFlash2)")
    jc.add_argument("--model", type=path, required=True, help="local Jeff checkpoint directory")
    jc.add_argument("--output", type=path, required=True, help="new empty directory (model/ + coreai/)")
    jc.add_argument("--ctx", type=int, default=2048, help="KV history of the prefill entry")
    jc.add_argument("--prefill", type=int, default=256, help="prefill rows (multiple of 8, > 8)")
    jc.add_argument("--quant", choices=("fp16", "int8"), default="fp16")
    jc.add_argument("--chunk-layers", type=int, default=4)
    jc.add_argument("--dry-run", action="store_true")
    js = sub.add_parser("jeff-smoke", help="Jeff prefill+readout smoke (host DecodeLayer; optional Core AI --build)")
    js.add_argument("--model", type=path, required=True, help="local Jeff checkpoint directory")
    js.add_argument("--build", type=path, help="coreai/ directory from jeff-convert")
    js.add_argument("--state", default="The disk on db-02 is 97 percent full and still growing.")
    js.add_argument("--options", default="page,wait,ignore")
    js.add_argument("--instructions", default="Choose the best next action.")
    js.add_argument("--ids", help="comma-separated token ids; skips tokenizer + prompt render")
    js.add_argument("--n-options", type=int, help="options of the --ids prompt")
    js.add_argument("--cases", type=path, help="scripts/jeff_reference.py output (ids + FP32 reference probabilities)")
    js.add_argument("--only", help="comma-separated case names from --cases")
    js.add_argument("--row", type=path, help="JSON decision row {state, question}")
    js.add_argument("--host", action="store_true", help="also run the host DecodeLayer reference")
    js.add_argument("--bench", type=int, default=0, help="extra timed Core AI prefills per case")
    js.add_argument("--out", type=path, help="results JSON")
    js.add_argument("--dry-run", action="store_true")
    jv = sub.add_parser("jeff-serve", help="local Jeff decision server (POST /v1/systemone) plus the browser demo")
    jv.add_argument("--model", type=path, required=True, help="local Jeff checkpoint directory")
    jv.add_argument("--build", type=path, required=True, help="coreai/ directory from jeff-convert")
    jv.add_argument("--host", default="127.0.0.1")
    jv.add_argument("--port", type=int, default=8787)
    jv.add_argument("--adapter", action="append", default=[], metavar="NAME=PATH",
                    help="extra merged Core AI build, name=coreai-dir. Repeatable. base is --build")
    jv.add_argument("--dry-run", action="store_true")
    jt = sub.add_parser("jeff-train-lora", help="sample LoRA on a Jeff checkpoint (snake oracle, or a JSONL dataset)")
    jt.add_argument("--model", type=path, required=True, help="local Jeff checkpoint directory")
    jt.add_argument("--output", type=path, required=True, help="new directory for adapter/, merged/, and report.json")
    jt.add_argument("--dataset", type=path, help="JSONL of {state, options, label, instructions}; default is the synthetic task")
    jt.add_argument("--task", choices=("snake", "tetris"), default="snake")
    jt.add_argument("--rank", type=int, default=16)
    jt.add_argument("--alpha", type=float, default=32)
    jt.add_argument("--train", type=int, default=256)
    jt.add_argument("--heldout", type=int, default=64)
    jt.add_argument("--epochs", type=int, default=2)
    jt.add_argument("--batch", type=int, default=4)
    jt.add_argument("--lr", type=float, default=2e-4, help="LoRA learning rate")
    jt.add_argument("--readout-lr", type=float, default=5e-6)
    jt.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    jt.add_argument("--seed", type=int, default=0)
    jt.add_argument("--play", type=int, default=0, help="PyTorch self-play games per model; 0 skips them")
    jt.add_argument("--max-steps", type=int, default=48)
    jt.add_argument("--skip-merge", action="store_true")
    jt.add_argument("--dry-run", action="store_true")
    for name in ("quantize", "convert", "chat", "serve"):
        q = sub.add_parser(name)
        q.add_argument("--model", type=path, required=True, help="original checkpoint directory")
        q.add_argument("--dry-run", action="store_true", help="print command and effective overrides only")
        if name == "quantize":
            q.add_argument("--wiki", type=path, required=True, help="wiki2_train.txt and wiki2_test.txt directory")
            q.add_argument("--output", type=path, required=True, help="new run directory")
            q.add_argument("--tag", default="qwen38-27b-vq2")
            q.add_argument("--plan", type=path, help="optional per-matrix mixed-bit plan JSON")
        elif name == "convert":
            q.add_argument("--export", type=path, required=True)
            q.add_argument("--output", type=path, required=True, help="new build parent; export name is appended")
            q.add_argument("--ctx", type=int, choices=(2048, 8192, 16384), default=16384)
        else:
            q.add_argument("--build", type=path, required=True)
            q.add_argument("--ctx", type=int, default=16384)
            if name == "serve":
                q.add_argument("--runtime", choices=("coreml", "coreai"), default="coreai")
                q.add_argument("--kv-cache-dtype", choices=("auto", "fp16", "v8", "kv8"), default="auto",
                               help="require a matching cache export; auto reads its manifest")
                modes = q.add_mutually_exclusive_group()
                modes.add_argument("--draft", type=path, help="Core AI DFlash2 package; default: bundle/drafter/dflash2_lut4_gptq.aimodel")
                modes.add_argument("--plain", action="store_true", help="diagnostic target-only generation")
                q.add_argument("--drafter", type=path, help="drafter config/selector directory; default: package parent")
                q.add_argument("--host", default="127.0.0.1")
                q.add_argument("--port", type=int, default=8765)
                q.add_argument("--summary-think", action="store_true",
                               help="keep thinking for context-summary requests (Pi's compaction; default: off)")
                q.add_argument("--summary-no-think", action="store_true", help=argparse.SUPPRESS)  # now the default
            else:
                q.add_argument("--prompt")
                q.add_argument("--no-think", action="store_true")
    return p


def _text_config(model: Path) -> dict:
    raw = json.loads((model / "config.json").read_text())
    cfg = raw.get("text_config")
    if isinstance(cfg, dict) and cfg.get("num_hidden_layers") and cfg.get("hidden_size"):
        return cfg
    if raw.get("num_hidden_layers") and raw.get("hidden_size"):
        return raw
    return {}


def _is_hybrid(cfg: dict) -> bool:
    types = cfg.get("layer_types") or []
    return "linear_attention" in types and "full_attention" in types


def prepare_jeff(a):
    """Jeff decision path: hybrid Qwen3.5 + readout. Does not use the 64 x 5120 gate."""
    if not (a.model / "config.json").is_file():
        raise ValueError(f"Missing checkpoint config: {a.model / 'config.json'}")
    cfg = _text_config(a.model)
    if not _is_hybrid(cfg):
        raise ValueError("jeff-convert / jeff-smoke / jeff-serve / jeff-train-lora expect a Qwen3.5 hybrid text config "
                         "(layer_types with linear_attention and full_attention).")
    env = {"MODEL": str(a.model)}
    published_embedding = a.model / "embed_tokens_fp16.npy"
    env["EMBED_NPY"] = str(published_embedding if published_embedding.is_file()
                           else a.model / ".anemll-forge" / "embed_tokens_fp16.npy")
    if a.command == "jeff-convert":
        if a.prefill <= 8 or a.prefill % 8:
            raise ValueError("--prefill must be a multiple of 8 and greater than 8")
        if a.ctx < a.prefill:
            raise ValueError("--ctx must be >= --prefill")
        dest = a.output
        if not a.dry_run and dest.exists() and any(dest.iterdir()):
            raise ValueError(f"Use a new or empty --output directory: {dest}")
        args = ["--model", str(a.model), "--output", str(a.output), "--ctx", str(a.ctx),
                "--prefill", str(a.prefill), "--quant", a.quant, "--chunk-layers", str(a.chunk_layers)]
        if a.dry_run:
            args.append("--dry-run")
        python = sys.executable if a.dry_run else coreai_python()
        return [python, str(ROOT / "scripts" / "jeff_coreai_convert.py"), *args], env
    if a.command == "jeff-train-lora":
        # Training is PyTorch in this interpreter (transformers + MPS or CPU). It does not load Core AI.
        args = ["--model", str(a.model), "--output", str(a.output), "--task", a.task,
                "--rank", str(a.rank), "--alpha", str(a.alpha), "--train", str(a.train),
                "--heldout", str(a.heldout), "--epochs", str(a.epochs), "--batch", str(a.batch),
                "--lr", str(a.lr), "--readout-lr", str(a.readout_lr),
                "--device", a.device, "--seed", str(a.seed), "--play", str(a.play),
                "--max-steps", str(a.max_steps)]
        if a.dataset is not None:
            args += ["--dataset", str(a.dataset)]
        if a.skip_merge:
            args.append("--skip-merge")
        return [sys.executable, str(ROOT / "scripts" / "jeff_lora_train.py"), *args], env
    if a.command == "jeff-serve":
        args = ["--model", str(a.model), "--build", str(a.build), "--host", a.host, "--port", str(a.port)]
        for item in a.adapter or []:
            if item.count("=") != 1 or not item.split("=", 1)[0]:
                raise ValueError("--adapter must be name=path to a coreai directory")
            args += ["--adapter", item]
        # The HTTP process is the Core AI interpreter (it loads the ANE packages). Tokenizing the Jeff
        # chat template needs transformers, which lives in the interpreter that launched forge.py.
        return [coreai_python(), str(ROOT / "scripts" / "jeff_serve.py"), *args], {
            **env, "TOKENIZER_PYTHON": sys.executable}
    args = ["--model", str(a.model), "--state", a.state, "--options", a.options,
            "--instructions", a.instructions]
    for flag, value in (("--build", a.build), ("--ids", a.ids), ("--n-options", a.n_options), ("--cases", a.cases),
                        ("--only", a.only), ("--row", a.row), ("--out", a.out)):
        if value is not None:
            args += [flag, str(value)]
    args += ["--host"] if a.host else []
    args += ["--bench", str(a.bench)] if a.bench else []
    python = coreai_python() if a.build else sys.executable
    return [python, str(ROOT / "scripts" / "jeff_coreai_smoke.py"), *args], env


def coreai_python() -> str:
    """The interpreter with the Core AI conversion SDK and runtime (coreai_torch, coreai.runtime): COREAI_PYTHON, else
    this one. The forge .venv has neither; the SDK ships its own environment."""
    return os.environ.get("COREAI_PYTHON") or sys.executable


def is_jeff_build(build: Path) -> bool:
    try:
        return json.loads((build / "manifest.json").read_text()).get("kind") == "jeff-decision"
    except (OSError, ValueError):
        return False


def prepare(a):
    """Return argv and env overrides without importing ML packages or writing files."""
    if a.command in ("jeff-convert", "jeff-smoke", "jeff-serve", "jeff-train-lora"):
        return prepare_jeff(a)
    if not (a.model / "config.json").is_file():
        raise ValueError(f"Missing checkpoint config: {a.model / 'config.json'}")
    cfg = json.loads((a.model / "config.json").read_text()).get("text_config", {})
    if cfg.get("num_hidden_layers") != 64 or cfg.get("hidden_size") != 5120:
        raise ValueError("This initial port expects the research checkpoint: 64 layers, hidden_size 5120.")
    env = {"MODEL": str(a.model), **NUMERICS}
    # A cache belongs to its checkpoint, never to an unrelated model in ~/Models.
    published_embedding = a.model / "embed_tokens_fp16.npy"
    env["EMBED_NPY"] = str(published_embedding if published_embedding.is_file()
                           else a.model / ".anemll-forge" / "embed_tokens_fp16.npy")
    args = []
    if a.command == "quantize":
        if Path(a.tag).name != a.tag or a.tag in ("", ".", ".."):
            raise ValueError("--tag must be a single directory name")
        for split in ("train", "test"):
            if not (a.wiki / f"wiki2_{split}.txt").is_file():
                raise ValueError(f"Missing wiki2_{split}.txt in {a.wiki}")
        if a.output.exists() and any(a.output.iterdir()):
            raise ValueError("Use a new or empty --output directory for a quantization run")
        env.update(WIKI=str(a.wiki), OUT=str(a.output), TAG=a.tag,
                   FORMAT="vector 2x16 + pcs", METHOD="gptq", BASIS="online", EXPORT="1",
                   MIXER="LUT4 per-tensor + pcs", KV_FMT="INT8 per-channel", HEAD="LUT4 per-tensor + pcs")
        if a.plan:
            plan = json.loads(a.plan.read_text())
            if not isinstance(plan, dict):
                raise ValueError("--plan must contain a JSON object")
            env["PLAN"] = str(a.plan)
        script = "qwen38_gptq_27b.py"
    elif a.command == "convert":
        if not a.export.is_dir():
            raise ValueError(f"Missing export directory: {a.export}")
        # The builder overwrites intermediates; a fresh destination makes that scope explicit.
        dest = a.output / a.export.name
        if dest.exists():
            raise ValueError(f"Build destination already exists; choose a new --output: {dest}")
        env.update(EXPORT_DIR=str(a.export), ANE_OUT=str(a.output), CTX=str(a.ctx),
                   CHUNK_PLAN=",".join(f"{i}-{i+3}" for i in range(0, 64, 4)))
        script, args = "qwen38_ane_model.py", ["build_v3"]
    else:
        if a.ctx <= 0:
            raise ValueError("--ctx must be positive")
        runtime = getattr(a, "runtime", "coreml")
        manifest = a.build / ("manifest.json" if runtime == "coreai" else f"manifest_ctx{a.ctx}_v4.json")
        if not manifest.is_file():
            raise ValueError(f"Missing build manifest: {manifest}")
        if runtime == "coreai":
            model_manifest = json.loads(manifest.read_text())
            contexts = model_manifest.get("ctxs", [])
            cache_format(model_manifest, getattr(a, "kv_cache_dtype", "auto"))
            if a.ctx not in contexts:
                raise ValueError(f"Unsupported Core AI --ctx {a.ctx}; manifest contexts: {contexts}")
        elif (a.build / f"manifest_ctx{a.ctx}_v5.json").exists():
            raise ValueError("This launcher validates v4 builds; a competing v5 manifest would take precedence. "
                             "Use a dedicated v4 build directory or the research runtime directly.")
        env["RUNTIME"] = runtime
        if runtime != "coreai" and getattr(a, "kv_cache_dtype", "auto") in ("v8", "kv8"):
            raise ValueError("V8 KV cache requires the Core AI Swift bridge runtime")
        script = "qwen38_server.py" if a.command == "serve" else "qwen38_chat.py"
        args = ["--hf", str(a.model), "--model-dir", str(a.build), "--ctx", str(a.ctx)]
        if a.command == "serve":
            args += ["--host", a.host, "--port", str(a.port), "--runtime", runtime,
                     "--kv-cache-dtype", a.kv_cache_dtype] + (["--summary-think"] if a.summary_think else [])
            if a.plain:
                if a.drafter:
                    raise ValueError("--drafter cannot be combined with diagnostic --plain")
                args += ["--plain"]
            else:
                if runtime != "coreai":
                    raise ValueError("The release uses Core AI target + drafter; --runtime coreml requires --plain diagnostics")
                pkg, directory = hf_release.drafter_paths(a.build, a.draft, a.drafter)
                hf_release.check_drafter_pair(pkg, directory, json.loads(manifest.read_text()), cfg)
                args += ["--draft", str(pkg), "--drafter", str(directory)]
                env["DRAFTER"] = str(directory)
                env["COREAI_DRAFTER_COMPUTE"] = "ane"
        else:
            if a.prompt is not None:
                args += ["--prompt", a.prompt]
            if a.no_think:
                args += ["--no-think"]
    return [sys.executable, str(ROOT / "scripts" / script), *args], env


def run_recovering(command, env):
    """Run a package-loading child. If it dies by a crash signal while loading a package (MPSGraph aborts on a cached
    specialization it cannot use), purge that package's cache for this Python and retry once: it recompiles."""
    fd, state = tempfile.mkstemp(prefix="anemll-forge-loading-")
    os.close(fd)
    env = {**env, G.STATE_ENV: state}
    try:
        for attempt in range(2):
            rc = subprocess.call(command, env=env, cwd=ROOT)
            pkg = Path(state).read_text().strip()
            if attempt or -rc not in G.CRASH_SIGNALS or not pkg:
                return rc
            n = G.purge(Path(pkg))
            print(f"{G.TAG} crashed (signal {-rc}) while loading {Path(pkg).name}; purged {n} cached specializations "
                  f"for '{G.process_key()}' and retrying once (it recompiles)", file=sys.stderr, flush=True)
    finally:
        os.unlink(state)


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    if a.command in hf_release.COMMANDS:
        try:
            return hf_release.run(a)
        except (ValueError, OSError, RuntimeError) as e:
            p.error(str(e))
    if a.command == "compile":
        if not (a.build / "manifest.json").is_file() and not a.follow:
            p.error(f"Missing build manifest: {a.build / 'manifest.json'}")
        draft = a.draft
        if draft is None:
            try:
                draft, _ = hf_release.drafter_paths(a.build)
            except ValueError:
                draft = None
        # The compile cache is keyed by the loading Python's identity: a Jeff build is loaded by jeff-smoke in the
        # Core AI SDK Python, so it is compiled there too.
        python = coreai_python() if is_jeff_build(a.build) else sys.executable
        command = [python, str(ROOT / "scripts" / "coreai_compile.py"), "--build", str(a.build)]
        command += ["--draft", str(draft)] if draft else []
        command += ["--force"] if a.force else []
        command += ["--follow"] if a.follow else []
        if a.dry_run:
            print(json.dumps(dict(argv=command), indent=2))
            return 0
        if sys.platform != "darwin":
            p.error("Core AI compilation requires macOS")
        env = os.environ.copy()   # before apply(): the child derives and logs the mode (not as an explicit override)
        try:
            SOC.apply(strict=True, log=lambda m: None)
        except SOC.UnsupportedSocError as e:
            p.error(str(e))
        return run_recovering(command, env)
    if a.command == "doctor":
        versions = {}
        for name in ("numpy", "torch", "coremltools", "transformers", "safetensors",
                     "scipy", "scikit-learn", "tokenizers", "ml_dtypes", "coreai-core"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        soc = SOC.detect_soc()
        try:
            mode = SOC.apply(strict=False, soc=soc)
        except SOC.UnsupportedSocError:
            mode = None
        print(json.dumps(dict(python=sys.version, platform=platform.platform(), packages=versions,
                             soc={"class": soc.klass, "generation": soc.generation, "source": soc.source},
                             ane_bonded_compile_mode=mode,
                             note="Presence is not compatibility or ANE-placement verification."), indent=2))
        return 0
    try:
        command, overrides = prepare(a)
    except (ValueError, OSError) as e:
        p.error(str(e))
    if a.command in ("serve", "chat"):
        try:
            SOC.apply(strict=True)
        except SOC.UnsupportedSocError as e:
            p.error(str(e))
        if SOC.MODE_ENV in os.environ:
            overrides[SOC.MODE_ENV] = os.environ[SOC.MODE_ENV]
    if a.dry_run:
        print(json.dumps(dict(argv=command, environment=overrides), indent=2))
        return 0
    linux_ok = a.command in ("quantize", "jeff-smoke", "jeff-train-lora")
    if not linux_ok and sys.platform != "darwin":
        p.error("Core ML/Core AI inference and conversion require macOS for this port "
                "(jeff-smoke host readout and --dry-run run anywhere)")
    env = os.environ.copy()
    # Clear research switches that could silently change the selected recipe or skip layers.
    for key in ("PLAN", "SWEEP", "ONLY", "NLAYERS", "MLP_DS_TABLE", "DBG_MIXER_IN", "DBG_GDN", "DBG_TAPS", "CTX_LADDER"):
        env.pop(key, None)
    env.update(overrides)
    if a.command in ("serve", "chat"):
        return run_recovering(command, env)
    return subprocess.call(command, env=env, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
