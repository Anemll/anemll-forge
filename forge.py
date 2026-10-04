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
    c.add_argument("--dry-run", action="store_true", help="print command only")
    hf_release.add_commands(sub)
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
                q.add_argument("--kv-cache-dtype", choices=("auto", "fp16", "v8"), default="auto",
                               help="require a matching cache export; auto reads its manifest")
                modes = q.add_mutually_exclusive_group()
                modes.add_argument("--draft", type=path, help="Core AI DFlash2 package; default: bundle/drafter/dflash2_lut4_gptq.aimodel")
                modes.add_argument("--plain", action="store_true", help="diagnostic target-only generation")
                q.add_argument("--drafter", type=path, help="drafter config/selector directory; default: package parent")
                q.add_argument("--host", default="127.0.0.1")
                q.add_argument("--port", type=int, default=8765)
            else:
                q.add_argument("--prompt")
                q.add_argument("--no-think", action="store_true")
    return p


def prepare(a):
    """Return argv and env overrides without importing ML packages or writing files."""
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
        if runtime != "coreai" and getattr(a, "kv_cache_dtype", "auto") == "v8":
            raise ValueError("V8 KV cache requires the Core AI Swift bridge runtime")
        script = "qwen38_server.py" if a.command == "serve" else "qwen38_chat.py"
        args = ["--hf", str(a.model), "--model-dir", str(a.build), "--ctx", str(a.ctx)]
        if a.command == "serve":
            args += ["--host", a.host, "--port", str(a.port), "--runtime", runtime,
                     "--kv-cache-dtype", a.kv_cache_dtype]
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
        if not (a.build / "manifest.json").is_file():
            p.error(f"Missing build manifest: {a.build / 'manifest.json'}")
        draft = a.draft
        if draft is None:
            try:
                draft, _ = hf_release.drafter_paths(a.build)
            except ValueError:
                draft = None
        command = [sys.executable, str(ROOT / "scripts" / "coreai_compile.py"), "--build", str(a.build)]
        command += ["--draft", str(draft)] if draft else []
        command += ["--force"] if a.force else []
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
    if a.command != "quantize" and sys.platform != "darwin":
        p.error("Core ML/Core AI inference and conversion require macOS for this port")
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
