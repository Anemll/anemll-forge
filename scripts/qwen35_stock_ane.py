#!/usr/bin/env python3
"""Compile and run the stock Qwen3.5-0.8B Core AI build. Prefill-only p256 chunks, split LM head.

Waits while another agent's ANE compile is active, then specializes this build.
Decode is one new token through the same 256-row prefill entry (Jeff's packages have no verify function).
Inference goes through the Swift bridge: each tensor is one IOSurface, bound once and rewritten in place.
"""
from __future__ import annotations

import argparse
import json
import os
import plistlib
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "coreai" / "swift_bridge"))

import ane_compile_mode as SOC
import coreai_bridge as B
import coreai_compile_guide as G
from jeff_prefix import resume_at
from qwen38_coreai_model import pick_package
from qwen38_kv_cache import put_rows

# Host-written tensors shared by every backbone chunk. Per-layer state and KV stay on the chunk.
SHARED_INPUTS = ("cos", "sin", "conv_sel", "commit", "commit_last", "conv_sel_out", "valid", "mask")

OPTION_WORDS = ("up", "down", "left", "right")


def foreign_compile_busy() -> list[str]:
    """Other Core AI compiles and a hot ANE compiler service. This process is ignored."""
    out = subprocess.run(["ps", "-ax", "-o", "pid=,pcpu=,command="], capture_output=True, text=True).stdout
    me = str(os.getpid())
    hits = []
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        pid, cpu, cmd = parts
        if pid == me:
            continue
        if "qwen35_stock" in cmd:
            continue
        if "coreai_compile.py" in cmd or "forge.py compile" in cmd:
            hits.append(f"pid {pid} {cpu}% {cmd[:160]}")
        elif "ANECompilerService" in cmd:
            try:
                hot = float(cpu) >= 25.0
            except ValueError:
                hot = False
            if hot:
                hits.append(f"ANECompilerService pid {pid} {cpu}%")
    return hits


def wait_for_ane(poll_s: float = 20.0) -> None:
    quiet = 0
    while True:
        hits = foreign_compile_busy()
        if not hits:
            quiet += 1
            if quiet >= 2:
                print("ANE compiler is idle", flush=True)
                return
        else:
            quiet = 0
            print("waiting for other ANE work: " + " | ".join(hits), flush=True)
        time.sleep(poll_s)


def compile_build(build: Path) -> None:
    SOC.apply(strict=True)
    man = json.loads((build / "manifest.json").read_text())
    files = [c["file"] for c in man["chunks"]] + [s["file"] for s in man["head"]["slices"]]
    print(f"compiling {len(files)} packages", flush=True)
    t0 = time.time()
    for name in files:
        target, _ = pick_package(build, name, None, print)
        t1 = time.time()
        print(f"compile {name}", flush=True)
        model = B.Model(target, compute="ane")
        del model
        print(f"compiled {name} in {time.time() - t1:.1f}s", flush=True)
    print(f"compile done in {time.time() - t0:.1f}s", flush=True)


def placement_of(package: Path, entries: list[str]) -> dict:
    cache = G.cache_dir(package)
    item = {"package": package.name, "status": "unknown", "ane_regions": 0, "gpu_regions": 0,
            "entries": {}, "cache": str(cache) if cache else None}
    if cache is None or not cache.is_dir():
        item["reason"] = "no cache directory"
        return item
    for mf in sorted(cache.glob("*/model.aimodelx/**/manifest.plist")):
        if ".mpsgraphpackage" not in str(mf):
            continue
        blob_root = mf.read_bytes()
        item["compile_modes"] = sorted(set(int(x) for x in re.findall(rb"aneBondedCompileMode\W+(\d+)", blob_root)))
        versions = plistlib.loads(blob_root).get("Package Version", {})
        ane_total = gpu_total = 0
        found = {}
        for fields in versions.values():
            for module in fields.get("Optimized Modules", {}).values():
                filename = module.get("File Name")
                ane_symbols = set()
                gpu_n = 0
                if filename:
                    graph = (mf.parent / filename).resolve()
                    if graph.is_file():
                        blob = graph.read_bytes()
                        ane_symbols = {x.decode("ascii") for x in re.findall(rb"[A-Za-z0-9_-]+_ANE_region_[A-Za-z0-9_]+", blob)}
                        gpu_n = len(set(re.findall(rb"[A-Za-z0-9_-]+_GPU_region_[A-Za-z0-9_]+", blob)))
                ane_total += len(ane_symbols)
                gpu_total += gpu_n
                attrs = module.get("Entry Function Attributes", {})
                for name, attr in attrs.items():
                    full = (isinstance(attr, list) and "mps.fullyPlacedOnANE" in attr
                            and "mps.noGPUActivity" in attr and any(s.startswith(name + "_ANE_region_") for s in ane_symbols))
                    found[name] = "fully_ane" if full else "other"
        item["ane_regions"] = ane_total
        item["gpu_regions"] = gpu_total
        for entry in entries:
            matches = {n: s for n, s in found.items() if n == entry or n.startswith(entry + "_")}
            if matches and all(s == "fully_ane" for s in matches.values()) and gpu_total == 0 and ane_total:
                item["entries"][entry] = "fully_ane"
            elif matches:
                item["entries"][entry] = "mixed" if gpu_total else "unknown"
            else:
                item["entries"][entry] = "missing"
        if gpu_total:
            item["status"] = "gpu_regions_present"
        elif item["entries"] and all(v == "fully_ane" for v in item["entries"].values()):
            item["status"] = "fully_ane"
        elif ane_total and not gpu_total:
            item["status"] = "ane_regions_cpu_or_unknown_leftover"
        break
    return item


def _as_np(buf: B.Buffer) -> np.ndarray:
    """A numpy view of a bridge buffer. The Buffer stays referenced by the caller, not by the view's owner cycle."""
    return np.asarray(buf)


class StockRunner:
    """Backbone chunks plus the split tied LM head. Bindings are created once and reused."""

    def __init__(self, build: Path, model_dir: Path):
        SOC.apply(strict=True)
        self.build = Path(build)
        self.man = json.loads((self.build / "manifest.json").read_text())
        if self.man.get("kind") != "qwen35-stock":
            raise ValueError(f"{build} is not a qwen35-stock package")
        cfg = json.loads((model_dir / "config.json").read_text())["text_config"]
        self.cfg = cfg
        self.emb = np.load(model_dir / "embed_tokens_fp16.npy")
        self.TP = int(self.man["TP"])
        self.ctx = int(self.man["pctxs"][0])
        self.L = int(self.man["pkv_len"][str(self.ctx)])
        self.entry = f"p{self.TP}_{self.ctx // 1024}k"
        self.hid = int(cfg["hidden_size"])
        self.nkv, self.hd = int(cfg["num_key_value_heads"]), int(cfg["head_dim"])
        rot = int(cfg["head_dim"] * cfg["rope_parameters"]["partial_rotary_factor"])
        self.rot = rot
        self.inv = 1.0 / cfg["rope_parameters"]["rope_theta"] ** (np.arange(0, rot, 2) / rot)
        t0 = time.time()
        self.shared: dict[str, B.Buffer] = {}
        self.shared_np: dict[str, np.ndarray] = {}
        self.chunks = []
        for ch in self.man["chunks"]:
            t1 = time.time()
            self.chunks.append(self._load_chunk(ch))
            print(f"loaded {ch['file']} in {time.time() - t1:.1f}s", flush=True)
        self.heads = []
        self.hx: B.Buffer | None = None
        self.hx_np: np.ndarray | None = None
        for sl in self.man["head"]["slices"]:
            t1 = time.time()
            self.heads.append(self._load_head(sl))
            print(f"loaded {sl['file']} in {time.time() - t1:.1f}s", flush=True)
        self.load_s = time.time() - t0
        self.pos = 0
        self._token_ids: list[int] = []
        self._rows: np.ndarray | None = None

    def _load_model(self, file: str) -> B.Model:
        target, _ = pick_package(self.build, file, None, print)
        return B.Model(target, compute="ane")

    def _shared_input(self, fn: B.Function, name: str) -> B.Buffer:
        spec = fn.inputs[name]
        existing = self.shared.get(name)
        if existing is None:
            buf = fn.buffer("input", name)
            self.shared[name] = buf
            self.shared_np[name] = _as_np(buf)
            return buf
        if (tuple(spec["shape"]) != existing.shape or tuple(spec["strides"]) != existing.strides
                or np.dtype(B.DTYPES[spec["dtype"]][1]) != existing.dtype):
            raise ValueError(f"shared input {name} layout differs across chunks")
        return existing

    def _load_chunk(self, ch: dict) -> dict:
        model = self._load_model(ch["file"])
        fn = model.function(self.entry)
        missing = [n for n in SHARED_INPUTS if n not in fn.inputs]
        if missing or "x" not in fn.inputs or "y" not in fn.outputs:
            raise ValueError(f"{ch['file']} {self.entry} missing {missing or ['x/y']}")
        inputs = {n: self._shared_input(fn, n) for n in SHARED_INPUTS}
        for name in fn.input_names:
            if name not in inputs:
                inputs[name] = fn.buffer("input", name)
        outputs = {n: fn.buffer("output", n) for n in fn.output_names}
        state_names = []
        kv_names = []
        for name in fn.output_names:
            if name == "y":
                continue
            if name.endswith("_out"):
                base = name[:-4]
                state_names.append(base)
            elif name.endswith("_new"):
                base = name[:-4]
                kv_names.append(base)
            else:
                raise ValueError(f"{ch['file']} unexpected output {name}")
            if base not in inputs or base in SHARED_INPUTS or base == "x":
                raise ValueError(f"{ch['file']} output {name} does not map to a per-layer input")
        y = outputs["y"]
        x = inputs["x"]
        if y.shape != x.shape:
            raise ValueError(f"{ch['file']} y {y.shape} != x {x.shape}")
        for name in kv_names:
            if inputs[name].shape[1] != self.L:
                raise ValueError(f"{ch['file']} {name} token axis {inputs[name].shape} != cache {self.L}")
        return {
            **ch,
            "model": model,
            "fn": fn,
            "inputs": inputs,
            "outputs": outputs,
            "in_np": {n: _as_np(b) for n, b in inputs.items()},
            "out_np": {n: _as_np(b) for n, b in outputs.items()},
            "binding": fn.bind(inputs, outputs),
            "state_names": state_names,
            "kv_names": kv_names,
        }

    def _load_head(self, sl: dict) -> dict:
        model = self._load_model(sl["file"])
        fn = model.function(sl["entry"])
        if list(fn.input_names) != ["x"] or "logits" not in fn.outputs:
            raise ValueError(f"{sl['file']} inputs {fn.input_names} outputs {fn.output_names}")
        if self.hx is None:
            self.hx = fn.buffer("input", "x")
            self.hx_np = _as_np(self.hx)
        elif tuple(fn.inputs["x"]["shape"]) != self.hx.shape or tuple(fn.inputs["x"]["strides"]) != self.hx.strides:
            raise ValueError(f"{sl['file']} x layout {fn.inputs['x']} != {self.hx.shape} {self.hx.strides}")
        outputs = {n: fn.buffer("output", n) for n in fn.output_names}
        return {
            **sl,
            "model": model,
            "fn": fn,
            "outputs": outputs,
            "out_np": {n: _as_np(b) for n, b in outputs.items()},
            "binding": fn.bind({"x": self.hx}, outputs),
        }

    def reset(self):
        for ch in self.chunks:
            for name in ch["state_names"]:
                ch["in_np"][name][:] = 0
        self.pos = 0
        self._token_ids = []
        self._rows = None

    def capture_state(self) -> dict:
        if self._rows is None or self.pos <= 0:
            raise RuntimeError("capture_state() needs a completed prefill")
        return {
            "pos": self.pos,
            "token_ids": list(self._token_ids),
            "hidden": self._rows[-1].copy(),
            "chunks": [
                {
                    "state": {k: ch["in_np"][k].copy() for k in ch["state_names"]},
                    "kv": {k: ch["in_np"][k].copy() for k in ch["kv_names"]},
                }
                for ch in self.chunks
            ],
        }

    def _restore(self, prefix: dict) -> None:
        chunks = prefix["chunks"]
        for ch, saved in zip(self.chunks, chunks):
            for k, arr in saved["state"].items():
                ch["in_np"][k][:] = arr
            for k, arr in saved["kv"].items():
                ch["in_np"][k][:] = arr
        self.pos = int(prefix["pos"])
        hidden = prefix.get("hidden")
        self._rows = None if hidden is None else np.asarray(hidden, np.float32).reshape(1, -1)

    def _fill_controls(self, p0: int, n: int) -> None:
        d = self.shared_np
        pos = np.minimum(np.arange(p0, p0 + self.TP), p0 + n - 1)
        ang = np.concatenate([np.outer(pos, self.inv)] * 2, axis=1)
        d["cos"][:], d["sin"][:] = np.cos(ang), np.sin(ang)
        d["conv_sel"][:] = 0
        d["conv_sel"][np.arange(3), np.arange(3)] = 1
        d["commit"][:] = 0
        d["commit_last"][:] = 0
        d["conv_sel_out"][:] = 0
        d["conv_sel_out"][np.arange(3), n + np.arange(3)] = 1
        d["valid"][:] = 0
        d["valid"][0, :n, 0] = 1
        d["mask"][:] = -1e4
        if p0:
            d["mask"][0, :p0] = 0

    def _absorb(self, ch: dict, n: int, p0: int) -> None:
        outs, ins = ch["out_np"], ch["in_np"]
        for name in ch["state_names"]:
            ins[name][:] = outs[f"{name}_out"]
        for name in ch["kv_names"]:
            put_rows(ins[name], outs[f"{name}_new"][:, :n], p0, n)

    def _block(self, ids: list[int]) -> np.ndarray:
        n, p0 = len(ids), self.pos
        if not 0 < n <= self.TP or p0 + n > self.L:
            raise ValueError(f"{p0 + n} positions exceed the {self.L}-row cache")
        x = self.chunks[0]["in_np"]["x"]
        x[:] = 0
        x[0, :, 0, :n] = self.emb[np.asarray(ids, np.int64)].T
        self._fill_controls(p0, n)
        prev_y = None
        for ch in self.chunks:
            if prev_y is not None:
                np.copyto(ch["in_np"]["x"], prev_y)
            B.run([ch["binding"]])
            self._absorb(ch, n, p0)
            prev_y = ch["out_np"]["y"]
        self.pos = p0 + n
        return np.array(prev_y[0, :, 0, :n].T, dtype=np.float32, copy=True)

    def prefill(self, token_ids: list[int], prefix: dict | None = None) -> dict:
        ids = [int(t) for t in token_ids]
        start = resume_at(prefix, ids)
        if prefix is None:
            self.reset()
            rows = []
        else:
            self._restore(prefix)
            rows = [] if self._rows is None else [self._rows]
        calls = []
        for i in range(start, len(ids), self.TP):
            t1 = time.perf_counter()
            block = self._block(ids[i:i + self.TP])
            calls.append(1e3 * (time.perf_counter() - t1))
            rows.append(block)
        self._token_ids = ids
        if not rows:
            raise ValueError("prefix covers the prompt but has no hidden state")
        self._rows = np.concatenate(rows, 0) if len(rows) > 1 or rows[0].ndim == 2 else rows[0]
        if self._rows.ndim == 1:
            self._rows = self._rows.reshape(1, -1)
        return {"calls_ms": calls, "hidden_rows": self._rows, "prefix_tokens": start}

    def head_logits(self, hidden_row: np.ndarray) -> tuple[np.ndarray, float]:
        if self.hx_np is None:
            raise RuntimeError("LM head is not loaded")
        self.hx_np[:] = 0
        self.hx_np[0, :, 0, 0] = np.asarray(hidden_row, np.float16)
        t1 = time.perf_counter()
        B.run([sl["binding"] for sl in self.heads])
        checked = []
        for sl in self.heads:
            part = np.array(sl["out_np"]["logits"], dtype=np.float32, copy=True).reshape(-1)
            if part.shape[0] != int(sl["rows"]):
                raise ValueError(f"{sl['file']} logits {part.shape[0]} != {sl['rows']}")
            checked.append(part)
        logits = np.concatenate(checked, 0)
        vocab = int(self.man["head"]["vocab"])
        if logits.shape[0] != vocab:
            raise ValueError(f"LM head logits {logits.shape[0]} != vocab {vocab}")
        return logits, 1e3 * (time.perf_counter() - t1)

    def prefill_logits(self, token_ids: list[int], all_positions: bool = False) -> dict:
        t0 = time.perf_counter()
        pre = self.prefill(token_ids)
        rows = pre["hidden_rows"]
        if all_positions:
            argmax = np.empty(len(rows), np.int32)
            head_ms = []
            final = None
            for i, row in enumerate(rows):
                logits, ms = self.head_logits(row)
                argmax[i] = int(logits.argmax())
                head_ms.append(ms)
                final = logits
        else:
            final, ms = self.head_logits(rows[-1])
            argmax = np.array([int(final.argmax())], np.int32)
            head_ms = [ms]
        return {
            "final_logits": final,
            "argmax": argmax,
            "pre_norm": rows[-1].astype(np.float32),
            "calls_ms": pre["calls_ms"],
            "head_ms": head_ms,
            "total_ms": 1e3 * (time.perf_counter() - t0),
            "n": len(token_ids),
        }


def kl_ref_est(ref: np.ndarray, est: np.ndarray) -> float:
    r = np.asarray(ref, np.float64)
    e = np.asarray(est, np.float64)
    r = r - r.max()
    e = e - e.max()
    pr = np.exp(r)
    pe = np.exp(e)
    pr /= pr.sum()
    pe /= pe.sum()
    return float(np.sum(pr * (np.log(pr + 1e-30) - np.log(pe + 1e-30))))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, np.float64).ravel()
    y = np.asarray(b, np.float64).ravel()
    return float(x.dot(y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30))


def score_words(logits: np.ndarray, ids: dict[str, int], surfaces: list[str]) -> str:
    vals = [float(logits[ids[s]]) for s in surfaces]
    return OPTION_WORDS[int(np.argmax(vals))]


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj))


def run_eval(runner: StockRunner, torch_dir: Path, out_dir: Path) -> dict:
    meta = json.loads((torch_dir / "torch_meta.json").read_text())
    torch_logits = np.load(torch_dir / "torch_final_logits.npy")
    torch_pre = np.load(torch_dir / "torch_pre_norm.npy")
    option_ids = {k: int(v) for k, v in meta["option_ids"].items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = meta["prompts"][0]["input_ids"]
    runner.reset()
    t0 = time.perf_counter()
    cold = runner.prefill(ids)
    cold_ms = 1e3 * (time.perf_counter() - t0)
    warm = []
    for _ in range(5):
        runner.reset()
        t1 = time.perf_counter()
        runner.prefill(ids)
        warm.append(1e3 * (time.perf_counter() - t1))
    prefix = runner.capture_state()
    t1 = time.perf_counter()
    hit = runner.prefill(ids, prefix=prefix)
    prefix_ms = 1e3 * (time.perf_counter() - t1)
    runner.reset()
    runner.prefill(ids)
    decode = []
    nxt = int(torch_logits[0].argmax())
    for _ in range(8):
        t1 = time.perf_counter()
        block = runner._block([nxt])
        backbone_ms = 1e3 * (time.perf_counter() - t1)
        logits, head_ms = runner.head_logits(block[-1])
        decode.append({"backbone_ms": backbone_ms, "head_ms": head_ms, "token": int(logits.argmax())})
        nxt = int(logits.argmax())
    gen_ids = list(meta["prompts"][1]["input_ids"])
    eos = meta.get("eos_token_id")
    runner.reset()
    runner.prefill(gen_ids)
    produced = []
    hidden = runner._rows[-1]
    for _ in range(24):
        logits, _ms = runner.head_logits(hidden)
        tok = int(logits.argmax())
        produced.append(tok)
        if eos is not None and tok == eos:
            break
        hidden = runner._block([tok])[-1]
    timing = {
        "prompt": meta["prompts"][0]["name"],
        "tokens": len(ids),
        "cold_prefill_ms": round(cold_ms, 2),
        "cold_calls_ms": [round(x, 2) for x in cold["calls_ms"]],
        "cached_prefill_ms": [round(x, 2) for x in warm],
        "cached_prefill_median_ms": round(float(np.median(warm)), 2),
        "prefix_hit_ms": round(prefix_ms, 2),
        "prefix_hit_calls": len(hit["calls_ms"]),
        "decode_steps": [{k: round(v, 2) if isinstance(v, float) else v for k, v in s.items()} for s in decode],
        "decode_backbone_median_ms": round(float(np.median([s["backbone_ms"] for s in decode])), 2),
        "decode_head_median_ms": round(float(np.median([s["head_ms"] for s in decode])), 2),
        "decode_note": "Each decode step is one new token on the p256_2k prefill entry plus 16 LM-head slices.",
    }
    print(json.dumps(timing, indent=2)[:4000], flush=True)
    _write_json(out_dir / "timing_partial.json", timing)
    parity = []
    for i, prompt in enumerate(meta["prompts"]):
        ids = prompt["input_ids"]
        print(f"ane parity {prompt['name']} n={len(ids)}", flush=True)
        got = runner.prefill_logits(ids, all_positions=True)
        ref_argmax = None
        agree = None
        # torch argmax stored in the meta? only npy object. Recompute from nothing:
        # The torch script saved argmax in an object npy. Load it if present.
        parity.append({
            "name": prompt["name"],
            "tokens": len(ids),
            "kl_fp32_ane": kl_ref_est(torch_logits[i], got["final_logits"]),
            "final_top1_ane": int(got["final_logits"].argmax()),
            "final_top1_torch": int(torch_logits[i].argmax()),
            "final_top1_match": bool(got["final_logits"].argmax() == torch_logits[i].argmax()),
            "pre_norm_cosine": cosine(torch_pre[i], got["pre_norm"]),
            "ane_argmax": got["argmax"].tolist(),
            "calls_ms": [round(x, 2) for x in got["calls_ms"]],
            "head_ms_sum": round(sum(got["head_ms"]), 2),
        })
        agree = parity[-1]
        print(json.dumps({k: agree[k] for k in ("name", "kl_fp32_ane", "final_top1_match", "pre_norm_cosine")}),
              flush=True)
    argmax_path = torch_dir / "torch_argmax.npy"
    if argmax_path.is_file():
        ref_rows = np.load(argmax_path, allow_pickle=True)
        for i, row in enumerate(parity):
            ref = np.asarray(ref_rows[i], np.int32)
            ane = np.asarray(row["ane_argmax"], np.int32)
            n = min(len(ref), len(ane))
            match = ref[:n] == ane[:n]
            row["per_position_top1"] = float(match.mean()) if n else None
            row["per_position_n"] = int(n)
            print(f"per-position top-1 {row['name']} {row['per_position_top1']:.4f} n={n}", flush=True)
            del row["ane_argmax"]
    parity_out = [{k: v for k, v in row.items() if k != "ane_argmax"} for row in parity]
    _write_json(out_dir / "parity_partial.json", {"load_s": round(runner.load_s, 2), "parity": parity_out,
                                                   "timing": timing, "generation": {"prompt": meta["prompts"][1]["name"],
                                                                                    "new_token_ids": produced}})
    snake = meta["snake"]
    snake_correct = 0
    snake_spaced = 0
    snake_vs_torch = 0
    for row in snake:
        got = runner.prefill_logits(row["input_ids"], all_positions=False)
        pred = score_words(got["final_logits"], option_ids, list(OPTION_WORDS))
        pred_sp = score_words(got["final_logits"], option_ids, [" " + w for w in OPTION_WORDS])
        snake_correct += int(pred == row["label"])
        snake_spaced += int(pred_sp == row["label"])
        snake_vs_torch += int(pred == row["pred"])
        if row["i"] % 32 == 0:
            done = row["i"] + 1
            print(f"ane snake {row['i']} pred={pred} label={row['label']} "
                  f"acc={snake_correct / done:.3f}", flush=True)
            _write_json(out_dir / "snake_partial.json", {
                "done": done, "correct": snake_correct, "spaced": snake_spaced, "vs_torch": snake_vs_torch,
            })
    n = len(snake)
    report = {
        "load_s": round(runner.load_s, 2),
        "parity": parity_out,
        "snake": {
            "n": n,
            "ane_bare_accuracy": snake_correct / n,
            "ane_spaced_accuracy": snake_spaced / n,
            "ane_vs_torch_bare": snake_vs_torch / n,
            "torch_bare_accuracy": meta["snake_accuracy"],
        },
        "timing": timing,
        "generation": {"prompt": meta["prompts"][1]["name"], "new_token_ids": produced},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ane_report.json").write_text(json.dumps(report))
    print(json.dumps({k: report[k] for k in ("snake", "timing")}, indent=2)[:4000], flush=True)
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, default=Path("/Users/anemll/Models/qwen35-0.8b-stock-coreai/coreai"))
    p.add_argument("--model", type=Path, default=Path("/Users/anemll/Models/qwen35-0.8b-stock-coreai/model"))
    p.add_argument("--torch", type=Path, default=Path("/Users/anemll/Models/qwen35-0.8b-stock-coreai/eval"))
    p.add_argument("--compile", action="store_true")
    p.add_argument("--placement-only", action="store_true")
    p.add_argument("--no-wait", action="store_true")
    args = p.parse_args(argv)
    build = args.build.expanduser().resolve()
    if (args.compile or not args.placement_only) and not args.no_wait:
        wait_for_ane()
    if args.compile:
        compile_build(build)
    man = json.loads((build / "manifest.json").read_text())
    place = [placement_of(build / c["file"], c["entries"]) for c in man["chunks"]]
    place += [placement_of(build / s["file"], [s["entry"]]) for s in man["head"]["slices"]]
    dest = args.torch.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "placement.json").write_text(json.dumps(place, indent=2))
    print(json.dumps([{k: r[k] for k in ("package", "status", "ane_regions", "gpu_regions")} for r in place], indent=2),
          flush=True)
    if args.placement_only:
        return 0
    runner = StockRunner(build, args.model.expanduser().resolve())
    run_eval(runner, dest, dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
