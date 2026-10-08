#!/usr/bin/env python3
"""Compile and run the stock Qwen3.5-0.8B Core AI build. Prefill-only p256 chunks, split LM head.

Waits while another agent's ANE compile is active, then specializes this build.
Decode is one new token through the same 256-row prefill entry (Jeff's packages have no verify function).
"""
from __future__ import annotations

import argparse
import asyncio
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
from coreai.runtime import AIModel
from jeff_prefix import resume_at
from qwen38_coreai_model import _spec, buffer, pick_package, writable
from qwen38_kv_cache import put_rows

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


class StockRunner:
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
        hid = int(cfg["hidden_size"])
        self.hid = hid
        self.nkv, self.hd = int(cfg["num_key_value_heads"]), int(cfg["head_dim"])
        nv = int(cfg["linear_num_value_heads"])
        dk, dv = int(cfg["linear_key_head_dim"]), int(cfg["linear_value_head_dim"])
        cdim = 2 * int(cfg["linear_num_key_heads"]) * dk + nv * dv
        gshapes = {"conv": (self.man["pend"] + 3, cdim), "rec": (nv, dk, dv),
                   "pend": (nv, 3 * self.man["pend"] + 1, dv)}
        rot = int(cfg["head_dim"] * cfg["rope_parameters"]["partial_rotary_factor"])
        self.rot = rot
        self.inv = 1.0 / cfg["rope_parameters"]["rope_theta"] ** (np.arange(0, rot, 2) / rot)
        self.loop = asyncio.new_event_loop()
        t0 = time.time()

        async def load(file, entries):
            target, _ = pick_package(self.build, file, None, print)
            model = await AIModel.load(target, specialization_options=_spec())
            return model, {e: model.load_function(e) for e in entries}

        self.chunks = []
        for ch in self.man["chunks"]:
            t1 = time.time()
            pkg, fns = self.loop.run_until_complete(load(ch["file"], ch["entries"]))
            state = {f"{n}{j}": buffer(gshapes[n]) for j in ch["gdn_j"] for n in gshapes}
            kv = {f"{s}{j}": buffer((self.nkv, self.L, self.hd)) for j in ch["att_j"] for s in ("k", "v")}
            self.chunks.append({**ch, "pkg": pkg, "fns": fns, "state": state, "kv": kv})
            print(f"loaded {ch['file']} in {time.time() - t1:.1f}s", flush=True)
        self.heads = []
        for sl in self.man["head"]["slices"]:
            t1 = time.time()
            pkg, fns = self.loop.run_until_complete(load(sl["file"], [sl["entry"]]))
            self.heads.append({**sl, "pkg": pkg, "fn": fns[sl["entry"]]})
            print(f"loaded {sl['file']} in {time.time() - t1:.1f}s", flush=True)
        self.load_s = time.time() - t0
        P = int(self.man["pend"])
        self.pin = {n: buffer(s) for n, s in (
            ("cos", (self.TP, rot)), ("sin", (self.TP, rot)), ("conv_sel", (3, P + 3)),
            ("commit", (1, P, 1)), ("commit_last", (1, P, 1)), ("conv_sel_out", (3, self.TP + 3)),
            ("valid", (1, self.TP, 1)), ("x", (1, hid, 1, self.TP)), ("xb", (1, hid, 1, self.TP)),
            ("hx", (1, hid, 1, 1)))}
        self.mask = buffer((1, self.L))
        self.pos = 0
        self._token_ids: list[int] = []
        self._rows: np.ndarray | None = None

    def reset(self):
        for ch in self.chunks:
            for _, w in ch["state"].values():
                w[:] = 0
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
                {"state": {k: v[1].copy() for k, v in ch["state"].items()},
                 "kv": {k: v[1].copy() for k, v in ch["kv"].items()}}
                for ch in self.chunks
            ],
        }

    def _restore(self, prefix: dict) -> None:
        chunks = prefix["chunks"]
        for ch, saved in zip(self.chunks, chunks):
            for k, arr in saved["state"].items():
                ch["state"][k][1][:] = arr
            for k, arr in saved["kv"].items():
                ch["kv"][k][1][:] = arr
        self.pos = int(prefix["pos"])
        hidden = prefix.get("hidden")
        self._rows = None if hidden is None else np.asarray(hidden, np.float32).reshape(1, -1)

    def _block(self, ids: list[int]) -> np.ndarray:
        d, TP, n, p0 = self.pin, self.TP, len(ids), self.pos
        if not 0 < n <= TP or p0 + n > self.L:
            raise ValueError(f"{p0 + n} positions exceed the {self.L}-row cache")
        d["x"][1][:] = 0
        d["x"][1][0, :, 0, :n] = self.emb[np.asarray(ids, np.int64)].T
        pos = np.minimum(np.arange(p0, p0 + TP), p0 + n - 1)
        ang = np.concatenate([np.outer(pos, self.inv)] * 2, axis=1)
        d["cos"][1][:], d["sin"][1][:] = np.cos(ang), np.sin(ang)
        d["conv_sel"][1][:] = 0
        d["conv_sel"][1][np.arange(3), np.arange(3)] = 1
        d["commit"][1][:] = 0
        d["commit_last"][1][:] = 0
        d["conv_sel_out"][1][:] = 0
        d["conv_sel_out"][1][np.arange(3), n + np.arange(3)] = 1
        d["valid"][1][:] = 0
        d["valid"][1][0, :n, 0] = 1
        self.mask[1][:] = -1e4
        self.mask[1][0, :p0] = 0
        shared = {k: v[0] for k, v in d.items() if k not in ("x", "xb", "hx")}
        shared["mask"] = self.mask[0]

        async def run():
            x = d["x"][0]
            for ch in self.chunks:
                ins = {**shared, "x": x, **{k: v[0] for k, v in ch["state"].items()},
                       **{k: v[0] for k, v in ch["kv"].items()}}
                out = await ch["fns"][self.entry](inputs=ins)
                for k, (_, w) in ch["state"].items():
                    w[:] = writable(out[f"{k}_out"])
                for k, (_, w) in ch["kv"].items():
                    put_rows(w, out[f"{k}_new"].numpy()[:, :n], p0, n)
                d["xb"][1][:] = writable(out["y"])
                x = d["xb"][0]

        self.loop.run_until_complete(run())
        self.pos = p0 + n
        return np.array(d["xb"][1][0, :, 0, :n].T, dtype=np.float32, copy=True)

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
        self.pin["hx"][1][0, :, 0, 0] = np.asarray(hidden_row, np.float16)
        t1 = time.perf_counter()

        async def run():
            parts = []
            for sl in self.heads:
                out = await sl["fn"](inputs={"x": self.pin["hx"][0]})
                parts.append(np.array(out["logits"].numpy(), dtype=np.float32, copy=True).reshape(-1))
            return parts

        parts = self.loop.run_until_complete(run())
        checked = []
        for sl, part in zip(self.heads, parts):
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


def run_eval(runner: StockRunner, torch_dir: Path, out_dir: Path) -> dict:
    meta = json.loads((torch_dir / "torch_meta.json").read_text())
    torch_logits = np.load(torch_dir / "torch_final_logits.npy")
    torch_pre = np.load(torch_dir / "torch_pre_norm.npy")
    option_ids = {k: int(v) for k, v in meta["option_ids"].items()}
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
            del row["ane_argmax"]
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
            print(f"ane snake {row['i']} pred={pred} label={row['label']}", flush=True)
    n = len(snake)
    # timings on the first parity prompt
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
    # one-token decode on the prefill graph
    runner.reset()
    runner.prefill(ids)
    decode = []
    # continue with the torch final top-1 token, then greedy from the ANE head
    nxt = int(torch_logits[0].argmax())
    for _ in range(8):
        t1 = time.perf_counter()
        block = runner._block([nxt])
        backbone_ms = 1e3 * (time.perf_counter() - t1)
        logits, head_ms = runner.head_logits(block[-1])
        decode.append({"backbone_ms": backbone_ms, "head_ms": head_ms, "token": int(logits.argmax())})
        nxt = int(logits.argmax())
    # short greedy from a fresh prompt, stop at eos or 24 tokens
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
    report = {
        "load_s": round(runner.load_s, 2),
        "parity": [{k: v for k, v in row.items() if k != "ane_argmax"} for row in parity],
        "snake": {
            "n": n,
            "ane_bare_accuracy": snake_correct / n,
            "ane_spaced_accuracy": snake_spaced / n,
            "ane_vs_torch_bare": snake_vs_torch / n,
            "torch_bare_accuracy": meta["snake_accuracy"],
        },
        "timing": {
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
        },
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
    if args.compile:
        if not args.no_wait:
            wait_for_ane()
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
