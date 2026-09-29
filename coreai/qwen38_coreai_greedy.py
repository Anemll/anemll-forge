"""Greedy chunk plan for the Core AI build of Qwen3.8-27B: chunks as large as fit one ANE program under a size limit.
From layer s, predict the largest chunk whose compiled ANE program (aned `modelSize`) stays under --limit, build it
(qwen38_coreai_build.py chunk), load it alone in a fresh process while streaming the aned log, run every entry once,
and read the real program size. Too big: one layer fewer, rebuild. Accepted: the next chunk starts after it, so the
plan ends up with a non-uniform number of layers per chunk. A failed load / entry or a program not fully on the ANE
stops the run for a look.
Progress is kept in <build>/greedy_state.json and <build>/manifest.json after every step (rerun to resume; a kernel
panic loses at most the chunk under test). Compile mode: MPSGRAPH_ANE_BONDED_COMPILE_MODE=2 (bonded only).
    .venv/bin/python qwen38_coreai_greedy.py --out ~/Models/vq27b/coreai_mixr6 --ctx 8192,16384,32768,65536
    .venv/bin/python qwen38_coreai_greedy.py probe <package.aimodel>        (the load test alone, prints JSON)
Prediction: per-layer program bytes of the mixr build's 4-layer chunks (aned modelSize, mode 2, entries v8/p64 at
16K/24K), times the largest measured / predicted ratio seen so far (first from a calibration chunk, --calibrate)."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = HERE / ".venv/bin/python"
CACHE = Path.home() / "Library/Caches/coreai-cache"
MODE = "2"
# aned modelSize of coreai_mixr/mix25in_mixr_lr64mix chunk_L00-03 ... chunk_L60-63 (mode 2, 4 entries), 2026-09-28
MIXR_PROG = [481280000, 481296384, 481280000, 481280000, 481280000, 592707584, 974897152, 863502336, 847085568,
             974897152, 974897152, 863469568, 863469568, 974913536, 752041984, 1069940736]
LAYER_PROG = [p / 4 for p in MIXR_PROG for _ in range(4)]
NL = 64


def log(msg: str, f=None):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    if f:
        with open(f, "a") as fh:
            fh.write(line + "\n")


def wired_gb() -> float:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    return next(int(l.split()[-1].rstrip(".")) for l in out.splitlines() if "wired down" in l) * 16384 / 2 ** 30


# ---- probe: one package, alone, in this (fresh) process ------------------------------------------------------------
def probe(pkg: Path) -> dict:
    os.environ["MPSGRAPH_ANE_BONDED_COMPILE_MODE"] = MODE
    sys.path.insert(0, str(HERE / "swift_bridge"))
    import numpy as np
    import coreai_bridge as B
    alog = pkg.parent / f".aned_{pkg.stem}.log"
    ls = subprocess.Popen(["/usr/bin/log", "stream", "--info", "--predicate", 'process == "aned"'],
                          stdout=open(alog, "w"), stderr=subprocess.STDOUT)
    time.sleep(2)
    res = {"package": pkg.name, "entries": {}, "error": None}
    w0, t0 = wired_gb(), time.time()
    try:
        model = B.Model(pkg, compute="ane")
        res["load_s"] = round(time.time() - t0, 1)
        for name in model.function_names:
            t = time.time()
            fn = model.function(name)
            ins = {n: fn.buffer("input", n) for n in fn.input_names}
            outs = {n: fn.buffer("output", n) for n in fn.output_names}
            sts = {n: fn.buffer("state", n) for n in fn.state_names}
            plan = B.Plan([fn.bind(ins, outs, sts)])
            plan.run()
            first = time.time() - t
            ts = []
            for _ in range(5):
                t1 = time.perf_counter()
                plan.run()
                ts.append(time.perf_counter() - t1)
            res["entries"][name] = {"first_s": round(first, 1), "ms": round(float(np.median(ts)) * 1e3, 2)}
            del plan, ins, outs, sts
        res["wired_gb"] = round(wired_gb() - w0, 2)
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    time.sleep(2)
    ls.terminate()
    stats = sorted({(int(a), int(b)) for a, b in re.findall(r"ANE Model Stats\] : modelSize=(\d+) : wiredMemory=(\d+)",
                                                             alog.read_text(errors="replace"))})
    res["ane_programs"] = [{"modelSize": a, "wiredMemory": b} for a, b in stats]
    res["modelSize"] = max((a for a, _ in stats), default=None)
    # placement and compile mode from this process's cached specialization
    digest = (pkg / "main.hash").read_bytes().hex()
    proc = Path(sys.executable).name.replace("_", "-")
    gpu, msgs, modes = 0, set(), set()
    for d in CACHE.glob(f"*/{proc}/{digest}/*/model.aimodelx"):
        for g in d.rglob("*.mpsgraph"):
            b = g.read_bytes()
            gpu += len(set(re.findall(rb"_GPU_region_\d+", b)))
            msgs.update(m.decode() for m in re.findall(rb"Unsupported [ -~]{5,120}", b))
        for mf in d.rglob("manifest.plist"):
            modes.update(int(m) for m in re.findall(rb"aneBondedCompileMode\W+(\d+)", mf.read_bytes()))
    res.update({"gpu_regions": gpu, "unsupported": sorted(msgs), "compile_modes": sorted(modes)})
    return res


# ---- greedy ----------------------------------------------------------------------------------------------------------
def pred(a: int, b: int, ratio: float) -> float:
    return sum(LAYER_PROG[a:b + 1]) * ratio


def largest(a: int, ratio: float, limit: float, cap: int | None = None) -> int:
    b = a
    while b + 1 < NL and pred(a, b + 1, ratio) <= limit and (cap is None or b + 1 <= cap):
        b += 1
    return b


class Run:
    def __init__(self, a):
        self.a = a
        self.out = Path(os.path.expanduser(a.out))
        self.export = Path(os.path.expanduser(a.export))
        self.build = self.out / self.export.name
        self.build.mkdir(parents=True, exist_ok=True)
        self.logf = self.build / "greedy.log"
        self.state_f = self.build / "greedy_state.json"
        self.state = json.loads(self.state_f.read_text()) if self.state_f.exists() else {
            "limit": a.limit, "ctxs": a.ctx, "pctxs": a.pctx, "ratio": None, "calibration": None,
            "accepted": [], "tried": []}
        assert self.state["ctxs"] == a.ctx and self.state["pctxs"] == a.pctx, "state was made with other contexts"

    def save(self):
        self.state_f.write_text(json.dumps(self.state, indent=1))

    def manifest(self):
        tmpl = json.loads((Path(os.path.expanduser(self.a.template)) / "manifest.json").read_text())
        ctxs, pctxs = self.state["ctxs"], self.state["pctxs"]
        cap = lambda c: min(c, 65536 - 64)  # noqa: E731  (qwen38_coreai_build.kv_len: one length per context)
        man = {k: tmpl[k] for k in ("version", "T", "TP", "pend", "taps")}
        man.update({"ctxs": ctxs, "pctxs": pctxs, "kv_len": {str(c): cap(c) for c in ctxs},
                    "pkv_len": {str(c): cap(c) for c in pctxs}, "export": str(self.export),
                    "chunks": [c["info"] for c in self.state["accepted"]], "head": tmpl["head"],
                    "plan": ",".join(f"{c['layers'][0]}-{c['layers'][1]}" for c in self.state["accepted"])})
        (self.build / "manifest.json").write_text(json.dumps(man, indent=1))

    def build_chunk(self, a: int, b: int, name: str | None = None) -> dict:
        env = {**os.environ, "EXPORT_DIR": str(self.export), "OUT": str(self.out)}
        cmd = [str(PY), str(HERE / "qwen38_coreai_build.py"), "chunk", f"{a}-{b}",
               "--ctx", ",".join(map(str, self.state["ctxs"])), "--pctx", ",".join(map(str, self.state["pctxs"]))]
        if name:
            cmd += ["--name", name]
        t = time.time()
        with open(self.build / "greedy_build.log", "a") as fh:
            fh.write(f"\n==== {' '.join(cmd)}\n")
            fh.flush()
            p = subprocess.run(cmd, env=env, cwd=HERE, stdout=subprocess.PIPE, stderr=fh, text=True)
            fh.write(p.stdout)
        if p.returncode:
            raise SystemExit(f"build {a}-{b} failed (exit {p.returncode}); see {self.build / 'greedy_build.log'}")
        info = json.loads([l for l in p.stdout.splitlines() if l.startswith("{")][-1])
        info["entries_ctx"] = [self.state["ctxs"], self.state["pctxs"]]
        log(f"  built {info['file']} ({info['mb']} MB, {len(info['entries'])} entries) in {time.time() - t:.0f}s",
            self.logf)
        return info

    def probe(self, pkg: Path) -> dict:
        env = {**os.environ, "MPSGRAPH_ANE_BONDED_COMPILE_MODE": MODE}
        t = time.time()
        p = subprocess.run([str(PY), str(Path(__file__).resolve()), "probe", str(pkg)], env=env, cwd=HERE,
                           capture_output=True, text=True, timeout=self.a.probe_timeout)
        lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
        if p.returncode or not lines:
            return {"error": f"probe exit {p.returncode}: {(p.stderr or p.stdout)[-600:]}", "modelSize": None}
        r = json.loads(lines[-1])
        r["probe_s"] = round(time.time() - t)
        return r

    def check(self, r: dict) -> str | None:
        """Why a probe result must stop the run (None: fine)."""
        if r.get("error"):
            return r["error"]
        if not r.get("modelSize"):
            return "no ANE program in the aned log (not on the ANE?)"
        if r.get("gpu_regions"):
            return f"{r['gpu_regions']} GPU regions: {r.get('unsupported')}"
        if r.get("compile_modes") and r["compile_modes"] != [int(MODE)]:
            return f"compile modes {r['compile_modes']}"
        return None

    def describe(self, r: dict) -> str:
        e = r.get("entries", {})
        ms = ", ".join(f"{k} {v['ms']}" for k, v in e.items())
        return (f"program {r['modelSize'] / 1e9:.3f} GB, wired {r.get('wired_gb')} GB (all entries' buffers), "
                f"load {r.get('load_s')}s, probe {r.get('probe_s')}s | ms/call: {ms}")

    def calibrate(self, a: int, b: int):
        if self.state["calibration"]:
            return
        name = f"cal_L{a:02d}-{b:02d}"
        log(f"calibration chunk {a}-{b} (predicted {pred(a, b, 1.0) / 1e9:.3f} GB with 4 entries)", self.logf)
        self.build_chunk(a, b, name)
        r = self.probe(self.build / f"{name}.aimodel")
        why = self.check(r)
        if why:
            self.state["tried"].append({"layers": [a, b], "calibration": True, "result": r})
            self.save()
            raise SystemExit(f"calibration {a}-{b} failed: {why}")
        ratio = r["modelSize"] / pred(a, b, 1.0)
        self.state.update({"ratio": ratio, "calibration": {"layers": [a, b], "result": r, "ratio": ratio}})
        self.save()
        log(f"  calibration: {self.describe(r)}; ratio to the 4-entry prediction {ratio:.3f}", self.logf)
        shutil.rmtree(self.build / f"{name}.aimodel", ignore_errors=True)

    def run(self):
        limit, target = self.state["limit"], self.state["limit"] * self.a.margin
        acc = self.state["accepted"]
        s = acc[-1]["layers"][1] + 1 if acc else 0
        cap = None
        while s < NL:
            ratio = self.state["ratio"]
            b = largest(s, ratio, target, cap)
            log(f"chunk from layer {s}: trying {s}-{b} ({b - s + 1} layers), predicted {pred(s, b, ratio) / 1e9:.3f} GB",
                self.logf)
            info = self.build_chunk(s, b)
            pkg = self.build / info["file"]
            r = self.probe(pkg)
            self.state["tried"].append({"layers": [s, b], "predicted": pred(s, b, ratio), "result": r})
            self.save()
            why = self.check(r)
            if why:
                raise SystemExit(f"chunk {s}-{b}: {why}")
            meas = r["modelSize"] / pred(s, b, 1.0)
            self.state["ratio"] = max(ratio, meas)
            if r["modelSize"] >= limit:
                log(f"  too big: {self.describe(r)}; dropping a layer", self.logf)
                shutil.rmtree(pkg, ignore_errors=True)
                cap = b - 1
                self.save()
                continue
            log(f"  accepted {s}-{b}: {self.describe(r)}", self.logf)
            acc.append({"layers": [s, b], "modelSize": r["modelSize"], "info": info, "result": r})
            self.save()
            self.manifest()
            s, cap = b + 1, None
        head = self.build / "head_T8.aimodel"
        if not head.exists():
            shutil.copytree(Path(os.path.expanduser(self.a.template)) / "head_T8.aimodel", head)
        self.manifest()
        plan = ", ".join(f"{c['layers'][0]}-{c['layers'][1]} ({c['modelSize'] / 1e9:.2f} GB)" for c in acc)
        log(f"done: {len(acc)} chunks: {plan}", self.logf)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "probe":
        print(json.dumps(probe(Path(sys.argv[2]).resolve())), flush=True)
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/Models/vq27b/coreai_mixr6")
    ap.add_argument("--export", default="~/Models/vq27b/export/mix25in_mixr_lr64mix")
    ap.add_argument("--template", default="~/Models/vq27b/coreai_mixr/mix25in_mixr_lr64mix",
                    help="build whose manifest settings and head package are reused")
    ap.add_argument("--ctx", default="8192,16384,32768,65536")
    ap.add_argument("--pctx", default=None, help="prefill contexts (default: --ctx)")
    ap.add_argument("--limit", type=float, default=2.0e9, help="max ANE program bytes (aned modelSize)")
    ap.add_argument("--margin", type=float, default=0.97, help="predict up to margin * limit")
    ap.add_argument("--calibrate", default="0-7", help="layers of the calibration chunk ('' = none, ratio 1.1)")
    ap.add_argument("--probe-timeout", type=int, default=5400)
    a = ap.parse_args()
    a.ctx = [int(x) for x in a.ctx.split(",")]
    a.pctx = [int(x) for x in (a.pctx or ",".join(map(str, a.ctx))).split(",")]
    r = Run(a)
    if a.calibrate:
        r.calibrate(*map(int, a.calibrate.split("-")))
    elif r.state["ratio"] is None:
        r.state["ratio"] = 1.1
    r.run()


if __name__ == "__main__":
    main()
