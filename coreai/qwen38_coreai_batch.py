"""Build Core AI chunk packages in parallel (source .aimodel only: the ANE compile happens at first load on the M6).
One qwen38_coreai_build.py process per chunk, --jobs at a time, in the given order. Each chunk's info JSON (the
manifest entry) goes to <build>/info/<file>.json and its log to <build>/logs/<file>.log; finished chunks are skipped
on a rerun. A job is "a-b" (chunk_Laa-bb.aimodel) or "name=a-b".
    EXPORT_DIR=~/Models/vq27b/export/mix25in_mixr_lr64mix OUT=~/Models/vq27b/coreai_mixr7 \\
      .venv/bin/python qwen38_coreai_batch.py --ctx 8192,16384,32768,65536 --jobs 3 0-11 cal_L00-07=0-7 24-31 ...
--template <build> writes <build>/manifest.json once every job is built (settings and head package reused from that build,
which must be of the same export).
Memory: a builder holds its layers as fp16 (~2 GB per layer with 8 entries), so order the jobs to keep the largest
chunks from running at the same time."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent


def job_file(job: str) -> tuple[str, int, int, str]:
    name, _, layers = job.rpartition("=")
    lo, hi = map(int, layers.split("-"))
    return name, lo, hi, f"{name or f'chunk_L{lo:02d}-{hi:02d}'}.aimodel"


def write_manifest(build: Path, template: Path, jobs: list[str], ctxs: list[int], pctxs: list[int], export: Path):
    infos = []
    for job in jobs:
        f = build / "info" / f"{job_file(job)[3]}.json"
        if not f.exists():
            print(f"manifest not written: {f.name} missing", flush=True)
            return
        infos.append(json.loads(f.read_text()))
    tmpl = json.loads((template / "manifest.json").read_text())
    cap = lambda c: min(c, 65536 - 64)  # noqa: E731  (qwen38_coreai_build.kv_len: one length per context)
    man = {k: tmpl[k] for k in ("version", "T", "TP", "pend", "taps")}
    infos.sort(key=lambda c: c["layers"][0])
    man.update({"ctxs": ctxs, "pctxs": pctxs, "kv_len": {str(c): cap(c) for c in ctxs},
                "pkv_len": {str(c): cap(c) for c in pctxs}, "export": str(export), "chunks": infos,
                "head": tmpl["head"], "plan": ",".join(f"{c['layers'][0]}-{c['layers'][1]}" for c in infos)})
    head = build / tmpl["head"]["file"]
    if not head.exists():
        shutil.copytree(template / tmpl["head"]["file"], head)
    (build / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"manifest -> {build / 'manifest.json'} ({len(infos)} chunks, head from {template})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs_", nargs="+", metavar="job")
    ap.add_argument("--ctx", default="8192,16384,32768,65536")
    ap.add_argument("--pctx", default=None, help="prefill contexts (default: --ctx)")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--template", default=None, help="build whose manifest settings and head package are reused")
    a = ap.parse_args()
    export = Path(os.path.expanduser(os.environ["EXPORT_DIR"]))
    build = Path(os.path.expanduser(os.environ["OUT"])) / export.name
    (build / "info").mkdir(parents=True, exist_ok=True)
    (build / "logs").mkdir(exist_ok=True)
    pctx = a.pctx or a.ctx

    def run(job: str):
        name, lo, hi, file = job_file(job)
        info_f = build / "info" / f"{file}.json"
        if info_f.exists() and (build / file).exists():
            print(f"{time.strftime('%H:%M:%S')} {file}: already built", flush=True)
            return
        cmd = [sys.executable, str(HERE / "qwen38_coreai_build.py"), "chunk", f"{lo}-{hi}", "--ctx", a.ctx,
               "--pctx", pctx] + (["--name", name] if name else [])
        t = time.time()
        print(f"{time.strftime('%H:%M:%S')} {file}: building layers {lo}-{hi}", flush=True)
        with open(build / "logs" / f"{file}.log", "w") as fh:
            p = subprocess.run(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=fh, text=True)
            fh.write(p.stdout)
        lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
        if p.returncode or not lines:
            print(f"{time.strftime('%H:%M:%S')} {file}: FAILED (exit {p.returncode}), see logs/{file}.log", flush=True)
            return
        info = json.loads(lines[-1])
        info["entries_ctx"] = [[int(x) for x in a.ctx.split(",")], [int(x) for x in pctx.split(",")]]
        info_f.write_text(json.dumps(info, indent=1))
        print(f"{time.strftime('%H:%M:%S')} {file}: done, {info['mb']} MB, {len(info['entries'])} entries, "
              f"{time.time() - t:.0f}s", flush=True)

    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(run, a.jobs_))
    print(f"{time.strftime('%H:%M:%S')} all jobs finished", flush=True)
    if a.template:
        write_manifest(build, Path(os.path.expanduser(a.template)) / export.name, a.jobs_,
                       [int(x) for x in a.ctx.split(",")], [int(x) for x in pctx.split(",")], export)


if __name__ == "__main__":
    main()
