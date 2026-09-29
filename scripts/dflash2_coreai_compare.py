"""Core ML vs Core AI DFlash2 drafter.
    check                torch reference (same GPTQ weights, dequantized) vs the Core ML and Core AI drafters on random
                         target-like features: hidden cosine, top-16 overlap, drafted tokens, per cycle
    replay coreml|coreai exact greedy acceptance on saved target traces (TRACES), draft-call time, memory (system wired
                         and the aned program size); run each in its own process
env: COREML_PKG, COREAI_PKG, TRACES, DRAFT_EXPORT (GPTQ export the Core ML package was built from)"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

E = os.path.expanduser
os.environ.setdefault("DRAFT_EXPORT", E("~/Models/dflash2/export/drafter_lut4_gptq_q7_cal"))
os.environ.setdefault("MPSGRAPH_ANE_BONDED_COMPILE_MODE", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent))
COREML_PKG = Path(E(os.environ.get("COREML_PKG", "~/Models/dflash2/ane/gptq_q7_cal/dflash2_lut4_gptq.mlpackage")))
COREAI_PKG = Path(E(os.environ.get("COREAI_PKG", "~/Models/dflash2/coreai/dflash2_lut4_gptq.aimodel")))
TRACES = Path(E(os.environ.get("TRACES", "~/Models/dflash2/traces/traces_bf16.npz")))


def wired_gb():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    return next(int(l.split()[-1].rstrip(".")) for l in out.splitlines() if "wired down" in l) * 16384 / 2 ** 30


def check():
    import dflash2_ane_drafter as D
    from dflash2_coreai_drafter import CoreAIDrafter
    from dflash2_drafter_ref import DFlash2Drafter, TargetShared
    cfg, w = D.load_drafter(D.DRAFTER, torch.bfloat16)
    _, deq = D.get_quant(cfg, w)
    del w
    scale = json.loads(COREAI_PKG.with_suffix(".json").read_text()).get("mask_scale", 1.0)
    os.environ["MASK_SCALE"] = str(scale)
    import dflash2_ref_replay as RR   # reference with the Core AI drafter's mask-row scale (Core ML drafter: 1.0)
    ref = RR.ScaledDrafter(cfg, deq)
    print(f"reference: mask scale {scale}, head {D.HEAD_EXPORT.parent.name}", flush=True)
    shared = TargetShared(D.MODEL, head=D.head_dequant_fp16())
    emb = shared.emb.to(torch.float16).numpy()
    ml = D.AneDrafter(COREML_PKG, cfg, deq, emb)
    ai = CoreAIDrafter(COREAI_PKG, cfg, deq, emb)
    gen = torch.Generator().manual_seed(0)
    chan = torch.exp(torch.randn(25600, generator=gen) * 0.7)
    ctx = ref.new_context()
    n0 = int(os.environ.get("CHECK_CTX", "150"))
    f = (torch.randn(n0, 25600, generator=gen) * chan).half()
    ref.add_context(ctx, f.float(), torch.arange(n0))
    for d in (ml, ai):
        d.add_context(f.numpy(), np.arange(n0))
    p, anchor = n0, 9707
    for cyc in range(int(os.environ.get("CYCLES", "6"))):
        toks_r, info_r = ref.propose(anchor, p, ctx, shared)
        hr = info_r["hidden"]
        lr = shared.head(hr)
        top_r = torch.topk(lr, 16).indices
        line = [f"cycle {cyc} @{p}"]
        res = {}
        for tag, d in (("coreml", ml), ("coreai", ai)):
            toks, info = d.propose(anchor, p)
            h = torch.from_numpy(info["hidden"][1:])
            cos = torch.nn.functional.cosine_similarity(hr, h, dim=-1)
            top = np.mean([len(set(a.tolist()) & set(b.tolist())) / 16
                           for a, b in zip(top_r, torch.topk(info["logits"], 16).indices)])
            res[tag] = toks
            match = sum(int(a == b) for a, b in zip(toks_r.tolist(), toks))
            line.append(f"{tag}: hidden cos min {float(cos.min()):.4f} rel err {float((h - hr).norm() / hr.norm()):.4f} "
                        f"top16 {top:.3f} tokens {match}/7")
        line.append(f"coreai==coreml tokens {sum(int(a == b) for a, b in zip(res['coreml'], res['coreai']))}/7")
        print(" | ".join(line), flush=True)
        k = 1 + cyc % 4 * 2
        fk = (torch.randn(k, 25600, generator=gen) * chan).half()
        ref.add_context(ctx, fk.float(), torch.arange(p, p + k))
        for d in (ml, ai):
            d.add_context(fk.numpy(), np.arange(p, p + k))
        p, anchor = p + k, int(toks_r[k - 1]) if k <= 7 else 55


def replay(which):
    import dflash2_ane_drafter as D
    from dflash2_target_ref import summarize
    alog = Path(E(f"~/Models/vq27b/tests/.aned_drafter_{which}.log"))
    ls = subprocess.Popen(["/usr/bin/log", "stream", "--info", "--predicate", 'process == "aned"'],
                          stdout=open(alog, "w"), stderr=subprocess.STDOUT)
    time.sleep(2)
    cfg = json.loads((D.DRAFTER / "config.json").read_text())
    w0 = wired_gb()
    t0 = time.time()
    if which == "coreml":
        d = D.AneDrafter(COREML_PKG, cfg, D.load_codebooks(), D.LazyEmbedding())
    else:
        from dflash2_coreai_drafter import CoreAIDrafter
        d = CoreAIDrafter(COREAI_PKG, cfg, D.load_codebooks(), D.LazyEmbedding())
    load_s, w_load = time.time() - t0, wired_gb() - w0
    traces = np.load(TRACES)
    n = len([k for k in traces.files if k.startswith("tokens_")])
    ms, t_call, t_ctx = [], [], []
    w_run = None
    for j in range(n):
        toks, plen = traces[f"tokens_{j}"], int(traces[f"plen_{j}"])
        feats = traces[f"feats_{j}"].reshape(-1, 25600)
        d.reset()
        t1 = time.perf_counter()
        d.add_context(feats[:plen], np.arange(plen))
        t_ctx.append((time.perf_counter() - t1) / max(1, (plen - 8) / 64))
        p = plen
        while p + 8 <= len(toks) and p + 8 <= feats.shape[0]:
            t1 = time.perf_counter()
            dr, _ = d.propose(int(toks[p]), p)
            t_call.append(time.perf_counter() - t1)
            m = 0
            while m < 7 and dr[m] == toks[p + 1 + m]:
                m += 1
            d.add_context(feats[p:p + m + 1], np.arange(p, p + m + 1))
            ms.append(m)
            p += m + 1
        if w_run is None:
            w_run = wired_gb() - w0
    time.sleep(1)
    ls.terminate()
    stats = sorted({(int(a), int(b)) for a, b in re.findall(r"modelSize=(\d+) : wiredMemory=(\d+)", alog.read_text(errors="replace"))})
    res = {"drafter": which, "pkg": str(COREML_PKG if which == "coreml" else COREAI_PKG), "load_s": round(load_s, 1),
           "wired_gb_after_load": round(w_load, 2), "wired_gb_running": round(w_run, 2),
           "ane_programs_mb": [(round(a / 1e6, 1), round(b / 1e6, 1)) for a, b in stats],
           "propose_ms_median": round(1e3 * float(np.median(t_call)), 2), "propose_ms_p90": round(1e3 * float(np.percentile(t_call, 90)), 2),
           "ctx_ms_per_64_rows": round(1e3 * float(np.median(t_ctx)), 2), **summarize(ms)}
    print(json.dumps(res), flush=True)
    with open(E("~/Models/vq27b/tests/drafter_compare.jsonl"), "a") as fh:
        fh.write(json.dumps({"time": time.strftime("%m-%d %H:%M"), **res}) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check()
    else:
        replay(sys.argv[2])
