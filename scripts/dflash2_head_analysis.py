"""How accurate is each drafter's LUT lm_head on the ANE? Runs one drafter (Core ML or Core AI, chosen compute units) on
fixed seeded inputs, saves its final hidden rows 1..7 and logits per cycle; `analyze` applies the exact dequantized LUT
head in fp32 to each run's OWN hidden rows and compares with the logits that run produced (the head's error alone).
    python dflash2_head_analysis.py collect coreml CPU_AND_NE|CPU_ONLY
    python dflash2_head_analysis.py collect coreai ane|gpu|cpu
    python dflash2_head_analysis.py analyze"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

E = os.path.expanduser
os.environ.setdefault("DRAFT_EXPORT", E("~/Models/dflash2/export/drafter_lut4_gptq_q7_cal"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "scripts"))
import ane_compile_mode  # noqa: E402
ane_compile_mode.apply(log=lambda m: None)  # the SoC's bonded compile mode (M6: 2, M5: 1) unless set explicitly
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(E("~/Models/vq27b/tests/head_analysis"))
COREML_PKG = Path(E("~/Models/dflash2/ane/gptq_q7_cal/dflash2_lut4_gptq.mlpackage"))
COREAI_PKG = Path(E("~/Models/dflash2/coreai/dflash2_lut4_gptq.aimodel"))
CYCLES, N_CTX = 12, 150


def collect(kind, units):
    torch.set_num_threads(1)
    import dflash2_ane_drafter as D
    cfg = json.loads((D.DRAFTER / "config.json").read_text())
    emb = D.LazyEmbedding()
    if kind == "coreml":
        import coremltools as ct
        d = D.AneDrafter(COREML_PKG, cfg, D.load_codebooks(), emb, getattr(ct.ComputeUnit, units))
    else:
        os.environ["COREAI_DRAFTER_COMPUTE"] = units
        from dflash2_coreai_drafter import CoreAIDrafter
        d = CoreAIDrafter(COREAI_PKG, cfg, D.load_codebooks(), emb)
    gen = torch.Generator().manual_seed(0)
    chan = torch.exp(torch.randn(25600, generator=gen) * 0.7)
    d.add_context((torch.randn(N_CTX, 25600, generator=gen) * chan).half().numpy(), np.arange(N_CTX))
    anchors = [9707, 1118, 13, 279, 5, 25, 3017, 220, 11, 1532, 510, 198]
    hs, ls = [], []
    p = N_CTX
    for c in range(CYCLES):
        _, info = d.propose(anchors[c], p)
        hs.append(info["hidden"][1:].astype(np.float32))                    # rows 1..7 (the head's input)
        ls.append(np.asarray(info["logits"], np.float32))
        k = 3
        d.add_context((torch.randn(k, 25600, generator=gen) * chan).half().numpy(), np.arange(p, p + k))
        p += k
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{kind}_{units}.npz", hidden=np.stack(hs), logits=np.stack(ls))
    print(f"saved {kind} {units}: hidden {np.stack(hs).shape}, logits {np.stack(ls).shape}", flush=True)


def analyze():
    torch.set_num_threads(8)
    import dflash2_ane_drafter as D
    w = D.head_dequant_fp16()                                                  # (V, 5120) exact dequantized LUT head
    print(f"{'run':22s} {'logit cos':>9s} {'rel err':>8s} {'max|err|':>9s} {'top1 agree':>10s} {'top16 overlap':>13s} {'max|logit|':>10s}")
    for f in sorted(OUT.glob("*.npz")):
        z = np.load(f)
        h = torch.from_numpy(z["hidden"].reshape(-1, 5120))
        lg = torch.from_numpy(z["logits"].reshape(-1, z["logits"].shape[-1]))
        ref = torch.empty_like(lg)
        for a in range(0, w.shape[0], 32768):
            ref[:, a:a + 32768] = h @ w[a:a + 32768].float().T
        cos = torch.nn.functional.cosine_similarity(ref, lg, dim=-1)
        err = (lg - ref).abs()
        t1 = (ref.argmax(-1) == lg.argmax(-1)).float().mean()
        tr, tl = torch.topk(ref, 16).indices, torch.topk(lg, 16).indices
        ov = np.mean([len(set(a.tolist()) & set(b.tolist())) / 16 for a, b in zip(tr, tl)])
        print(f"{f.stem:22s} {float(cos.mean()):9.5f} {float((lg - ref).norm() / ref.norm()):8.4f} {float(err.max()):9.3f} "
              f"{float(t1):10.3f} {ov:13.3f} {float(ref.abs().max()):10.2f}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect(sys.argv[2], sys.argv[3])
    else:
        analyze()
