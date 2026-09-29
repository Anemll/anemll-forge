"""Mean KL divergence D_KL(p_bf16 || p_quant) of quantized Qwen3.8-27B variants on a self-generated in-domain trace.

    generate   bf16 model answers PROMPTS (thinking on) -> trace.npz (token ids, prompt lengths)
    reference  bf16 teacher-forced pass over the trace -> ref.npz: per position the top-K log-probs + ids
    eval       same pass with quantized weights from an export dir (qwen38_gptq_27b.py EXPORT), dequantized
               (MLP rotation folded back, mixers, INT8 K/V, lm_head) -> kl_<tag>.json
    eval without EXPORT_DIR: bf16 against the reference = run-to-run numerical noise floor

KL over the top-K reference tokens plus one tail bucket (the tail's contribution is below 1e-4 at K=256).
Size axis (as in the reference chart): quantized weight bytes excluding embeddings, including the output head.

    TRACE=/path/to/data/vq27b/kl python qwen38_kl.py generate
    python qwen38_kl.py reference
    EXPORT_DIR=/path/to/data/vq27b/runs/export/full_mix25_mixer4_head4 python qwen38_kl.py eval
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from scipy.linalg import hadamard

MODEL = Path(os.environ.get("MODEL", "/path/to/data/Qwen3.8-27B"))
TRACE = Path(os.environ.get("TRACE", "/path/to/data/vq27b/kl"))
EXPORT_DIR = Path(os.environ["EXPORT_DIR"]) if os.environ.get("EXPORT_DIR") else None
# ablation: apply only these parts of the export (mlp, attn = full-attention projections, gdn = DeltaNet
# projections, head), optionally only in layers QLAYERS="lo-hi" (inclusive); the rest stays bf16
PARTS = set(os.environ.get("PARTS", "mlp,attn,gdn,head").split(","))
QLAYERS = tuple(int(x) for x in os.environ["QLAYERS"].split("-")) if os.environ.get("QLAYERS") else (0, 10 ** 9)
# LR_RANK=r: add the rank-r SVD of the quantization error (W_bf16 - W_q) to every applied matrix (closed-form
# low-rank / LoRA-style error compensation, fp16 A @ B next to the LUT weight on the ANE)
LR_RANK = int(os.environ.get("LR_RANK", "0"))
LR_PARTS = set(os.environ.get("LR_PARTS", "mlp,attn,gdn").split(","))  # parts that get the low-rank correction


def exported_lowrank(t, key, w):
    """w + lr_a @ lr_b when the export carries trained low-rank factors for this matrix (qwen38_blockrecon.py)."""
    if f"{key}.lr_a" in t:
        return w + t[f"{key}.lr_a"].float() @ t[f"{key}.lr_b"].float()
    return w


def with_lowrank(orig, q, part="mlp"):
    """q (fp32, CPU) + the best rank-LR_RANK approximation of orig - q (for parts in LR_PARTS)."""
    if not LR_RANK or part not in LR_PARTS:
        return q
    q = q.float().cpu()
    e = orig.float().cpu() - q                                          # CPU: QR / SVD support on MPS varies
    u, sv, v = torch.svd_lowrank(e, q=LR_RANK + 16, niter=4)
    return q + (u[:, :LR_RANK] * sv[:LR_RANK]) @ v[:, :LR_RANK].T
TOPK = int(os.environ.get("TOPK", "256"))
MAX_NEW = int(os.environ.get("MAX_NEW", "768"))
BATCH = int(os.environ.get("GEN_BATCH", "16"))
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.set_grad_enabled(False)

PROMPTS = [
    # code
    "Write a Python function that returns the longest palindromic substring of a string, with a short explanation.",
    "Explain the difference between a process and a thread, with an example in C.",
    "Find the bug: `def mean(xs): return sum(xs) / len(xs) if xs else 0` fails for generators. Fix it.",
    "Write a SQL query that returns the top 3 customers by total order value per country.",
    "Implement an LRU cache in Rust with O(1) get and put.",
    "What does `git rebase --onto` do? Give a concrete example.",
    "Write a bash one-liner that finds the 10 largest files under the current directory.",
    "Explain Python's GIL and when multiprocessing is better than threading.",
    "Convert this JavaScript to TypeScript with proper types: function merge(a, b) { return {...a, ...b}; }",
    "Write a CUDA kernel outline for a row-wise softmax and explain the memory access pattern.",
    "Review this code for security issues: `os.system('ping ' + request.args['host'])`.",
    "Implement binary search over a rotated sorted array in Go.",
    "Explain how a Swift actor prevents data races.",
    "Write a Core ML conversion script outline for a PyTorch image classifier.",
    "What is the time complexity of building a heap from n elements, and why?",
    "Write a regular expression that validates an IPv4 address and explain it.",
    # math / reasoning
    "A train leaves at 3:15 pm going 80 km/h; another leaves the same station at 4:00 pm going 110 km/h. When does the second catch up?",
    "Prove that the square root of 2 is irrational.",
    "How many ways can 8 rooks be placed on a chessboard so that none attack each other?",
    "If a fair coin is flipped 10 times, what is the probability of at least 7 heads?",
    "Solve for x: 3^(2x) - 10*3^x + 9 = 0.",
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much is the ball?",
    "What is the expected number of rolls of a fair die to see all six faces?",
    "Integrate x * e^(2x) dx.",
    "Three friends split a bill; one pays twice as much as another, and the third pays $10 more than the first. The bill is $90. Who pays what?",
    "Explain Bayes' theorem with a medical test example (1% prevalence, 95% sensitivity, 90% specificity).",
    "Is 2^61 - 1 prime? Explain how you would check.",
    "Find the eigenvalues of [[2, 1], [1, 2]] and interpret them.",
    # science / knowledge
    "Why is the sky blue but sunsets red?",
    "Explain how mRNA vaccines work.",
    "What limits the clock speed of modern CPUs?",
    "Explain transformer attention to a software engineer who knows linear algebra.",
    "What is the difference between weather and climate?",
    "How does a lithium-ion battery degrade over time?",
    "Explain quantization of neural network weights and why 4-bit models can still work well.",
    "What is the Apple Neural Engine and how does it differ from a GPU?",
    "Explain the CAP theorem with examples of real databases.",
    "What causes the seasons on Earth?",
    "How does public-key cryptography allow two strangers to share a secret?",
    "Explain what a hash table is and how collisions are handled.",
    # writing / instruction following
    "Write a haiku about debugging at 3 am.",
    "Summarize the plot of Hamlet in five sentences.",
    "Draft a polite email declining a meeting and proposing two alternative times.",
    "Give me a 5-day beginner workout plan with rest days.",
    "Write a product description for a waterproof hiking backpack, under 80 words.",
    "List pros and cons of remote work in a table.",
    "Explain recursion to a 10-year-old.",
    "Write a short story opening (100 words) set on a generation starship.",
    "Rewrite this sentence to be more concise: 'In order to be able to make a decision, we need to have more information.'",
    "Create a JSON object describing a book with title, author, year, and genres.",
    # agentic / tools-like
    "You have tools read_file(path) and run(cmd). Plan the steps to find why `make test` fails, then list the first tool call as JSON.",
    "Given a repository with a failing CI lint step, describe how you would diagnose and fix it.",
    "Break down the task 'add dark mode to a React app' into concrete steps.",
    "Write a commit message for a change that fixes an off-by-one error in pagination.",
    # multilingual
    "Explique la photosynthèse en trois phrases.",
    "Erkläre den Unterschied zwischen 'seit' und 'seitdem' mit Beispielen.",
    "用简单的语言解释什么是区块链。",
    "日本の四季について短く説明してください。",
    "Explica qué es la inflación y cómo afecta a los ahorros.",
    "Объясни, что такое рекурсия, на примере.",
    # longer reasoning
    "Design a rate limiter for an API with 1000 requests per minute per user; discuss trade-offs of token bucket vs sliding window.",
    "Compare B-trees and LSM trees for a write-heavy workload.",
    "A company's revenue grew 20% then fell 20%. Is it back where it started? Explain generally.",
    "Plan a test strategy for a payment service, covering unit, integration, and chaos testing.",
]


def load_model():
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(str(MODEL))
    model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    return tok, model.eval().to(DEVICE)


def generate():
    tok, model = load_model()
    tok.padding_side = "left"
    seqs, plens, t0 = [], [], time.time()
    for b in range(0, len(PROMPTS), BATCH):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False,
                                         enable_thinking=True) for p in PROMPTS[b:b + BATCH]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEVICE)
        out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
        for i in range(out.shape[0]):
            prompt = enc["input_ids"][i][enc["attention_mask"][i].bool()].tolist()
            gen = out[i, enc["input_ids"].shape[1]:].tolist()
            eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
            gen = gen[:next((j + 1 for j, t in enumerate(gen) if t in eos), len(gen))]
            seqs.append(np.array(prompt + gen, np.int32))
            plens.append(len(prompt))
        print(f"batch {b // BATCH}: {sum(len(s) for s in seqs)} tokens so far ({time.time() - t0:.0f}s)", flush=True)
    TRACE.mkdir(parents=True, exist_ok=True)
    np.savez(TRACE / "trace.npz", lengths=np.array([len(s) for s in seqs]), plens=np.array(plens),
             ids=np.concatenate(seqs))
    print(f"trace: {len(seqs)} sequences, {sum(plens)} input + {sum(len(s) for s in seqs) - sum(plens)} output tokens")


def trace():
    d = np.load(TRACE / "trace.npz")
    ends = np.cumsum(d["lengths"])
    return [d["ids"][e - n:e] for e, n in zip(ends, d["lengths"])]


def logprobs(model, ids):
    out = model(input_ids=torch.from_numpy(ids.astype(np.int64))[None].to(DEVICE), use_cache=False)
    return F.log_softmax(out.logits[0, :-1].float(), dim=-1)  # predicts ids[1:]


def reference():
    tok, model = load_model()
    top_i, top_v, nll = [], [], []
    for n, s in enumerate(trace()):
        lp = logprobs(model, s)
        v, i = lp.topk(TOPK, dim=-1)
        top_i.append(i.to(torch.int32).cpu().numpy()), top_v.append(v.cpu().numpy())
        nll.append(-lp.gather(1, torch.from_numpy(s[1:].astype(np.int64))[:, None].to(DEVICE))[:, 0].cpu().numpy())
        print(f"ref {n}: {len(s)} tokens", flush=True) if n % 8 == 0 else None
    np.savez(TRACE / "ref.npz", ids=np.concatenate(top_i), lp=np.concatenate(top_v), nll=np.concatenate(nll))


def rotation(n, seed, block=1024):
    h = hadamard(block) / np.sqrt(block)
    s = np.random.default_rng(seed).choice([-1.0, 1.0], n)
    m = np.zeros((n, n), np.float32)
    for b in range(n // block):
        sl = slice(b * block, (b + 1) * block)
        m[sl, sl] = s[sl, None] * h
    return torch.from_numpy(m)


def dequant(t, key):
    if f"{key}.weight" in t:
        return t[f"{key}.weight"].float()
    if f"{key}.int8" in t:
        return t[f"{key}.int8"].float() * t[f"{key}.scale"].float()[:, None]
    lut, idx = t[f"{key}.lut"].float(), t[f"{key}.idx"].long()
    cd = lut.shape[1]
    w = lut[idx].permute(0, 2, 1).reshape(idx.shape[0] * cd, idx.shape[1])
    return w * t[f"{key}.scale"].float()[:, None] if f"{key}.scale" in t else w


def apply_export(model):
    """Replace weights with the export's dequantized ones; returns quantized bytes (excl. embeddings, incl. head)."""
    text, nbytes = model.model.language_model, 0.0
    for f in sorted(EXPORT_DIR.glob("*.safetensors")):
        t = load_file(f)
        for k, v in t.items():
            if k.endswith(".idx"):
                nbytes += v.numel() * np.ceil(np.log2(t[k[:-4] + ".lut"].shape[0])) / 8
            elif k.endswith(".int8"):
                nbytes += v.numel()
            elif k.endswith((".weight", ".scale", ".lut")):
                nbytes += v.numel() * 2
        if f.name == "lm_head.safetensors":
            if "head" in PARTS:
                model.lm_head.weight.data = dequant(t, "lm_head").to(DEVICE, torch.bfloat16)
            continue
        i = int(f.name.split("_")[1][:2])
        layer = text.layers[i]
        in_range = QLAYERS[0] <= i <= QLAYERS[1]
        if f.name.endswith("_mixer.safetensors"):
            for key in {k.rsplit(".", 1)[0] for k in t}:
                part = "attn" if key.startswith("self_attn") else "gdn"
                if in_range and part in PARTS:
                    mod = layer.get_submodule(key)
                    w = exported_lowrank(t, key, dequant(t, key))
                    mod.weight.data = with_lowrank(mod.weight.data, w, part).to(DEVICE, torch.bfloat16)
            continue
        if not (in_range and "mlp" in PARTS):
            continue
        with safe_open(f, framework="pt") as fh:
            meta = fh.metadata()
        for m in ("gate", "up", "down"):
            w = dequant(t, m)
            if meta.get("basis") == "online":
                w = w @ rotation(w.shape[1], int(meta["seed_in"] if m != "down" else meta["seed_mid"])).T
            w = exported_lowrank(t, m, w)  # trained factors live in the original (unrotated) basis
            mod = getattr(layer.mlp, f"{m}_proj")
            mod.weight.data = with_lowrank(mod.weight.data, w, "mlp").to(DEVICE, torch.bfloat16)
    # unquantized projections and the head count at bf16
    names = ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj", "linear_attn.in_proj_a",
             "linear_attn.in_proj_b", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    exported = {k.rsplit(".", 1)[0] for f in EXPORT_DIR.glob("layer_*_mixer.safetensors") for k in load_file(f)}
    for i, layer in enumerate(text.layers):
        for n in names:
            try:
                mod = layer.get_submodule(n)
            except AttributeError:
                continue
            if n in ("linear_attn.in_proj_a", "linear_attn.in_proj_b") or (not n.startswith("mlp") and n not in exported):
                nbytes += mod.weight.numel() * 2
    if not (EXPORT_DIR / "lm_head.safetensors").exists():
        nbytes += model.lm_head.weight.numel() * 2
    return nbytes


def evaluate():
    tok, model = load_model()
    nbytes = apply_export(model) if EXPORT_DIR else None
    ref = np.load(TRACE / "ref.npz")
    off, kls, agree, nll = 0, [], [], []
    for s in trace():
        lq = logprobs(model, s)
        n = len(s) - 1
        ids = torch.from_numpy(ref["ids"][off:off + n].astype(np.int64)).to(DEVICE)
        lp = torch.from_numpy(ref["lp"][off:off + n]).to(DEVICE)
        lq_k = lq.gather(1, ids)
        p = lp.exp()
        p_tail = (1 - p.sum(1)).clamp_min(1e-12)
        q_tail = (1 - lq_k.exp().sum(1)).clamp_min(1e-12)
        kl = (p * (lp - lq_k)).sum(1) + p_tail * (p_tail.log() - q_tail.log())
        kls.append(kl.cpu().numpy())
        agree.append((lq.argmax(1) == ids[:, 0]).cpu().numpy())
        nll.append(-lq.gather(1, torch.from_numpy(s[1:].astype(np.int64))[:, None].to(DEVICE))[:, 0].cpu().numpy())
        off += n
    kl, ag, nl = np.concatenate(kls), np.concatenate(agree), np.concatenate(nll)
    tag = os.environ.get("TAG") or (EXPORT_DIR.name if EXPORT_DIR else "bf16")
    res = {"tag": tag, "tokens": int(len(kl)), "mean_kl": float(kl.mean()), "median_kl": float(np.median(kl)),
           "p99_kl": float(np.quantile(kl, 0.99)), "top1_agree": float(ag.mean()),
           "ppl": float(np.exp(nl.mean())), "ref_ppl": float(np.exp(ref["nll"].mean())),
           "size_gib": None if nbytes is None else nbytes / 2 ** 30}
    (TRACE / f"kl_{tag}.json").write_text(json.dumps(res, indent=1))
    np.save(TRACE / f"klpos_{tag}.npy", kl.astype(np.float32))  # per position, for tail analysis
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    {"generate": generate, "reference": reference, "eval": evaluate}[sys.argv[1]]()
