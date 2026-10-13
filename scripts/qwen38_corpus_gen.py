"""Phase-2 QAT corpus (QUANT_LOG §9.4): bf16 answers (thinking on, sampled T 0.6 / top-p 0.95 / top-k 20 as the KL traces
and qwen38_calib_gen.py) to CORPUS_PROMPTS ([{"p", "tools", "cat"}], prompts checked against kl / kl_test / dev2 for
8-gram overlap), tool prompts rendered with qwen38_calib_gen's tool schemas + system prompt. Packed into SEQ-token rows:
CAL_OUT = train rows followed by the validation rows (VAL_FRAC of the sequences, chosen at random, packed separately,
so no validation row shares a sequence with a train row); CAL_OUT.json records n_train / n_val; CAL_OUT.seqs.npz keeps
the raw sequences (ids, lengths, plens, cats) for overlap checks.
    MODEL=... CORPUS_PROMPTS=../configs/kl/qat_corpus_prompts.json CAL_OUT=<dir>/corpus_ids.npy python qwen38_corpus_gen.py"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import qwen38_kl as K
from qwen38_calib_gen import SYSTEM, TOOLS

SEQ = int(os.environ.get("SEQ", "1024"))
MAX_NEW = int(os.environ.get("MAX_NEW", "1792"))
BATCH = int(os.environ.get("GEN_BATCH", "64"))
VAL_FRAC = float(os.environ.get("VAL_FRAC", "0.05"))
CAL_OUT = Path(os.environ["CAL_OUT"])
PROMPTS = json.loads(Path(os.environ["CORPUS_PROMPTS"]).read_text())


def main():
    CAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(os.environ.get("SEED", "1234")))
    tok, model = K.load_model()
    tok.padding_side = "left"
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    seqs, plens, t0 = [], [], time.time()
    part = CAL_OUT.with_suffix(".partial.npz")
    if part.exists():  # resume after a crash: finished batches are kept
        d = np.load(part)
        ends = np.cumsum(d["lengths"])
        seqs = [d["ids"][e - n:e].tolist() for e, n in zip(ends, d["lengths"])]
        plens = d["plens"].tolist()
        print(f"resuming after {len(seqs)} sequences", flush=True)
    for b in range(len(seqs), len(PROMPTS), BATCH):
        texts = []
        for x in PROMPTS[b:b + BATCH]:
            msgs = ([{"role": "system", "content": SYSTEM}] if x["tools"] else []) + [{"role": "user", "content": x["p"]}]
            texts.append(tok.apply_chat_template(msgs, tools=TOOLS if x["tools"] else None, add_generation_prompt=True,
                                                 tokenize=False, enable_thinking=True))
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(K.DEVICE)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
        for i in range(out.shape[0]):
            prompt = enc["input_ids"][i][enc["attention_mask"][i].bool()].tolist()
            gen = out[i, enc["input_ids"].shape[1]:].tolist()
            gen = gen[:next((j + 1 for j, t in enumerate(gen) if t in eos), len(gen))]
            seqs.append(prompt + gen)
            plens.append(len(prompt))
        np.savez(part, ids=np.concatenate([np.array(s, np.int64) for s in seqs]), lengths=[len(s) for s in seqs],
                 plens=plens)
        print(f"batch {b // BATCH}: {len(seqs)} sequences, {sum(len(s) for s in seqs)} tokens ({time.time() - t0:.0f}s)",
              flush=True)
    rng = np.random.default_rng(0)
    val = set(rng.choice(len(seqs), max(1, int(round(VAL_FRAC * len(seqs)))), replace=False).tolist())

    def pack(idx):
        flat = np.concatenate([np.array(seqs[i], np.int64) for i in idx])
        return flat[: len(flat) // SEQ * SEQ].reshape(-1, SEQ)
    tr = pack([i for i in range(len(seqs)) if i not in val])
    va = pack(sorted(val))
    np.save(CAL_OUT, np.concatenate([tr, va]))
    np.savez(CAL_OUT.with_suffix(".seqs.npz"), ids=np.concatenate([np.array(s, np.int64) for s in seqs]),
             lengths=[len(s) for s in seqs], plens=plens, cats=[x["cat"] for x in PROMPTS[:len(seqs)]],
             val=sorted(val))
    meta = {"n_train": int(len(tr)), "n_val": int(len(va)), "sequences": len(seqs), "val_sequences": len(val),
            "tokens": int(sum(len(s) for s in seqs)), "seq": SEQ, "max_new": MAX_NEW,
            "truncated": int(sum(len(s) - p >= MAX_NEW for s, p in zip(seqs, plens))), "s": round(time.time() - t0)}
    CAL_OUT.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
