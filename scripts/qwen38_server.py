#!/usr/bin/env python3
"""OpenAI-compatible server for the quantized Qwen3.8-27B running on the Apple Neural Engine (M6).

POST /v1/chat/completions (stream or not, tools / tool_calls, chat_template_kwargs), GET /v1/models, GET /health.
Prompts are rendered with the model's own chat template (tools, enable_thinking, reasoning_effort,
preserve_thinking); Qwen3.8's XML tool calls (<tool_call><function=..><parameter=..>) are returned as OpenAI
tool_calls, typed by each tool's JSON schema; thinking is returned as reasoning_content.

One model, one request at a time. Prompt cache: the Core ML states continue from the previous request when the
new prompt extends it; the Gated DeltaNet states are also snapshotted at the end of every prompt, so a follow-up
that re-renders the last answer differently still reuses everything up to that prompt (KV rows past the
position are masked, so only the DeltaNet recurrent / conv states need the snapshot).

    python qwen38_server.py --hf /path/to/bundle/model --model-dir /path/to/bundle/coreai --ctx 8192
"""
import argparse
import json
import os
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

MODEL_ID = "qwen38-27b-ane"
# The drafter's ANE call stalls (300-700 ms, a few per 100 cycles, the next verify too) when it is submitted
# within ~2 ms of the target verify's return: greedy leaves ~1.5 ms of host work there, sampling ~6 ms. A 3 ms
# minimum gap removes the stalls (M6, Core AI target + drafter, 16K greedy, 4 runs each: 10 drafter + 4 verify
# stalls vs none; historical drafter_gap_test.py harness, not distributed).
DRAFT_GAP = float(os.environ.get("DRAFT_GAP_MS", "3")) / 1e3
# Thinking budget: a reasoning that reaches its budget without "</think>" is closed with Qwen's budget phrase and the
# answer follows (the quantized model can deliberate past pi's 16K maxTokens: 2026-09-28, 16384 tokens of thinking,
# no answer). ANSWER_RESERVE tokens of max_tokens stay for the answer.
THINK_STOP = ("\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly "
              "now.\n</think>\n\n")
ANSWER_RESERVE = 4096


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--model-dir", required=True, help="Core AI target package directory")
    p.add_argument("--hf", required=True, help="checkpoint/bundle dir: tokenizer, chat template, embedding")
    p.add_argument("--runtime", choices=("coreai", "coreml"), default="coreai")
    p.add_argument("--kv-cache-dtype", choices=("auto", "fp16", "v8"), default="auto",
                   help="require matching model cache inputs; auto reads manifest.json")
    p.add_argument("--ctx", type=int, default=8192, help="context length the chunks were built for")
    p.add_argument("--think", action="store_true", help="enable thinking by default (clients can override)")
    p.add_argument("--max-tokens", type=int, default=4096, help="default completion limit")
    p.add_argument("--think-budget", default="low=2048,medium=6144,xhigh=12288",
                   help="reasoning tokens per reasoning_effort before the server closes the thinking (no effort: "
                        "medium); at most max(max_tokens - 4096, max_tokens / 2); 0 = no limit")
    p.add_argument("--presence", type=float, default=0.0, help="default presence_penalty (requests can override)")
    p.add_argument("--dry", type=float, default=0.0, help="DRY repetition penalty multiplier (0 = off; requests: "
                   "dry_multiplier); penalty = mult * base^(match - allowed) for tokens extending a repeated run. "
                   "Off by default: code repeats legitimately (0.8 turned 'for (let x = 0; x' into 'for (let x = 0; <')")
    p.add_argument("--dry-base", type=float, default=1.75)
    p.add_argument("--dry-allowed", type=int, default=8, help="repeated run length (tokens) tolerated before DRY acts")
    p.add_argument("--loop-guard", type=int, default=6, help="stop when the output tail repeats this many times with "
                   "the same period (0 = off)")
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--draft", help="Core AI DFlash2 package (default: bundle/drafter/dflash2_lut4_gptq.aimodel)")
    modes.add_argument("--plain", action="store_true", help="diagnostic target-only generation")
    p.add_argument("--drafter", help="drafter config/selector directory (default: drafter package parent)")
    args = p.parse_args(argv)
    if args.plain and args.drafter:
        p.error("--drafter cannot be combined with --plain")
    if args.runtime != "coreai" and not args.plain:
        p.error("Core ML is a plain diagnostic path; supply --plain")
    if args.runtime != "coreai" and args.kv_cache_dtype == "v8":
        p.error("V8 KV cache requires the Core AI Swift bridge runtime")
    return args


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dry_matches(seq, breakers=frozenset(), cap=32, window=4096):
    """DRY: for each token that followed an earlier occurrence of the current suffix, the length of that repeated
    suffix -> {token: match length}. Appending such a token would extend a verbatim repeat. As in the reference DRY,
    a repeat never spans a sequence breaker (tokens with a newline, ':', '"' or '*'): nothing after a breaker, and
    the suffix stops at the last one."""
    s = np.asarray(seq[-window:], np.int64)
    L = len(s)
    if L < 2 or int(s[-1]) in breakers:
        return {}
    P = np.nonzero(s[:-1] == s[-1])[0]
    if not len(P):
        return {}
    m, alive = np.ones(len(P), np.int64), np.ones(len(P), bool)
    for k in range(1, cap):
        if int(s[L - 1 - k]) in breakers:  # k <= L - 1: every match dies once P - k < 0
            break
        idx = P - k
        ok = alive & (idx >= 0)
        ok[ok] = s[idx[ok]] == s[L - 1 - k]
        m += ok
        alive = ok
        if not alive.any():
            break
    best = {}
    for t, mm in zip(s[P + 1].tolist(), m.tolist()):
        if mm > best.get(t, 0):
            best[t] = mm
    return best


def loop_period(out, reps, min_period=4, max_period=256):
    """Period of an exact repetition covering the last reps periods of the output, else 0."""
    if reps <= 0 or len(out) < min_period * reps:
        return 0
    a = np.asarray(out[-max_period * reps:], np.int64)
    for P in range(min_period, min(max_period, len(a) // reps) + 1):
        t = a[-P * reps:]
        if np.array_equal(t[P:], t[:-P]):
            return P
    return 0


class Engine:
    def __init__(self, a):
        mdir = Path(os.path.expanduser(a.model_dir))
        hf = Path(os.path.expanduser(a.hf))
        # Fail on a wrong/missing head or tap pairing before allocating either large model.
        self.runtime = a.runtime
        from qwen38_kv_cache import cache_format
        if self.runtime == "coreai":
            cache_format(json.loads((mdir / "manifest.json").read_text()), a.kv_cache_dtype)
        dpath, ddir, dcfg = None, None, None
        if not a.plain:
            from hf_release import drafter_paths, check_drafter_pair
            dpath, ddir = drafter_paths(mdir, a.draft, a.drafter)
            target = json.loads((mdir / "manifest.json").read_text())
            target_cfg = json.loads((hf / "config.json").read_text())["text_config"]
            dcfg, _ = check_drafter_pair(dpath, ddir, target, target_cfg)
            os.environ.update(DRAFTER=str(ddir), COREAI_DRAFTER_COMPUTE="ane")
            a.draft = str(dpath)
        if self.runtime == "coreai":
            from hf_release import coreai_bridge_environment
            os.environ.update(coreai_bridge_environment())
            sys.path.insert(0, os.environ["COREAI_BRIDGE_DIR"])
            import coreai_bridge
            coreai_bridge.lib()  # Check native loadability and ABI before loading target or drafter.
        os.environ.update(ANE_OUT=str(mdir.parent), EXPORT_DIR=mdir.name, CTX=str(a.ctx), MODEL=str(hf))
        prepared_embedding = hf / "embed_tokens_fp16.npy"
        if prepared_embedding.is_file():
            os.environ["EMBED_NPY"] = str(prepared_embedding)
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import qwen38_ane_model as M
        from transformers import AutoTokenizer
        t = time.time()
        if self.runtime == "coreai":  # Core AI build (qwen38_coreai_build.py): all context / prefill entry points of a
            os.environ["COREAI_DIR"] = str(mdir)  # chunk share one weight copy; the context grows through the ladder
            import qwen38_coreai_model as A
            man = json.loads((mdir / "manifest.json").read_text())
            import coreai_compile_guide as G
            extra = [] if dpath is None else [(f"drafter {Path(dpath).name}", Path(dpath), G.DRAFTER_S)]
            self.model = A.CoreAIQwen(ladder=[c for c in man["ctxs"] if c <= a.ctx], log=log,
                                     kv_cache_dtype=a.kv_cache_dtype,  # [ctx] switches, timestamped
                                     extra_packages=extra)
        else:
            self.model = M.load_model()
        self.v2 = hasattr(self.model, "snapshot")   # v2 / v3: batched prefill + zero-copy DeltaNet states
        self.drafter, self.cycles, self.verify_end = None, 0, 0.0
        self.tm = dict.fromkeys(("draft", "verify", "sample", "ctx", "host"), 0.0)
        self.acc_hist = [0] * 8  # accepted drafts per cycle (0..7), this request
        if not a.plain:
            assert hasattr(self.model, "call"), "--draft needs a v3/v4 build (one T=8 function per chunk)"
            import dflash2_ane_drafter as D
            from dflash2_coreai_drafter import CoreAIDrafter
            guide = getattr(self.model, "compile_guide", None)
            make = lambda: CoreAIDrafter(dpath, dcfg, D.load_codebooks(ddir), self.model.emb)  # noqa: E731
            self.drafter = guide.load(f"drafter {Path(dpath).name}", make) if guide else make()
            log(f"drafter: Core AI {dpath.name}")
        self.tok = AutoTokenizer.from_pretrained(str(hf))
        self.a = a
        # usable positions: a Core AI 64K entry holds 65472 KV rows (the ANE caps [history | block] at 65536)
        self.ctx = min(a.ctx, self.model.cap(self.model.ladder[-1])) if hasattr(self.model, "cap") else a.ctx
        self.stops = {self.tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")} | {248044}
        self.budgets = {} if str(a.think_budget) in ("0", "off", "") else \
            {k: int(v) for k, v in (kv.split("=") for kv in a.think_budget.split(","))}
        self.think_stop = self.tok.encode(THINK_STOP, add_special_tokens=False)
        self.think_forced = 0
        # DRY sequence breakers: byte-level BPE tokens containing a newline (\u010a), ':', '"' or '*'
        self.breakers = frozenset(i for i, t in enumerate(self.tok.convert_ids_to_tokens(list(range(len(self.tok)))))
                                  if t and any(c in t for c in "\u010a:\"*"))
        self.im_start = self.tok.convert_tokens_to_ids("<|im_start|>")
        # Gated DeltaNet state names per chunk (from the manifest's layer ranges); KV needs no snapshot
        cfg = json.loads((hf / "config.json").read_text())["text_config"]
        names = [] if self.v2 else json.loads((mdir / f"manifest_ctx{a.ctx}.json").read_text())["chunks"]
        self.gdn_states = []
        for n in names:
            lo, hi = (int(x) for x in re.search(r"L(\d+)-(\d+)", n).groups())
            self.gdn_states.append([s for j, l in enumerate(range(lo, hi + 1))
                                    if cfg["layer_types"][l] == "linear_attention" for s in (f"conv{j}", f"rec{j}")])
        self.fed, self.snaps = [], {}   # snapshots: "system" (before the first user turn), "turn" (before the last header)
        self.user_hdr = self.tok.encode("<|im_start|>user", add_special_tokens=False)
        self.lock = threading.Lock()
        print(f"[{MODEL_ID}] loaded {'batched prefill' if self.v2 else f'{len(names)} chunks'} + head"
              f"{' + DFlash2 drafter ' + Path(a.draft).name if a.draft else ''} from "
              f"{mdir.name} (ctx {a.ctx}) in {time.time() - t:.0f}s", flush=True)

    # ---- prompt rendering ----
    @staticmethod
    def _text(content):
        if content is None:
            return ""
        if isinstance(content, list):
            return "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return str(content)

    def render(self, messages, tools, kw):
        msgs = []
        for m in messages:
            m = dict(m)
            m["role"] = "system" if m.get("role") == "developer" else m.get("role")
            m["content"] = self._text(m.get("content"))
            if m.get("reasoning") and not m.get("reasoning_content"):
                m["reasoning_content"] = m["reasoning"]
            if m.get("tool_calls"):
                calls = []
                for c in m["tool_calls"]:
                    f = dict(c.get("function", c))
                    if isinstance(f.get("arguments"), str):
                        try:
                            f["arguments"] = json.loads(f["arguments"]) if f["arguments"].strip() else {}
                        except json.JSONDecodeError:
                            f["arguments"] = {"arguments": f["arguments"]}
                    calls.append({"type": "function", "function": f})
                m["tool_calls"] = calls
            msgs.append(m)
        text = self.tok.apply_chat_template(msgs, tools=tools or None, add_generation_prompt=True, tokenize=False, **kw)
        return self.tok.encode(text, add_special_tokens=False)

    # ---- state / prompt cache ----
    def _snapshot(self, key, ids):
        if self.v2:
            self.snaps[key] = {"ids": list(ids), **self.model.snapshot()}
            if self.drafter is not None:  # the drafter's context ring: which slot holds which position, and the queue
                self.snaps[key]["draft"] = (self.drafter.slot_pos.copy(), list(self.drafter.pending))
            return
        self.snaps[key] = {"ids": list(ids), "pos": self.model.pos, "half": getattr(self.model, "half", 0),
                     "states": [{n: st.read_state(n) for n in names}
                                for st, names in zip(self.model.states, self.gdn_states)]}

    def _restore(self, snap):
        if self.v2:
            self.model.restore(snap)
            if self.drafter is not None:
                slot_pos, pending = snap["draft"]
                d = self.drafter
                # slots rewritten since the snapshot hold positions of an abandoned continuation: hide them (the
                # queued features of the snapshot are written again on the next flush)
                d.slot_pos = np.where(d.slot_pos == slot_pos, slot_pos, -1)
                d.pending = list(pending)
        else:
            for st, saved in zip(self.model.states, snap["states"]):
                for n, v in saved.items():
                    st.write_state(n, v)
            self.model.pos = snap["pos"]
            self.model.half = snap.get("half", 0)
        self.fed = list(snap["ids"])

    def _feed(self, ids):
        """Tokens into the model; logits after the last one (v2+: batched prefill; with the drafter, the tap
        features of every token also go into the drafter's context)."""
        if self.drafter is not None:
            return self.model.feed(ids, on_features=self.drafter.add_context)
        if self.v2:
            return self.model.feed(ids)
        logits = None
        for t in ids:
            logits = self.model.step(t)
        return logits

    def prefill(self, ids):
        """Bring the model state to `ids`; returns (logits after the last token, tokens reused).
        The DeltaNet state cannot be rewound, so the state is snapshotted just before the final
        '<|im_start|>assistant' header: the next request re-renders the previous answer in its own way, but
        everything up to that header is identical and restores from the snapshot."""
        n = len(self.fed)
        if n and not (len(ids) > n and ids[:n] == self.fed):  # cache miss past the model state: log where and why
            j = next((i for i, (a, b) in enumerate(zip(ids, self.fed)) if a != b), min(len(ids), n))
            show = lambda t: self.tok.decode(t[max(0, j - 12):j + 24], skip_special_tokens=False)  # noqa: E731
            print(f"[cache] model state {n} tok, prompt {len(ids)} tok, first difference at {j}:\n"
                  f"   state : {show(self.fed)!r}\n   prompt: {show(ids)!r}", flush=True)
        best = max((s_ for s_ in self.snaps.values()
                    if len(ids) > len(s_["ids"]) and ids[:len(s_["ids"])] == s_["ids"]),
                   key=lambda s_: len(s_["ids"]), default=None)
        if n and len(ids) > n and ids[:n] == self.fed and (best is None or n >= len(best["ids"])):
            reused, how = n, "continue (model state is a prefix of the prompt)"
        elif best is not None:
            key = next(k for k, v in self.snaps.items() if v is best)
            self._restore(best)
            reused, how = len(self.fed), f"restore '{key}' snapshot"
        else:
            self.model.reset()
            if self.drafter is not None:
                self.drafter.reset()
            self.fed, reused, how = [], 0, "cold (no reusable state)"
        log(f"request: prompt {len(ids)} tok | cache: {how}, reused {reused} | prefilling {len(ids) - reused} tok ...")
        t_pre = time.perf_counter()
        turn_cut = max((i for i, t in enumerate(ids) if t == self.im_start), default=0)
        h = self.user_hdr
        sys_cut = next((i for i in range(len(ids) - len(h) + 1) if ids[i:i + len(h)] == h), 0)
        cuts = []
        if sys_cut > reused and ("system" not in self.snaps or self.snaps["system"]["ids"] != ids[:sys_cut]):
            cuts.append((sys_cut, "system"))
        if turn_cut > reused:
            cuts.append((turn_cut, "turn"))
        logits, i = None, reused
        for cut, key in sorted(cuts) + [(len(ids), None)]:
            if cut > i:  # feed up to the snapshot point, then snapshot
                logits = self._feed(ids[i:cut])
                self.fed.extend(ids[i:cut])
                i = cut
            if key:
                self._snapshot(key, ids[:cut])
        dt = time.perf_counter() - t_pre
        self.prefill_stats = {"tokens": len(ids) - reused, "seconds": dt,
                              "tokens_per_second": (len(ids) - reused) / max(dt, 1e-9),
                              "cached_tokens": reused,
                              "logits_finite": bool(np.isfinite(logits).all()) if logits is not None else None}
        log(f"prefill: {len(ids) - reused} tok in {dt:.1f}s ({(len(ids) - reused) / max(dt, 1e-9):.0f} tok/s) | decoding ...")
        return logits, reused

    # ---- sampling / generation ----
    def dist(self, logits, temp, top_p, top_k, seen=None, presence=0.0, hist=None, dry=None):
        """Sampling distribution of one row: top-k, presence penalty, DRY repetition penalty (hist = the output so
        far, dry = (multiplier, base, allowed)), temperature, top-p -> (ids, probs); None for greedy. Plain and
        speculative decoding sample from exactly this."""
        if temp <= 0:
            return None
        logits = logits.astype(np.float32)
        k = min(top_k or len(logits), len(logits) - 1)
        idx = np.argpartition(-logits, k)[:k]
        z = logits[idx].astype(np.float64)
        if presence and seen is not None:
            z = z - presence * seen[idx]
        if dry and dry[0] > 0 and hist is not None and len(hist) > dry[2]:
            mult, base, allowed = dry
            for t, mm in dry_matches(hist, self.breakers).items():
                if mm >= allowed:
                    hit = np.nonzero(idx == t)[0]
                    if len(hit):
                        z[hit[0]] -= mult * base ** min(mm - allowed, 24)
        z = z / temp
        p = np.exp(z - z.max())
        p /= p.sum()
        order = np.argsort(-p)
        keep = order[: int(np.searchsorted(np.cumsum(p[order]), top_p)) + 1]
        return idx[keep], p[keep] / p[keep].sum()

    def sample(self, logits, temp, top_p, top_k, rng, seen=None, presence=0.0, hist=None, dry=None):
        d = self.dist(logits, temp, top_p, top_k, seen, presence, hist, dry)
        if d is None:
            return int(np.argmax(logits))
        return int(d[0][rng.choice(len(d[0]), p=d[1])])

    def accept_rate(self):
        """Share of drafted tokens accepted this request (7 drafts per cycle), %."""
        c = sum(self.acc_hist)
        return 100.0 * sum(k * v for k, v in enumerate(self.acc_hist)) / max(1, 7 * c)

    def cycle_ms(self):
        """Mean ms per draft cycle by phase: drafter propose, target verify, accept / sampling, context update (drafter
        add_context + target accept), host work between cycles (detokenize, stream)."""
        n = max(1, self.cycles)
        parts = " ".join(f"{k} {1e3 * v / n:.1f}" for k, v in self.tm.items())
        return f"ms/cycle {1e3 * sum(self.tm.values()) / n:.0f} ({parts})"

    def accept_hist(self):
        """Distribution of accepted drafts per cycle, e.g. '0:31% 1:18% ... 7:9%'."""
        c = max(1, sum(self.acc_hist))
        return " ".join(f"{k}:{100 * v / c:.0f}%" for k, v in enumerate(self.acc_hist))

    def draft_cycle(self, anchor, samp, rng, seen, hist=None, dry=None):
        """One speculative cycle after `anchor` (not yet in the model): draft 7, verify [anchor + 7] in one target
        call, accept (greedy: equal to the target's argmax; sampling: with probability p(draft), the first rejection
        resamples from p without the draft: same output distribution as plain sampling), commit 1 + accepted rows.
        An accepted stop token ends the cycle uncommitted. Returns (new tokens, committed tokens)."""
        m, d = self.model, self.drafter
        p = m.pos
        t0 = time.perf_counter()
        if t0 < self.verify_end + DRAFT_GAP:           # see DRAFT_GAP; counted in the draft phase
            time.sleep(self.verify_end + DRAFT_GAP - t0)
        drafts = [int(t) for t in d.propose(anchor, p)[0]]
        t1 = time.perf_counter()
        lg = m.call([anchor] + drafts)
        t2 = self.verify_end = time.perf_counter()
        k, nxt = 0, None
        while k < 7:
            temp, top_p, top_k, presence = samp
            h = None if hist is None else hist + drafts[:k]           # row k's context: output + accepted drafts
            dk, dist = drafts[k], self.dist(lg[k], temp, top_p, top_k, seen, presence, h, dry)
            if dist is None:
                ok = int(np.argmax(lg[k])) == dk
            else:
                hit = np.where(dist[0] == dk)[0]
                ok = bool(len(hit)) and rng.random() < float(dist[1][hit[0]])
            if not ok:
                if dist is None:
                    nxt = int(np.argmax(lg[k]))
                else:
                    ids_, pr = dist
                    if len(hit):
                        pr = pr.copy()
                        pr[hit[0]] = 0.0
                        pr /= pr.sum()
                    nxt = int(ids_[rng.choice(len(ids_), p=pr)])
                break
            if dk in self.stops:
                nxt = dk
                break
            seen[dk] = 1.0
            k += 1
        if nxt is None:
            nxt = self.sample(lg[7], *samp[:3], rng, seen, samp[3], None if hist is None else hist + drafts, dry)
        t3 = time.perf_counter()
        d.add_context(m.features(k + 1), np.arange(p, p + k + 1))
        m.accept(k + 1)
        t4 = time.perf_counter()
        for key, dt in (("draft", t1 - t0), ("verify", t2 - t1), ("sample", t3 - t2), ("ctx", t4 - t3)):
            self.tm[key] += dt
        self.cycles += 1
        self.acc_hist[k] += 1
        return drafts[:k] + [nxt], [anchor] + drafts[:k]

    def think_budget(self, thinking, effort, max_tokens):
        """Reasoning tokens allowed before the server closes the thinking (0: no limit)."""
        b = self.budgets.get(effort or "medium", self.budgets.get("medium", 0)) if thinking else 0
        return min(b, max(max_tokens - ANSWER_RESERVE, max_tokens // 2)) if b else 0

    def generate(self, ids, max_tokens, temp, top_p, top_k, stop, seed, on_text, presence=None, dry=None,
                 think_budget=0):
        logits, reused = self.prefill(ids)
        self.think_forced = 0
        decode_started = time.perf_counter()
        t_gen, out, text, finish = time.time(), [], "", "length"
        rng = np.random.default_rng(seed)
        presence = self.a.presence if presence is None else float(presence)
        samp = (temp, top_p, top_k, presence)
        dry = (self.a.dry if dry is None else float(dry), self.a.dry_base, self.a.dry_allowed)
        seen = np.zeros(len(logits), np.float64)
        self.cycles, self.loop = 0, 0
        self.acc_hist = [0] * 8
        self.tm = dict.fromkeys(("draft", "verify", "sample", "ctx", "host"), 0.0)
        t_host = time.perf_counter()
        next_check, next_log = 64, time.time() + 10
        anchor = self.sample(logits, temp, top_p, top_k, rng, seen, presence, out, dry)
        new, done = [anchor], False
        while not done:
            for nxt in new:
                if nxt in self.stops:
                    finish, done = "stop", True
                    break
                out.append(nxt)
                seen[nxt] = 1.0
                full = self.tok.decode(out, skip_special_tokens=False)
                delta, text = full[len(text):], full
                if stop and any(s_ in text for s_ in stop):
                    text = text[:min(text.index(s_) for s_ in stop if s_ in text)]
                    finish, done = "stop", True
                    break
                if on_text(delta) is False:
                    finish, done = "stop", True
                    break
                if len(out) >= max_tokens:
                    done = True
                    break
            if time.time() >= next_log:  # live decode progress
                next_log = time.time() + 10
                el = time.time() - t_gen
                log(f"decode: {len(out)} tok, {el:.0f}s, {len(out) / max(el, 1e-9):.1f} tok/s"
                    + (f", {len(out) / max(1, self.cycles):.2f} tok/call, accept {self.accept_rate():.0f}%, {self.cycle_ms()}"
                       if self.drafter is not None else ""))
            if not done and len(out) >= next_check:  # loop guard: the tail repeats with a fixed period
                next_check = len(out) + 32
                per = loop_period(out, self.a.loop_guard)
                if per:
                    self.loop, finish, done = per, "stop", True
            if done:
                break
            if think_budget and not self.think_forced and len(out) >= think_budget and "</think>" not in text:
                # close the reasoning: the last token + THINK_STOP into the model (and the drafter's context), the
                # answer's first token sampled after it
                self.think_forced = len(out)
                seq = [new[-1]] + self.think_stop
                logits = self._feed(seq)
                self.fed.extend(seq)
                out.extend(self.think_stop)
                seen[self.think_stop] = 1.0
                full = self.tok.decode(out, skip_special_tokens=False)
                delta, text = full[len(text):], full
                log(f"[think] reasoning budget {think_budget} reached at {self.think_forced} tokens: thinking closed")
                if on_text(delta) is False:
                    finish = "stop"
                    break
                new = [self.sample(logits, temp, top_p, top_k, rng, seen, presence, out, dry)]
                continue
            anchor = new[-1]
            if self.drafter is not None:
                self.tm["host"] += time.perf_counter() - t_host
                new, committed = self.draft_cycle(anchor, samp, rng, seen, out, dry)
                t_host = time.perf_counter()
                self.fed.extend(committed)
            else:
                logits = self.model.step(anchor)
                self.fed.append(anchor)
                new = [self.sample(logits, temp, top_p, top_k, rng, seen, presence, out, dry)]
        decode_seconds = time.perf_counter() - decode_started
        last_logits = getattr(self.model, "logits", None)
        if not isinstance(last_logits, np.ndarray):
            last_logits = logits
        self.decode_stats = {"tokens": len(out), "seconds": decode_seconds,
                             "tokens_per_second": len(out) / max(decode_seconds, 1e-9),
                             "cycles": self.cycles,
                             "tokens_per_call": len(out) / self.cycles if self.cycles else None,
                             "draft_accept_percent": self.accept_rate() if self.drafter is not None else None,
                             "draft_accept_histogram": list(self.acc_hist),
                             "phase_seconds": dict(self.tm), "finish_reason": finish,
                             "last_logits_finite": bool(np.isfinite(last_logits).all())}
        return text, finish, len(out), reused, decode_seconds


def split_output(text, thinking):
    """-> (reasoning, content, [(name, {param: raw string})])"""
    reasoning = ""
    if thinking or text.lstrip().startswith("<think>"):
        # with thinking on the chat template already ends the prompt with "<think>\n"
        body = text.split("<think>", 1)[1] if text.lstrip().startswith("<think>") else text
        if "</think>" in body:
            reasoning, text = body.split("</think>", 1)
        else:
            reasoning, text = body, ""
    calls = []
    for fn, body in re.findall(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", text, re.S):
        params = {k: v for k, v in re.findall(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", body, re.S)}
        calls.append((fn, params))
    content = text.split("<tool_call>", 1)[0] if calls else text
    return reasoning.strip(), content.strip(), calls


def typed_args(name, params, tools):
    schema = {}
    for t in tools or []:
        f = t.get("function", t)
        if f.get("name") == name:
            schema = f.get("parameters", {}).get("properties", {})
    out = {}
    for k, v in params.items():
        typ = schema.get(k, {}).get("type")
        if typ in ("integer", "number", "boolean", "array", "object", "null"):
            try:
                out[k] = json.loads(v)
                continue
            except json.JSONDecodeError:
                pass
        out[k] = v
    return out


PARAM_CLOSE = "\n</parameter>"


def stream_calls(region, tools, final):
    """Tool calls in the generated text after the first <tool_call>, as (name, arguments JSON so far, complete).
    The arguments string only ever grows by appending as more text arrives (string values stream JSON-escaped
    with a partial closing tag held back; typed values appear once complete), so it can be streamed as deltas."""
    schema = {}
    for t in tools or []:
        f = t.get("function", t)
        schema[f.get("name")] = f.get("parameters", {}).get("properties", {})
    calls = []
    for part in region.split("<tool_call>")[1:]:
        m = re.search(r"<function=([^>\s]+)>", part)
        if not m:
            break
        name, body = m.group(1), part[m.end():]
        props = schema.get(name, {})
        args, n, complete = "", 0, False
        for pm in re.finditer(r"<parameter=([^>\s]+)>", body):
            key = pm.group(1)
            rest = body[pm.end():]
            rest = rest[1:] if rest.startswith("\n") else rest
            end = rest.find("</parameter>")
            args += ("{" if n == 0 else ", ") + json.dumps(key) + ": "
            n += 1
            typed = props.get(key, {}).get("type") in ("integer", "number", "boolean", "array", "object", "null")
            if end < 0:  # value still being generated
                if not typed:
                    v = rest if final else rest[:max(0, len(rest) - len(PARAM_CLOSE))]
                    args += '"' + json.dumps(v)[1:-1]
                break
            v = rest[:end]
            v = v[:-1] if v.endswith("\n") else v
            if typed:
                try:
                    args += json.dumps(json.loads(v))
                    continue
                except json.JSONDecodeError:
                    pass
            args += json.dumps(v)
        else:
            if "</function>" in body:
                args += "}" if n else "{}"
                complete = True
        calls.append((name, args, complete))
        if not complete:
            break
    return calls


class StreamSplitter:
    """Incrementally route generated text into reasoning / content deltas, holding back possible partial tags
    and everything from <tool_call> on (tool calls are sent once complete)."""

    def __init__(self, thinking, tools=None):
        self.acc, self.sent_r, self.sent_c, self.thinking = "", 0, 0, thinking
        self.tools, self.calls = tools, []  # calls: [id, name, arguments sent so far]

    def feed(self, delta, final=False):
        self.acc += delta
        acc, out = self.acc, {}
        hold = 0 if final else 12
        if self.thinking or acc.lstrip().startswith("<think>"):
            # thinking on: the prompt ends with "<think>\n", so output is reasoning until "</think>"
            body = acc.split("<think>", 1)[1] if acc.lstrip().startswith("<think>") else acc
            end = body.find("</think>")
            r = body if end < 0 else body[:end]
            limit = len(r) if end >= 0 else max(0, len(r) - hold)
            if limit > self.sent_r:
                out["reasoning_content"] = r[self.sent_r:limit].lstrip("\n") if self.sent_r == 0 else r[self.sent_r:limit]
                self.sent_r = limit
            if end < 0:
                return out
            content = body[end + len("</think>"):]
        elif not final and "<think>".startswith(acc.lstrip()) and acc.strip():
            return out
        else:
            content = acc
        stopper = content.find("<tool_call>")
        if stopper >= 0:  # tool calls stream as OpenAI tool_calls deltas (name first, then argument fragments)
            deltas = []
            for i, (name, args, _) in enumerate(stream_calls(content[stopper:], self.tools, final)):
                if i == len(self.calls):
                    self.calls.append([f"call_{uuid.uuid4().hex[:12]}", name, ""])
                    deltas.append({"index": i, "id": self.calls[i][0], "type": "function",
                                   "function": {"name": name, "arguments": ""}})
                sent = self.calls[i][2]
                if len(args) > len(sent) and args.startswith(sent):
                    deltas.append({"index": i, "function": {"arguments": args[len(sent):]}})
                    self.calls[i][2] = args
            if deltas:
                out["tool_calls"] = deltas
        limit = stopper if stopper >= 0 else max(0, len(content) - (0 if final else hold))
        c = content[:limit]
        if self.sent_c == 0:
            c_strip = c.lstrip("\n")
            skipped = len(c) - len(c_strip)
        else:
            skipped = 0
        if limit > self.sent_c + skipped:
            out["content"] = content[self.sent_c + skipped:limit]
            self.sent_c = limit
        return out


def make_handler(engine):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                self._json(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "anemll",
                                                             "max_model_len": engine.ctx}]})
            elif self.path.rstrip("/") in ("/health", "/v1/health"):
                self._json(200, {"status": "ok", "model": MODEL_ID, "context": engine.ctx,
                                 "position": engine.model.pos,
                                 "kv_cache_dtype": getattr(engine.model, "kv_cache_dtype", "fp16"),
                                 "kv_cache_formats": list(getattr(engine.model, "kv_cache_formats", ("fp16",))),
                                 "target_graph": getattr(engine.model, "graph", None),
                                 "active_context_entry": getattr(engine.model, "ctx", engine.ctx),
                                 "prefill": getattr(engine, "prefill_stats", None),
                                 "decode": getattr(engine, "decode_stats", None)})
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
                return self._json(404, {"error": {"message": "not found"}})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except json.JSONDecodeError as e:
                return self._json(400, {"error": {"message": f"bad json: {e}"}})
            kw = dict(req.get("chat_template_kwargs") or {})
            kw.setdefault("enable_thinking", engine.a.think)
            effort = req.get("reasoning_effort") or kw.get("reasoning_effort")
            if effort:
                kw["reasoning_effort"] = {"minimal": "low", "low": "low", "medium": "medium"}.get(effort, "xhigh")
            thinking = bool(kw["enable_thinking"])
            tools = req.get("tools") if req.get("tool_choice") != "none" else None
            stop = req.get("stop")
            stop = [stop] if isinstance(stop, str) else (stop or [])
            temp = req.get("temperature")
            temp = (1.0 if thinking else 0.7) if temp is None else float(temp)
            top_p = req.get("top_p")
            top_p = (0.95 if thinking else 0.8) if top_p is None else float(top_p)
            top_k = int(req.get("top_k") or 20)
            presence = req.get("presence_penalty")
            dry = req.get("dry_multiplier")
            stream = bool(req.get("stream"))
            include_usage = bool((req.get("stream_options") or {}).get("include_usage"))
            rid, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())
            with engine.lock:
                try:
                    ids = engine.render(req.get("messages", []), tools, kw)
                except Exception as e:  # noqa: BLE001
                    return self._json(400, {"error": {"message": f"template error: {e}"}})
                # a verify writes T = 8 rows past the committed position; plain decode one
                room = engine.ctx - len(ids) - (8 if engine.drafter is not None else 1)
                if room <= 0:
                    return self._json(400, {"error": {"message": f"prompt is {len(ids)} tokens; context is {engine.ctx}",
                                                      "code": "context_length_exceeded"}})
                max_tokens = min(req.get("max_completion_tokens") or req.get("max_tokens") or engine.a.max_tokens, room)
                budget = int(req["thinking_budget"]) if thinking and req.get("thinking_budget") is not None else \
                    engine.think_budget(thinking, kw.get("reasoning_effort"), max_tokens)
                t0 = time.time()
                if stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()

                    def emit(delta, finish=None, usage=None):
                        chunk = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                                 "choices": [] if usage else [{"index": 0, "delta": delta, "finish_reason": finish}]}
                        if usage:
                            chunk["usage"] = usage
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.flush()

                    splitter = StreamSplitter(thinking, tools)
                    try:
                        emit({"role": "assistant", "content": ""})

                        def on_text(d):
                            try:
                                out = splitter.feed(d)
                                if out:
                                    emit(out)
                            except (BrokenPipeError, ConnectionResetError):
                                return False
                        text, finish, n, reused, dt = engine.generate(ids, max_tokens, temp, top_p, top_k, stop,
                                                                      req.get("seed"), on_text, presence, dry, budget)
                        tail = splitter.feed("", final=True)
                        if tail:
                            emit(tail)
                        if splitter.calls:  # streamed incrementally above
                            finish = "tool_calls"
                        else:
                            _, _, calls = split_output(text, thinking)
                            if calls:
                                emit({"tool_calls": [{"index": i, "id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                                                      "function": {"name": fn, "arguments": json.dumps(typed_args(fn, p, tools))}}
                                                     for i, (fn, p) in enumerate(calls)]})
                                finish = "tool_calls"
                        emit({}, finish=finish)
                        usage = {"prompt_tokens": len(ids), "completion_tokens": n, "total_tokens": len(ids) + n,
                                 "prompt_tokens_details": {"cached_tokens": reused}}
                        if include_usage:
                            emit({}, usage=usage)
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return
                else:
                    text, finish, n, reused, dt = engine.generate(ids, max_tokens, temp, top_p, top_k, stop,
                                                                  req.get("seed"), lambda d: None, presence, dry, budget)
                    reasoning, content, calls = split_output(text, thinking)
                    msg = {"role": "assistant", "content": content or (None if calls else "")}
                    if reasoning:
                        msg["reasoning_content"] = reasoning
                    if calls:
                        msg["tool_calls"] = [{"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                                              "function": {"name": fn, "arguments": json.dumps(typed_args(fn, p, tools))}}
                                             for fn, p in calls]
                        finish = "tool_calls"
                    self._json(200, {"id": rid, "object": "chat.completion", "created": created, "model": MODEL_ID,
                                     "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                                     "usage": {"prompt_tokens": len(ids), "completion_tokens": n,
                                               "total_tokens": len(ids) + n,
                                               "prompt_tokens_details": {"cached_tokens": reused}}})
                t_pre = time.time() - t0 - dt
                print(f"[{time.strftime('%H:%M:%S')}] prompt {len(ids)} (cached {reused}, prefill "
                      f"{len(ids) - reused} in {t_pre:.1f}s) | gen {n} in {dt:.1f}s ({n / max(dt, 1e-9):.1f} tok/s"
                      + (f", {n / max(1, engine.cycles):.2f} tok/call, accept {engine.accept_rate():.0f}% "
                         f"(accepted/7 per cycle: {engine.accept_hist()}), {engine.cycle_ms()}" if engine.drafter is not None else "")
                      + ") | "
                      + (f"LOOP period {engine.loop} stopped | " if engine.loop else "")
                      + (f"THINK closed at {engine.think_forced} (budget) | " if engine.think_forced else "")
                      + f"{finish} | ctx {engine.model.pos}/{engine.ctx}"
                      + (f" ({engine.model.ctx // 1024}K entry)" if getattr(engine.model, "ladder", None) else ""), flush=True)

    return Handler


def startup_banner(a):
    mdir = Path(os.path.expanduser(a.model_dir))
    if a.runtime == "coreai" and (mdir / "manifest.json").exists():
        man = json.loads((mdir / "manifest.json").read_text())
        build = (f"Core AI: verify-8 contexts {[c for c in man['ctxs'] if c <= a.ctx]}, prefill-{man.get('TP', 0)} "
                 f"contexts {man.get('pctxs', [])}, KV {man.get('kv_cache', {}).get('format', 'fp16')}, one weight copy")
    else:
        build = "build manifests " + (", ".join(sorted(f.name for f in mdir.glob(f"manifest_ctx{a.ctx}_v*.json")))
                                      or "v1 (none)")
    lines = [
        f"started {time.strftime('%Y-%m-%d %H:%M:%S')}  pid {os.getpid()}  listen {a.host}:{a.port}",
        f"model dir     {mdir}  (max ctx {a.ctx}; {build})",
        f"checkpoint    {os.path.expanduser(a.hf)}",
        f"speculative   {'ON  drafter ' + os.path.expanduser(a.draft) if a.draft else 'OFF (no drafter)'}",
        f"thinking      default {'on' if a.think else 'off'} (clients override via chat_template_kwargs.enable_thinking)",
        f"think budget  {a.think_budget} tokens per reasoning_effort (request field thinking_budget overrides; "
        f"then closed with Qwen's budget phrase)",
        f"sampling      defaults: temperature 1.0 thinking / 0.7 non-thinking, top_p 0.95 / 0.8, top_k 20; "
        f"max_tokens {a.max_tokens}",
        f"repetition    presence_penalty {a.presence}; DRY multiplier {a.dry} base {a.dry_base} allowed "
        f"{a.dry_allowed} tokens; loop guard {'off' if not a.loop_guard else f'{a.loop_guard} repeats'}",
        f"python        {sys.executable}",
    ]
    print("\n".join(f"[{MODEL_ID}] {l}" for l in lines), flush=True)


def main():
    a = parse()
    startup_banner(a)
    engine = Engine(a)
    server = ThreadingHTTPServer((a.host, a.port), make_handler(engine))
    print(f"[{MODEL_ID}] serving OpenAI API on http://{a.host}:{a.port}/v1 "
          f"({'speculative ON' if a.draft else 'speculative OFF'}, ctx {a.ctx})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
