"""JetSpec++ acceptance on OUR target by replay (queue 7, 19 Nov 2026).

The public JetSpec++ torch reference (JetSpec 41a30357, draft/reference: JetSpecPPDrafter, RoundDriver, the hybrid
anchor decoder) runs unchanged through the reference reader (bench/jetspec_plus_plus/reader.py read_passes ->
jetspec.core.llm.LLM.generate_tree, width-one B16 trees). Only the target is replaced, by a replay of one captured
trace of our served model (v8_server_tapdump.py): the prompt + greedy continuation token ids and the five tap
features (residual stream after layers 5 / 19 / 33 / 47 / 61, [T, 5 x 5120] fp16) of every position.

Why the replay is exact (temperature 0): generate_tree's verify only uses (a) the target's argmax at each tree node
and (b) the tap features of the ACCEPTED nodes. A width-one tree is a chain; node i is accepted iff every node up to
i equals the captured greedy stream, so every accepted node is on the stream, and its argmax (the next stream token)
and its taps are in the trace. Nodes off the stream get a sentinel argmax (V - 1, an unused id, so no draft can match an unknown position)
and zero taps; neither is ever read.

The target object (ReplayTarget) holds only what the drafter side reads from a target: embed_tokens (kept on the
CPU), layers 62 (GDN) and 63 (full attention), the final norm and lm_head, from the BF16 checkpoint at its pinned
revision (the decoder manifest's sha256 pins are checked by the reference build_drafter). target._cache stays None,
so the anchor decoder builds its layer 62 / 63 prefix state from the tap-61 features, as the reference does when it
has no live target cache.

MPS: anchor_offers and blend compute in float64, which MPS lacks; they run on the CPU (wrappers below). Everything
else runs where the weights are.
    PY jsp_replay.py --traces DIR [--release R] [--target-root T] [--schedule J3-R3] [--device mps]
                     [--max-new N] [--limit K] [--out OUT.json] [--tiny]   (--tiny: no checksum pins, eos off)"""
import argparse, glob, hashlib, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

SRC = Path(os.environ.get("JETSPEC_SRC", "/Volumes/SSD4TB/anemll-forge-research/jetspec_pp/src/JetSpec"))
sys.path.insert(0, str(SRC))
from jetspec.core.llm import LLM  # noqa: E402
from jetspec.engines.jetspec_plus_plus.draft.method.adapter import HybridC1RMSNorm  # noqa: E402
from jetspec.engines.jetspec_plus_plus.draft.method.cache_rows import DeviceOffers  # noqa: E402
from jetspec.engines.jetspec_plus_plus.draft.reference import drafter as RD  # noqa: E402
from jetspec.engines.jetspec_plus_plus.draft.reference.models import hybrid as H  # noqa: E402
from jetspec.engines.jetspec_plus_plus.draft.reference.round import JetSpecPPConfig  # noqa: E402

RELEASE = Path("/Volumes/SSD4TB/anemll-forge-research/jetspec_pp/release")
TARGET = Path("/Volumes/SSD4TB/Qwen3.8-27B")
TAPS = [5, 19, 33, 47, 61]
EOS = {248044, 248046}   # the served stops: <|endoftext|> (248044) and <|im_end|> (248046)


# ---------------------------------------------------------------- float64 parts on the CPU (MPS has no float64)
def _cpu(x):
    return x.cpu() if torch.is_tensor(x) else x


_anchor_offers, _blend = RD.anchor_offers, RD.blend


def anchor_offers_cpu(logits, drafts, **kw):
    return _anchor_offers(logits.cpu(), drafts.cpu(), **kw)


def blend_cpu(p_top1, logits, draft_ids, probabilities, sizes, *, coefficients):
    return _blend(_cpu(p_top1), _cpu(logits), _cpu(draft_ids), _cpu(probabilities), _cpu(sizes),
                  coefficients=coefficients)


def topk_pair_to_lists_cpu(topk_tok_t, topk_lp_t):
    # accum_logp stacks tokens and log-probs as float64 on the device; the same values through the CPU
    return topk_tok_t.cpu().tolist(), topk_lp_t.cpu().double().tolist()


def patch_cpu_scoring():
    from jetspec.methods.jetspec.tree.baselines import accum_logp
    RD.anchor_offers, RD.blend = anchor_offers_cpu, blend_cpu
    accum_logp._topk_pair_to_lists = topk_pair_to_lists_cpu


# ---------------------------------------------------------------- the captured trace
class Trace:
    """tokens.npy int [T] (prompt + greedy stream), taps.npy fp16 [T, 5 x H], filled.npy bool [T], meta.json
    (prompt_len, index, finish, ...)."""

    def __init__(self, d):
        self.dir = Path(d)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.tokens = np.load(self.dir / "tokens.npy").astype(np.int64)
        self.taps = np.load(self.dir / "taps.npy", mmap_mode="r")
        self.filled = np.load(self.dir / "filled.npy")
        self.P = int(self.meta["prompt_len"])
        assert self.taps.shape[0] >= self.filled.shape[0] and self.filled.shape[0] <= len(self.tokens) + 1
        # the generated stream may end with a stop token the server did not feed: then it is known
        self.known = len(self.tokens)

    def last_tapped(self):
        """first position without taps (every position before it has them)"""
        bad = np.flatnonzero(~self.filled)
        return int(bad[0]) if len(bad) else int(self.filled.shape[0])


# ---------------------------------------------------------------- the replay target
class CpuEmbedding(nn.Module):
    """The 2.5 GB table stays on the CPU; looked-up rows go to the compute device."""

    def __init__(self, vocab, width, device, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab, width, dtype=dtype), requires_grad=False)
        self.out_device = device

    def forward(self, ids):
        return torch.nn.functional.embedding(ids.cpu(), self.weight).to(self.out_device)


class ReplayTarget(nn.Module):
    """What the JetSpec++ drafter side reads from a qwen3_5_text target, nothing else."""

    def __init__(self, root, *, device, dtype, tail=(62, 63)):
        super().__init__()
        root = Path(root).resolve(strict=True)
        cfg = H.text_config(json.loads((root / "config.json").read_text()))
        cfg._name_or_path = str(root)
        self.config, self.tail = cfg, tuple(tail)
        self.model = nn.Module()
        with torch.device("meta"):
            layers = [nn.Identity() for _ in range(cfg.num_hidden_layers)]
            for i in self.tail:
                layers[i] = H.HybridLayer(cfg, cfg.layer_types[i])
            self.model.layers = nn.ModuleList(layers)
            self.model.norm = HybridC1RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.model.layers.to(dtype=dtype); self.model.norm.to(dtype=dtype); self.lm_head.to(dtype=dtype)
        self.model.layers.to_empty(device=device); self.model.norm.to_empty(device=device)
        self.lm_head.to_empty(device=device)
        self.model.embed_tokens = CpuEmbedding(cfg.vocab_size, cfg.hidden_size, device, dtype)
        self.generation_config = SimpleNamespace(eos_token_id=getattr(cfg, "eos_token_id", None))
        self._cache = None   # the anchor decoder then builds its prefix from the tap-61 features
        self._load(root, device, dtype)
        self.eval().requires_grad_(False)

    def _load(self, root, device, dtype):
        from safetensors import safe_open
        index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        own = {k: v for k, v in self.state_dict().items()
               if not (k.startswith("model.layers.") and int(k.split(".")[2]) not in self.tail)}

        def source(name):
            return "lm_head.weight" if name == "lm_head.weight" else "model.language_model." + name[len("model."):]
        missing = [n for n in own if source(n) not in index]
        assert not missing, f"target lacks {missing}"
        for shard in sorted({index[source(n)] for n in own}):
            with safe_open(str(root / shard), framework="pt", device="cpu") as f:
                for n, t in own.items():
                    if index[source(n)] == shard:
                        x = f.get_tensor(source(n))
                        assert tuple(x.shape) == tuple(t.shape), (n, x.shape, t.shape)
                        t.copy_(x.to(dtype=dtype))
        self.loaded = sorted(own)


class ReplayCache:
    """The cache seam of generate_tree: a length, and commit_tree as HybridCache (prefix + accepted chain)."""

    def __init__(self):
        self.length = 0

    def get_seq_length(self):
        return self.length

    def commit_tree(self, prefix_len, accepted):
        count = int(accepted.numel())
        if not torch.equal(accepted.cpu(), torch.arange(count)):
            raise ValueError("replay supports width-one chains only")
        self.length = int(prefix_len) + count


class ReplayRunner:
    """ModelRunner.forward for a captured trace."""

    def __init__(self, target, device, dtype):
        self.target, self.device, self.dtype = target, device, dtype
        self.trace, self.stats = None, {}
        self.V = target.config.vocab_size
        self.SENT = self.V - 1

    def _hidden(self, positions, on):
        tr = self.trace
        width = tr.taps.shape[1]
        out = np.zeros((len(positions), width), np.float16)
        for j, (p, ok) in enumerate(zip(positions, on)):
            if ok:
                if p >= tr.filled.shape[0] or not tr.filled[p]:
                    raise RuntimeError(f"trace {tr.dir.name}: no taps at accepted position {p}")
                out[j] = tr.taps[p]
        return torch.from_numpy(out).to(device=self.device, dtype=self.dtype)[None]

    def _logits(self, greedy):
        lg = torch.zeros((1, len(greedy), self.V), dtype=torch.float32)
        lg[0, torch.arange(len(greedy)), torch.as_tensor(greedy)] = 1.0
        return lg.to(self.device)

    @torch.inference_mode()
    def forward(self, input_ids, past_key_values, position_ids=None, attention_mask=None, cache_position=None,
                output_hidden_states=False, target_layer_ids=None, last_position_logits_only=False):
        tr, cache = self.trace, past_key_values
        if target_layer_ids is not None and list(target_layer_ids) != TAPS:
            raise ValueError(f"taps {target_layer_ids} != captured {TAPS}")
        ids = input_ids[0].tolist()
        start, n = cache.get_seq_length(), len(ids)
        if position_ids is not None:
            exp = torch.arange(start, start + n, device=position_ids.device)[None]
            if not torch.equal(position_ids, exp):
                raise ValueError("replay supports consecutive chain positions only")
        need = output_hidden_states and target_layer_ids is not None
        if start == 0:   # prefill: the prompt
            if n != tr.P or ids != tr.tokens[:n].tolist():
                raise ValueError(f"trace {tr.dir.name}: the prompt differs from the captured one")
            greedy = [int(tr.tokens[n])]
            hidden = self._hidden(range(n), [True] * n) if need else None
            cache.length = n
            self.stats["prefill"] = self.stats.get("prefill", 0) + 1
            return self._logits(greedy), cache, hidden
        on, greedy = [], []
        for i, t in enumerate(ids):
            p = start + i
            ok = p < tr.known and t == int(tr.tokens[p]) and (i == 0 or on[-1])
            if i == 0 and not ok:
                raise RuntimeError(f"trace {tr.dir.name}: the anchor at {p} is not the captured token")
            on.append(ok)
            greedy.append(int(tr.tokens[p + 1]) if ok and p + 1 < tr.known else self.SENT)
            if ok and p + 1 >= tr.known:
                self.stats["past_end"] = self.stats.get("past_end", 0) + 1
        hidden = self._hidden([start + i for i in range(n)], on) if need else None
        cache.length = start + n
        self.stats["verify"] = self.stats.get("verify", 0) + 1
        return self._logits(greedy), cache, hidden


def make_llm(target_root, device, dtype, *, eos=EOS):
    llm = LLM.__new__(LLM)
    llm.model = ReplayTarget(target_root, device=device, dtype=dtype)
    llm.tokenizer = None
    llm.runner = ReplayRunner(llm.model, device, dtype)
    llm.device, llm.dtype = device, dtype
    llm.eos_token_ids = set(eos)
    llm.model.new_cache = ReplayCache   # make_cache() uses model.new_cache when present
    return llm


def load_adapter_lowmem(path, device, dtype):
    """The reference load_adapter (draft/reference/models/adapter.py) without its peak: it builds CtxKVC1Head in
    fp32 on the CPU (4.47 B parameters = 17.9 GB) and loads the 8.9 GB checkpoint beside it, about 27 GB on a 32 GB
    Mac. Here the module is built on the meta device, every checkpoint tensor goes straight to the device, and the
    only buffers (the rotary inv_freq vectors, which depend on the rotary width and theta only) come from a narrow
    copy of the same module (hidden 64) on the CPU. The same checks, then the reference's own .to(device, dtype):
    the result equals the reference loader's (checked on the tiny artifacts: check_adapter_loader)."""
    import copy as _copy
    import math
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers import Qwen3Config
    from jetspec.engines.jetspec_plus_plus.draft.method.adapter import CtxKVC1Head
    directory = Path(path)
    for name in ("config.json", "head.safetensors", "converter.safetensors"):
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
    raw = json.loads((directory / "config.json").read_text())
    if not isinstance(raw, dict) or raw.get("model_type") != "qwen3":
        raise ValueError("adapter config must describe qwen3")
    parameters = raw.get("rope_parameters")
    rope_type = (parameters or {}).get("rope_type", raw.get("rope_type"))
    if rope_type not in (None, "default"):
        raise ValueError(f"unsupported rope_type {rope_type!r}")
    if any(n in raw or n in (parameters or {}) for n in ("partial_rotary_factor", "partial_rotary_ratio", "partial_rotary_dim")):
        raise ValueError("partial rotary is unsupported for the drafter")
    theta = raw.get("rope_theta", (parameters or {}).get("rope_theta"))
    if not isinstance(theta, (int, float)) or isinstance(theta, bool) or not math.isfinite(theta) or theta <= 0:
        raise ValueError("rope_theta is absent or invalid")

    def make_config(r):
        c = Qwen3Config.from_dict(r)
        c.rope_theta, c._attn_implementation = float(theta), "sdpa"
        return c
    config = make_config(raw)
    with safe_open(directory / "head.safetensors", framework="pt") as checkpoint:
        if (checkpoint.metadata() or {}).get("PREFIX_KV_SOURCE") is not None:
            raise ValueError("head PREFIX_KV_SOURCE requires target cache K/V")
    with torch.device("meta"):
        head = CtxKVC1Head(config)
    converter = load_file(directory / "converter.safetensors", device="cpu")
    if set(converter) != {"fc.weight", "hidden_norm.weight"}:
        raise ValueError(f"adapter converter keys differ: {sorted(converter)}")
    for name, shape in {"fc.weight": (head.hidden_size, head.hspace_dim), "hidden_norm.weight": (head.hidden_size,)}.items():
        if tuple(converter[name].shape) != shape or not converter[name].is_floating_point():
            raise ValueError(f"adapter converter {name} must be floating point with shape {shape}")
    # the buffers: a narrow copy (hidden 64, MLP 128; the same heads, head_dim, rotary and stream layout)
    narrow = _copy.deepcopy(raw)
    narrow.update(hidden_size=64, intermediate_size=128)
    ck = narrow["dflash_config"]["ctxkv_head"]
    ck["target_text_config"] = dict(ck["target_text_config"], hidden_size=64, intermediate_size=128)
    small = CtxKVC1Head(make_config(narrow))
    small_buffers = dict(small.named_buffers())
    names = {k for k, _ in head.named_buffers()}
    if names != set(small_buffers):
        raise ValueError("narrow adapter copy has different buffers")
    expected = head.state_dict()
    state = {}
    with safe_open(directory / "head.safetensors", framework="pt", device="cpu") as f:
        keys = set(f.keys())
        if keys != set(expected):
            raise RuntimeError(f"adapter head keys differ: {sorted(keys ^ set(expected))[:8]}")
        for k in sorted(keys):
            t = f.get_tensor(k)
            if not t.is_floating_point() or tuple(t.shape) != tuple(expected[k].shape):
                raise ValueError(f"adapter head tensor {k}: {t.dtype} {tuple(t.shape)}")
            state[k] = t.to(device=device, dtype=dtype)
            del t
    head.load_state_dict(state, strict=True, assign=True)
    for k in names:
        mod, attr = (head.get_submodule(k.rsplit(".", 1)[0]), k.rsplit(".", 1)[1])
        mod._buffers[attr] = small_buffers[k].clone()
    head.to(device=device, dtype=dtype)   # as the reference (params already there; buffers cast as it casts them)
    head.eval()
    head.requires_grad_(False)
    return head


def patch_adapter_loader():
    from jetspec.engines.jetspec_plus_plus.draft.reference.models import adapter as A
    A.load_adapter = load_adapter_lowmem


def check_adapter_loader(path, device="cpu", dtype=torch.bfloat16):
    """The low-memory loader against the reference one: every parameter and buffer bit-identical."""
    from jetspec.engines.jetspec_plus_plus.draft.reference.models.adapter import load_adapter
    a, b = load_adapter(path, device, dtype), load_adapter_lowmem(path, device, dtype)
    sa, sb = dict(a.named_parameters()) | dict(a.named_buffers()), dict(b.named_parameters()) | dict(b.named_buffers())
    assert set(sa) == set(sb), set(sa) ^ set(sb)
    bad = [k for k in sa if sa[k].dtype != sb[k].dtype or sa[k].device != sb[k].device or not torch.equal(sa[k], sb[k])]
    return len(sa), bad


def build(llm, *, release, schedule, dataset=None):
    from jetspec.engines.jetspec_plus_plus.draft.reference.drafter import build_drafter
    patch_adapter_loader()
    config = JetSpecPPConfig.paper_eval(schedule=schedule, model_family="27b")
    t0 = time.time()
    drafter = build_drafter(llm=llm, config=config, drafter_checkpoint=str(release),
                            adapter_artifact=str(Path(release) / "adapter"),
                            heads_bundle=str(Path(release) / "calibration"), dataset=dataset)
    return drafter, config, time.time() - t0


class _UnusedRow:
    """A keyed row the true outcome cannot select (rows = outcome): never read."""

    def __getattr__(self, name):
        raise RuntimeError("an unused keyed row was selected: the outcome filter is wrong")


class OutcomeRows:
    """rows = outcome: draft only the keyed rows that can be this round's true outcome.

    The reference drafts up to 16 keyed rows per round (k accepted drafts, correction anchor) and keeps the one
    that verification selects. Each row is an independent RoundDriver.run whose random context block comes from a
    fresh generator (seed ^ prompt_index), so a row's result does not depend on which other rows ran. The replay
    knows the stream, so a row (k, a) is a candidate iff the block's drafts 1..k equal the stream and a is the
    stream token after them; the true outcome is always a candidate. Candidates run as in the reference, every other
    key gets a placeholder that raises if it is ever read. Same rounds, hits and outputs; far fewer passes."""

    def __init__(self, drafter, runner):
        self.drafter, self.runner, self.run0 = drafter, runner, drafter.driver.run
        self.ran = self.skipped = 0
        drafter.driver.run = self

    def __call__(self, context_ids, exact, **kw):
        anchor, tail = kw.get("anchor"), kw.get("tail_tokens")
        if anchor is None or tail is None or kw.get("admission"):
            return self.run0(context_ids, exact, **kw)
        tr, C = self.runner.trace, context_ids.shape[1]
        k = tail.shape[1] - 1
        drafts = tail[0, 1:].tolist()
        stream = tr.tokens[C:C + k + 1].tolist()
        candidate = len(stream) == k + 1 and drafts == stream[:k] and int(anchor) == stream[k]
        if candidate:
            self.ran += 1
            return self.run0(context_ids, exact, **kw)
        self.skipped += 1
        return _UnusedRow()


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


def run_trace(llm, drafter, config, tr, *, max_new):
    from bench.jetspec_plus_plus.reader import read_passes
    llm.runner.trace, llm.runner.stats = tr, {}
    sync(llm.device)
    t0 = time.time()
    rows = [{"index": int(tr.meta.get("index", 0)), "token_ids": tr.tokens[:tr.P].tolist()}]
    found = read_passes(llm, rows, drafter_factory=lambda **_: drafter, config=config, tokens=max_new,
                        target_layer_ids=TAPS)[0]
    sync(llm.device)
    secs = time.time() - t0
    gen = found["token_ids"]
    want = tr.tokens[tr.P:tr.P + len(gen)].tolist()
    lossless = gen == want
    acc = found["accept_lengths"]
    return dict(trace=tr.dir.name, index=found["prompt_index"], prompt_len=tr.P, generated=len(gen),
                rounds=len(acc), tau=sum(acc) / len(acc) if acc else None, accept_lengths=acc,
                passes_run=found["passes_run"], converged_rounds=found["converged_rounds"],
                counts=dict(drafter.counts), driver_counts=dict(drafter.driver.counts),
                runner=dict(llm.runner.stats), seconds=secs, s_per_round=secs / max(1, len(acc)),
                lossless_vs_trace=lossless)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True, help="a capture directory (one subdirectory per trace)")
    ap.add_argument("--release", default=str(RELEASE))
    ap.add_argument("--target-root", default=str(TARGET))
    ap.add_argument("--schedule", default="J3-R3")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dataset", default="gsm8k", help="calibration entry (gsm8k, math500, ...)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiny", action="store_true", help="tiny random artifacts (make_tiny.py): no stop tokens")
    ap.add_argument("--rows", choices=["outcome", "all"], default="outcome",
                    help="outcome: draft only keyed rows that can be the true outcome (same result, see OutcomeRows)")
    a = ap.parse_args()
    dtype = torch.bfloat16
    if a.device == "mps":
        patch_cpu_scoring()
    dirs = sorted(d for d in glob.glob(f"{a.traces}/*") if Path(d, "meta.json").exists())
    if a.limit:
        dirs = dirs[:a.limit]
    t0 = time.time()
    llm = make_llm(a.target_root, a.device, dtype, eos=set() if a.tiny else EOS)
    t_target = time.time() - t0
    drafter, config, t_build = build(llm, release=a.release, schedule=a.schedule, dataset=a.dataset)
    rows = OutcomeRows(drafter, llm.runner) if a.rows == "outcome" else None
    banner = dict(jetspec_src=str(SRC), jetspec_commit=os.popen(f"git -C {SRC} rev-parse HEAD").read().strip(),
                  rows=a.rows, dataset=a.dataset,
                  release=a.release, target_root=a.target_root, schedule=a.schedule, device=a.device,
                  dtype=str(dtype), config=config.__dict__, max_new=a.max_new, loaded_target_tensors=len(llm.model.loaded),
                  target_load_s=round(t_target, 1), drafter_build_s=round(t_build, 1), torch=torch.__version__)
    print("BANNER " + json.dumps(banner), flush=True)
    res = []
    for d in dirs:
        tr = Trace(d)
        cap = min(a.max_new, tr.known - tr.P - (0 if tr.meta.get("stop_appended") else 16))
        r = run_trace(llm, drafter, config, tr, max_new=cap)
        if rows is not None:
            r["rows_ran"], r["rows_skipped"] = rows.ran, rows.skipped
            rows.ran = rows.skipped = 0
        res.append(r)
        print(f"{r['trace']}: prompt {r['prompt_len']}, {r['generated']} tokens in {r['rounds']} rounds, tau "
              f"{r['tau']:.3f}, hits {r['counts'].get('hits', 0)} / misses {r['counts'].get('misses', 0)}, "
              f"{r['seconds']:.1f} s ({r['s_per_round']:.2f} s / round), lossless {r['lossless_vs_trace']}", flush=True)
        Path(a.out).write_text(json.dumps(dict(banner=banner, results=res), indent=1))
    acc = [x for r in res for x in r["accept_lengths"]]
    summary = dict(traces=len(res), rounds=len(acc), tau_pooled=sum(acc) / len(acc) if acc else None,
                   all_lossless=all(r["lossless_vs_trace"] for r in res))
    print("SUMMARY " + json.dumps(summary), flush=True)
    Path(a.out).write_text(json.dumps(dict(banner=banner, summary=summary, results=res), indent=1))


if __name__ == "__main__":
    main()
