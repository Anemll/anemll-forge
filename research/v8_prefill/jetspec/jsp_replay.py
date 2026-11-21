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
def mmap_bf16(path, name):
    """A BF16 tensor of a safetensors file as a read-only memory map (file-backed pages, nothing copied)."""
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    meta = hdr[name]
    assert meta["dtype"] == "BF16", meta["dtype"]
    a, b = meta["data_offsets"]
    arr = np.memmap(path, dtype=np.uint16, mode="r", offset=8 + n + a, shape=tuple(meta["shape"]))
    assert arr.nbytes == b - a
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.from_numpy(arr).view(torch.bfloat16)


class CpuEmbedding(nn.Module):
    """The 2.5 GB table stays on the CPU, memory-mapped from the checkpoint (20 Nov 2026: file-backed pages instead
    of a 2.5 GB copy); looked-up rows go to the compute device."""

    def __init__(self, vocab, width, device, dtype):
        super().__init__()
        self.weight = None
        self.shape, self.out_device = (vocab, width), device

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
        en = "model.language_model.embed_tokens.weight"
        w = mmap_bf16(str(root / index[en]), en)
        assert tuple(w.shape) == self.model.embed_tokens.shape and dtype == torch.bfloat16
        self.model.embed_tokens.weight = w

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
        self.loaded = sorted(own) + ["model.embed_tokens.weight (memory-mapped)"]


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


_CHECKED = set()


def check_only_load_target_tensors(root, names, *, file_pins=None, prefixes=()):
    """hybrid.HybridAnchorDecoder.from_checkpoint's verification read (20 Nov 2026): the reference loads the tail
    tensors (layers 62 / 63, norm, lm_head: 3.5 GB) only to check them and drops them (the decoder views the target's
    own). Same checks (file SHA256 pins, the index, missing / unexpected names), shapes from the safetensors headers,
    nothing materialized; returns {} (the caller ignores the value)."""
    from safetensors import safe_open
    from jetspec.engines.jetspec_plus_plus.draft.reference.models import decoder as D
    root = Path(root).resolve(strict=True)
    key = (str(root), tuple(names), tuple(sorted((file_pins or {}).items())), tuple(prefixes))
    if key in _CHECKED:
        return {}
    _CHECKED.add(key)

    def checked(name):
        path = (root / name).resolve(strict=True)
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"target-file path escapes the target directory: {name}")
        if file_pins is not None:
            if name not in file_pins:
                raise ValueError(f"target-slice decoder lacks pin for {name}")
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != file_pins[name]:
                raise ValueError(f"target-file SHA256 mismatch: {name}")
        return path
    index = root / "model.safetensors.index.json"
    weight_map = json.loads(checked(index.name).read_text())["weight_map"]
    missing = set(names) - set(weight_map)
    if missing:
        raise ValueError(f"target-slice decoder missing tensors: {sorted(missing)}")
    unexpected = {n for n in weight_map if n.startswith(prefixes)} - set(names)
    if unexpected:
        raise ValueError(f"target-slice decoder unexpected tensors: {sorted(unexpected)}")
    for shard in sorted({weight_map[n] for n in names}):
        with safe_open(str(checked(shard)), framework="pt") as handle:
            for n in names:
                if weight_map[n] == shard:
                    handle.get_slice(n).get_shape()
    del D
    return {}


PRELOADED = {}


def preload_drafter(release, device, dtype):
    """The B16 drafter head before the target (20 Nov 2026): the reference load_drafter builds it on the CPU and loads the
    whole state on the CPU before moving it (6.4 GB transient); done first, that peak does not stack on the target."""
    from jetspec.engines.jetspec_plus_plus.draft.reference.models import drafter as DR
    head = DR.load_drafter(release, device=device, dtype=dtype, model_family="27b")
    PRELOADED["head"] = (str(release), str(device), dtype, head)
    load0 = DR.load_drafter

    def load_drafter(checkpoint, *, device, dtype, model_family="auto", **kw):
        r, d, t, h = PRELOADED["head"]
        if (str(checkpoint), str(device), dtype) == (r, d, t) and model_family in ("27b", "auto") and not kw:
            return h
        return load0(checkpoint, device=device, dtype=dtype, model_family=model_family, **kw)
    DR.load_drafter = load_drafter


def preload_adapter(release, device, dtype):
    """The adapter (8.6 GB) before the target too (20 Nov 2026): its file streams through the page cache while the most
    memory is free; build_drafter then gets the preloaded module (same low-memory loader, same checks)."""
    from jetspec.engines.jetspec_plus_plus.draft.reference.models import adapter as A
    path = Path(release) / "adapter"
    mod = load_adapter_lowmem(path, device, dtype)
    key = (str(path), str(device), dtype)

    def load_adapter(p, device=None, dtype=None, *a_, **k_):
        if (str(p), str(device), dtype) == key and not a_ and not k_:
            return mod
        return load_adapter_lowmem(p, device, dtype)
    load_adapter.preloaded = True
    A.load_adapter = load_adapter
    return mod


def mem(tag):
    import gc, resource
    gc.collect()
    mps = ""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        mps = f", MPS {torch.mps.current_allocated_memory() / 1e9:.1f} GB (driver {torch.mps.driver_allocated_memory() / 1e9:.1f})"
    print(f"MEM {tag}: max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.1f} GB{mps}", flush=True)


def fake_quant(module, bits, group=128, skip=("embed", "lm_head", "norm")):
    """Round-to-nearest symmetric quantization of every Linear weight in module, per output row and group of `group`
    input channels, written back in place in the weights' own dtype (20 Nov 2026: does the drafter survive the
    bit width an ANE port would use?). Returns (tensors, parameters) quantized."""
    n_t = n_p = 0
    qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for name, m in module.named_modules():
            if not isinstance(m, nn.Linear) or any(k in name for k in skip):
                continue
            w = m.weight
            o, i = w.shape
            g = group if i % group == 0 else i
            x = w.float().reshape(o, i // g, g)
            sc = x.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qmax
            w.copy_((torch.clamp(torch.round(x / sc), -qmax - 1, qmax) * sc).reshape(o, i).to(w.dtype))
            n_t += 1; n_p += w.numel()
            del x, sc
    return n_t, n_p


def fake_quant_spec(module, spec):
    """spec: '<bits>' (RTN symmetric, g128), '8c' (RTN symmetric per output channel, 8 bits), 'lut<bits>' (a 2^bits-entry
    palette per output row fit by k-means, 12 Lloyd iterations, quantile init: the ANE's LUT form). In place; returns
    (tensors, parameters)."""
    if spec == "8c":
        return fake_quant(module, 8, group=10 ** 9)
    if spec.startswith("lut") and spec.endswith("rf"):  # 'lut4rf': rotated LUT files (quant_adapter_gptq.py): W = (lut[idx] s) R
        d = Path(os.environ["JSP_LUT_DIR"]); n_t = n_p = 0
        with torch.no_grad():
            for name, m in module.named_modules():
                if not isinstance(m, nn.Linear) or any(k in name for k in ("embed", "lm_head", "norm")):
                    continue
                z = np.load(d / f"{name}.weight.npz")
                wr = torch.from_numpy(z["lut"].astype(np.float32)).reshape(-1)[torch.from_numpy(z["idx"].astype(np.int64))] \
                    * torch.from_numpy(z["scale"].astype(np.float32))[:, None]
                blk = int(z["block"]); sign = torch.from_numpy(z["sign"].astype(np.float32))
                h = torch.ones(1, 1)
                while h.shape[0] < blk:
                    h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
                h = h / blk ** 0.5
                o, i = wr.shape
                w = (wr.reshape(o, i // blk, blk) @ h).reshape(o, i) * sign     # R^T w': undo the rotation
                assert tuple(w.shape) == tuple(m.weight.shape), name
                m.weight.copy_(w.to(device=m.weight.device, dtype=m.weight.dtype))
                n_t += 1; n_p += m.weight.numel()
        return n_t, n_p
    if spec.startswith("lut") and spec.endswith("f"):   # 'lut6f': the ANE build's own LUT files (JSP_LUT_DIR, quant_adapter_lut6.py)
        d = Path(os.environ["JSP_LUT_DIR"]); n_t = n_p = 0
        with torch.no_grad():
            for name, m in module.named_modules():
                if not isinstance(m, nn.Linear) or any(k in name for k in ("embed", "lm_head", "norm")):
                    continue
                z = np.load(d / f"{name}.weight.npz")
                w = (torch.from_numpy(z["lut"].astype(np.float32)).reshape(-1)[torch.from_numpy(z["idx"].astype(np.int64))]
                     * torch.from_numpy(z["scale"].astype(np.float32))[:, None])
                assert tuple(w.shape) == tuple(m.weight.shape), name
                m.weight.copy_(w.to(device=m.weight.device, dtype=m.weight.dtype))
                n_t += 1; n_p += m.weight.numel()
        return n_t, n_p
    if spec.startswith("lut") and spec.endswith("t"):   # 'lut6t': the ANE build's form (lut_tensor.py), per tensor + channel scale
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from lut_tensor import lut_quantize, lut_dequantize
        bits = int(spec[3:-1]); n_t = n_p = 0
        with torch.no_grad():
            for name, m in module.named_modules():
                if not isinstance(m, nn.Linear) or any(k in name for k in ("embed", "lm_head", "norm")):
                    continue
                lut, idx, sc = lut_quantize(m.weight.detach().cpu(), bits)   # on the CPU: torch.histc fails on MPS
                m.weight.copy_(lut_dequantize(lut, idx, sc).to(device=m.weight.device, dtype=m.weight.dtype))
                n_t += 1; n_p += m.weight.numel()
                del lut, idx, sc
        return n_t, n_p
    if not spec.startswith("lut"):
        return fake_quant(module, int(spec))
    bits = int(spec[3:]); K = 2 ** bits
    n_t = n_p = 0
    with torch.no_grad():
        for name, m in module.named_modules():
            if not isinstance(m, nn.Linear) or any(k in name for k in ("embed", "lm_head", "norm")):
                continue
            w = m.weight
            o, i = w.shape
            step = max(1, (1 << 24) // (i * K))            # rows per chunk (distance tensor about 64 MB fp32)
            for r0 in range(0, o, step):
                x = w[r0:r0 + step].float()                                            # (r, i)
                q = torch.linspace(0, 1, K + 2, device=x.device)[1:-1]
                c = torch.quantile(x, q, dim=1).T.contiguous()                         # (r, K)
                for _ in range(12):
                    a = (x.unsqueeze(-1) - c.unsqueeze(1)).abs().argmin(-1)            # (r, i)
                    sums = torch.zeros_like(c).scatter_add_(1, a, x)
                    cnt = torch.zeros_like(c).scatter_add_(1, a, torch.ones_like(x))
                    c = torch.where(cnt > 0, sums / cnt.clamp_min(1), c)
                a = (x.unsqueeze(-1) - c.unsqueeze(1)).abs().argmin(-1)
                w[r0:r0 + step].copy_(torch.gather(c, 1, a).to(w.dtype))
                del x, c, a
            n_t += 1; n_p += w.numel()
    return n_t, n_p


def install_actq(adapter, mode, scales_path=None, parts="mlp"):
    """E2 (20 Nov 2026): INT8 activations into the adapter's Linear layers, simulated exactly as an ANE W4A8 build of the
    rotated LUT4 files (spec lut4rf, JSP_LUT_DIR) computes them: the weight is W = W' R (R = blockdiag(H_blk) diag(sign),
    the matrix's own rotation), so W x = W' (R x); the hook replaces x by R^T q(R x), q = INT8 fake-quantization:
      tok     per token (row): s = max|R x| / 127, codes round((R x) / s) clipped to +-127 (lut3.AQd's recipe)
      tensor  per tensor: a constant s = A / 127 per matrix input from JSP_ACTQ_SCALES (json {linear name: A}), clipped
    parts: 'mlp' (gate / up / down only) or 'all' (every adapter Linear). Hooks on the module (installed once)."""
    d = Path(os.environ["JSP_LUT_DIR"])
    sc = json.loads(Path(scales_path).read_text()) if mode == "tensor" else None
    hooks, n = [], 0
    for name, m in adapter.named_modules():
        if not isinstance(m, nn.Linear) or any(k in name for k in ("embed", "lm_head", "norm")):
            continue
        if parts == "mlp" and not ("." + name).count(".mlp."):
            continue
        z = np.load(d / f"{name}.weight.npz")
        blk = int(z["block"])
        h = torch.ones(1, 1)
        while h.shape[0] < blk:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        h = (h / blk ** 0.5)
        dev = m.weight.device
        hs = h.to(dev, torch.float32)
        sign = torch.from_numpy(z["sign"].astype(np.float32)).to(dev)
        A_ = float(sc[name]) if sc is not None else None

        def fn(mod, args, hs=hs, sign=sign, blk=blk, A_=A_):
            x = args[0]
            sh = x.shape
            xf = x.float().reshape(-1, sh[-1])
            xr = ((xf * sign).reshape(xf.shape[0], -1, blk) @ hs.T).reshape(xf.shape)
            s = (xr.abs().amax(-1, keepdim=True).clamp_min(2 ** -9) / 127.0) if A_ is None else torch.full_like(xr[:, :1], A_ / 127.0)
            q = torch.clamp(torch.round(xr / s), -127, 127) * s
            xb = ((q.reshape(xf.shape[0], -1, blk) @ hs).reshape(xf.shape)) * sign      # R^T q (H orthonormal, symmetric)
            return (xb.to(x.dtype).reshape(sh),) + tuple(args[1:])
        hooks.append(m.register_forward_pre_hook(fn))
        n += 1
    adapter._actq_hooks = hooks
    print(f"ACTQ adapter: INT8 activations ({mode}{', scales ' + str(scales_path) if sc else ''}) into {n} Linear layers ({parts}), "
          f"rotated basis from {d}", flush=True)
    return n


def patch_adapter_loader():
    from jetspec.engines.jetspec_plus_plus.draft.reference.models import adapter as A
    if not getattr(A.load_adapter, "preloaded", False):   # keep a preloaded adapter
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
    ap.add_argument("--multi", default="", help="research: ds=TRACEDIR,... in one process (models loaded once); --out gets _<ds>")
    ap.add_argument("--quant", default="", help="research: fake-quantize drafter parts, e.g. adapter:4,head:4 (RTN, g128)")
    ap.add_argument("--rows", choices=["outcome", "all"], default="outcome",
                    help="outcome: draft only keyed rows that can be the true outcome (same result, see OutcomeRows)")
    a = ap.parse_args()
    import socket   # D1 (20 Nov): the replay (~17.6 GB of GPU memory) beside a served forge (19 GB wired) swaps, and forge's
    with socket.socket() as so_:   # swap watchdog then kills forge (09:02:55): refuse while 8765 serves, unless forced
        so_.settimeout(0.5)
        if so_.connect_ex(("127.0.0.1", 8765)) == 0 and os.environ.get("JSP_ALLOW_FORGE") != "1":
            sys.exit("forge is serving on 8765: stop it first (one model at a time), or set JSP_ALLOW_FORGE=1")
    dtype = torch.bfloat16
    if a.device == "mps":
        patch_cpu_scoring()
    jobs = [tuple(x.split("=", 1)) for x in a.multi.split(",") if x] if a.multi else [(a.dataset, a.traces)]
    t0 = time.time()
    H.load_target_tensors = check_only_load_target_tensors
    if not a.tiny:
        preload_drafter(a.release, a.device, dtype)
        mem("drafter head preloaded")
        preload_adapter(a.release, a.device, dtype)
        mem("adapter preloaded")
    llm = make_llm(a.target_root, a.device, dtype, eos=set() if a.tiny else EOS)
    mem("target loaded")
    t_target = time.time() - t0
    qinfo = {}
    for ds, tdir in jobs:
        run_one(a, llm, dtype, t_target, ds, tdir, qinfo, multi=bool(a.multi))


def run_one(a, llm, dtype, t_target, dataset, tdir, qinfo, multi):
    """one dataset: build the drafter (the preloaded head / adapter are reused), quantize once, replay its traces"""
    drafter, config, t_build = build(llm, release=a.release, schedule=a.schedule, dataset=dataset)
    mem(f"drafter built ({dataset})")
    for part in [x for x in a.quant.split(",") if x]:
        what, spec = part.split(":")
        if what in qinfo:
            continue                       # the preloaded modules are shared: quantized once
        mod = {"adapter": drafter.driver.adapter, "head": drafter.driver.head}[what]
        t_q = time.time()
        qinfo[what] = (spec,) + fake_quant_spec(mod, spec)
        kind = {"8c": "RTN per channel, 8 bits"}.get(spec, (f"LUT {spec.rstrip('rft')[3:]} bits per tensor + channel scale (ANE form{', Hadamard-rotated files' if spec.endswith('rf') else ', the ANE build files' if spec.endswith('f') else ''})" if spec.endswith(("t", "f"))
                                                            else f"LUT {spec[3:]} bits per row (k-means)") if spec.startswith("lut") else f"RTN g128, {spec} bits")
        print(f"QUANT {what}: {qinfo[what][1]} Linear weights, {qinfo[what][2] / 1e9:.2f}B parameters, {kind}, in {time.time() - t_q:.0f}s", flush=True)
        mem(f"quantized {what}")
    skip = [x for x in os.environ.get("JSP_SKIP_BLOCKS", "").split(",") if x]   # C3: adapter blocks as identity (u -> u)
    for name in skip:
        blk = drafter.driver.adapter.get_submodule(name)
        if not getattr(blk, "_skipped", False):
            blk.forward = (lambda **kw: kw["u"]); blk._skipped = True
    if skip:
        print(f"SKIP adapter blocks (identity): {skip}", flush=True)
    cap_dir = os.environ.get("JSP_CAPTURE")   # C4: sampled inputs of every adapter Linear (GPTQ calibration)
    if cap_dir and not getattr(drafter.driver.adapter, "_cap_hooks", None):
        capd, per_call, cap_max = {}, int(os.environ.get("JSP_CAPTURE_PER_CALL", "48")), int(os.environ.get("JSP_CAPTURE_ROWS", "4096"))
        gen = torch.Generator().manual_seed(0)

        def hook(name):
            def fn(mod, args):
                x = args[0].detach().reshape(-1, args[0].shape[-1])
                have = sum(t.shape[0] for t in capd.get(name, []))
                if have >= cap_max:
                    return
                k = min(per_call, x.shape[0], cap_max - have)
                sel = torch.randperm(x.shape[0], generator=gen)[:k].to(x.device)
                capd.setdefault(name, []).append(x[sel].to("cpu", torch.float16))
            return fn
        hooks = [m.register_forward_pre_hook(hook(n)) for n, m in drafter.driver.adapter.named_modules() if isinstance(m, nn.Linear)]
        drafter.driver.adapter._cap_hooks = (hooks, capd, Path(cap_dir))
    actq = os.environ.get("JSP_ACTQ")   # E2: INT8 activations of the adapter MLPs (W4A8), in each matrix's rotated basis
    if actq and not getattr(drafter.driver.adapter, "_actq_hooks", None):
        install_actq(drafter.driver.adapter, actq, os.environ.get("JSP_ACTQ_SCALES"), os.environ.get("JSP_ACTQ_PARTS", "mlp"))
    rows = OutcomeRows(drafter, llm.runner) if a.rows == "outcome" else None
    out = Path(a.out) if not multi else Path(a.out).with_name(Path(a.out).stem + f"_{dataset}.json")
    banner = dict(jetspec_src=str(SRC), jetspec_commit=os.popen(f"git -C {SRC} rev-parse HEAD").read().strip(),
                  rows=a.rows, dataset=dataset, traces=str(tdir),
                  release=a.release, target_root=a.target_root, schedule=a.schedule, device=a.device,
                  dtype=str(dtype), config=config.__dict__, max_new=a.max_new, loaded_target_tensors=len(llm.model.loaded),
                  target_load_s=round(t_target, 1), drafter_build_s=round(t_build, 1), torch=torch.__version__,
                  quant=a.quant or None, skip_blocks=os.environ.get("JSP_SKIP_BLOCKS") or None,
                  actq=(f"{actq}:{os.environ.get('JSP_ACTQ_PARTS', 'mlp')}" + (f":{os.environ.get('JSP_ACTQ_SCALES')}" if actq == "tensor" else "")) if actq else None,
                  mps_allocated_gb=round(torch.mps.current_allocated_memory() / 1e9, 2)
                  if torch.backends.mps.is_available() else None)
    print("BANNER " + json.dumps(banner), flush=True)
    dirs = sorted(d for d in glob.glob(f"{tdir}/*") if Path(d, "meta.json").exists())
    if a.limit:
        dirs = dirs[:a.limit]
    res = []
    if out.exists():   # resume (20 Nov 2026): keep the traces an earlier run of the same command finished
        old = json.loads(out.read_text())
        ob = old.get("banner", {})
        if (ob.get("schedule") == a.schedule and ob.get("rows") == a.rows and ob.get("quant") == (a.quant or None)
                and ob.get("actq") == banner.get("actq")):
            res = old.get("results", [])
            print(f"RESUME: {len(res)} traces already in {out}", flush=True)
    done = {r["trace"] for r in res}
    for d in dirs:
        if Path(d).name in done:
            continue
        tr = Trace(d)
        cap = min(a.max_new, tr.known - tr.P - (0 if tr.meta.get("stop_appended") else 16))
        if cap < 1:   # 20 Nov 2026: a reply shorter than the reader's 16-token margin (e.g. mt-bench 0, 14 tokens)
            print(f"{Path(d).name}: SKIP (stream {tr.known - tr.P} tokens, cap {cap})", flush=True)
            continue
        r = run_trace(llm, drafter, config, tr, max_new=cap)
        if rows is not None:
            r["rows_ran"], r["rows_skipped"] = rows.ran, rows.skipped
            rows.ran = rows.skipped = 0
        res.append(r)
        print(f"{r['trace']}: prompt {r['prompt_len']}, {r['generated']} tokens in {r['rounds']} rounds, tau "
              f"{r['tau']:.3f}, hits {r['counts'].get('hits', 0)} / misses {r['counts'].get('misses', 0)}, "
              f"{r['seconds']:.1f} s ({r['s_per_round']:.2f} s / round), lossless {r['lossless_vs_trace']}", flush=True)
        out.write_text(json.dumps(dict(banner=banner, results=res), indent=1))
    acc = [x for r in res for x in r["accept_lengths"]]
    summary = dict(dataset=dataset, traces=len(res), rounds=len(acc), tau_pooled=sum(acc) / len(acc) if acc else None,
                   tok_per_round=sum(r["generated"] for r in res) / max(1, sum(r["rounds"] for r in res)),
                   all_lossless=all(r["lossless_vs_trace"] for r in res))
    print("SUMMARY " + json.dumps(summary), flush=True)
    out.write_text(json.dumps(dict(banner=banner, summary=summary, results=res), indent=1))
    ch = getattr(drafter.driver.adapter, "_cap_hooks", None)
    if ch:
        _, cap, d = ch; d.mkdir(parents=True, exist_ok=True)
        for n, ts in cap.items():
            np.save(d / f"{n}.npy", torch.cat(ts).numpy())
        print(f"CAPTURE: {len(cap)} Linear inputs, {sum(sum(t.shape[0] for t in v) for v in cap.values())} rows -> {d}", flush=True)

if __name__ == "__main__":
    main()
