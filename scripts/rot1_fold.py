"""rot1: the one shared fold of a residual-stream rotation R into Qwen3.8-27B (RUNBOOK R0.1, option b3: fold on the fly).

Every consumer (GPTQ loader, precision checker, proxy, QAT student, KL eval) folds the original BF16 weights with these
functions, in FP32, so two consumers can never fold differently. Only R.npy + basis.json (+ the rotated embedding table
and small checkpoint, scripts/qwen38_rotate_checkpoint.py) are persisted.

Convention: residual h (column) -> h' = R h; with row vectors h' = h R^T.
  readers  W' = W diag(1 + g) R^T   (input_layernorm gain into GDN in_proj_qkv / z / a / b and attention q / k / v,
                                     post_attention_layernorm gain into gate / up, the final norm gain into lm_head)
  writers  W' = R W                 (GDN out_proj, attention o_proj, mlp down_proj)
  embed    E' = E R^T               (rows e' = R e)
  norms    gain (1 + w) -> 1, i.e. w = 0 (Qwen3_5RMSNorm computes x_hat * (1 + w))
RMSNorm commutes with R (rms(R h) = rms(h)), so the folded model computes the same function.

R_KIND: identity | had20x256 (H_20 (Paley) kron H_256, / sqrt(5120), times diag(random signs from R_SEED))
        | block1024 (blockdiag(H_1024) / 32 times signs: the transpose of qwen38_kl.rotation(n, seed)) | file:<path.npy>
identity skips the matmul (exact). On CUDA, strict_fp32() disables TF32 (otherwise FP32 matmuls silently use TF32).

basis.json = the basis ID: name, r_kind, r_seed, n, fold_version, fold_norms, r_sha256 (of R.npy's float32 bytes),
probe_seed, fingerprint, basis_id (sha256 of the canonical JSON of all other fields). The fingerprint holds probe values of three
folded matrices (first GDN reader in_proj_qkv, first attention writer o_proj, lm_head): W' p for a fixed random p, plus
||W'||_F. A sha256 of FP32 GEMM output is not reproducible across BLAS libraries / devices, so consumers re-fold the
three matrices on their own device and compare the probes to a relative tolerance (check_basis).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

FOLD_VERSION = "rot1-fold-v1"
HIDDEN = 5120
PREFIX = "model.language_model."
READERS = {"linear_attention": ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_a",
                                "linear_attn.in_proj_b"),
           "full_attention": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")}
WRITERS = {"linear_attention": ("linear_attn.out_proj",), "full_attention": ("self_attn.o_proj",)}
MLP_READERS = ("mlp.gate_proj", "mlp.up_proj")
MLP_WRITERS = ("mlp.down_proj",)
NORM_IN, NORM_POST = "input_layernorm", "post_attention_layernorm"
TAPS = (5, 19, 33, 47, 61)  # DFlash2 hidden-state taps (layer outputs)
PROBE_SEED = 12345        # fingerprint probe vector seed (recorded in basis.json)
FP_TOL = 1e-4               # fingerprint relative tolerance (FP32 fold differences across devices are ~1e-6)


def strict_fp32():
    """Real FP32 on CUDA (RUNBOOK section 4): no TF32 in matmul or cuDNN."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def hadamard_matrix(n: int) -> np.ndarray:
    """Unnormalized +-1 Hadamard matrix of order n: Sylvester for powers of two, Paley I for q + 1 (q prime, 3 mod 4),
    and Kronecker products of those (e.g. 20 = Paley(19))."""
    if n == 1:
        return np.ones((1, 1))
    if n & (n - 1) == 0:
        h = np.ones((1, 1))
        while len(h) < n:
            h = np.block([[h, h], [h, -h]])
        return h
    q = n - 1
    if q % 4 == 3 and all(q % d for d in range(2, int(q ** 0.5) + 1)):
        sq = {(x * x) % q for x in range(1, q)}
        chi = np.array([0] + [1 if x in sq else -1 for x in range(1, q)])
        jac = np.array([[chi[(j - i) % q] for j in range(q)] for i in range(q)], dtype=float)
        s = np.zeros((n, n))
        s[0, 1:], s[1:, 0], s[1:, 1:] = 1, -1, jac
        h = np.eye(n) + s
    else:
        raise ValueError(f"no Hadamard construction for order {n}")
    assert np.array_equal(h @ h.T, n * np.eye(n)), n
    return h


def make_rotation(kind: str, seed: int | None, n: int = HIDDEN) -> np.ndarray | None:
    """R as float64 (n, n), or None for identity. h' = R h."""
    if kind == "identity":
        return None
    if kind.startswith("file:"):
        r = np.load(kind[5:]).astype(np.float64)
        assert r.shape == (n, n), r.shape
        return r
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], n)
    if kind == "had20x256":
        assert n == 20 * 256, n
        k = np.kron(hadamard_matrix(20), hadamard_matrix(256)) / np.sqrt(n)
    elif kind == "block1024":
        k = np.kron(np.eye(n // 1024), hadamard_matrix(1024)) / 32.0
    else:
        raise ValueError(kind)
    return k * signs[None, :]  # R = K diag(s)


def r_sha256(r32: np.ndarray | None) -> str:
    return "identity" if r32 is None else hashlib.sha256(np.ascontiguousarray(r32, np.float32).tobytes()).hexdigest()


# ---- the fold -----------------------------------------------------------------------------------------------------
def fold_reader(w: torch.Tensor, gain: torch.Tensor | None, r: torch.Tensor | None) -> torch.Tensor:
    """W diag(1 + g) R^T in FP32 (gain = the RMSNorm weight w; None: no gain fold)."""
    x = w.float()
    if gain is not None:
        x = x * (1.0 + gain.float().to(x.device))[None, :]
    if r is not None:
        x = x @ r.to(x.device, torch.float32).T
    return x


def fold_writer(w: torch.Tensor, r: torch.Tensor | None) -> torch.Tensor:
    x = w.float()
    return x if r is None else r.to(x.device, torch.float32) @ x


def fold_embed(e: torch.Tensor, r: torch.Tensor | None, rows: int = 16384) -> torch.Tensor:
    """E R^T in FP32, in row chunks (the table is 248K x 5120)."""
    if r is None:
        return e.float()
    rt = r.to(e.device, torch.float32).T
    return torch.cat([e[i:i + rows].float() @ rt for i in range(0, len(e), rows)])


def fold_layer_tensors(t: dict, kind: str, r: torch.Tensor | None, fold_norms: bool = True) -> dict:
    """Fold one decoder layer given as {layer-relative key: tensor} (e.g. 'mlp.gate_proj.weight'); returns a new dict
    with the folded matrices in FP32 and the two norm weights set to 0 (if fold_norms); other tensors unchanged."""
    out = dict(t)
    g_in = t[f"{NORM_IN}.weight"] if fold_norms else None
    g_post = t[f"{NORM_POST}.weight"] if fold_norms else None
    for name in READERS[kind]:
        out[f"{name}.weight"] = fold_reader(t[f"{name}.weight"], g_in, r)
    for name in MLP_READERS:
        out[f"{name}.weight"] = fold_reader(t[f"{name}.weight"], g_post, r)
    for name in WRITERS[kind] + MLP_WRITERS:
        out[f"{name}.weight"] = fold_writer(t[f"{name}.weight"], r)
    if fold_norms:
        for nm in (NORM_IN, NORM_POST):
            out[f"{nm}.weight"] = torch.zeros_like(t[f"{nm}.weight"])
    return out


@torch.no_grad()
def fold_layer_module(layer, kind: str, r: torch.Tensor | None, fold_norms: bool = True, out_dtype=None):
    """In place on a transformers Qwen3_5DecoderLayer: compute the fold in FP32 on the parameters' device and store it
    in out_dtype (default: each parameter's own dtype, i.e. the pipeline's cast: bf16 for a bf16 model, fp32 for a
    layer moved to fp32 first)."""
    names = [f"{n}.weight" for n in READERS[kind] + MLP_READERS + WRITERS[kind] + MLP_WRITERS]
    names += [f"{NORM_IN}.weight", f"{NORM_POST}.weight"]
    t = {n: layer.get_parameter(n).data for n in names}
    new = fold_layer_tensors(t, kind, r, fold_norms)
    for n in names:
        p = layer.get_parameter(n)
        p.data = new[n].to(p.device, out_dtype or p.dtype)


@torch.no_grad()
def fold_globals(text, lm_head, r: torch.Tensor | None, fold_norms: bool = True, embed_dtype=None, head_dtype=None):
    """Embedding rows, final-norm gain into lm_head, final norm -> 0 (in place)."""
    e = text.embed_tokens.weight
    e.data = fold_embed(e.data, r).to(embed_dtype or e.dtype)
    g = text.norm.weight.data if fold_norms else None
    w = lm_head.weight
    w.data = fold_reader(w.data, g, r).to(head_dtype or w.dtype)
    if fold_norms:
        text.norm.weight.data = torch.zeros_like(text.norm.weight.data)


@torch.no_grad()
def fold_model(model, r: torch.Tensor | None, fold_norms: bool = True, device="cpu"):
    """Fold a whole transformers Qwen3_5ForConditionalGeneration in place, layer by layer (FP32 compute on `device`,
    stored back in each parameter's dtype and original device)."""
    text = model.model.language_model
    for i, layer in enumerate(text.layers):
        home = next(layer.parameters()).device
        layer.to(device)
        fold_layer_module(layer, text.config.layer_types[i], r, fold_norms)
        layer.to(home)
    fold_globals(text, model.lm_head, r, fold_norms)


# ---- basis ID -----------------------------------------------------------------------------------------------------
def _wmap(model_dir: Path) -> dict:
    return json.loads((Path(model_dir) / "model.safetensors.index.json").read_text())["weight_map"]


def get_tensor(model_dir: Path, key: str, wmap: dict | None = None) -> torch.Tensor:
    from safetensors import safe_open
    wmap = wmap or _wmap(model_dir)
    with safe_open(Path(model_dir) / wmap[key], framework="pt") as f:
        return f.get_tensor(key)


def layer_types(model_dir: Path) -> list:
    c = json.loads((Path(model_dir) / "config.json").read_text())
    return c.get("text_config", c)["layer_types"]


def fingerprint(model_dir: Path, r: torch.Tensor | None, fold_norms: bool = True, device="cpu") -> dict:
    """Probe values of three folded matrices (see module doc)."""
    model_dir, wmap = Path(model_dir), _wmap(model_dir)
    lt = layer_types(model_dir)
    g, a = lt.index("linear_attention"), lt.index("full_attention")
    rr = None if r is None else r.to(device)
    out = {}
    specs = (("gdn_reader", f"{PREFIX}layers.{g}.linear_attn.in_proj_qkv.weight", f"{PREFIX}layers.{g}.{NORM_IN}.weight", "r"),
             ("attn_writer", f"{PREFIX}layers.{a}.self_attn.o_proj.weight", None, "w"),
             ("lm_head", "lm_head.weight", f"{PREFIX}norm.weight", "r"))
    for name, key, gkey, role in specs:
        w = get_tensor(model_dir, key, wmap).to(device)
        if role == "r":
            gain = get_tensor(model_dir, gkey, wmap).to(device) if fold_norms else None
            wf = fold_reader(w, gain, rr)
        else:
            wf = fold_writer(w, rr)
        p = torch.from_numpy(np.random.default_rng(PROBE_SEED).standard_normal(wf.shape[1])).to(device, torch.float64)
        rows = torch.linspace(0, wf.shape[0] - 1, 16).long().to(device)
        probe = wf[rows].double() @ p
        out[name] = {"key": key, "rows": rows.tolist(), "probe": [float(v) for v in probe.cpu()],
                     "fro": float(wf.double().norm())}
        del w, wf
    return out


def basis_id(b: dict) -> str:
    body = {k: v for k, v in b.items() if k != "basis_id"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_basis(rot_dir: Path, device="cpu"):
    """(R as an FP32 tensor or None for identity, basis dict); verifies R.npy's sha256 and the basis_id."""
    rot_dir = Path(rot_dir)
    b = json.loads((rot_dir / "basis.json").read_text())
    assert b["fold_version"] == FOLD_VERSION, f"{rot_dir}: fold version {b['fold_version']} != {FOLD_VERSION}"
    assert basis_id(b) == b["basis_id"], f"{rot_dir}/basis.json: basis_id does not match its content"
    if b["r_kind"] == "identity":
        return None, b
    r = np.load(rot_dir / "R.npy")
    assert r_sha256(r) == b["r_sha256"], f"{rot_dir}/R.npy: sha256 mismatch with basis.json"
    return torch.from_numpy(np.ascontiguousarray(r, np.float32)).to(device), b


def compare_fingerprint(a: dict, b: dict, tol: float = FP_TOL) -> float:
    """Max relative probe / norm difference between two fingerprints; raises if above tol."""
    worst = 0.0
    for k in b:
        pa, pb = np.array(a[k]["probe"]), np.array(b[k]["probe"])
        d = max(np.abs(pa - pb).max() / max(np.abs(pb).max(), 1e-30), abs(a[k]["fro"] - b[k]["fro"]) / b[k]["fro"])
        worst = max(worst, float(d))
        if not d <= tol:
            raise RuntimeError(f"basis fingerprint mismatch on {k}: rel diff {d:.3e} > {tol:g} (wrong R, fold code or "
                               f"source checkpoint)")
    return worst


def check_basis(rot_dir: Path, model_dir: Path, device="cpu"):
    """Load the basis and re-fold the fingerprint matrices here; returns (R, basis dict). Every consumer calls this."""
    if device != "cpu" and torch.device(device).type == "cuda":
        strict_fp32()
    r, b = load_basis(rot_dir, device)
    d = compare_fingerprint(fingerprint(model_dir, r, b["fold_norms"], device), b["fingerprint"])
    print(f"rot1 basis {b['name']} ({b['r_kind']}, seed {b['r_seed']}) id {b['basis_id'][:16]}: fingerprint ok "
          f"(rel diff {d:.1e})", flush=True)
    return r, b


def require_same_basis(export_dir: Path | None, b: dict | None, what: str = "export"):
    """Refuse a basis mismatch between an export (its basis.json sidecar) and the consumer's ROT_DIR basis."""
    if export_dir is None:
        return
    side = Path(export_dir) / "basis.json"
    eb = json.loads(side.read_text()) if side.exists() else None
    if eb is None and b is None:
        return
    if eb is None:
        raise RuntimeError(f"{what} {export_dir} has no basis.json but ROT_DIR is basis {b['basis_id'][:16]}")
    if b is None:
        raise RuntimeError(f"{what} {export_dir} is in basis {eb['basis_id'][:16]} ({eb['name']}); set ROT_DIR to it")
    if eb["basis_id"] != b["basis_id"]:
        raise RuntimeError(f"basis mismatch: {what} {eb['basis_id'][:16]} ({eb['name']}) vs ROT_DIR "
                           f"{b['basis_id'][:16]} ({b['name']})")


def basis_meta(b: dict | None) -> dict:
    """Safetensors header metadata carrying the basis ID verbatim (R0.5); empty without a basis (exports unchanged)."""
    return {} if b is None else {"rot1_basis_id": b["basis_id"], "rot1_basis": json.dumps(b, sort_keys=True)}


def write_basis_sidecar(b: dict | None, out_dir: Path):
    if b is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "basis.json").write_text(json.dumps(b, indent=1, sort_keys=True))


def from_env(model_dir: Path, device="cpu"):
    """ROT_DIR env -> (R, basis) via check_basis, or (None, None) when unset."""
    d = os.environ.get("ROT_DIR")
    return check_basis(Path(os.path.expanduser(d)), model_dir, device) if d else (None, None)
