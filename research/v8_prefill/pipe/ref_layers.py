"""Float32 reference of the release target's layers in canonical form (what a GPU implementation must compute), checked
against the ANE's own chunk outputs on real text. Mirrors qwen38_coreai_build LayerW / GDNW / AttnW numerics without
the ANE fp16 tricks (GDN_SQ / GDN_SV scalings are exact and dropped; attention in fp32 with an unquantized cache).
    BUILDPY ref_layers.py HIDDEN.npz FIRST_LAYER [N_LAYERS]     (default 4 = one chunk)"""
import sys, time
import numpy as np, torch
from scipy.linalg import hadamard

V02 = "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02"
sys.path[:0] = [V02 + "/coreai", V02 + "/scripts"]
import qwen38_ane_model as M  # noqa: E402

CFG = M.cfg()
EPS = CFG["rms_norm_eps"]
HID, INTER = CFG["hidden_size"], CFG["intermediate_size"]
NK, NV, DK, DV = 16, 48, 128, 128
NH, NKV, HD = 24, 4, 256
ROT = int(HD * CFG["rope_parameters"]["partial_rotary_factor"])
THETA = CFG["rope_parameters"]["rope_theta"]
H1024 = torch.from_numpy(hadamard(1024).astype(np.float32))
f32 = torch.float32


def rot(x, seed):  # (N, n): per 1024 block H (s * x) / 32 (builder Hadamard)
    if seed is None or int(seed) < 0:
        return x
    n = x.shape[1]
    s = torch.from_numpy(np.random.default_rng(int(seed)).choice([-1.0, 1.0], n).astype(np.float32))
    return ((x * s).reshape(x.shape[0], n // 1024, 1024) @ H1024.T / 32).reshape(x.shape)


def rms(x, w, eps=EPS):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * w


def silu(x):
    return 0.5 * x * (1 + torch.tanh(0.5 * x))


def t(a):
    return torch.from_numpy(np.asarray(a, np.float32))


def proj(W, key, x):
    """QConv in float32: LUT (lut[idx] * scale), INT8 (codes * scale) or dense, + the low-rank a (b x)."""
    if key + "/lut" in W:
        lut, idx = t(W[key + "/lut"]), torch.from_numpy(W[key + "/idx"].astype(np.int64))
        w = torch.stack([lut[:, j][idx] for j in range(lut.shape[1])], 1).reshape(-1, idx.shape[1])
        y = x @ w.T
        if key + "/scale" in W:
            y = y * t(W[key + "/scale"])
    elif key + "/int8" in W:
        y = (x @ t(W[key + "/int8"]).T) * t(W[key + "/scale"])
    else:
        y = x @ t(W[key + "/dense"] if key + "/dense" in W else W[key]).T
    if key + "/lr_a" in W:
        y = y + (x @ t(W[key + "/lr_b"]).T) @ t(W[key + "/lr_a"]).T
    return y


def gdn(W, i, h, s_in, s_out):
    p = f"{i}/linear_attn."
    N = h.shape[0]
    hq = rot(h, s_in)
    qkv, z = proj(W, p + "in_proj_qkv.weight", hq), proj(W, p + "in_proj_z.weight", hq)
    b, a = h @ t(W[p + "in_proj_b.weight"]).T, h @ t(W[p + "in_proj_a.weight"]).T
    cw = t(W[p + "conv1d.weight"][:, 0].T)                                       # (4, cdim)
    rows = torch.cat([torch.zeros(3, qkv.shape[1]), qkv])
    conv = silu(sum(rows[j:j + N] * cw[j] for j in range(4)))
    kd = NK * DK
    q, k, v = conv[:, :kd], conv[:, kd:2 * kd], conv[:, 2 * kd:]
    l2 = lambda u, s: u * torch.rsqrt((u * u).sum(-1, keepdim=True) + 1e-6) * s  # noqa: E731
    q = l2(q.reshape(N, NK, DK), DK ** -0.5).repeat_interleave(NV // NK, 1)    # (N, NV, DK): v head h <- k head h // 3
    k = l2(k.reshape(N, NK, DK), 1.0).repeat_interleave(NV // NK, 1)
    v = v.reshape(N, NV, DV)
    beta = torch.sigmoid(b)                                                       # (N, NV)
    g = torch.nn.functional.softplus(a + t(W[p + "dt_bias"])) * -torch.exp(t(W[p + "A_log"]))
    S, o = torch.zeros(NV, DK, DV), torch.empty(N, NV, DV)
    dec = torch.exp(g)
    for n in range(N):
        S = S * dec[n].reshape(NV, 1, 1)
        u = beta[n].reshape(NV, 1) * (v[n] - torch.einsum("hk,hkv->hv", k[n], S))
        S = S + k[n].unsqueeze(2) * u.unsqueeze(1)
        o[n] = torch.einsum("hk,hkv->hv", q[n], S)
    o = rms(o, t(W[p + "norm.weight"])) * silu(z.reshape(N, NV, DV))
    return proj(W, p + "out_proj.weight", rot(o.reshape(N, NV * DV), s_out))


def attn(W, i, h, s_in, s_out):
    p = f"{i}/self_attn."
    N = h.shape[0]
    hq = rot(h, s_in)
    qg = proj(W, p + "q_proj.weight", hq).reshape(N, NH, 2 * HD)
    q, gate = rms(qg[..., :HD], 1 + t(W[p + "q_norm.weight"])), qg[..., HD:].reshape(N, NH * HD)
    k = rms(proj(W, p + "k_proj.weight", hq).reshape(N, NKV, HD), 1 + t(W[p + "k_norm.weight"]))
    v = proj(W, p + "v_proj.weight", hq).reshape(N, NKV, HD)
    inv = 1.0 / THETA ** (np.arange(0, ROT, 2) / ROT)
    ang = np.concatenate([np.outer(np.arange(N), inv)] * 2, 1)
    c, s = t(np.cos(ang)).reshape(N, 1, ROT), t(np.sin(ang)).reshape(N, 1, ROT)

    def rope(x):
        r = x[..., :ROT]
        return torch.cat([r * c + torch.cat([-r[..., ROT // 2:], r[..., :ROT // 2]], -1) * s, x[..., ROT:]], -1)
    q, k = rope(q), rope(k)
    kk, vv = k.repeat_interleave(NH // NKV, 1), v.repeat_interleave(NH // NKV, 1)  # q head h <- kv head h // 6
    sc = torch.einsum("nhd,mhd->hnm", q, kk) * HD ** -0.5
    sc = sc + torch.triu(torch.full((N, N), -1e30), 1)
    o = torch.einsum("hnm,mhd->nhd", torch.softmax(sc, -1), vv).reshape(N, NH * HD) * torch.sigmoid(gate)
    return proj(W, p + "o_proj.weight", rot(o, s_out))


def mlp(W, i, x, s_in, s_mid):
    h = rot(rms(x, 1 + t(W[f"{i}/post_attention_layernorm.weight"])), s_in)
    a = rot(silu(proj(W, f"{i}/mlp.gate_proj.weight", h)) * proj(W, f"{i}/mlp.up_proj.weight", h), s_mid)
    return x + proj(W, f"{i}/mlp.down_proj.weight", a)


def layer(W, i, x):
    m_in, m_out = W.get(f"{i}/mix.rotation", [-1, -1])
    r_in, r_mid = W.get(f"{i}/mlp.rotation", [-1, -1])
    h = rms(x, 1 + t(W[f"{i}/input_layernorm.weight"]))
    y = gdn(W, i, h, m_in, m_out) if CFG["layer_types"][i] == "linear_attention" else attn(W, i, h, m_in, m_out)
    return mlp(W, i, x + y, r_in, r_mid)


PER_LAYER = []


def main():
    import qwen38_coreai_build as B
    Z = np.load(sys.argv[1])
    first, n = int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 4
    torch.set_num_threads(8)
    ck = M.Checkpoint()
    x = torch.from_numpy(Z[f"after_layer_{first - 1}"].astype(np.float32)) if first else \
        torch.from_numpy(np.asarray(ck.embed_table()[Z["ids"]], np.float32))
    for i in range(first, first + n):
        t0 = time.time()
        W = B.layer_arrays(ck, i)
        with torch.no_grad():
            x = layer(W, i, x)
        key = f"after_layer_{i}"
        msg = f"layer {i} ({CFG['layer_types'][i]}) {time.time() - t0:.0f}s"
        PER_LAYER.append(x.clone())
        if key in Z.files:
            ane = torch.from_numpy(Z[key].astype(np.float32))
            d = (x - ane).norm(dim=1) / ane.norm(dim=1)
            msg += (f" | vs ANE chunk output: rel err {float((x - ane).norm() / ane.norm()):.4f}, per-token median "
                    f"{float(d.median()):.4f}, p99 {float(d.quantile(0.99)):.4f}, cos {float(torch.nn.functional.cosine_similarity(x.double().flatten(), ane.double().flatten(), 0)):.6f}")
        print(msg, flush=True)
    np.save(sys.argv[1].replace(".npz", f"_ref_{first}_{first + n - 1}.npy"), x.numpy().astype(np.float16))


if __name__ == "__main__":
    main()
