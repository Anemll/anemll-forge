"""Lazy-commit Gated DeltaNet state for DFlash2 block verification: one GDN layer, several functions sharing
ONE state layout (prototype for the target rebuild).

State per GDN layer (P = pending capacity = verify block, H = 3 conv history rows):
  conv (H + P, 10240): rows 0..2 = the 3 raw qkv rows before the last call's first position, rows 3.. = the
       last call's raw qkv rows (last P of them if T > P). Every call writes [prev3 | its rows]; prev3 is picked
       from the stored buffer by a host one-hot prev_sel (3, H + P), so any start alignment works.
  rec  (48, 128 + 3P + 1, 128): rows 0..127 = committed recurrent state S (k-major), then per pending row its key
       k_j, its UT-transformed value u_j and key wk_j (computed by the call that produced the rows, from its own
       inputs: u = (I + N)^-1 (beta v), wk = (I + N)^-1 (beta k exp(cum)), N_ij = beta_i k_i.k_j exp(cum_i - cum_j),
       i > j), and one row [cum_0 .. cum_P-1, 0 ...] (inclusive log-decay cumsum).
Every call first commits the first k pending rows into S (host inputs: commit (1, P, 1) = [1]*k + [0]*(P-k) and
commit_last (1, P, 1) = one-hot of row k-1, all zero for k = 0):
  S' = S exp(cum_k-1) + (k_j c_j exp(min(cum_k-1 - cum_j, 0)))^T (u - wk S)
Exact for any k: for a unit lower-triangular system the leading rows of the solution do not depend on later rows.
No inverse is computed from state-derived values (a state-derived N @ N fails MIL->EIR with -14). Then the call
processes its own T rows from that committed state:
  lazy (decode T=1, verify T=8):   its rows become the new pending rows; outputs from the RETURNED state
  prefill_side (T > P):            commits its own rows (final state written), outputs from the in-graph
                                   intermediate states (consumers off to the side of the rec update)
  prefill_scratch (T <= 3P + 1):   same, but the outputs are routed through the rec scratch rows and read
                                   back from the returned value; the next call must pass commit = 0
ANE state rules respected: one update per state; raw reads only feed their own state's update value; every
other consumer uses the returned value (except prefill_side, which tests exactly that).

    python dflash2_gdn_lazy.py            # build, run a mixed call sequence, compare with qwen38_decode_ref
Env: UNITS (CPU_ONLY / CPU_AND_NE), LAYER (a DeltaNet layer of the test tensors), VARIANTS (side,scratch).
"""
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import Function, Program, types
import qwen38_ane_chunk as C
from qwen38_decode_ref import DecodeLayer, load_layer_weights, text_config

LAYER = int(os.environ.get("LAYER", "30"))
P, H = 8, 3
OUT = Path(os.path.expanduser(os.environ.get("OUT", "~/Models/dflash2/gdn_lazy")))
f16 = np.float16
torch.set_grad_enabled(False)



def _softplus(x):  # overflow-safe on the ANE (fp16 softplus returns 0 for x >~ 11); see qwen38_ane_chunk.softplus
    return mb.add(x=mb.relu(x=x), y=mb.log(x=mb.add(x=mb.exp(x=mb.mul(x=mb.abs(x=x), y=np.float16(-1))), y=np.float16(1))))

def dims(cfg):
    nk, nv, dk, dv = (cfg[k] for k in ("linear_num_key_heads", "linear_num_value_heads",
                                        "linear_key_head_dim", "linear_value_head_dim"))
    return nk, nv, dk, dv, 2 * nk * dk + nv * dv


# ---- gated delta rule on one chunk of T <= 8 rows (UT transform), split into reusable pieces -----------------
# (chunkwise WY / UT-transform form: Songlin Yang, "DeltaNet Explained (Part II)", https://sustcsonglin.github.io/blog/2024/deltanet-2/)
def delta_core(kh, vh, beta, g, T, nv):
    """cum (nv,T,1) inclusive log-decay cumsum, pair (nv,T,T) = exp(cum_i - cum_j) [i >= j], and u, wk such that
    the rows' effective values given the chunk's initial state S0 are vn = u - wk @ S0."""
    i, j = np.meshgrid(np.arange(T), np.arange(T), indexing="ij")
    l_inc, l_str = (i >= j).astype(f16), (i > j).astype(f16)
    cum = mb.reshape(x=mb.matmul(x=mb.reshape(x=g, shape=(nv, 1, T)), y=np.ascontiguousarray(l_inc.T)),
                     shape=(nv, T, 1))
    pair = mb.mul(x=mb.exp(x=mb.minimum(x=mb.sub(x=cum, y=mb.reshape(x=cum, shape=(nv, 1, T))), y=f16(0))),
                  y=l_inc)
    kb, vb = mb.mul(x=kh, y=beta), mb.mul(x=vh, y=beta)
    n = mb.mul(x=mb.matmul(x=kb, y=kh, transpose_y=True), y=mb.mul(x=pair, y=l_str))
    eye = np.eye(T, dtype=f16)
    inv, npow, m = mb.sub(x=eye, y=n), n, 1
    while 2 * m < T:                                   # (I + N)^-1 = (I - N)(I + N^2)(I + N^4), N nilpotent
        # N^2 as I - (I - N)(I + N): a state-derived N @ N fails MIL->EIR (-14), this form loads
        npow = mb.sub(x=eye, y=mb.matmul(x=mb.sub(x=eye, y=npow), y=mb.add(x=eye, y=npow)))
        inv = mb.matmul(x=inv, y=mb.add(x=eye, y=npow))
        m *= 2
    return cum, pair, mb.matmul(x=inv, y=vb), mb.matmul(x=inv, y=mb.mul(x=kb, y=mb.exp(x=cum)))


def delta_state(s, kh, T, nv, core):
    """State after the chunk: S0 exp(total) + (k exp(total - cum))^T vn."""
    cum, _, u, wk = core
    vn = mb.sub(x=u, y=mb.matmul(x=wk, y=s))
    total = mb.slice_by_index(x=cum, begin=[0, T - 1, 0], end=[nv, T, 1])
    kd = mb.mul(x=kh, y=mb.exp(x=mb.sub(x=total, y=cum)))
    return mb.add(x=mb.mul(x=s, y=mb.exp(x=total)), y=mb.matmul(x=kd, y=vn, transpose_x=True))


def delta_out(s, qh, kh, core):
    """Row outputs q_t S_t of the chunk: (q exp(cum)) S0 + ((q k^T) * pair) vn."""
    cum, pair, u, wk = core
    vn = mb.sub(x=u, y=mb.matmul(x=wk, y=s))
    return mb.add(x=mb.matmul(x=mb.mul(x=qh, y=mb.exp(x=cum)), y=s),
                  y=mb.matmul(x=mb.mul(x=mb.matmul(x=qh, y=kh, transpose_y=True), y=pair), y=vn))


def rows_slice(t, a, b, nv, d):
    return mb.slice_by_index(x=t, begin=[0, a, 0], end=[nv, b, d])


def pad_rows(t, T, n, nv, d):
    return t if T == n else mb.concat(values=[t, np.zeros((nv, n - T, d), f16)], axis=1)


def gdn_lazy(cfg, w, q, h, conv_st, rec_st, commit, commit_last, prev_sel, T, mode):
    nk, nv, dk, dv, cdim = dims(cfg)
    kd, vd, eps = nk * dk, nv * dv, cfg["rms_norm_eps"]
    qkv = mb.reshape(x=C.lut_linear(h, q["linear_attn.in_proj_qkv.weight"]), shape=(cdim, T))
    z = mb.transpose(x=mb.reshape(x=C.lut_linear(h, q["linear_attn.in_proj_z.weight"]), shape=(nv, dv, T)),
                     perm=[0, 2, 1])
    b = mb.reshape(x=C.dense_linear(h, w["linear_attn.in_proj_b.weight"]), shape=(nv, T, 1))
    a = mb.reshape(x=C.dense_linear(h, w["linear_attn.in_proj_a.weight"]), shape=(nv, T, 1))
    beta = mb.sigmoid(x=b)
    neg_a = (-w["linear_attn.A_log"].exp()).numpy().astype(f16).reshape(nv, 1, 1)
    dt = w["linear_attn.dt_bias"].numpy().astype(f16).reshape(nv, 1, 1)
    g = mb.mul(x=_softplus(mb.add(x=a, y=dt)), y=neg_a)                          # log decay (nv, T, 1)

    # conv buffer: write [prev3 | this call's rows]; prev3 read back from the returned value
    rows = mb.transpose(x=qkv, perm=[1, 0])                                          # (T, cdim)
    tail = mb.slice_by_index(x=rows, begin=[T - P, 0], end=[T, cdim]) if T > P else (
        rows if T == P else mb.concat(values=[rows, np.zeros((P - T, cdim), f16)], axis=0))
    conv_ret = mb.coreml_update_state(state=conv_st, value=mb.concat(
        values=[mb.matmul(x=prev_sel, y=mb.read_state(input=conv_st)), tail], axis=0))
    prev = mb.transpose(x=mb.slice_by_index(x=conv_ret, begin=[0, 0], end=[H, cdim]), perm=[1, 0])
    seq = mb.concat(values=[prev, qkv], axis=1)                                      # (cdim, T + 3)
    cw = w["linear_attn.conv1d.weight"][:, 0].numpy().astype(f16)
    conv = None
    for j in range(4):
        term = mb.mul(x=mb.slice_by_index(x=seq, begin=[0, j], end=[cdim, j + T]), y=cw[:, j:j + 1])
        conv = term if conv is None else mb.add(x=conv, y=term)
    qq, kk, vv = mb.split(x=C.silu(conv), split_sizes=[kd, kd, vd], axis=0)
    rep = nv // nk

    def heads(t):  # (nk*dk, T) -> (nv, T, dk)
        t = mb.transpose(x=mb.reshape(x=t, shape=(nk, dk, T)), perm=[0, 2, 1])
        return mb.reshape(x=mb.tile(x=mb.reshape(x=t, shape=(nk, 1, T, dk)), reps=[1, rep, 1, 1]), shape=(nv, T, dk))

    def l2n(t, scale):
        ss = mb.reduce_sum(x=mb.mul(x=t, y=t), axes=[-1], keep_dims=True)
        return mb.mul(x=mb.mul(x=t, y=mb.rsqrt(x=ss, epsilon=1e-6)), y=f16(scale))

    qh, kh = l2n(heads(qq), dk ** -0.5), l2n(heads(kk), 1.0)
    vh = mb.transpose(x=mb.reshape(x=vv, shape=(nv, dv, T)), perm=[0, 2, 1])

    # commit the first k pending rows (raw read -> rec update value only)
    r = mb.read_state(input=rec_st)
    s0 = rows_slice(r, 0, dk, nv, dv)
    kp, up = rows_slice(r, dk, dk + P, nv, dk), rows_slice(r, dk + P, dk + 2 * P, nv, dv)
    wkp = rows_slice(r, dk + 2 * P, dk + 3 * P, nv, dk)
    cum_p = mb.reshape(x=mb.slice_by_index(x=r, begin=[0, dk + 3 * P, 0], end=[nv, dk + 3 * P + 1, P]), shape=(nv, P, 1))
    total = mb.reduce_sum(x=mb.mul(x=cum_p, y=commit_last), axes=[1], keep_dims=True)          # cum_k-1 (nv, 1, 1)
    kd = mb.mul(x=mb.mul(x=kp, y=commit), y=mb.exp(x=mb.minimum(x=mb.sub(x=total, y=cum_p), y=f16(0))))
    s1 = mb.add(x=mb.mul(x=s0, y=mb.exp(x=total)),
                y=mb.matmul(x=kd, y=mb.sub(x=up, y=mb.matmul(x=wkp, y=s0)), transpose_x=True))

    def pending(kh_, core, n):  # rows to store for the next call's commit
        cum, _, u, wk = core
        crow = mb.concat(values=[t for t in (mb.reshape(x=cum, shape=(nv, 1, n)), np.zeros((nv, 1, dv - n), f16))
                                 if not (isinstance(t, np.ndarray) and t.size == 0)], axis=2)
        return [pad_rows(kh_, n, P, nv, dk), pad_rows(u, n, P, nv, dv), pad_rows(wk, n, P, nv, dk), crow]

    if mode == "lazy":
        assert T <= P
        core = delta_core(kh, vh, beta, g, T, nv)
        ret = mb.coreml_update_state(state=rec_st, value=mb.concat(values=[s1] + pending(kh, core, T), axis=1))
        o = delta_out(rows_slice(ret, 0, dk, nv, dv), qh, kh, core)
    else:
        assert T % P == 0
        s, outs = s1, []
        for c0 in range(0, T, P):
            sl = [rows_slice(t, c0, c0 + P, nv, d) for t, d in ((qh, dk), (kh, dk), (vh, dv), (beta, 1), (g, 1))]
            core = delta_core(sl[1], sl[2], sl[3], sl[4], P, nv)
            outs.append(delta_out(s, sl[0], sl[1], core))
            s = delta_state(s, sl[1], P, nv, core)
        o = mb.concat(values=outs, axis=1) if len(outs) > 1 else outs[0]            # (nv, T, dv)
        if mode == "prefill_side":
            mb.coreml_update_state(state=rec_st, value=mb.concat(values=[s, np.zeros((nv, 3 * P + 1, dv), f16)], axis=1))
        else:  # prefill_scratch: route the outputs through the state (next call: commit = 0)
            assert T <= 3 * P + 1
            ret = mb.coreml_update_state(state=rec_st, value=mb.concat(
                values=[s, pad_rows(o, T, 3 * P + 1, nv, dv)], axis=1))
            o = rows_slice(ret, dk, dk + T, nv, dv)
    o = mb.mul(x=C.rms_last(o, w["linear_attn.norm.weight"].numpy().astype(f16), eps), y=C.silu(z))
    o = mb.reshape(x=mb.transpose(x=o, perm=[0, 2, 1]), shape=(1, vd, 1, T))
    return C.lut_linear(o, q["linear_attn.out_proj.weight"])


def gdn_lazy_io(cfg, w, q, h, conv_in, rec_in, pend_in, commit, commit_last, T, lazy):
    """Host-owned-buffer version (GDN_IO): conv_in (3, cdim) = raw qkv rows of the 3 tokens before this block,
    rec_in (nv, dk, dv) = S committed as of the last call, pend_in (nv, 3P + 1, dv) = that call's pending rows
    [k | u | wk | cum]. Returns (y, conv rows (T + 3, cdim), rec_out = S' (commit applied; for a non-lazy call
    also this block's rows), pend_out (this block's rows if lazy, else zeros)). No state rules apply."""
    nk, nv, dk, dv, cdim = dims(cfg)
    kd, vd, eps = nk * dk, nv * dv, cfg["rms_norm_eps"]
    qkv, z, b, a = C.gdn_proj(cfg, w, q, h, T)
    rows = mb.concat(values=[conv_in, mb.transpose(x=qkv, perm=[1, 0])], axis=0)         # (T + 3, cdim)
    cw = w["linear_attn.conv1d.weight"][:, 0].numpy().astype(f16).T
    conv = None
    for j in range(4):
        term = mb.mul(x=mb.slice_by_index(x=rows, begin=[j, 0], end=[j + T, cdim]), y=np.ascontiguousarray(cw[j:j + 1]))
        conv = term if conv is None else mb.add(x=conv, y=term)
    conv = mb.transpose(x=C.silu(conv), perm=[1, 0])
    qq, kk, vv = mb.split(x=conv, split_sizes=[kd, kd, vd], axis=0)
    rep = nv // nk

    def heads(t):
        t = mb.transpose(x=mb.reshape(x=t, shape=(nk, dk, T)), perm=[0, 2, 1])
        return mb.reshape(x=mb.tile(x=mb.reshape(x=t, shape=(nk, 1, T, dk)), reps=[1, rep, 1, 1]), shape=(nv, T, dk))

    def l2n(t, scale):
        ss = mb.reduce_sum(x=mb.mul(x=t, y=t), axes=[-1], keep_dims=True)
        return mb.mul(x=mb.mul(x=t, y=mb.rsqrt(x=ss, epsilon=1e-6)), y=f16(scale))

    qh, kh = l2n(heads(qq), dk ** -0.5), l2n(heads(kk), 1.0)
    vh = mb.transpose(x=mb.reshape(x=vv, shape=(nv, dv, T)), perm=[0, 2, 1])
    beta = mb.sigmoid(x=b)
    neg_a = (-w["linear_attn.A_log"].exp()).numpy().astype(f16).reshape(nv, 1, 1)
    dt = w["linear_attn.dt_bias"].numpy().astype(f16).reshape(nv, 1, 1)
    g = mb.mul(x=_softplus(mb.add(x=a, y=dt)), y=neg_a)
    # commit the first k pending rows of the previous call
    kp, up = rows_slice(pend_in, 0, P, nv, dk), rows_slice(pend_in, P, 2 * P, nv, dv)
    wkp = rows_slice(pend_in, 2 * P, 3 * P, nv, dk)
    cum_p = mb.reshape(x=mb.slice_by_index(x=pend_in, begin=[0, 3 * P, 0], end=[nv, 3 * P + 1, P]), shape=(nv, P, 1))
    total = mb.reduce_sum(x=mb.mul(x=cum_p, y=commit_last), axes=[1], keep_dims=True)
    kd_ = mb.mul(x=mb.mul(x=kp, y=commit), y=mb.exp(x=mb.minimum(x=mb.sub(x=total, y=cum_p), y=f16(0))))
    s1 = mb.add(x=mb.mul(x=rec_in, y=mb.exp(x=total)),
                y=mb.matmul(x=kd_, y=mb.sub(x=up, y=mb.matmul(x=wkp, y=rec_in)), transpose_x=True))
    if lazy:
        assert T <= P
        core = delta_core(kh, vh, beta, g, T, nv)
        cum, _, u, wk = core
        crow = mb.concat(values=[t for t in (mb.reshape(x=cum, shape=(nv, 1, T)), np.zeros((nv, 1, dv - T), f16))
                                 if not (isinstance(t, np.ndarray) and t.size == 0)], axis=2)
        pend_out = mb.concat(values=[pad_rows(kh, T, P, nv, dk), pad_rows(u, T, P, nv, dv), pad_rows(wk, T, P, nv, dk),
                                     crow], axis=1)
        o, s_out = delta_out(s1, qh, kh, core), s1
    else:
        s, outs = s1, []
        for c0 in range(0, T, P):
            n = min(P, T - c0)
            sl = [rows_slice(t, c0, c0 + n, nv, d) for t, d in ((qh, dk), (kh, dk), (vh, dv), (beta, 1), (g, 1))]
            core = delta_core(sl[1], sl[2], sl[3], sl[4], n, nv)
            outs.append(delta_out(s, sl[0], sl[1], core))
            s = delta_state(s, sl[1], n, nv, core)
        o = mb.concat(values=outs, axis=1) if len(outs) > 1 else outs[0]
        pend_out, s_out = mb.mul(x=pend_in, y=f16(0)), s
    o = mb.mul(x=C.rms_last(o, w["linear_attn.norm.weight"].numpy().astype(f16), eps), y=C.silu(z))
    o = mb.reshape(x=mb.transpose(x=o, perm=[0, 2, 1]), shape=(1, vd, 1, T))
    return C.lut_linear(o, q["linear_attn.out_proj.weight"]), rows, s_out, pend_out


def build_fn_io(cfg, w, q, T, lazy, path):
    nk, nv, dk, dv, cdim = dims(cfg)
    specs = {"h": mb.TensorSpec((1, cfg["hidden_size"], 1, T), types.fp16),
             "commit": mb.TensorSpec((1, P, 1), types.fp16), "commit_last": mb.TensorSpec((1, P, 1), types.fp16),
             "conv_in": mb.TensorSpec((H, cdim), types.fp16), "rec_in": mb.TensorSpec((nv, dk, dv), types.fp16),
             "pend_in": mb.TensorSpec((nv, 3 * P + 1, dv), types.fp16)}
    with Function(specs, opset_version=ct.target.iOS18) as fn:
        y, rows, s_out, pend_out = gdn_lazy_io(cfg, w, q, fn.inputs["h"], fn.inputs["conv_in"], fn.inputs["rec_in"],
                                               fn.inputs["pend_in"], fn.inputs["commit"], fn.inputs["commit_last"], T, lazy)
        fn.set_outputs([mb.identity(x=y, name="y"), mb.identity(x=rows, name="conv_out"),
                        mb.identity(x=s_out, name="rec_out"), mb.identity(x=pend_out, name="pend_out")])
    prog = Program()
    prog.add_function("main", fn)
    pipeline = ct.PassPipeline.DEFAULT
    pipeline.remove_passes(["common::canonicalize_quantized_lut_pattern"])
    ct.convert(prog, minimum_deployment_target=ct.target.iOS18, skip_model_load=True,
               pass_pipeline=pipeline).save(str(path))
    return path


def run_io():
    """GDN_IO variant: decode1 / verify8 lazy, prefill16 committing; host passes the buffers through."""
    cfg = text_config()
    w = load_layer_weights([LAYER])[LAYER]
    q = {k: ("dense", w[k].numpy()) for k in ("linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
                                               "linear_attn.out_proj.weight")}
    units = getattr(ct.ComputeUnit, os.environ.get("UNITS", "CPU_ONLY"))
    OUT.mkdir(parents=True, exist_ok=True)
    fns = {"decode1": (1, True), "verify8": (8, True), "prefill16": (16, False)}
    models = {}
    for n, (T, lazy) in fns.items():
        p = build_fn_io(cfg, w, q, T, lazy, OUT / f"L{LAYER}_io_{n}.mlpackage")
        models[n] = ct.models.MLModel(str(p), compute_units=units)
        print(f"[io] {n} loaded on {units}: {models[n].__proxy__ is not None}", flush=True)
    nk, nv, dk, dv, cdim = dims(cfg)
    rng = np.random.default_rng(0)
    ref = DecodeLayer(cfg, LAYER, w, ctx=8)
    buf = {"conv_rows": np.zeros((H, cdim), f16), "rec": np.zeros((nv, dk, dv), f16),
           "pend": np.zeros((nv, 3 * P + 1, dv), f16)}

    def call(name, xs, commit_k):
        c, last = np.zeros((1, P, 1), f16), np.zeros((1, P, 1), f16)
        c[0, :commit_k] = 1
        if commit_k:
            last[0, commit_k - 1] = 1
        # conv_in = the 3 rows ending at the committed length of the previous call's rows
        conv_in = buf["conv_rows"][buf["conv_keep"]:buf["conv_keep"] + H] if "conv_keep" in buf else buf["conv_rows"][-H:]
        out = models[name].predict({"h": xs.T[None, :, None, :], "commit": c, "commit_last": last, "conv_in": conv_in,
                                    "rec_in": buf["rec"], "pend_in": buf["pend"]})
        buf.update(conv_rows=out["conv_out"].astype(f16), rec=out["rec_out"].astype(f16), pend=out["pend_out"].astype(f16))
        return out["y"].astype(np.float32)[0, :, 0, :].T

    def ref_rows(d, xs):
        return np.stack([d.gdn(torch.from_numpy(x.astype(np.float32))).numpy() for x in xs])

    def clone(d):
        e = DecodeLayer.__new__(DecodeLayer)
        e.__dict__.update(d.__dict__)
        e.conv_state, e.state = d.conv_state.clone(), d.state.clone()
        return e

    worst = 1.0

    def report(tag, y, yr):
        nonlocal worst
        cos = [float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)) for a, b in zip(y, yr)]
        worst = min(worst, min(cos))
        print(f"[io] {tag:34s} min cos {min(cos):.5f}", flush=True)

    xs = (rng.standard_normal((16, cfg["hidden_size"]))).astype(f16)
    report("prefill16 @0", call("prefill16", xs, 0), ref_rows(ref, xs))
    buf["conv_keep"] = 16                        # all 16 rows committed: conv_in = rows 16..18 of (19, cdim)
    pending = 0
    for k_acc in (3, 8, 1, 5):
        xs = (rng.standard_normal((8, cfg["hidden_size"]))).astype(f16)
        yr = ref_rows(clone(ref), xs)
        report(f"verify8 (commit {pending})", call("verify8", xs, pending), yr)
        ref_rows(ref, xs[:k_acc])
        buf["conv_keep"], pending = k_acc, k_acc  # next conv_in = rows k..k+2 of (11, cdim)
    xs = (rng.standard_normal((1, cfg["hidden_size"]))).astype(f16)
    report(f"decode1 (commit {pending})", call("decode1", xs, pending), ref_rows(ref, xs))
    buf["conv_keep"], pending = 1, 1
    xs = (rng.standard_normal((16, cfg["hidden_size"]))).astype(f16)
    report(f"prefill16 (commit {pending})", call("prefill16", xs, pending), ref_rows(ref, xs))
    print(f"[io] worst token cos {worst:.5f}", flush=True)


def build_fn(cfg, w, q, T, mode, path):
    nk, nv, dk, dv, cdim = dims(cfg)
    specs = {"h": mb.TensorSpec((1, cfg["hidden_size"], 1, T), types.fp16),
             "commit": mb.TensorSpec((1, P, 1), types.fp16), "commit_last": mb.TensorSpec((1, P, 1), types.fp16),
             "prev_sel": mb.TensorSpec((H, H + P), types.fp16),
             "conv0": mb.StateTensorSpec((H + P, cdim), types.fp16),
             "rec0": mb.StateTensorSpec((nv, dk + 3 * P + 1, dv), types.fp16)}
    with Function(specs, opset_version=ct.target.iOS18) as fn:
        y = gdn_lazy(cfg, w, q, fn.inputs["h"], fn.inputs["conv0"], fn.inputs["rec0"], fn.inputs["commit"],
                     fn.inputs["commit_last"], fn.inputs["prev_sel"], T, mode)
        fn.set_outputs([mb.identity(x=y, name="y")])
    prog = Program()
    prog.add_function("main", fn)
    pipeline = ct.PassPipeline.DEFAULT
    pipeline.remove_passes(["common::canonicalize_quantized_lut_pattern"])
    ct.convert(prog, minimum_deployment_target=ct.target.iOS18, skip_model_load=True,
               pass_pipeline=pipeline).save(str(path))
    return path


class Host:
    """Host bookkeeping: absolute positions held by the conv buffer rows, pending rows count."""

    def __init__(self):
        self.buf_pos = [None] * (H + P)

    def inputs(self, p0, commit_k):
        sel = np.zeros((H, H + P), f16)
        for r, pos in enumerate(range(p0 - H, p0)):
            if pos >= 0:
                sel[r, self.buf_pos.index(pos)] = 1
        c, last = np.zeros((1, P, 1), f16), np.zeros((1, P, 1), f16)
        c[0, :commit_k] = 1
        if commit_k:
            last[0, commit_k - 1] = 1
        return {"commit": c, "commit_last": last, "prev_sel": sel}

    def after(self, p0, T):
        new = list(range(p0, p0 + T))[-P:]
        self.buf_pos = list(range(p0 - H, p0)) + new + [None] * (P - len(new))


def main():
    cfg = text_config()
    assert cfg["layer_types"][LAYER] == "linear_attention"
    w = load_layer_weights([LAYER])[LAYER]
    q = {k: ("dense", w[k].numpy()) for k in ("linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
                                               "linear_attn.out_proj.weight")}
    units = getattr(ct.ComputeUnit, os.environ.get("UNITS", "CPU_ONLY"))
    OUT.mkdir(parents=True, exist_ok=True)
    for variant in os.environ.get("VARIANTS", "scratch,side").split(","):
        t0 = time.time()
        fns = {"decode1": (1, "lazy"), "verify8": (8, "lazy"), "prefill16": (16, f"prefill_{variant}")}
        pkgs = {n: build_fn(cfg, w, q, T, mode, OUT / f"L{LAYER}_{n}_{variant}.mlpackage")
                for n, (T, mode) in fns.items()}
        desc = ct.utils.MultiFunctionDescriptor()
        for n, p in pkgs.items():
            desc.add_function(str(p), "main", n)
        desc.default_function_name = "decode1"
        mf = OUT / f"L{LAYER}_gdn_lazy_{variant}.mlpackage"
        shutil.rmtree(mf, ignore_errors=True)
        ct.utils.save_multifunction(desc, str(mf))
        for p in pkgs.values():
            shutil.rmtree(p, ignore_errors=True)
        print(f"[{variant}] built in {time.time() - t0:.0f}s", flush=True)
        try:
            models = {n: ct.models.MLModel(str(mf), compute_units=units, function_name=n) for n in fns}
        except Exception as e:  # noqa: BLE001
            print(f"[{variant}] LOAD FAILED on {units}: {str(e)[:300]}", flush=True)
            continue
        print(f"[{variant}] loaded on {units} ({time.time() - t0:.0f}s)", flush=True)
        run_sequence(cfg, w, models, variant)


def run_sequence(cfg, w, models, variant):
    rng = np.random.default_rng(0)
    ref = DecodeLayer(cfg, LAYER, w, ctx=8)
    host, state = Host(), models["decode1"].make_state()

    def x_rows(T):
        return (rng.standard_normal((T, cfg["hidden_size"])) * 1.0).astype(f16)

    def ref_rows(d, xs):
        out = []
        for x in xs:
            out.append(d.gdn(torch.from_numpy(x.astype(np.float32))).numpy())
        return np.stack(out)

    def call(name, xs, p0, commit_k):
        feed = {"h": xs.T[None, :, None, :], **host.inputs(p0, commit_k)}
        y = models[name].predict(feed, state=state)["y"].astype(np.float32)[0, :, 0, :].T
        host.after(p0, len(xs))
        return y

    def clone(d):
        e = DecodeLayer.__new__(DecodeLayer)
        e.__dict__.update(d.__dict__)
        e.conv_state, e.state = d.conv_state.clone(), d.state.clone()
        return e

    def report(tag, y, yr):
        cos = [float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)) for a, b in zip(y, yr)]
        rel = float(np.linalg.norm(y - yr) / np.linalg.norm(yr))
        print(f"[{variant}] {tag:38s} min cos {min(cos):.5f}  rel err {rel:.4f}", flush=True)
        return min(cos)

    worst = 1.0
    xs = x_rows(16)
    worst = min(worst, report("prefill16 @0", call("prefill16", xs, 0, 0), ref_rows(ref, xs)))
    p, pending = 16, 0
    for k_acc in (3, 8, 1, 5):                       # verify cycles: accept k_acc rows of each block
        xs = x_rows(8)
        tent = clone(ref)
        yr = ref_rows(tent, xs)
        worst = min(worst, report(f"verify8 @{p} (commit {pending})", call("verify8", xs, p, pending), yr))
        ref_rows(ref, xs[:k_acc])                    # committed reference advances k_acc rows
        p, pending = p + k_acc, k_acc
    for _ in range(2):
        xs = x_rows(1)
        worst = min(worst, report(f"decode1 @{p} (commit {pending})", call("decode1", xs, p, pending), ref_rows(ref, xs)))
        p, pending = p + 1, 1
    xs = x_rows(8)
    tent = clone(ref)
    worst = min(worst, report(f"verify8 @{p} (commit {pending})", call("verify8", xs, p, pending), ref_rows(tent, xs)))
    ref_rows(ref, xs[:2])
    p, pending = p + 2, 2
    xs = x_rows(16)
    worst = min(worst, report(f"prefill16 @{p} (commit {pending})", call("prefill16", xs, p, pending), ref_rows(ref, xs)))
    p, pending = p + 16, 0
    xs = x_rows(1)
    worst = min(worst, report(f"decode1 @{p} (commit {pending})", call("decode1", xs, p, pending), ref_rows(ref, xs)))
    print(f"[{variant}] worst token cos {worst:.5f}", flush=True)


if __name__ == "__main__":
    run_io() if os.environ.get("IO") == "1" else main()
