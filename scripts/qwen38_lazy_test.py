"""One T=8 lazy-commit function (LAZY=1, GDN_IO=1, KV MLState) for decode / prefill / verify, on one layer group.
Path A: one token per call (1 committed row per call). Path B: verify-style 8-row blocks with a random number of
accepted rows, the next block starting right after them. Outputs of accepted rows must match path A.
    RANGE=60-63 CTX=8192 python qwen38_lazy_test.py"""
import os
import shutil
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("EXPORT_DIR", str(Path("~/Models/vq27b/export/full_mix25_mixer4_head4").expanduser()))
import coremltools as ct  # noqa: E402
import qwen38_ane_chunk as C  # noqa: E402
import qwen38_ane_model as M  # noqa: E402

a_, b_ = (int(x) for x in os.environ.get("RANGE", "60-63").split("-"))
LAYERS = list(range(a_, b_ + 1))
CTX, T = int(os.environ.get("CTX", "8192")), 8
KV_IN = os.environ.get("KV_IN", "0") == "1"   # read-only KV inputs + new-row outputs, host commits rows
HOST_KV = os.environ.get("HOST_KV", "1") == "1"  # copy committed rows into the caches (off: timing only)
OUT = Path(__file__).parent / "qwen38_prefill" / "lazy"
SA = ct.models.SharedArray


class Runner:
    def __init__(self, m, cfg, gdn, att):
        self.m, self.cfg, self.gdn = m, cfg, gdn
        nv, dk, dv = (cfg[k] for k in ("linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim"))
        cdim = 2 * cfg["linear_num_key_heads"] * dk + nv * dv
        shapes = {}
        for j in gdn:
            shapes |= {f"conv{j}": (T + 3, cdim), f"rec{j}": (nv, dk, dv), f"pend{j}": (nv, 3 * C.PEND + 1, dv)}
        self.cur = {k: SA(v) for k, v in shapes.items()}
        self.nxt = {k: SA(v) for k, v in shapes.items()}
        kvs = (cfg["num_key_value_heads"], CTX, cfg["head_dim"])
        self.kv = {f"{s_}{j}": SA(kvs) for j in att for s_ in ("k", "v")} if KV_IN else {}
        self.kv_new = {n: SA((kvs[0], T, kvs[2])) for n in self.kv}
        rot = int(cfg["head_dim"] * cfg["rope_parameters"]["partial_rotary_factor"])
        self.inv = 1.0 / cfg["rope_parameters"]["rope_theta"] ** (np.arange(0, rot, 2) / rot)
        hid = cfg["hidden_size"]
        self.inp = {"x": SA((1, hid, 1, T)), "cos": SA((T, rot)), "sin": SA((T, rot)),
                    "mask": SA((1, CTX) if KV_IN else (T, CTX)), "conv_sel": SA((3, T + 3)),
                    "commit": SA((1, C.PEND, 1)), "commit_last": SA((1, C.PEND, 1))}
        if not KV_IN:
            self.inp["kv_write"] = SA((T, CTX))
        self.y = SA((1, hid, 1, T))
        self.state = None if KV_IN else m.make_state()
        self.pos, self.pending = 0, 0  # pending = rows of the last call that the next call commits

    def call(self, xs):
        """Run len(xs) <= T rows at self.pos; returns their outputs. The caller then sets accept(k)."""
        n, p0, k = len(xs), self.pos, self.pending
        pos = np.minimum(np.arange(p0, p0 + T), p0 + n - 1)
        f = np.outer(pos, self.inv)
        ang = np.concatenate([f, f], axis=1)
        x = np.zeros((1, self.cfg["hidden_size"], 1, T), np.float16)
        x[0, :, 0, :n] = np.asarray(xs).T
        kvw = np.zeros((T, CTX), np.float16)
        kvw[np.arange(n), np.arange(p0, p0 + n)] = 1
        sel = np.zeros((3, T + 3), np.float16)
        sel[np.arange(3), k + np.arange(3)] = 1
        com, last = np.zeros((1, C.PEND, 1), np.float16), np.zeros((1, C.PEND, 1), np.float16)
        com[0, :k] = 1
        if k:
            last[0, k - 1] = 1
        mask = np.where(np.arange(CTX)[None, :] < p0, 0, -1e4).astype(np.float16) if KV_IN else \
            np.where(np.arange(CTX)[None, :] <= pos[:, None], 0, -1e4).astype(np.float16)
        for name, v in (("x", x), ("cos", np.cos(ang).astype(np.float16)), ("sin", np.sin(ang).astype(np.float16)),
                        ("mask", mask), ("conv_sel", sel), ("commit", com), ("commit_last", last)) + \
                (() if KV_IN else (("kv_write", kvw),)):
            self.inp[name].write(v)
        self.m.predict({**self.inp, **self.cur, **self.kv}, state=self.state,
                       output_backings={"y": self.y} | {f"{k_}_out": v for k_, v in self.nxt.items()} |
                       {f"{n_}_new": v for n_, v in self.kv_new.items()})
        self.cur, self.nxt = self.nxt, self.cur
        return self.y.to_numpy().astype(np.float32)[0, :, 0, :n].T

    def accept(self, k):
        if KV_IN and HOST_KV and k:  # commit the block's first k k / v rows into the caches (test: numpy round trip)
            for n_, buf in self.kv.items():
                a = buf.to_numpy()
                a[:, self.pos:self.pos + k] = self.kv_new[n_].to_numpy()[:, :k]
                buf.write(a)
        self.pending, self.pos = k, self.pos + k


def main():
    cfg, ck = M.cfg(), M.Checkpoint()
    C.CTX, C.REPEAT, C.OUT, C.GDN_IO, C.KV_IO, C.LAZY, C.LAYERS, C.JOFF, C.TAPS = CTX, 1, OUT / "tmp", True, False, True, LAYERS, 0, []
    C.KV_IN = KV_IN
    weights = {i: ck.layer(i) for i in LAYERS}
    quant = {i: M.layer_quant(ck, i, weights[i]) for i in LAYERS}
    pkg = C.build(cfg, weights, quant, T=T, compile_model=False)
    mlc = OUT / f"lazy{'_kvin' if KV_IN else ''}_L{a_}-{b_}_ctx{CTX}_T{T}.mlmodelc"
    shutil.rmtree(mlc, ignore_errors=True)
    ct.models.utils.compile_model(str(pkg), str(mlc))
    shutil.rmtree(C.OUT, ignore_errors=True)
    t0 = time.time()
    m = ct.models.CompiledMLModel(str(mlc), compute_units=getattr(ct.ComputeUnit, os.environ.get("UNITS", "CPU_AND_NE")))
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    gdn = [j for j, l in enumerate(LAYERS) if cfg["layer_types"][l] == "linear_attention"]
    att = [j for j in range(len(LAYERS)) if j not in gdn]
    rng = np.random.default_rng(0)
    N = 96
    xs = (rng.standard_normal((N + 8, cfg["hidden_size"])) * 2).astype(np.float16)

    ra = Runner(m, cfg, gdn, att)
    ya = []
    for t in range(N):
        ya.append(ra.call(xs[t:t + 1])[0])
        ra.accept(1)
    rb, got, i = Runner(m, cfg, gdn, att), {}, 0
    accs = []
    while i < N:
        yb = rb.call(xs[i:i + T])
        k = int(min(rng.integers(1, T + 1), N - i))
        for r in range(k):
            got[i + r] = yb[r]
        rb.accept(k)
        accs.append(k)
        i += k

    def cos(a, b):
        return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    cs = [cos(got[t], ya[t]) for t in range(N)]
    print(f"verify-style blocks (accepted per block {accs[:12]}...) vs one token per call over {N} tokens: "
          f"cos min {min(cs):.5f} mean {np.mean(cs):.5f}", flush=True)
    # speed: 1-row call (decode) and 8-row call (prefill / verify)
    global HOST_KV
    HOST_KV = False  # timing of the call itself
    r = Runner(m, cfg, gdn, att)
    for nrow in (1, 8):
        ts = []
        for _ in range(20):
            t1 = time.perf_counter()
            r.call(xs[:nrow])
            ts.append(1e3 * (time.perf_counter() - t1))
            r.accept(nrow if r.pos + nrow < CTX - 16 else 0)
        print(f"{nrow}-row call: median {np.median(ts):.2f} ms  max {max(ts):.2f}", flush=True)


if __name__ == "__main__":
    main()
