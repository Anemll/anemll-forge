"""Resize without copy: bind a smaller context entry's KV history inputs as VIEWS of the largest entry's buffers.

Package: the shipped-form layer-3 attention (kv_state_L03_1/attn_L03_release.aimodel: KV_KEYS_T=1 keys (nkv, hd, L),
values (nkv, L, hd), scales (nkv, L); entries v8 / p64 at 16K, 32K, 96K). Big buffers: the 96K entry's inputs in its
preferred layout. For each smaller entry: an exact-size binding (its own preferred buffers) and a view binding (shape of
the entry, strides of the 96K buffer, the same IOSurface: libcoreai_bridge_view cai_buffer_view), both holding the same
dense KV8 history. Checks: the binding is accepted, the call runs, y / k_new / v_new equal the exact binding's bitwise;
then ms per call for both. Model load journaled (deploy_journal v7).
    COREAI_BRIDGE_LIB=.../libcoreai_bridge_view.dylib MainPY view_resize_check.py --out OUT.json"""
import argparse, ctypes as _c, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

C = Path("/Users/anemll/Documents/Codex/ios-prefill-20261030")
V02 = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02")
for p in (V02 / "coreai/swift_bridge", V02 / "scripts", C / "claude_full96_prep/freeze_eight_row_v4"):
    sys.path.insert(0, str(p))
import ane_compile_mode  # noqa: E402
ane_compile_mode.apply(log=lambda m: None)
import coreai_bridge as B  # noqa: E402
import deploy_journal as DJ  # noqa: E402
from qwen38_kv_cache import quantize_values  # noqa: E402

PKG = C / "kv_state_L03_1/attn_L03_release.aimodel"
KV = ("k", "v", "ks", "vs")


def view(base, shape, strides):
    L = B.lib()
    f = L.cai_buffer_view
    f.argtypes = [_c.c_void_p, _c.c_int32, _c.POINTER(_c.c_int64), _c.POINTER(_c.c_int64), _c.c_char_p, _c.c_int32]
    f.restype = _c.c_void_p
    e = _c.create_string_buffer(2048)
    h = f(base._h, len(shape), (_c.c_int64 * len(shape))(*shape), (_c.c_int64 * len(shape))(*strides), e, len(e))
    if not h:
        raise B.BridgeError(e.value.decode())
    o = object.__new__(B.Buffer)
    o._h, o.shape, o.dtype, o.strides = h, tuple(shape), base.dtype, tuple(strides)
    o.address, o.nbytes, o._base = base.address, base.nbytes, base
    return o


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); a = ap.parse_args()
    res = {"package": str(PKG), "main_hash": (PKG / "main.hash").read_bytes().hex(), "lib": str(B.LIB_PATH), "entries": {}}
    save = lambda: a.out.write_text(json.dumps(res, indent=1, default=str) + "\n")  # noqa: E731
    J = DJ.Journal({"driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "plan_sha256": "n/a",
                    "out": str(a.out), "mode": "view_resize_check"}, f"viewresize-{os.getpid()}-{int(time.time())}")
    dev, pkg = DJ.m6_device(), {"path": str(PKG), "main_hash": res["main_hash"], "label": "attn L3 release form (view resize)"}
    aid = J.attempt(dev, pkg, functions=None, request={"compute": "ane"})
    t0 = time.monotonic()
    try:
        m = B.Model(PKG, compute="ane")
        fns = {n: m.function(n) for n in m.function_names}
    except Exception as e:  # noqa: BLE001
        J.result(dev, aid, pkg, "failed", error=f"{type(e).__name__}: {e}"); raise
    J.result(dev, aid, pkg, "ok", model_load_s=time.monotonic() - t0)
    rng = np.random.default_rng(7)
    for T, big_e in ((8, "v8_96k"), (64, "p64_96k")):
        big = {n: fns[big_e].buffer("input", n) for n in KV}
        res.setdefault("big_layout", {})[big_e] = {n: {"shape": big[n].shape, "strides": big[n].strides} for n in KV}
        for e in (f"{'v8' if T == 8 else 'p64'}_{k}k" for k in (16, 32)):
            f = fns[e]
            sp = {n: f.spec("input", n) for n in KV}
            L = sp["v"]["shape"][1]
            fill = L - 2 * T
            r = {"L": L, "fill": fill, "exact_layout": {n: {"shape": sp[n]["shape"], "strides": sp[n]["strides"]} for n in KV}}
            exact = {n: f.buffer("input", n) for n in KV}
            for x in list(exact.values()) + list(big.values()):
                x.zero()
            kc, ks = quantize_values(rng.standard_normal((4, fill, 256)).astype(np.float16))
            vc, vs = quantize_values((rng.standard_normal((4, fill, 256)) * 0.5).astype(np.float16))
            for d in (exact, big):
                d["k"].np[:, :, :fill] = kc.transpose(0, 2, 1); d["v"].np[:, :fill] = vc
                d["ks"].np[:, :fill] = ks; d["vs"].np[:, :fill] = vs
            common = {n: f.buffer("input", n) for n in ("x", "cos", "sin", "mask")}
            common["x"].np[:] = rng.standard_normal(common["x"].shape).astype(np.float16)
            ang = np.outer(np.arange(fill, fill + T), 1.0 / 1e7 ** (np.arange(0, 64, 2) / 64))
            common["cos"].np[:] = np.concatenate([np.cos(ang)] * 2, 1); common["sin"].np[:] = np.concatenate([np.sin(ang)] * 2, 1)
            common["mask"].np[:] = np.where(np.arange(L)[None] < fill, 0, -1e4)
            outs = {k_: {n: f.buffer("output", n) for n in ("y", "k_new", "v_new")} for k_ in ("exact", "view")}
            be = f.bind({**common, **exact}, outs["exact"])
            try:
                VIEW = os.environ.get("VIEW", "k,v,ks,vs").split(",")   # which inputs are views; the rest are exact buffers
                views = {n: (view(big[n], sp[n]["shape"], big[n].strides) if n in VIEW else exact[n]) for n in KV}
                r["views"] = VIEW
                bv = f.bind({**common, **views}, outs["view"])
                B.Plan([bv]).run()
                r["view_bind_run"] = "ok"
            except Exception as ex:  # noqa: BLE001
                r["view_bind_run"] = f"{type(ex).__name__}: {ex}"[:600]
                res["entries"][e] = r; save(); print(e, r["view_bind_run"], flush=True)
                continue
            B.Plan([be]).run()
            r["equal"] = {n: bool(np.array_equal(outs["exact"][n].np, outs["view"][n].np)) for n in ("y", "k_new", "v_new")}
            r["y_maxdiff"] = float(np.abs(outs["exact"]["y"].np.astype(np.float32) - outs["view"]["y"].np.astype(np.float32)).max())
            tm = {}
            for k_, b in (("exact", be), ("view", bv)):
                p = B.Plan([b]); ts = []
                for _ in range(22):
                    t = time.perf_counter(); p.run(); ts.append((time.perf_counter() - t) * 1e3)
                tm[k_] = float(np.median(ts[2:]))
            r["ms"] = tm
            res["entries"][e] = r; save()
            print(f"{e} (L {L}, view strides {[big[n].strides for n in KV]}): {r['view_bind_run']}; equal {r['equal']} "
                  f"maxdiff {r['y_maxdiff']:.3g}; ms exact {tm['exact']:.2f} view {tm['view']:.2f}", flush=True)
    save()


if __name__ == "__main__":
    main()
