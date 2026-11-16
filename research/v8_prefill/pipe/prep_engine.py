"""Manifest + weight files for gpu_layers over layers [L0, L1): palettized matrices as tables into the ANE's compiled
programs (model.hwx of the build's chunks, found via m6_hwx_inspect map), the rest from the release export /
checkpoint (exactly the builder's layer_arrays).
    EXPORT_DIR=<release export> BUILDPY prep_engine.py BUILD_DIR L0 L1 OUT_DIR [Lmax]"""
import json, os, re, struct, subprocess, sys
from pathlib import Path
import numpy as np

V02 = "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02"
sys.path[:0] = [V02 + "/coreai", V02 + "/scripts"]
import qwen38_coreai_build as B  # noqa: E402
import qwen38_ane_model as M  # noqa: E402
S = Path(__file__).resolve().parents[1]
PARSER = str(S / "hwx/src/hwx_dump/hwx_parsing")


def kern(p):
    b = open(p, "rb").read(65536); ncmds, off = struct.unpack_from("<I", b, 16)[0], 32
    for _ in range(ncmds):
        cmd, size = struct.unpack_from("<2I", b, off)
        if cmd == 0x19 and b[off + 8:off + 24].rstrip(b"\0").decode().startswith("__KERN"):
            return struct.unpack_from("<4Q", b, off + 24)[2:]
        off += size


def unpack(buf, bits):
    return np.unpackbits(np.frombuffer(buf, np.uint8), bitorder="little").reshape(-1, bits) @ (1 << np.arange(bits))


def table(K, lut, idx, sc):
    """HWX layout -> (tab (pal_off, sg_off, row0) per subgroup, G, fold, cd). Every subgroup's rows are identified by
    lookup in the export's index rows (no layout formula); raises if a subgroup does not hold G consecutive rows."""
    R, cin = idx.shape
    cd, bits = lut.shape[1], int(np.log2(lut.shape[0]))
    hits = [m.start() for m in re.finditer(re.escape(np.ascontiguousarray(lut, np.float16).tobytes()), K)]
    look = {bytes(idx[r]): r for r in range(R)}

    def first_row(h, palb, G):  # the export row this block's first subgroup starts with, or None
        o = h + palb + 2 * G * cd
        if o + G * cin * bits // 8 > len(K):
            return None
        rows = unpack(K[o: o + G * cin * bits // 8], bits).reshape(cin, G).T.astype(np.uint8)
        r = [look.get(bytes(x)) for x in rows]
        return r[0] if r[0] is not None and r == list(range(r[0], r[0] + G)) else None
    found = None
    for h0 in hits:  # a palette shared with another matrix: find a block of this one first
        for palb in sorted({lut.nbytes, -(-lut.nbytes // 64) * 64}):
            for G in (32, 16, 8):
                if first_row(h0, palb, G) is not None:
                    found = (palb, G, 2 * G * cd + G * cin * bits // 8)
                    break
            if found:
                break
        if found:
            break
    if not found:
        raise ValueError("layout not recognized")
    palb, G, sgb = found
    seen, keep = set(), []
    for h in hits:  # this matrix's blocks only; the compiler may store a block twice (identical bytes): the first copy
        r = first_row(h, palb, G)
        if r is not None and r not in seen:
            seen.add(r); keep.append(h)
    hits = keep
    if len(hits) != 32:
        raise ValueError(f"{len(hits)} blocks of this matrix (want 32)")
    nsg = R // 32 // G
    tab, ratios = [], []
    for h in hits:
        for s_ in range(nsg):
            so = h + palb + s_ * sgb
            rows = unpack(K[so + 2 * G * cd: so + sgb], bits).reshape(cin, G).T.astype(np.uint8)
            r0, rl = look.get(bytes(rows[0])), look.get(bytes(rows[-1]))
            if r0 is None or rl != r0 + G - 1:
                raise ValueError(f"subgroup at {so:#x}: rows {r0} .. {rl}")
            tab.append((h, so, r0))
            if len(ratios) < 24:
                st = np.frombuffer(K[so: so + 2 * G * cd], np.float16).astype(np.float64)
                ratios.append(st / sc[cd * r0: cd * r0 + G * cd].astype(np.float64))
    if sorted(t[2] for t in tab) != list(range(0, R, G)):
        raise ValueError("subgroups do not cover every row once")
    ratios = np.concatenate(ratios)
    if np.abs(ratios / ratios.mean() - 1).max() > 1e-3:
        raise ValueError(f"non-uniform scale fold {ratios.min()} .. {ratios.max()}")
    return np.asarray(tab, np.uint32), G, float(ratios.mean()), cd


def main():
    build, l0, l1, out = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
    lmax = int(sys.argv[5]) if len(sys.argv) > 5 else 2048
    out.mkdir(parents=True, exist_ok=True)
    man = json.loads((build / "manifest.json").read_text())
    cfg = M.cfg()
    ck = M.Checkpoint()
    hwx, hwx_idx, kern_bytes = [], {}, {}
    res = {"dir": str(out.resolve()), "Lmax": lmax, "eps": cfg["rms_norm_eps"], "hwx": hwx, "layers": []}
    rot = int(cfg["head_dim"] * cfg["rope_parameters"]["partial_rotary_factor"])
    res["rope_inv"] = (1.0 / cfg["rope_parameters"]["rope_theta"] ** (np.arange(0, rot, 2) / rot)).astype(np.float32).tolist()
    w32 = lambda name, a: (np.ascontiguousarray(a, np.float32).tofile(out / name), name)[1]  # noqa: E731
    w16 = lambda name, a: (np.ascontiguousarray(a, np.float16).tofile(out / name), name)[1]  # noqa: E731
    for i in range(l0, l1):
        ch = next(c for c in man["chunks"] if c["layers"][0] <= i <= c["layers"][1])
        if ch["file"] not in hwx_idx:
            mp = subprocess.run([sys.executable, V02 + "/scripts/m6_hwx_inspect.py", "map", str(build / ch["file"])],
                                capture_output=True, text=True, env={**os.environ, "HWX_PARSING": PARSER}).stdout
            p = re.findall(r"(/Library/Caches/com\.apple\.aned/\S+/model\.hwx)", mp)[0]
            fo, fs = kern(p)
            hwx_idx[ch["file"]] = len(hwx)
            hwx.append({"path": p, "off": fo, "size": fs})
            f = open(p, "rb"); f.seek(fo); kern_bytes[ch["file"]] = f.read(fs)
        K, hi = kern_bytes[ch["file"]], hwx_idx[ch["file"]]
        W = B.layer_arrays(ck, i)
        kind = "gdn" if cfg["layer_types"][i] == "linear_attention" else "attn"
        L = {"i": i, "kind": kind, "ln1": w32(f"{i}_ln1.f32", 1 + W[f"{i}/input_layernorm.weight"]),
             "ln2": w32(f"{i}_ln2.f32", 1 + W[f"{i}/post_attention_layernorm.weight"]), "rot": {}, "mats": {}, "dense": {}}
        m_in, m_out = [int(v) for v in W.get(f"{i}/mix.rotation", [-1, -1])]
        r_in, r_mid = [int(v) for v in W.get(f"{i}/mlp.rotation", [-1, -1])]
        for key, seed, n in (("mix_in", m_in, 5120), ("mix_out", m_out, 6144), ("mlp_in", r_in, 5120), ("mlp_mid", r_mid, 17408)):
            if seed >= 0:
                L["rot"][key] = w32(f"{i}_rot_{key}.f32", np.random.default_rng(seed).choice([-1.0, 1.0], n))
        # PREP_NO_FFN=1: mixer matrices only (V8 runs every FFN on the ANE; a shared-weight build stores the FFN chunked)
        mats = {} if os.environ.get("PREP_NO_FFN") == "1" else {"gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}
        mats.update({"qkv": "linear_attn.in_proj_qkv", "z": "linear_attn.in_proj_z", "out": "linear_attn.out_proj"} if kind == "gdn"
                    else {"q": "self_attn.q_proj", "o": "self_attn.o_proj"})
        for name, key in mats.items():
            p = f"{i}/{key}.weight"
            lut, idx, sc = W[p + "/lut"], W[p + "/idx"], W[p + "/scale"]
            tab, G, fold, cd = table(K, lut, idx, sc)
            tab.tofile(out / f"{i}_{name}.tab")
            m = {"hwx": hi, "tab": f"{i}_{name}.tab", "bits": int(np.log2(lut.shape[0])), "G": G, "cin": int(idx.shape[1]),
                 "cout": int(idx.shape[0] * cd), "cd": cd, "fmul": 1.0 / fold, "lr_a": None, "lr_b": None}
            if p + "/lr_a" in W:
                m["lr_a"], m["lr_b"] = w16(f"{i}_{name}.lra", W[p + "/lr_a"]), w16(f"{i}_{name}.lrb", W[p + "/lr_b"])
            L["mats"][name] = m
            print(f"layer {i} {name}: {m['bits']}-bit cd{cd} G{G} {m['cout']}x{m['cin']} fold {fold:g}", flush=True)
        if kind == "gdn":
            p = f"{i}/linear_attn."
            ab = np.zeros((128, 5120), np.float32)
            ab[:48], ab[64:112] = W[p + "in_proj_b.weight"], W[p + "in_proj_a.weight"]
            L["dense"]["ab"] = {"w": w16(f"{i}_ab.f16", ab), "cout": 128, "cin": 5120, "lr_a": None, "lr_b": None}
            L["conv_w"] = w32(f"{i}_convw.f32", W[p + "conv1d.weight"][:, 0].T)
            L["neg_a"] = w32(f"{i}_nega.f32", -np.exp(W[p + "A_log"]))
            L["dt"] = w32(f"{i}_dt.f32", W[p + "dt_bias"])
            L["normw"] = w32(f"{i}_normw.f32", W[p + "norm.weight"])
        else:
            p = f"{i}/self_attn."
            for n in ("k", "v"):
                q = p + f"{n}_proj.weight"
                wd = W[q + "/int8"].astype(np.float32) * W[q + "/scale"].astype(np.float32)[:, None]
                L["dense"][n] = {"w": w16(f"{i}_{n}.f16", wd), "cout": 1024, "cin": 5120,
                                 "lr_a": w16(f"{i}_{n}.lra", W[q + "/lr_a"]) if q + "/lr_a" in W else None,
                                 "lr_b": w16(f"{i}_{n}.lrb", W[q + "/lr_b"]) if q + "/lr_a" in W else None}
            L["qn"] = w32(f"{i}_qn.f32", 1 + W[p + "q_norm.weight"])
            L["kn"] = w32(f"{i}_kn.f32", 1 + W[p + "k_norm.weight"])
        res["layers"].append(L)
    (out / os.environ.get("MANIFEST_NAME", "manifest.json")).write_text(json.dumps(res, indent=1))
    print("manifest", out / "manifest.json", "hwx", [Path(h["path"]).parent.name[:8] for h in hwx])


if __name__ == "__main__":
    main()
