"""One-line run banner for the V8 GPU engine (18 Nov 2026): the attention variant as the LOADED dylib reports it
(v8_variant: its own globals and kernel names, not the caller's environment), the dyld image actually mapped (path +
sha256), whether the decode keep-alive dylib is in the process, and the Metal sources (compiled at load; no metallib).
    line, ok = banner(v8, dylib_name, expect)     expect: off | c | v | w | x (EXPECT_F8)"""
import ctypes as C_, hashlib, os
from pathlib import Path


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dyld_images():
    lib = C_.CDLL(None)
    lib._dyld_image_count.restype = C_.c_uint32
    lib._dyld_get_image_name.restype, lib._dyld_get_image_name.argtypes = C_.c_char_p, [C_.c_uint32]
    return [lib._dyld_get_image_name(i).decode() for i in range(lib._dyld_image_count())]


WANT = {"off": ("attn_softmax8n", "attn_pv_w", "fp16, 2 B", "every element"),
        "x": ("attn_softmax8n_f8x", "attn_pv_f8v", "e4m3", "boundary chunk only"),
        "x+w128": ("folded", "attn_pv_f8m_w128", "e4m3", "boundary chunk only")}


def banner(v8, dylib_name, expect):
    g = v8.g
    if not hasattr(g, "v8_variant"):
        return f"BANNER expect F8={expect} | {dylib_name} has no v8_variant | MISMATCH", False
    buf = C_.create_string_buffer(4096)
    g.v8_variant.restype, g.v8_variant.argtypes = C_.c_int32, [C_.c_char_p, C_.c_int32]
    r = g.v8_variant(buf, 4096)
    var = buf.value.decode()
    imgs = dyld_images()
    eng = [p for p in imgs if p.endswith("/" + dylib_name)]
    ka = [p for p in imgs if "libdec_gpu" in p]
    kp = Path(os.environ["V8_KERNELS"])
    srcs = [kp, kp.parent / "kernels_fold.metal", kp.parent / "kernels_f8.metal"] + ([kp.parent / "kernels_f8fold.metal"] if "f8fold" in var else []) \
        + ([kp.parent / "kernels_qk.metal"] if "kernels_qk" in var else []) + ([kp.parent / "kernels_gdnp.metal"] if "kernels_gdnp" in var else [])
    got = var.split()[0].split("=")[1]
    f8f = var.split()[1].split("=")[1] if var.split()[1].startswith("F8F=") else "off"
    if f8f != "off":
        got += "+" + f8f
    ok = r >= 0 and len(eng) == 1 and got == expect and not ka
    w = WANT.get(expect)
    if w:
        ok = ok and f"softmax={w[0]} " in var and f"pv={w[1]} " in var and f"V copy={w[2]}" in var and f"mask={w[3]}" in var
    eqk = os.environ.get("EXPECT_QK")   # 19 Nov: the Q.K kernel too (qk=attn_qk8isk | attn_qk8isk_tg)
    if eqk:
        ok = ok and f"qk={eqk} " in var
    elut = os.environ.get("EXPECT_LUT")   # queue 9: the LUT GEMM kernel (lut=lut_gemm32_i8t_k256 | ..._k256ws)
    if elut:
        ok = ok and f"lut={elut} " in var
        eqk = (eqk or "") + f" lut={elut}"
    line = (f"BANNER expect F8={expect}{' qk=' + eqk if eqk else ''} | loaded {var} | keep-alive: "
            f"{'LOADED ' + ','.join(ka) if ka else 'off (libdec_gpu* not loaded)'}"
            f" | dylib {eng[0] if eng else 'NOT MAPPED'} sha256 {sha(eng[0])[:16] if eng else '-'}"
            f" | metal (source, compiled at load; no metallib) " + " ".join(f"{p.name}:{sha(p)[:16]}" for p in srcs)
            + f" | {'MATCH' if ok else 'MISMATCH'}")
    return line, ok
