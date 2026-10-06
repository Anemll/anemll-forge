"""SHA-256 of every weight array the Core AI builder (coreai/qwen38_coreai_build.py) takes from MODEL and EXPORT_DIR:
layer_arrays() of all 64 layers and the final norm, exactly as the conversion consumes them. Two inputs with the same
digests give the builder identical weights (its output bytes still vary slightly from run to run: the converter's
serialization, not the weights).

    MODEL=<dir> EXPORT_DIR=<export> python scripts/qwen38_weights_digest.py --out digest.json
    MODEL=<dir> EXPORT_DIR=<export> python scripts/qwen38_weights_digest.py --check weights_digest.json
--drafter digests what coreai/dflash2_coreai_build.py consumes instead (env DRAFTER, DRAFT_EXPORT, HEAD_EXPORT, MODEL):
the drafter's weights with the build's folds, the mask-token embedding row and the output head."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coreai"))
sys.path.insert(0, str(ROOT / "scripts"))


def digest() -> dict:
    import qwen38_coreai_build as Bld
    ck = Bld.M.Checkpoint()
    out = {}
    for i in range(Bld.CFG["num_hidden_layers"]):
        for k, v in Bld.layer_arrays(ck, i).items():
            a = np.ascontiguousarray(v)
            out[k] = f"{a.dtype}{a.shape}:" + hashlib.sha256(a.view(np.uint8).data if a.size else b"").hexdigest()
    n = np.ascontiguousarray(ck.get("model.language_model.norm.weight").float().numpy())
    out["norm"] = hashlib.sha256(n.data).hexdigest()
    return out


def digest_drafter() -> dict:
    import dflash2_coreai_build as D
    from safetensors import safe_open
    cfg = json.loads((D.DRAFTER / "config.json").read_text())
    arrays = {**D.load_weights(cfg), "mask_embedding": D.mask_embedding(cfg)}
    with safe_open(D.HEAD_EXPORT, "np") as f:
        arrays.update({k: f.get_tensor(k) for k in ("lm_head.lut", "lm_head.idx", "lm_head.scale")})
    out = {}
    for k, v in arrays.items():
        a = np.ascontiguousarray(v)
        out[k] = f"{a.dtype}{a.shape}:" + hashlib.sha256(a.view(np.uint8).data if a.size else b"").hexdigest()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--out", type=Path, help="write the digests")
    g.add_argument("--check", type=Path, help="compare with a published digest file")
    ap.add_argument("--drafter", action="store_true", help="the DFlash2 drafter build's inputs instead of the target's")
    a = ap.parse_args()
    d = digest_drafter() if a.drafter else digest()
    if a.out:
        a.out.write_text(json.dumps(d, indent=0, sort_keys=True))
        print(f"{len(d)} arrays -> {a.out}")
        return 0
    ref = json.loads(a.check.read_text())
    bad = sorted(k for k in set(d) | set(ref) if d.get(k) != ref.get(k))
    print(f"{len(d)} arrays checked against {len(ref)}: " + ("identical" if not bad else f"{len(bad)} differ, e.g. {bad[:5]}"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
