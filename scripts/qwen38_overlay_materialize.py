"""Materialize a QAT overlay export (qwen38_qat_kl.py: links to the source export + overlay.qat with the trained
lut / scale / lr tensors) into a self-contained export of full files, the layout the Core AI builder and the drafter
tools read. Tensors and metadata are copied; the overlay's "<file>::<tensor>" entries replace their tensors.

    python qwen38_overlay_materialize.py <overlay export dir> <out dir>
"""
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"Output must be a new directory: {out}")
    over = load_file(src / "overlay.qat")
    by_file = {}
    for k, v in over.items():
        f, key = k.split("::", 1)
        by_file.setdefault(f, {})[key] = v
    # zero-byte bf16-model parameters (norms etc., TRAIN_PARTS norm / qk / ab) have no source file: they go to
    # overlay_model.qat (read by qwen38_kl.apply_export). NOTE: the Core AI converter reads norms from MODEL, so an ANE
    # build of such an export needs a replacement small checkpoint (scripts/qwen38_small_checkpoint.py on main).
    model_t = by_file.pop("model", {})
    if model_t:
        Path(sys.argv[2]).mkdir(parents=True, exist_ok=True)
        save_file({f"model::{k}": v for k, v in model_t.items()}, str(Path(sys.argv[2]) / "overlay_model.qat"))
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("*.safetensors")):
        t = load_file(f)
        with safe_open(f, framework="pt") as fh:
            meta = fh.metadata()
        new = by_file.pop(f.name, {})
        missing = [k for k in new if k not in t]
        assert not missing, f"{f.name}: overlay tensors not in the source: {missing[:5]}"
        for k, v in new.items():
            assert v.shape == t[k].shape and v.dtype == t[k].dtype, (f.name, k, v.shape, t[k].shape)
        t.update(new)
        save_file(t, str(out / f.name), metadata=meta)
    assert not by_file, f"overlay files without a source file: {sorted(by_file)}"
    if (src / "basis.json").exists():  # rot1 basis ID sidecar (R0.5): travels with the export
        import shutil
        shutil.copyfile(src / "basis.json", out / "basis.json")
        print(f"basis.json copied ({__import__('json').loads((src / 'basis.json').read_text())['basis_id'][:16]})")
    print(f"{len(over)} overlay tensors applied -> {out}")


if __name__ == "__main__":
    main()
