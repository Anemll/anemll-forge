"""CPU references at 512 rows for the 17 Nov LUT3 row-sweep packages (18 Nov 2026): the served FFN (layer 36, chain of 4
passes) and the LUT3 FP16 / INT8 modules on the same inputs (x_512 of lut3_rows/ref.npz), written to a NEW file
lut3_rows/ref512.npz (ref.npz untouched). BUILD env (as v8/build_dec8_one.sh): BUILDPY ref512.py"""
import sys, time
from pathlib import Path
import numpy as np
W8 = Path(__file__).resolve().parent
sys.path.insert(0, str(W8))
import qdq_probe as QP  # noqa: E402

OUT = Path("/Volumes/SSD4TB/anemll-forge-research/a8probe/lut3_rows")
L, HID, CHAIN, T = QP.L, QP.HID, 4, 512
torch, B, M, V, CM, Q = QP.builder()
import lut3  # noqa: E402
layer = QP.served_layer(torch, B, M, V, CM)
Wd = B.layer_arrays(M.Checkpoint(), L)
Wn = lut3.layer_lutw(Wd, L, V.SPLIT)
del Wd
x = np.load(OUT / "ref.npz")["x_%d" % T]
xt = torch.from_numpy(x).view(1, HID, 1, T)
res = {"x_%d" % T: x}
with torch.no_grad():
    for name, mod in (("served", layer.cmlp), ("fp16", lut3.ServedL3(B, layer, Wn, "fp16").eval()),
                      ("i8gud", lut3.ServedL3(B, layer, Wn, "i8gud").eval())):
        t0 = time.time(); y = xt
        for _ in range(CHAIN):
            y = mod(y)
        res["%s_%d" % (name, T)] = y.view(HID, T).float().numpy()
        print("%s: %.0f s" % (name, time.time() - t0), flush=True)
        np.savez(OUT / "ref512.npz", **res)
print("saved", OUT / "ref512.npz")
