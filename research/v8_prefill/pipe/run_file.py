import os, sys, subprocess, numpy as np
NT = int(os.environ.get('NT', '64'))
man, l0, l1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
Z = np.load("../w8a8/hidden_1024.npz")
x = Z[f"after_layer_{l0 - 1}"].astype(np.float16)                     # (1024, 5120)
nb = x.shape[0] // NT
np.ascontiguousarray(x.reshape(nb, NT, 5120).transpose(0, 2, 1)).tofile("blk_in.f16")
r = subprocess.run(["engine/gpu_layers", man, "file", "blk_in.f16", "blk_out.f16", str(nb)], capture_output=True, text=True)
print(r.stdout.strip(), r.stderr[-1500:])
y = np.fromfile("blk_out.f16", np.float16).reshape(nb, 5120, NT).transpose(0, 2, 1).reshape(-1, 5120).astype(np.float64)
rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)
ane = Z[f"after_layer_{l1 - 1}"].astype(np.float64)
print(f"GPU vs ANE chunk output: rel err {rel(y, ane):.4f}")
try:
    ref = np.load(f"../w8a8/hidden_1024_ref_{l0}_{l1 - 1}.npy").astype(np.float64)
    print(f"GPU vs fp32 reference: rel err {rel(y, ref):.2e} | reference vs ANE {rel(ref, ane):.4f}")
    d = np.linalg.norm(y - ref, axis=1) / np.linalg.norm(ref, axis=1)
    print("   per-token GPU vs ref: first", np.round(d[:3], 5), "max", d.max(), "at", d.argmax())
except FileNotFoundError:
    pass
