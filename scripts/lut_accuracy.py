"""Scalar vs vector (Cout-axis) k-means palettization: weight SNR at equal bits/weight.
Only ANE-valid LUTs (vector <= 16, entries*vector <= 256). Per-tensor LUT."""
import numpy as np
import torch
from sklearn.cluster import KMeans

CONFIGS = [  # (cluster_dim, n_bits)
    (1, 4), (2, 6),            # 4, 3 bits
    (1, 3),                    # 3
    (1, 2), (2, 4),            # 2
    (4, 6),                    # 1.5
    (1, 1), (2, 2), (4, 4),    # 1
    (4, 2), (8, 4),            # 0.5
    (16, 4), (8, 2),           # 0.25
]


def vq(w2d: np.ndarray, cd: int, nb: int, fit_n: int = 200_000, seed: int = 0) -> np.ndarray:
    """w2d (Cout, Cin); vectors = cd consecutive output channels at one input index."""
    cout, cin = w2d.shape
    vecs = w2d.reshape(cout // cd, cd, cin).transpose(0, 2, 1).reshape(-1, cd)
    rng = np.random.default_rng(seed)
    sample = vecs[rng.choice(len(vecs), min(fit_n, len(vecs)), replace=False)]
    km = KMeans(1 << nb, n_init=1, random_state=seed, max_iter=100).fit(sample)
    q = km.cluster_centers_[km.predict(vecs)]
    return q.reshape(cout // cd, cin, cd).transpose(0, 2, 1).reshape(cout, cin)


def snr_db(w, q):
    return 10 * np.log10((w**2).sum() / ((w - q) ** 2).sum())


def main():
    sd = torch.load("/path/to/user/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth", map_location="cpu")
    mats = {
        "resnet50 layer4.2.conv3 (2048x512)": sd["layer4.2.conv3.weight"].reshape(2048, -1).detach().numpy(),
        "resnet50 layer4.0.conv2 (512x4608, 3x3)": sd["layer4.0.conv2.weight"].reshape(512, -1).detach().numpy(),
        "resnet50 fc (1000x2048)": sd["fc.weight"].detach().numpy()[:992],
        "gaussian (1024x1024)": np.random.default_rng(1).standard_normal((1024, 1024)).astype(np.float32),
    }
    names = list(mats)
    print(f"{'config':14s} {'bits/w':>6s} " + " ".join(f"{n.split(' (')[0][-18:]:>18s}" for n in names))
    for cd, nb in CONFIGS:
        row = []
        for n in names:
            w = mats[n].astype(np.float64)
            row.append(snr_db(w, vq(w, cd, nb)))
        label = f"{'scalar' if cd == 1 else f'vec{cd}'} {1 << nb}-entry"
        print(f"{label:14s} {nb / cd:6g} " + " ".join(f"{v:18.2f}" for v in row), flush=True)


if __name__ == "__main__":
    main()
