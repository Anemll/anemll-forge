"""LUT quantization in the ANE build's native form (20 Nov 2026, B8): one 2^bits-entry scalar palette per tensor over
the row-normalized weights, plus a per-output-channel scale (qwen38_coreai_build.QConv: lut[idx] * scale, cd = 1).
The palette is fit by Lloyd's algorithm on a 8192-bin histogram of w / max|w_row| (weighted quantile init, 40
iterations); every weight takes the nearest palette entry. Used both by the replay (dequantized in place:
the ANE build's exact weight values) and by the ANE build (lut / idx / scale).
    lut, idx, scale = lut_quantize(w, bits)       w: (out, in) torch tensor; returns fp16 lut (K, 1), uint8 idx, fp16 scale"""
import torch


def lut_quantize(w, bits=6, bins=8192, iters=40):
    K = 2 ** bits
    x = w.detach().float()
    scale = x.abs().amax(1, keepdim=True).clamp_min(1e-12)
    z = (x / scale).reshape(-1)
    h = torch.histc(z, bins=bins, min=-1.0, max=1.0)
    centers = torch.linspace(-1.0, 1.0, bins + 1, device=z.device)
    centers = 0.5 * (centers[1:] + centers[:-1])
    cdf = torch.cumsum(h, 0) / h.sum()
    q = (torch.arange(K, device=z.device, dtype=torch.float32) + 0.5) / K
    c = centers[torch.searchsorted(cdf, q).clamp(max=bins - 1)].clone()
    for _ in range(iters):
        c, _ = torch.sort(c)
        mid = 0.5 * (c[1:] + c[:-1])
        a = torch.bucketize(centers, mid)
        num = torch.zeros(K, device=z.device).scatter_add_(0, a, h * centers)
        den = torch.zeros(K, device=z.device).scatter_add_(0, a, h)
        c = torch.where(den > 0, num / den.clamp_min(1e-12), c)
    c, _ = torch.sort(c)
    mid = 0.5 * (c[1:] + c[:-1])
    idx = torch.bucketize(x / scale, mid).to(torch.uint8)
    return c.to(torch.float16).reshape(K, 1), idx, scale.reshape(-1).to(torch.float16)


def lut_dequantize(lut, idx, scale):
    return lut.reshape(-1).float()[idx.long()] * scale.float()[:, None]
