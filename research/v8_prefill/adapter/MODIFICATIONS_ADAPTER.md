## Super-adapter build (anemll/anemll-forge-qwen3.8-27B-V8-96K-SuperAdapter, 24 Nov 2026)

Everything above applies unchanged; this repository adds the harness-hub super-adapter
(`harness-hub/Qwen3.8-27B-super-adapter`, Apache-2.0; revision in `adapter/ADAPTER_SOURCE.json`):

- **Weights.** The release 0.2 quantized weights (indices, LUTs, scales, INT8 k / v, the head) are unchanged. Each
  matrix the adapter targets (q / k / v / o of the 16 attention layers, gate / up / down of the 64 MLPs) gets fp16
  low-rank factors `lr_a = (alpha / r) B` (Cout x r) and `lr_b = A` (r x Cin), r = 96, scale 2.0, computed by the
  builder as two 1x1 convolutions next to the quantized weight. On the attention layers they are appended to the
  release's rank-64 corrections and zero-padded to rank 192. Export: `release_vq3pA_sa_p64`
  (`research/v8_prefill/adapter/adapter_export.py`, `PAD_ATTN=64`), record of every matrix in `adapter/adapter_export.json`.
- **Build.** `coreai/m6-96k-dec8ts-sa`: `v8/build_dec8ts_all.sh` with `EXPORT_DIR` set to that export (same entry
  set, numerics and compile settings as `m6-96k-dec8ts`); `v8/chunkmlp.py` carries the MLP factors through the
  chunked FFN. GPU tables `gpu/w_dec8ts_sa` (`prep_engine_par.py` + `prep_kv8.py` from the same export).
- **Runtime.** `pipe/engine/gpu_v8_f8q.swift`: the GPU's low-rank correction reads its rank from the table files
  (multiple of 64, up to 192); the same source serves the base build unchanged.
- **Checks.** Layer 3 against BF16 + adapter: adapted minus base export = the adapter's change to 3e-4 (fp16), builder
  path exact; chunked MLP with factors = reference MLP to 1.9e-6. 8K KL-512: 0.0533 (base 0.0518), top-1 91.67 %
  (91.89 %). 96K A/B and Pi sessions: `research/v8_prefill/results/24nov/adapter/README.md`.
