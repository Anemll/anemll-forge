> Historical sensitivity prior, imported 2026-09-29. Its recommendation to prioritize late MLP layers for 2-bit quantization is superseded by the later in-domain experiments in [QUANTIZATION_NOTES.md](QUANTIZATION_NOTES.md). Scalar grouped-LUT sensitivity does not establish vector-LUT sensitivity.

# Qwen3.8-27B relative sensitivity for 2-bit VQ

Source: [p4ik/Qwen3.8-27B-MLX-OptiQ-5bit](https://huggingface.co/p4ik/Qwen3.8-27B-MLX-OptiQ-5bit) `optiq/sensitivity.json`.
Metric: KL vs bf16 at scalar 4-bit (group 64). **`dens4 = KL@4 / (params in billions)`** — higher = more fragile per weight.

**Use as relative priority only** (scalar group-quant KL ≠ vector-LUT SNR). Cross-check with APEX ΔKL/GB and True2456 concentration.

## Role ranking (tensors ≥1M params)

| Priority | Role | Mean dens | Rel. vs mlp_down med | Params | 2-bit VQ guidance |
|---:|---|---:|---:|---:|---|
| 1 | `full_attn_v` | 1.315 | ×19.63 | 0.08B | Highest dens among attentions — **do not 2-bit first**; keep 4–8b |
| 2 | `full_attn_k` | 0.810 | ×12.09 | 0.08B | Very high — protect with V |
| 3 | `gdn_in_z` | 0.157 | ×2.35 | 1.51B | ~2.3× MLP — last GDN to crush |
| 4 | `gdn_out` | 0.155 | ×2.31 | 1.51B | ~2.3× MLP — sensitive |
| 5 | `full_attn_o` | 0.147 | ×2.20 | 0.50B | ~2.2× MLP — careful |
| 6 | `gdn_in_qkv` | 0.115 | ×1.71 | 2.52B | ~1.7× MLP — careful 2-bit |
| 7 | `full_attn_q` | 0.096 | ×1.43 | 1.01B | ~1.4× MLP — careful |
| 8 | `mlp_down` | 0.058 | ×0.87 | 5.70B | Bulk candidate — primary 2-bit VQ target |
| 9 | `mlp_up` | 0.058 | ×0.86 | 5.70B | Bulk candidate |
| 10 | `mlp_gate` | 0.057 | ×0.85 | 5.70B | Bulk candidate |
| 11 | `lm_head` | 0.002 | ×0.03 | 1.27B | OptiQ dens looks mild; **still protect** (APEX dearest / 248k vocab) |

## Layer position (same metric)

OptiQ dens is **higher early**, lower late:
- MLP dens ~0.08–0.09 at L0–7 → ~0.005–0.01 at L56–63
- Same pattern for GDN / full-attn

**Crush late MLP first** for 2-bit VQ; keep early attn/GDN richer.

## Practical run order for 2-bit VQ (`nbits=4, cluster_dim=2` → 2 bpw)

1. **First (safest bulk):** `layers.48–63.mlp.{gate,up,down}_proj`
2. **Then:** `layers.24–47.mlp.*`
3. **Then:** `layers.8–23.mlp.*` (watch quality)
4. **Only if needed:** late `gdn_in_qkv`, `gdn_out`, `full_attn_o/q`
5. **Hold out of 2-bit:** `full_attn_{k,v}` (esp. early/mid), early `gdn_in_z`/`out`, all `in_proj_a/b`, **`lm_head`**, vision, MTP, norms

## Cross-checks from other cards

| Source | Extra signal |
|---|---|
| APEX ΔKL/GB | `output` ≫ full-attn ≫ FFN edge ≫ FFN middle ≫ `token_embd` |
| True2456 PR | attn QKV most concentrated (135× vs down_proj); GDN in_proj 2nd; down_proj flattest |
| GPTQ Int4 | worst absolute module loss: `L61 mlp.gate_proj`; cluster L58–61 gate |

Note: GPTQ late-gate MSE ≠ OptiQ early-high KL dens — different objectives. For **bit allocation**, prefer OptiQ dens + APEX; for **debugging a bad late layer after VQ**, check GPTQ hotspots.

## Files in this folder
- `tensor_sensitivity_ranked.csv`
- `role_band_summary.csv`
- `vq2bit_tiers.txt`
- `sensitivity.json`, `metadata.json` (upstream)