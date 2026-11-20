| drafter weights | gsm8k tok/round (vs bf16) | humaneval tok/round (vs bf16) | mbpp tok/round (vs bf16) | lossless |
|---|---|---|---|---|
| bf16 | 10.23 | 13.19 | 10.55 | yes |
| int8 per channel, adapter + head | 10.17 (-0.6%) | 13.26 (+0.5%) | 10.66 (+1.0%) | yes |
| 6-bit g128, adapter + head | 10.20 (-0.3%) | 13.19 (+0.0%) | 10.45 (-1.0%) | yes |
| LUT4 per row, adapter (head bf16) | 9.05 (-11.5%) | 12.23 (-7.3%) | 8.72 (-17.4%) | yes |
| LUT3 per row, adapter (head bf16) | 2.67 (bf16 on the same 2: 11.58, -77%) | not run | not run | yes |
| our DFlash2 chain (tok/verify, same prompts) | 6.08 | 7.30 | 5.89 | |
