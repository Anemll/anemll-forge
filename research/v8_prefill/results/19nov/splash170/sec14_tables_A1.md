| ctx | prompt | prefill tok/s, Splash / forge | W gross, S / F | W net, S / F | J per prompt token gross, S / F | J net, S / F | samples, S / F |
|---|---|---|---|---|---|---|---|
| 2K | 1,529 | 484 / 598 (+24%) | 27.6 / 15.4 | 22.7 / 11.2 | 0.057 / 0.026 | 0.047 / 0.019 | 3 / 2 |
| 8K | 7,673 | 483 / 659 (+36%) | 47.2 / 45.1 | 42.3 / 40.9 | 0.098 / 0.068 | 0.088 / 0.062 | 13 / 10 |
| 16K | 15,865 | 448 / 579 (+29%) | 43.2 / 43.5 | 38.3 / 39.3 | 0.096 / 0.075 | 0.086 / 0.068 | 30 / 23 |
| 32K | 32,249 | 415 / 545 (+31%) | 43.9 / 45.4 | 39.0 / 41.2 | 0.106 / 0.083 | 0.094 / 0.076 | 66 / 49 |
| 64K | 64,953 | 358 / 459 (+28%) | 44.1 / 45.5 | 39.2 / 41.3 | 0.123 / 0.099 | 0.110 / 0.090 | 155 / 119 |
| 96K | 97,785 | 311 / 374 (+20%) | 43.1 / 44.5 | 38.9 / 39.5 | 0.139 / 0.119 | 0.125 / 0.106 | 268 / 220 |

| ctx | decode tok/s, Splash / forge | W gross, S / F | W net, S / F | J per generated token gross, S / F | J net, S / F | samples, S / F |
|---|---|---|---|---|---|---|
| 2K | 61.0 / 60.6 (-1%) | 43.1 / 24.4 | 38.2 / 20.1 | 0.705 / 0.400 | 0.625 / 0.331 | 3 / 4 |
| 8K | 60.6 / 59.1 (-2%) | 44.0 / 24.7 | 39.1 / 20.5 | 0.724 / 0.416 | 0.643 / 0.345 | 3 / 4 |
| 16K | 54.5 / 50.6 (-7%) | 41.1 / 24.6 | 36.2 / 20.4 | 0.751 / 0.484 | 0.661 / 0.401 | 4 / 4 |
| 32K | 54.1 / 57.5 (+6%) | 41.9 / 24.9 | 37.0 / 20.6 | 0.771 / 0.431 | 0.681 / 0.358 | 4 / 4 |
| 64K | 47.9 / 52.7 (+10%) | 42.6 / 24.5 | 37.7 / 20.3 | 0.886 / 0.464 | 0.784 / 0.384 | 5 / 4 |
| 96K | 43.3 / 46.4 (+7%) | 41.5 / 28.4 | 37.3 / 23.4 | 0.954 / 0.609 | 0.858 / 0.503 | 5 / 5 |

idle W (each run's own 30 s baseline): Splash splash170_ud_iq3xxs.json 4.91; Splash splash170_ud_iq3xxs_96k.json 4.21; forge forge_qk.json 4.21; forge bench_A1.json 4.93
Splash 2K decode repeats tok/s [61.0, 61.0, 60.9] (splash170_ud_iq3xxs.json)
Splash 8K decode repeats tok/s [60.9, 60.6, 60.2] (splash170_ud_iq3xxs.json)
Splash 16K decode repeats tok/s [54.3, 54.5, 54.8] (splash170_ud_iq3xxs.json)
Splash 32K decode repeats tok/s [54.1, 52.6, 54.2] (splash170_ud_iq3xxs.json)
Splash 64K decode repeats tok/s [47.9, 47.9, 48.0] (splash170_ud_iq3xxs.json)
Splash 96K decode repeats tok/s [44.2, 43.2, 43.3] (splash170_ud_iq3xxs_96k.json)
forge 2K decode repeats tok/s [60.7, 60.6, 60.6] (forge_qk.json)
forge 8K decode repeats tok/s [59.1, 59.1, 59.1] (forge_qk.json)
forge 16K decode repeats tok/s [50.4, 50.7, 50.6] (forge_qk.json)
forge 32K decode repeats tok/s [57.5, 57.5, 57.6] (forge_qk.json)
forge 64K decode repeats tok/s [52.7, 52.7, 52.6] (forge_qk.json)
forge 96K decode repeats tok/s [43.5, 46.4, 46.7] (bench_A1.json)
