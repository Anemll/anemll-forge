| ctx | prompt | prefill tok/s, Splash / forge | W gross, S / F | W net, S / F | J per prompt token gross, S / F | J net, S / F | samples, S / F |
|---|---|---|---|---|---|---|---|
| 2K | 1,529 | 481 / 599 (+24%) | 19.0 / 15.4 | 14.8 / 11.6 | 0.040 / 0.026 | 0.031 / 0.019 | 2 / 2 |
| 8K | 7,673 | 489 / 685 (+40%) | 43.1 / 48.2 | 38.8 / 44.1 | 0.088 / 0.070 | 0.079 / 0.064 | 13 / 9 |
| 16K | 15,865 | 464 / 650 (+40%) | 46.6 / 48.7 | 42.3 / 44.6 | 0.100 / 0.075 | 0.091 / 0.069 | 29 / 20 |
| 32K | 32,249 | 425 / 576 (+36%) | 45.7 / 48.0 | 41.5 / 43.8 | 0.108 / 0.083 | 0.098 / 0.076 | 64 / 47 |
| 64K | 64,953 | 360 / 465 (+29%) | 44.2 / 46.9 | 39.9 / 42.7 | 0.123 / 0.101 | 0.111 / 0.092 | 153 / 117 |
| 96K | 97,785 | 314 / 389 (+24%) | 44.0 / 45.4 | 39.6 / 41.5 | 0.140 / 0.116 | 0.126 / 0.107 | 264 / 211 |

| ctx | decode tok/s, Splash / forge | W gross, S / F | W net, S / F | J per generated token gross, S / F | J net, S / F | samples, S / F |
|---|---|---|---|---|---|---|
| 2K | 59.8 / 62.2 (+4%) | 42.1 / 23.9 | 37.9 / 20.2 | 0.702 / 0.384 | 0.631 / 0.324 | 4 / 3 |
| 8K | 61.2 / 59.0 (-4%) | 43.6 / 24.6 | 39.3 / 20.6 | 0.710 / 0.416 | 0.640 / 0.347 | 4 / 4 |
| 16K | 57.8 / 58.3 (+1%) | 44.4 / 24.6 | 40.1 / 20.5 | 0.766 / 0.421 | 0.692 / 0.351 | 3 / 4 |
| 32K | 53.2 / 53.1 (-0%) | 41.8 / 24.7 | 37.5 / 20.4 | 0.782 / 0.463 | 0.703 / 0.383 | 5 / 4 |
| 64K | 46.9 / 49.9 (+6%) | 42.7 / 24.6 | 38.5 / 20.4 | 0.909 / 0.492 | 0.818 / 0.407 | 5 / 4 |
| 96K | 44.4 / 45.3 (+2%) | 43.2 / 27.4 | 38.9 / 23.5 | 0.970 / 0.602 | 0.873 / 0.518 | 5 / 5 |

idle W (each run's own 30 s baseline): Splash splash170_16384.json 4.29; Splash splash170_2048.json 4.25; Splash splash170_32768.json 4.21; Splash splash170_65536.json 4.26; Splash splash170_8192.json 4.31; Splash splash170_98304.json 4.32; forge forge_16384.json 4.09; forge forge_2048.json 3.73; forge forge_32768.json 4.27; forge forge_65536.json 4.24; forge forge_8192.json 4.06; forge forge_98304.json 3.83
Splash 2K decode repeats tok/s [59.8, 59.7, 61.3] (splash170_2048.json)
Splash 8K decode repeats tok/s [61.4, 61.2, 61.2] (splash170_8192.json)
Splash 16K decode repeats tok/s [57.6, 57.8, 57.8] (splash170_16384.json)
Splash 32K decode repeats tok/s [53.2, 51.9, 53.4] (splash170_32768.json)
Splash 64K decode repeats tok/s [46.8, 46.9, 48.0] (splash170_65536.json)
Splash 96K decode repeats tok/s [44.4, 44.4, 44.3] (splash170_98304.json)
forge 2K decode repeats tok/s [62.2, 62.2, 62.2] (forge_2048.json)
forge 8K decode repeats tok/s [58.9, 59.0, 59.0] (forge_8192.json)
forge 16K decode repeats tok/s [58.2, 58.3, 58.4] (forge_16384.json)
forge 32K decode repeats tok/s [53.1, 53.1, 53.2] (forge_32768.json)
forge 64K decode repeats tok/s [49.9, 49.9, 49.8] (forge_65536.json)
forge 96K decode repeats tok/s [44.8, 45.5, 45.3] (forge_98304.json)
