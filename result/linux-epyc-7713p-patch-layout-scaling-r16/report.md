# Balanced versus shared-y-only patch scaling

These are contraction-free measurements on factor-4 padded, fixed-`R=16` inputs. Both layouts use patch cap 128 and patch reconstruction tolerance `1e-6`. Operation times are not measured here; work proxies are structural sums over actual compatible patch pairs.

## Default-compression N sweep

| N | χ | balanced patches L/R | balanced pairs | balanced outputs | y-only patches L/R | y-only pairs | y-only outputs | pair ratio B/Y | parameter-product proxy ratio Y/B |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 115 | 1/1 | 1 | 1 | 1/1 | 1 | 1 | 1.00 | 1.000 |
| 1,024 | 130 | 1/2 | 2 | 1 | 1/2 | 2 | 1 | 1.00 | 1.000 |
| 2,048 | 198 | 4/4 | 8 | 4 | 8/8 | 8 | 1 | 1.00 | 0.426 |
| 4,096 | 263 | 32/32 | 128 | 16 | 16/16 | 16 | 1 | 8.00 | 0.437 |
| 8,192 | 283 | 32/32 | 128 | 16 | 32/32 | 32 | 1 | 4.00 | 0.549 |
| 12,000 | 282 | 32/32 | 128 | 16 | 32/32 | 32 | 1 | 4.00 | 0.434 |

## Local ranks, storage, and exact patch reconstruction error

| N | layout | max χ L/R | median χ L/R | p90 χ L/R | saturated L/R | parameters L+R | reconstruction error L/R | build (s) | validation (s) |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | balanced_xyz | 114/114 | 114/114 | 114/114 | 0/0 | 169,196 | 1.383e-06/1.312e-06 | 0.573 | 0.087 |
| 512 | shared_y_only | 114/114 | 114/114 | 114/114 | 0/0 | 169,196 | 1.383e-06/1.312e-06 | 0.564 | 0.088 |
| 1,024 | balanced_xyz | 128/119 | 128/119 | 128/119 | 1/0 | 272,464 | 1.882e-06/2.412e-06 | 1.158 | 0.212 |
| 1,024 | shared_y_only | 128/119 | 128/119 | 128/119 | 1/0 | 272,464 | 1.882e-06/2.412e-06 | 1.137 | 0.200 |
| 2,048 | balanced_xyz | 88/87 | 85/85 | 85/85 | 0/0 | 486,228 | 3.299e-06/3.313e-06 | 3.496 | 0.990 |
| 2,048 | shared_y_only | 84/85 | 34/33 | 84/85 | 0/0 | 524,580 | 3.652e-06/3.587e-06 | 6.194 | 2.928 |
| 4,096 | balanced_xyz | 121/121 | 25/25 | 117/118 | 0/0 | 1,152,216 | 3.793e-06/3.873e-06 | 16.605 | 44.914 |
| 4,096 | shared_y_only | 120/119 | 31/30 | 119/118 | 0/0 | 1,062,292 | 3.684e-06/3.719e-06 | 10.378 | 13.289 |
| 8,192 | balanced_xyz | 128/128 | 19/20 | 127/127 | 4/2 | 1,478,476 | 3.983e-06/4.163e-06 | 21.538 | 46.328 |
| 8,192 | shared_y_only | 127/127 | 22/22 | 125/125 | 0/0 | 2,171,320 | 3.616e-06/3.932e-06 | 19.222 | 95.156 |
| 12,000 | balanced_xyz | 128/128 | 14/15 | 128/128 | 5/6 | 1,755,552 | 3.814e-06/3.911e-06 | 26.903 | 44.960 |
| 12,000 | shared_y_only | 128/107 | 15/18 | 107/107 | 2/0 | 2,349,388 | 3.471e-06/3.605e-06 | 24.496 | 86.642 |

## Fixed-N compressed-χ sweep (N=12,000)

| input rtol | χ | layout | patches L/R | pairs | outputs | max χ L/R | parameter-product proxy | cubed max-bond proxy | reconstruction error max |
|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| 1e-06 | 282 | balanced_xyz | 32/32 | 128 | 16 | 128/128 | 179,873,276,512 | 6.982e+13 | 3.911e-06 |
| 1e-06 | 282 | shared_y_only | 32/32 | 32 | 1 | 128/107 | 78,054,235,072 | 2.568e+13 | 3.605e-06 |
| 1e-08 | 423 | balanced_xyz | 32/32 | 128 | 16 | 128/128 | 187,697,370,000 | 6.956e+13 | 4.839e-06 |
| 1e-08 | 423 | shared_y_only | 32/32 | 32 | 1 | 128/111 | 87,043,763,520 | 3.217e+13 | 4.635e-06 |
| 1e-09 | 499 | balanced_xyz | 32/32 | 128 | 16 | 128/128 | 187,697,370,000 | 6.956e+13 | 4.839e-06 |
| 1e-09 | 499 | shared_y_only | 32/32 | 32 | 1 | 128/111 | 87,043,763,520 | 3.217e+13 | 4.635e-06 |
| 1e-10 | 583 | balanced_xyz | 32/32 | 128 | 16 | 128/128 | 187,697,370,000 | 6.956e+13 | 4.840e-06 |
| 1e-10 | 583 | shared_y_only | 32/32 | 32 | 1 | 128/111 | 87,043,763,520 | 3.217e+13 | 4.635e-06 |

## Interpretation

- The observed balanced staircase reaches 32 patches/operand, 128 compatible pairs, and 16 output groups at N=4,096, then remains unchanged through N=12,000.
- Shared-y-only reaches 16 patches/operand at N=4,096 and 32 at N=8,192. From N=8,192 onward it has 32 compatible pairs and one output group, versus balanced 128 and 16.
- At N=12,000, shared-y-only has 0.434× the balanced parameter-product proxy and 0.368× the cubed max-bond proxy. This suggests less pairwise contraction work, but it does not include the rank of each contracted contribution or the cost of fitting all y contributions into one global x/z output group.
- Increasing compressed input χ from 282 to 583 at fixed N does not change either patch layout. The patch representation is truncated with its separate fixed `patch_input_rtol=1e-6`, so additional global-input precision is discarded before the hypothetical contraction.
- Exact reconstruction errors are approximately 1.3e-6 to 4.8e-6 and therefore exceed the nominal 1e-6 patch tolerance. The records mark `patch_tolerance_met=false`; no contraction estimate should hide this effective patch-error floor.
- The bounded N=16,384 attempt timed out during padded global-TCI input construction before producing a patch record. Therefore the next N-driven staircase is only bounded as greater than 12,000 in this single-core, 570-second workflow.

The compatible-pair and structural proxy ratios are not measured contraction speedups.
