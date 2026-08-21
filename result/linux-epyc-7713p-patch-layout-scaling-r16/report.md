# Balanced versus shared-y-only patch scaling

These are contraction-free measurements on factor-4 padded, fixed-`R=16` inputs. Both layouts use patch cap 128 and patch reconstruction tolerance `1e-6`. Operation times are not measured here; work proxies are structural sums over actual compatible patch pairs.

## Default-compression N sweep

| N | χ | balanced patches L/R | balanced pairs | balanced outputs | y-only patches L/R | y-only pairs | y-only outputs | pair ratio B/Y | parameter-product proxy ratio Y/B |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 115 | 1/1 | 1 | 1 | 1/1 | 1 | 1 | 1.00 | 1.000 |
| 1,024 | 130 | 4/4 | 8 | 4 | 8/12 | 8 | 1 | 1.00 | 0.548 |
| 2,048 | 198 | 4/4 | 8 | 4 | 8/8 | 8 | 1 | 1.00 | 0.418 |
| 4,096 | 263 | 32/32 | 128 | 16 | 32/32 | 32 | 1 | 4.00 | 0.494 |
| 8,192 | 283 | 32/32 | 128 | 16 | 32/32 | 32 | 1 | 4.00 | 0.322 |
| 12,000 | 282 | 32/32 | 128 | 16 | 32/32 | 32 | 1 | 4.00 | 0.361 |

## Local ranks, storage, and exact patch reconstruction error

| N | layout | max χ L/R | median χ L/R | p90 χ L/R | saturated L/R | parameters L+R | reconstruction error L/R | build (s) | validation (s) |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | balanced_xyz | 115/115 | 115/115 | 115/115 | 0/0 | 171,392 | 0.000e+00/7.018e-08 | 0.543 | 0.078 |
| 512 | shared_y_only | 115/115 | 115/115 | 115/115 | 0/0 | 171,392 | 0.000e+00/7.018e-08 | 0.518 | 0.085 |
| 1,024 | balanced_xyz | 91/128 | 89/92 | 91/128 | 0/2 | 452,484 | 4.138e-07/3.569e-07 | 2.374 | 0.788 |
| 1,024 | shared_y_only | 82/128 | 50/128 | 81/128 | 0/8 | 1,015,528 | 5.448e-07/4.243e-07 | 3.868 | 7.940 |
| 2,048 | balanced_xyz | 95/92 | 90/92 | 91/92 | 0/0 | 532,768 | 3.763e-07/4.100e-07 | 3.408 | 1.038 |
| 2,048 | shared_y_only | 87/87 | 45/44 | 87/87 | 0/0 | 609,864 | 5.642e-07/5.804e-07 | 5.451 | 2.679 |
| 4,096 | balanced_xyz | 128/127 | 41/42 | 125/124 | 1/0 | 1,513,988 | 5.226e-07/5.485e-07 | 14.996 | 68.434 |
| 4,096 | shared_y_only | 127/126 | 47/48 | 121/123 | 0/0 | 2,145,940 | 5.125e-07/5.286e-07 | 12.585 | 119.147 |
| 8,192 | balanced_xyz | 128/128 | 39/42 | 128/128 | 6/7 | 1,824,052 | 5.116e-07/5.129e-07 | 21.865 | 75.464 |
| 8,192 | shared_y_only | 87/87 | 40/46 | 87/87 | 0/0 | 2,167,456 | 4.795e-07/4.878e-07 | 20.379 | 109.896 |
| 12,000 | balanced_xyz | 128/128 | 34/38 | 128/128 | 6/7 | 2,051,768 | 5.817e-07/5.815e-07 | 28.619 | 78.196 |
| 12,000 | shared_y_only | 108/108 | 35/42 | 108/108 | 0/0 | 2,535,956 | 5.479e-07/5.649e-07 | 26.620 | 135.860 |

## Selected fixed-N compressed-χ checks

| N | input rtol | χ | layout | patches L/R | pairs | outputs | max χ L/R | parameter-product proxy | cubed max-bond proxy | reconstruction error max |
|---:|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 1e-06 | 283 | balanced_xyz | 32/32 | 128 | 16 | 128/128 | 152,207,577,904 | 7.539e+13 | 5.129e-07 |
| 8,192 | 1e-06 | 283 | shared_y_only | 32/32 | 32 | 1 | 87/87 | 48,975,778,640 | 7.014e+12 | 4.878e-07 |
| 8,192 | 3e-10 | 418 | balanced_xyz | 32/32 | 128 | 16 | 128/128 | 193,077,552,128 | 7.091e+13 | 8.935e-07 |
| 8,192 | 3e-10 | 418 | shared_y_only | 32/32 | 32 | 1 | 105/106 | 70,844,638,864 | 1.977e+13 | 8.536e-07 |

## Interpretation

- The layouts are identical only at N=512. At N=1,024, balanced uses 4/4 patches while shared-y-only uses 8/12, although both still have eight compatible pairs.
- The observed balanced staircase reaches 32 patches/operand, 128 compatible pairs, and 16 output groups at N=4,096, then remains unchanged through N=12,000.
- Shared-y-only reaches 32 patches/operand at N=4,096. From that point onward it has 32 compatible pairs and one output group, versus balanced 128 and 16.
- At N=12,000, shared-y-only has 0.361× the balanced parameter-product proxy and 0.339× the cubed max-bond proxy. This suggests less pairwise contraction work, but it does not include the rank of each contracted contribution or the cost of fitting all y contributions into one global x/z output group.
- At N=8,192, increasing compressed input χ from 283 to 418 does not change either patch layout. The higher-rank inputs remain within the independently checked patch tolerance. N=12,000 χ≥423 probes exceeded the 570-second construction/validation bound and are not reported as measurements.
- The requested whole-input `patch_input_rtol=1e-6` is converted to a local SVD tolerance by distributing its squared budget over one sweep's two visits to each of the 15 chain edges. Every recorded exact reconstruction residual is at most 8.94e-7 and has `patch_tolerance_met=true`.
- The bounded N=16,384 attempt timed out during padded global-TCI input construction before producing a patch record. Therefore the next N-driven staircase is only bounded as greater than 12,000 in this single-core, 570-second workflow.

The compatible-pair and structural proxy ratios are not measured contraction speedups.
