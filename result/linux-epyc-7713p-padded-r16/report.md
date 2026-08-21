# Factor-4 padded Gaussian patch scaling and small-N MPO contraction

All values use fixed `R=16`, patch cap 128, one pinned EPYC 7713P core, and single-threaded Rayon/BLAS. Gaussian centers occupy the central active box; the computational half-width is four times larger. Patch-scaling rows perform no MPO contraction. Operation timings exclude input generation, patch preparation, reference construction/cache I/O, output conversion, and validation.

## N versus balanced patch layout

| N | input χ (L/R) | (Px, PyL, PyR, Pz) | patches (L/R) | compatible contractions | output projectors | patch build (s) | retained output Gaussians | candidate / N² | retained cache estimate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 115/115 | (1, 1, 1, 1) | 1/1 | 1 | 1 | 0.566 | 190,459 | 1.000 | 8.7 MiB |
| 1,024 | 130/129 | (1, 1, 2, 1) | 1/2 | 2 | 1 | 1.147 | 602,344 | 1.000 | 27.6 MiB |
| 2,048 | 198/198 | (2, 2, 2, 2) | 4/4 | 8 | 4 | 3.534 | 1,891,887 | 0.997 | 86.6 MiB |
| 4,096 | 262/263 | (4, 8, 8, 4) | 32/32 | 128 | 16 | 15.285 | 5,774,560 | 0.937 | 264.3 MiB |
| 8,192 | 283/279 | (4, 8, 8, 4) | 32/32 | 128 | 16 | 21.446 | 17,289,354 | 0.799 | 791.4 MiB |
| 12,000 | 282/282 | (4, 8, 8, 4) | 32/32 | 128 | 16 | 26.639 | 31,241,586 | 0.707 | 1430.1 MiB |

Across N=512 to 12000, retained integrated Gaussians scale empirically as approximately `N^1.62`. This is closer to the expected fixed-density y-overlap law `N^(3/2)` than to linear scaling; storing every retained component is already about 1.40 GiB at N=12,000. The cell-list candidate fraction decreases with N, but the rigorous global `1e-12` tail extent is broad at these sizes.

The patch count is a rank-cap staircase rather than a smooth power law: one patch at N=512, an asymmetric transition at N=1024, 4 patches per operand at N=2048, and 32 regular Cartesian patches per operand from N=4096 through N=12000. Therefore these data do not support a single continuous `N_p ∝ N^α` fit.

## Profiled small-N contractions

| N | input χ | global (s) | patched (s) | speedup | global error | patched error | compatible contributions | output patches |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 115 | 6.755 | 2.620 | 2.578x | 1.964e-06 | 2.664e-06 | 1 | 1 |
| 4,096 | 263 | 70.155 | 68.743 | 1.021x | 1.922e-06 | 2.080e-06 | 128 | 16 |

## Fit/contraction timing breakdown

| N | global zipup init (s) | global sweep (s) | contribution contractions (s) | fit_sum (s) | patched remainder (s) |
|---:|---:|---:|---:|---:|---:|
| 512 | 4.821 | 1.934 | 2.490 (1 calls) | 0.064 (1 calls) | 0.066 |
| 4,096 | 46.953 | 23.160 | 56.961 (128 calls) | 3.037 (16 calls) | 8.745 |

At N=4096 the compatible contribution contractions dominate the patched arm; `fit_sum` is a small fraction. The current factor-4 padded input reaches χ≈263 at N=4096. Larger-N contraction timing was intentionally not run: this profile is for patch scaling plus repeatable small-N timing, with every measurement command bounded below ten minutes.
