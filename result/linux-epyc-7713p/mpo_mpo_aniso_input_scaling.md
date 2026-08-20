# Anisotropic Gaussian MPO input and patch scaling

This focused case-6 input study measures adaptive patching through compressed input bond dimension `chi_in ≈ 1000`. It uses the unchanged constant-density random anisotropic family (`sigma_minor = 0.05`, log-uniform aspect ratio in `[1, 8]`, independent angle, center, and positive weight per Gaussian).

## Construction and accuracy

The input pipeline is:

1. fused two-variable TCI at relative tolerance `1e-8`;
2. global relative-L2 SVD truncation at `1e-6`;
3. adaptive input truncation with per-patch cap 128.

TCI evaluates the same random mixture through a spatial index. A component is omitted only outside a radius where its individual Gaussian is below a common exponent threshold. Since all weights are positive, the sum of all omitted tails is bounded pointwise by `1e-12`. The committed sampled input errors include this bound and remain below `1e-5` after global SVD truncation.

The alternative multiscale constructor builds one randomly rotated Gaussian directly. It marks points every half minor-axis standard deviation along the major-axis ridge as unsafe. The focused test covers `rho = 8`, a 45-degree rotation, and `R = 10`, and requires maximum sampled absolute error below `5e-8` at polynomial degree 28. Mixtures can be formed by adding these single-Gaussian QTTs with intermediate SVD compression. This path preserves the instance family but is not used for the large sweep because its per-Gaussian construction cost scales linearly with `N`; spatially indexed TCI reaches the requested rank regime efficiently.

## Reproducible settings

These input-only records were collected separately from the profile's earlier contraction sweep. The settings below, including core 0, are authoritative for these records rather than the contraction affinity in `run.yaml`.

- Machine: AMD EPYC 7713P 64-Core Processor
- CPU affinity: core 0 (`taskset -c 0`)
- `RAYON_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`
- `BENCH_INPUT_GENERATOR=tci`
- `BENCH_INPUT_TCI_RTOL=1e-8`
- `BENCH_TCI_LOCAL_ABS_TOL=1e-12`
- `BENCH_TCI_INITIAL_PIVOTS=8` (deterministic center-derived pivots)
- `BENCH_INPUT_SVD_RTOL=1e-6`
- `BENCH_PATCH_MAX_BOND=128`
- `BENCH_MAX_INPUT_CHI=1400`
- `BENCH_MAX_BOND=512, 512, 1280, 3072, 3584` for `N=512, 4096, 20480, 100000, 125000`
- one deterministic seed (`BENCH_SEED=0`)
- `BENCH_ERROR_SAMPLES=256` at every size

Input generation, global SVD truncation, sampled validation, and adaptive patch preparation are all included in `preparation time`; no contraction is run (`BENCH_INPUT_ONLY=1`).

## Results

Each pair is `(left, right)` for the two independently drawn mixtures.

| N | R | raw chi | compressed chi | global parameters | patches | max patch chi | patch parameters | sampled relative error | preparation time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 9 | 181 | 89 | (48,060, 47,608) | (1, 1) | (84, 84) | (45,056, 45,056) | (3.36e-06, 2.56e-06) | 5.08 s |
| 4,096 | 10 | 280 | 256 | (254,476, 255,736) | (6, 6) | (124, 124) | (352,704, 352,088) | (1.78e-06, 3.38e-06) | 46.36 s |
| 20,480 | 11 | 889 | 458 | (888,076, 886,408) | (30, 30) | (121, 121) | (1,775,032, 1,772,136) | (2.75e-06, 6.95e-06) | 263.99 s |
| 100,000 | 12 | 1,074 | 900 | (3,450,320, 3,454,660) | (120, 120) | (120, 121) | (9,866,732, 9,982,956) | (3.06e-06, 3.74e-06) | 1,568.20 s |
| 125,000 | 13 | 1,342 | 900 | (4,120,944, 4,113,404) | (120, 120) | (121, 123) | (11,553,076, 11,596,696) | (5.51e-06, 9.14e-06) | 2,539.63 s |

The compressed-rank plateau reaches `chi_in = 900`, within 10% of the requested `chi_in ≈ 1000` regime. Patch count grows stepwise from 1 to 6, 30, and 120 while every patch remains below cap 128. The `N = 125,000`, `R = 13` point increases raw rank and parameter count but remains on the same compressed-rank and patch-count plateau, so increasing `N` alone does not produce a smooth patch-count curve.

Raw JSON records are in [`raw/`](raw/) with filenames `mpo_mpo_aniso_input-tci-n*-chi*.json`.
