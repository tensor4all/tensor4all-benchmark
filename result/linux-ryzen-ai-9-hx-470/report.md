# Tensor4all benchmark results

Profile: `linux-ryzen-ai-9-hx-470`. CPU: AMD Ryzen AI 9 HX 470 w/ Radeon 890M. Threads: 1. CPU affinity: 0. Source revision: `05ebd78bdc40e09f1534a4bab2ac72131ce271d3`. tensor4all-rs revision: `a5157385f58048355a28a6a003d4cb63d3b1349b`.

The source revision identifies the clean code that was measured. The commit adding these generated records necessarily follows that revision.

All timings exclude input construction, cache I/O, format conversion, patch preparation, output conversion and accuracy evaluation. Gaussian inputs use independent two-dimensional interpolative decompositions, balanced pairwise addition, final relative-L2/SVD tolerance `1e-6`, and fixed patch cap 128.

## Case 1: Fourier elementwise

| input χ | K | arm | time (s) | sampled relative L2 | output χ | parameters | allocations | allocated (MiB) |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 5 | 4 | aci | 0.051511 | 1.166e-08 | 9 | 1026 | 935093 | 92.31 |
| 5 | 4 | fit | 0.012445 | 1.213e-08 | 8 | 606 | 142866 | 23.60 |
| 5 | 4 | naive | 0.000918 | 1.211e-08 | 8 | 606 | 5323 | 2.61 |
| 5 | 4 | zipup | 0.006449 | 1.197e-08 | 8 | 606 | 80441 | 12.09 |
| 8 | 8 | aci | 0.073725 | 8.181e-09 | 16 | 1702 | 1293258 | 129.49 |
| 8 | 8 | fit | 0.012954 | 1.066e-08 | 8 | 726 | 143003 | 24.22 |
| 8 | 8 | naive | 0.001320 | 1.066e-08 | 8 | 726 | 5323 | 3.85 |
| 8 | 8 | zipup | 0.006676 | 1.080e-08 | 8 | 726 | 80573 | 12.56 |
| 8 | 16 | aci | 0.063717 | 6.086e-09 | 16 | 2206 | 1095665 | 111.28 |
| 8 | 16 | fit | 0.013237 | 1.620e-08 | 10 | 900 | 142989 | 24.91 |
| 8 | 16 | naive | 0.001710 | 1.620e-08 | 10 | 900 | 5323 | 5.12 |
| 8 | 16 | zipup | 0.006923 | 1.636e-08 | 10 | 900 | 80547 | 13.03 |
| 11 | 32 | aci | 0.074269 | 4.931e-09 | 23 | 3318 | 1253379 | 136.21 |
| 11 | 32 | fit | 0.013954 | 1.806e-08 | 13 | 1200 | 143389 | 26.29 |
| 11 | 32 | naive | 0.002830 | 1.806e-08 | 13 | 1200 | 5346 | 8.20 |
| 11 | 32 | zipup | 0.007409 | 2.001e-08 | 13 | 1182 | 80911 | 14.02 |
| 14 | 64 | aci | 0.070777 | 4.480e-09 | 29 | 4752 | 1149586 | 129.83 |
| 14 | 64 | fit | 0.014982 | 1.813e-08 | 16 | 1638 | 143592 | 28.05 |
| 14 | 64 | naive | 0.004957 | 1.813e-08 | 16 | 1638 | 5363 | 13.72 |
| 14 | 64 | zipup | 0.007748 | 1.856e-08 | 16 | 1638 | 81012 | 15.25 |
| 16 | 128 | aci | 0.071915 | 3.247e-09 | 32 | 6808 | 1115945 | 135.96 |
| 16 | 128 | fit | 0.022185 | 1.044e-08 | 18 | 2258 | 143793 | 31.91 |
| 16 | 128 | naive | 0.011175 | 1.044e-08 | 18 | 2258 | 5374 | 23.45 |
| 16 | 128 | zipup | 0.010941 | 1.064e-08 | 18 | 2258 | 81754 | 18.20 |

## Case 2: Gaussian elementwise, global versus patched

| input χ | raw χ | N | R | arm | time (s) | sampled relative L2 | input patches | input max patch χ | output patches | parameters | allocations | allocated (MiB) | speedup |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 32 | 2 | 6 | global_aci | 0.004388 | 1.398e-06 | 2 | 16 | 1 | 6560 | 57157 | 11.46 | 1.000 |
| 16 | 32 | 2 | 6 | global_fit | 0.007957 | 2.293e-06 | 2 | 16 | 1 | 2300 | 57856 | 13.03 | 1.000 |
| 16 | 32 | 2 | 6 | patched_aci | 0.007052 | 2.292e-06 | 2 | 16 | 1 | 2300 | 82738 | 17.63 | 0.622 |
| 16 | 32 | 2 | 6 | patched_fit | 0.013680 | 2.292e-06 | 2 | 16 | 1 | 2300 | 82983 | 24.03 | 0.582 |
| 24 | 50 | 8 | 7 | global_aci | 0.011431 | 2.942e-06 | 2 | 24 | 1 | 23268 | 90766 | 29.69 | 1.000 |
| 24 | 50 | 8 | 7 | global_fit | 0.013576 | 3.462e-06 | 2 | 24 | 1 | 4776 | 65431 | 22.37 | 1.000 |
| 24 | 50 | 8 | 7 | patched_aci | 0.014890 | 3.459e-06 | 2 | 24 | 1 | 4908 | 117462 | 39.44 | 0.768 |
| 24 | 50 | 8 | 7 | patched_fit | 0.033404 | 3.459e-06 | 2 | 24 | 1 | 4908 | 110716 | 55.35 | 0.406 |
| 41 | 64 | 32 | 8 | global_aci | 0.034806 | 3.468e-06 | 2 | 41 | 1 | 98992 | 107105 | 75.53 | 1.000 |
| 41 | 64 | 32 | 8 | global_fit | 0.040618 | 4.661e-06 | 2 | 41 | 1 | 11072 | 99690 | 56.84 | 1.000 |
| 41 | 64 | 32 | 8 | patched_aci | 0.028499 | 4.366e-06 | 2 | 41 | 1 | 11740 | 111749 | 68.33 | 1.221 |
| 41 | 64 | 32 | 8 | patched_fit | 0.106389 | 4.365e-06 | 2 | 41 | 1 | 11740 | 157617 | 149.48 | 0.382 |
| 64 | 99 | 128 | 9 | global_aci | 0.149108 | 2.705e-06 | 2 | 64 | 1 | 317712 | 139715 | 226.03 | 1.000 |
| 64 | 99 | 128 | 9 | global_fit | 0.153660 | 3.993e-06 | 2 | 64 | 1 | 25476 | 110346 | 174.30 | 1.000 |
| 64 | 99 | 128 | 9 | patched_aci | 0.069497 | 3.179e-06 | 2 | 64 | 1 | 28028 | 140133 | 134.67 | 2.146 |
| 64 | 99 | 128 | 9 | patched_fit | 0.407905 | 3.136e-06 | 2 | 64 | 1 | 28028 | 209300 | 346.28 | 0.377 |

## Case 3: Gaussian MPO-MPO contraction, global versus patched

| input χ | raw χ | N | R | arm | time (s) | sampled relative L2 | input patches | input max patch χ | output patches | parameters | allocations | allocated (MiB) | speedup |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 32 | 2 | 6 | global_fit | 0.007252 | 5.943e-07 | 2 | 16 | 1 | 2008 | 37373 | 12.17 | 1.000 |
| 16 | 32 | 2 | 6 | patched_fit | 0.006535 | 5.943e-07 | 2 | 16 | 1 | 2008 | 37831 | 10.73 | 1.110 |
| 24 | 50 | 8 | 7 | global_fit | 0.014745 | 7.318e-07 | 2 | 24 | 1 | 3652 | 42131 | 28.71 | 1.000 |
| 24 | 50 | 8 | 7 | patched_fit | 0.015607 | 7.318e-07 | 2 | 24 | 1 | 3652 | 47310 | 24.44 | 0.945 |
| 41 | 64 | 32 | 8 | global_fit | 0.040755 | 1.343e-06 | 2 | 41 | 1 | 7340 | 47988 | 61.42 | 1.000 |
| 41 | 64 | 32 | 8 | patched_fit | 0.042206 | 1.773e-06 | 2 | 41 | 1 | 7192 | 56270 | 59.30 | 0.966 |
| 64 | 99 | 128 | 9 | global_fit | 0.202155 | 1.525e-06 | 2 | 64 | 1 | 17604 | 56339 | 203.17 | 1.000 |
| 64 | 99 | 128 | 9 | patched_fit | 0.146369 | 2.108e-06 | 2 | 64 | 1 | 16764 | 71594 | 145.10 | 1.381 |
