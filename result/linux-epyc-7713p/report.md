# Tensor4all benchmark results

Profile: `linux-epyc-7713p`. CPU: AMD EPYC 7713P 64-Core Processor. Threads: 1. CPU affinity: 0. Source revision: `63a6345444662f7b85b919bdf525a37ec1d82d86`. tensor4all-rs revision: `9e9aedaebe0d3918b34dd399ff0981e337f3835b`.

All timings exclude input construction, cache I/O, format conversion, patch preparation, output conversion and accuracy evaluation. Gaussian inputs use independent two-dimensional interpolative decompositions, balanced pairwise addition, final relative-L2/SVD tolerance `1e-6`, and fixed patch cap 128.

## Case 1: Fourier elementwise

| input χ | K | arm | time (s) | sampled relative L2 | output χ | parameters |
|---:|---:|---|---:|---:|---:|---:|
| 5 | 4 | aci | 0.079583 | 1.163e-08 | 9 | 1026 |
| 5 | 4 | fit | 0.093877 | 1.213e-08 | 8 | 606 |
| 5 | 4 | naive | 0.002272 | 1.211e-08 | 8 | 606 |
| 5 | 4 | zipup | 0.043693 | 1.197e-08 | 8 | 606 |
| 8 | 8 | aci | 0.091403 | 8.173e-09 | 15 | 1628 |
| 8 | 8 | fit | 0.096802 | 1.066e-08 | 8 | 726 |
| 8 | 8 | naive | 0.003046 | 1.066e-08 | 8 | 726 |
| 8 | 8 | zipup | 0.044871 | 1.080e-08 | 8 | 726 |
| 8 | 16 | aci | 0.085566 | 6.089e-09 | 16 | 2206 |
| 8 | 16 | fit | 0.098068 | 1.620e-08 | 10 | 900 |
| 8 | 16 | naive | 0.003682 | 1.620e-08 | 10 | 900 |
| 8 | 16 | zipup | 0.045233 | 1.636e-08 | 10 | 900 |
| 11 | 32 | aci | 0.081603 | 4.939e-09 | 22 | 3324 |
| 11 | 32 | fit | 0.094314 | 1.806e-08 | 13 | 1200 |
| 11 | 32 | naive | 0.006426 | 1.806e-08 | 13 | 1200 |
| 11 | 32 | zipup | 0.046418 | 2.001e-08 | 13 | 1182 |
| 14 | 64 | aci | 0.084678 | 4.480e-09 | 29 | 4752 |
| 14 | 64 | fit | 0.099391 | 1.813e-08 | 16 | 1638 |
| 14 | 64 | naive | 0.009786 | 1.813e-08 | 16 | 1638 |
| 14 | 64 | zipup | 0.044601 | 1.856e-08 | 16 | 1638 |
| 16 | 128 | aci | 0.093596 | 3.255e-09 | 32 | 6728 |
| 16 | 128 | fit | 0.105228 | 1.044e-08 | 18 | 2258 |
| 16 | 128 | naive | 0.019192 | 1.044e-08 | 18 | 2258 |
| 16 | 128 | zipup | 0.050193 | 1.064e-08 | 18 | 2258 |

## Case 2: Gaussian elementwise, global versus patched

| input χ | raw χ | N | R | arm | time (s) | sampled relative L2 | input patches | input max patch χ | output patches | parameters | speedup |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 16 | 32 | 2 | 6 | global_aci | 0.010260 | 1.398e-06 | 2 | 16 | 1 | 6560 | 1.000 |
| 16 | 32 | 2 | 6 | global_fit | 0.033338 | 2.293e-06 | 2 | 16 | 1 | 2300 | 1.000 |
| 16 | 32 | 2 | 6 | patched_aci | 0.023649 | 2.292e-06 | 2 | 16 | 1 | 2300 | 0.434 |
| 16 | 32 | 2 | 6 | patched_fit | 0.053148 | 2.292e-06 | 2 | 16 | 1 | 2300 | 0.627 |
| 24 | 50 | 8 | 7 | global_aci | 0.021533 | 2.942e-06 | 2 | 24 | 1 | 23268 | 1.000 |
| 24 | 50 | 8 | 7 | global_fit | 0.048685 | 3.462e-06 | 2 | 24 | 1 | 4776 | 1.000 |
| 24 | 50 | 8 | 7 | patched_aci | 0.274936 | 3.459e-06 | 2 | 24 | 1 | 4908 | 0.078 |
| 24 | 50 | 8 | 7 | patched_fit | 0.122505 | 3.459e-06 | 2 | 24 | 1 | 4908 | 0.397 |
| 41 | 64 | 32 | 8 | global_aci | 0.058162 | 3.468e-06 | 2 | 41 | 1 | 98992 | 1.000 |
| 41 | 64 | 32 | 8 | global_fit | 0.092868 | 4.661e-06 | 2 | 41 | 1 | 11072 | 1.000 |
| 41 | 64 | 32 | 8 | patched_aci | 2.442332 | 4.366e-06 | 2 | 41 | 1 | 11740 | 0.024 |
| 41 | 64 | 32 | 8 | patched_fit | 0.611421 | 4.365e-06 | 2 | 41 | 1 | 11740 | 0.152 |
| 64 | 99 | 128 | 9 | global_aci | 0.214291 | 2.705e-06 | 2 | 64 | 1 | 317712 | 1.000 |
| 64 | 99 | 128 | 9 | global_fit | 0.310755 | 3.993e-06 | 2 | 64 | 1 | 25476 | 1.000 |
| 64 | 99 | 128 | 9 | patched_aci | 8.684410 | 3.168e-06 | 2 | 64 | 1 | 28028 | 0.025 |
| 64 | 99 | 128 | 9 | patched_fit | 4.077012 | 3.136e-06 | 2 | 64 | 1 | 28028 | 0.076 |

## Case 3: Gaussian MPO-MPO contraction, global versus patched

| input χ | raw χ | N | R | arm | time (s) | sampled relative L2 | input patches | input max patch χ | output patches | parameters | speedup |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 16 | 32 | 2 | 6 | global_fit | 0.019848 | 5.943e-07 | 2 | 16 | 1 | 2008 | 1.000 |
| 16 | 32 | 2 | 6 | patched_fit | 0.022947 | 5.943e-07 | 2 | 16 | 1 | 2008 | 0.865 |
| 24 | 50 | 8 | 7 | global_fit | 0.041170 | 7.318e-07 | 2 | 24 | 1 | 3652 | 1.000 |
| 24 | 50 | 8 | 7 | patched_fit | 0.043008 | 7.318e-07 | 2 | 24 | 1 | 3652 | 0.957 |
| 41 | 64 | 32 | 8 | global_fit | 0.098708 | 1.343e-06 | 2 | 41 | 1 | 7340 | 1.000 |
| 41 | 64 | 32 | 8 | patched_fit | 0.096481 | 1.773e-06 | 2 | 41 | 1 | 7192 | 1.023 |
| 64 | 99 | 128 | 9 | global_fit | 0.445703 | 1.525e-06 | 2 | 64 | 1 | 17604 | 1.000 |
| 64 | 99 | 128 | 9 | patched_fit | 0.282489 | 2.108e-06 | 2 | 64 | 1 | 16764 | 1.578 |
