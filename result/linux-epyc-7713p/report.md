# Tensor4all benchmark results

> **Archived legacy profile.** These records use the earlier per-Gaussian interpolative input family and variable `R`; they are not current factor-4 padded, fixed-`R=16` results.

Profile: `linux-epyc-7713p`. CPU: AMD EPYC 7713P 64-Core Processor. Threads: 1. CPU affinity: 0. Source revision: `4ed4ba6597639797b9f622cb7c5ab429e7b92647`. tensor4all-rs revision: `9e9aedaebe0d3918b34dd399ff0981e337f3835b`.

The source revision identifies the clean code that was measured. The commit adding these generated records necessarily follows that revision.

All timings exclude input construction, cache I/O, format conversion, patch preparation, output conversion and accuracy evaluation. Gaussian inputs use independent two-dimensional interpolative decompositions, balanced pairwise addition, final relative-L2/SVD tolerance `1e-6`, and fixed patch cap 128.

## Case 1: Fourier elementwise

| input χ | K | arm | time (s) | sampled relative L2 | output χ | parameters |
|---:|---:|---|---:|---:|---:|---:|
| 5 | 4 | aci | 0.073347 | 1.163e-08 | 9 | 1026 |
| 5 | 4 | fit | 0.078786 | 1.213e-08 | 8 | 606 |
| 5 | 4 | naive | 0.001891 | 1.211e-08 | 8 | 606 |
| 5 | 4 | zipup | 0.036320 | 1.197e-08 | 8 | 606 |
| 8 | 8 | aci | 0.077033 | 8.173e-09 | 15 | 1628 |
| 8 | 8 | fit | 0.080764 | 1.066e-08 | 8 | 726 |
| 8 | 8 | naive | 0.002576 | 1.066e-08 | 8 | 726 |
| 8 | 8 | zipup | 0.037141 | 1.080e-08 | 8 | 726 |
| 8 | 16 | aci | 0.072607 | 6.089e-09 | 16 | 2206 |
| 8 | 16 | fit | 0.085960 | 1.620e-08 | 10 | 900 |
| 8 | 16 | naive | 0.003332 | 1.620e-08 | 10 | 900 |
| 8 | 16 | zipup | 0.038438 | 1.636e-08 | 10 | 900 |
| 11 | 32 | aci | 0.080893 | 4.939e-09 | 22 | 3324 |
| 11 | 32 | fit | 0.087356 | 1.806e-08 | 13 | 1200 |
| 11 | 32 | naive | 0.006156 | 1.806e-08 | 13 | 1200 |
| 11 | 32 | zipup | 0.038147 | 2.001e-08 | 13 | 1182 |
| 14 | 64 | aci | 0.080989 | 4.480e-09 | 29 | 4752 |
| 14 | 64 | fit | 0.096748 | 1.813e-08 | 16 | 1638 |
| 14 | 64 | naive | 0.009583 | 1.813e-08 | 16 | 1638 |
| 14 | 64 | zipup | 0.044973 | 1.856e-08 | 16 | 1638 |
| 16 | 128 | aci | 0.089005 | 3.255e-09 | 32 | 6728 |
| 16 | 128 | fit | 0.094404 | 1.044e-08 | 18 | 2258 |
| 16 | 128 | naive | 0.016861 | 1.044e-08 | 18 | 2258 |
| 16 | 128 | zipup | 0.044607 | 1.064e-08 | 18 | 2258 |

## Case 2: Gaussian elementwise, global versus patched

| input χ | raw χ | N | R | arm | time (s) | sampled relative L2 | input patches | input max patch χ | output patches | parameters | speedup |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 16 | 32 | 2 | 6 | global_aci | 0.008655 | 1.398e-06 | 2 | 16 | 1 | 6560 | 1.000 |
| 16 | 32 | 2 | 6 | global_fit | 0.028473 | 2.293e-06 | 2 | 16 | 1 | 2300 | 1.000 |
| 16 | 32 | 2 | 6 | patched_aci | 0.021844 | 2.292e-06 | 2 | 16 | 1 | 2300 | 0.396 |
| 16 | 32 | 2 | 6 | patched_fit | 0.044864 | 2.292e-06 | 2 | 16 | 1 | 2300 | 0.635 |
| 24 | 50 | 8 | 7 | global_aci | 0.019403 | 2.942e-06 | 2 | 24 | 1 | 23268 | 1.000 |
| 24 | 50 | 8 | 7 | global_fit | 0.042136 | 3.462e-06 | 2 | 24 | 1 | 4776 | 1.000 |
| 24 | 50 | 8 | 7 | patched_aci | 0.270262 | 3.459e-06 | 2 | 24 | 1 | 4908 | 0.072 |
| 24 | 50 | 8 | 7 | patched_fit | 0.112770 | 3.459e-06 | 2 | 24 | 1 | 4908 | 0.374 |
| 41 | 64 | 32 | 8 | global_aci | 0.065374 | 3.468e-06 | 2 | 41 | 1 | 98992 | 1.000 |
| 41 | 64 | 32 | 8 | global_fit | 0.088816 | 4.661e-06 | 2 | 41 | 1 | 11072 | 1.000 |
| 41 | 64 | 32 | 8 | patched_aci | 2.430336 | 4.366e-06 | 2 | 41 | 1 | 11740 | 0.027 |
| 41 | 64 | 32 | 8 | patched_fit | 0.593427 | 4.365e-06 | 2 | 41 | 1 | 11740 | 0.150 |
| 64 | 99 | 128 | 9 | global_aci | 0.212690 | 2.705e-06 | 2 | 64 | 1 | 317712 | 1.000 |
| 64 | 99 | 128 | 9 | global_fit | 0.329209 | 3.993e-06 | 2 | 64 | 1 | 25476 | 1.000 |
| 64 | 99 | 128 | 9 | patched_aci | 8.942323 | 3.168e-06 | 2 | 64 | 1 | 28028 | 0.024 |
| 64 | 99 | 128 | 9 | patched_fit | 4.083926 | 3.136e-06 | 2 | 64 | 1 | 28028 | 0.081 |

## Case 3: Gaussian MPO-MPO contraction, global versus patched

| input χ | raw χ | N | R | arm | time (s) | sampled relative L2 | input patches | input max patch χ | output patches | parameters | speedup |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 16 | 32 | 2 | 6 | global_fit | 0.021572 | 5.943e-07 | 2 | 16 | 1 | 2008 | 1.000 |
| 16 | 32 | 2 | 6 | patched_fit | 0.022959 | 5.943e-07 | 2 | 16 | 1 | 2008 | 0.940 |
| 24 | 50 | 8 | 7 | global_fit | 0.046198 | 7.318e-07 | 2 | 24 | 1 | 3652 | 1.000 |
| 24 | 50 | 8 | 7 | patched_fit | 0.048666 | 7.318e-07 | 2 | 24 | 1 | 3652 | 0.949 |
| 41 | 64 | 32 | 8 | global_fit | 0.093784 | 1.343e-06 | 2 | 41 | 1 | 7340 | 1.000 |
| 41 | 64 | 32 | 8 | patched_fit | 0.098161 | 1.773e-06 | 2 | 41 | 1 | 7192 | 0.955 |
| 64 | 99 | 128 | 9 | global_fit | 0.435670 | 1.525e-06 | 2 | 64 | 1 | 17604 | 1.000 |
| 64 | 99 | 128 | 9 | patched_fit | 0.299308 | 2.108e-06 | 2 | 64 | 1 | 16764 | 1.456 |
