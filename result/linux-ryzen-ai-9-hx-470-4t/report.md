# Tensor4all benchmark results

Profile: `linux-ryzen-ai-9-hx-470-4t`. CPU: AMD Ryzen AI 9 HX 470 w/ Radeon 890M. Threads: 4. CPU affinity: 0-3. Source revision: `05ebd78bdc40e09f1534a4bab2ac72131ce271d3`. tensor4all-rs revision: `a5157385f58048355a28a6a003d4cb63d3b1349b`.

The source revision identifies the clean code that was measured. The commit adding these generated records necessarily follows that revision.

All timings exclude input construction, cache I/O, format conversion, patch preparation, output conversion and accuracy evaluation. Gaussian inputs use independent two-dimensional interpolative decompositions, balanced pairwise addition, final relative-L2/SVD tolerance `1e-6`, and fixed patch cap 128.

## Case 1: Fourier elementwise

| input χ | K | arm | time (s) | sampled relative L2 | output χ | parameters | allocations | allocated (MiB) |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 5 | 4 | aci | 0.056474 | 1.166e-08 | 9 | 1026 | 935094 | 92.31 |
| 5 | 4 | fit | 0.025642 | 1.213e-08 | 8 | 606 | 142954 | 23.65 |
| 5 | 4 | naive | 0.001582 | 1.211e-08 | 8 | 606 | 5467 | 2.61 |
| 5 | 4 | zipup | 0.014577 | 1.197e-08 | 8 | 606 | 80520 | 12.11 |
| 8 | 8 | aci | 0.077047 | 8.181e-09 | 16 | 1702 | 1293260 | 129.49 |
| 8 | 8 | fit | 0.018698 | 1.066e-08 | 8 | 726 | 143245 | 24.26 |
| 8 | 8 | naive | 0.001533 | 1.066e-08 | 8 | 726 | 5599 | 3.85 |
| 8 | 8 | zipup | 0.009921 | 1.080e-08 | 8 | 726 | 80806 | 12.58 |
| 8 | 16 | aci | 0.066019 | 6.086e-09 | 16 | 2206 | 1095666 | 111.28 |
| 8 | 16 | fit | 0.019670 | 1.620e-08 | 10 | 900 | 143319 | 24.94 |
| 8 | 16 | naive | 0.002000 | 1.620e-08 | 10 | 900 | 5644 | 5.13 |
| 8 | 16 | zipup | 0.010695 | 1.636e-08 | 10 | 900 | 80869 | 13.09 |
| 11 | 32 | aci | 0.077827 | 4.931e-09 | 23 | 3318 | 1253380 | 136.21 |
| 11 | 32 | fit | 0.019988 | 1.806e-08 | 13 | 1200 | 144030 | 26.34 |
| 11 | 32 | naive | 0.003920 | 1.806e-08 | 13 | 1200 | 6139 | 8.22 |
| 11 | 32 | zipup | 0.011348 | 2.001e-08 | 13 | 1182 | 81544 | 14.05 |
| 14 | 64 | aci | 0.074522 | 4.480e-09 | 29 | 4752 | 1149587 | 129.84 |
| 14 | 64 | fit | 0.021687 | 1.813e-08 | 16 | 1638 | 143911 | 28.08 |
| 14 | 64 | naive | 0.005417 | 1.813e-08 | 16 | 1638 | 6685 | 13.75 |
| 14 | 64 | zipup | 0.011935 | 1.856e-08 | 16 | 1638 | 81873 | 15.28 |
| 16 | 128 | aci | 0.075587 | 3.247e-09 | 32 | 6808 | 1115933 | 135.92 |
| 16 | 128 | fit | 0.023958 | 1.044e-08 | 18 | 2258 | 145567 | 31.98 |
| 16 | 128 | naive | 0.009562 | 1.044e-08 | 18 | 2258 | 7403 | 23.51 |
| 16 | 128 | zipup | 0.016043 | 1.064e-08 | 18 | 2258 | 83392 | 18.25 |

## Case 2: Gaussian elementwise, global versus patched

| input χ | raw χ | N | R | arm | time (s) | sampled relative L2 | input patches | input max patch χ | output patches | parameters | allocations | allocated (MiB) | speedup |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 32 | 2 | 6 | global_aci | 0.004899 | 1.398e-06 | 2 | 16 | 1 | 6560 | 57207 | 11.53 | 1.000 |
| 16 | 32 | 2 | 6 | global_fit | 0.010786 | 2.293e-06 | 2 | 16 | 1 | 2300 | 60186 | 114.89 | 1.000 |
| 16 | 32 | 2 | 6 | patched_aci | 0.008488 | 2.292e-06 | 2 | 16 | 1 | 2300 | 81260 | 17.26 | 0.577 |
| 16 | 32 | 2 | 6 | patched_fit | 0.017352 | 2.292e-06 | 2 | 16 | 1 | 2300 | 84856 | 26.06 | 0.622 |
| 24 | 50 | 8 | 7 | global_aci | 0.012319 | 2.942e-06 | 2 | 24 | 1 | 23268 | 87494 | 29.34 | 1.000 |
| 24 | 50 | 8 | 7 | global_fit | 0.017572 | 3.462e-06 | 2 | 24 | 1 | 4776 | 70531 | 26.53 | 1.000 |
| 24 | 50 | 8 | 7 | patched_aci | 0.016877 | 3.459e-06 | 2 | 24 | 1 | 4908 | 116482 | 38.87 | 0.730 |
| 24 | 50 | 8 | 7 | patched_fit | 0.036035 | 3.459e-06 | 2 | 24 | 1 | 4908 | 123259 | 64.21 | 0.488 |
| 41 | 64 | 32 | 8 | global_aci | 0.038988 | 3.468e-06 | 2 | 41 | 1 | 98992 | 110020 | 78.88 | 1.000 |
| 41 | 64 | 32 | 8 | global_fit | 0.042459 | 4.661e-06 | 2 | 41 | 1 | 11072 | 110564 | 73.16 | 1.000 |
| 41 | 64 | 32 | 8 | patched_aci | 0.032272 | 4.366e-06 | 2 | 41 | 1 | 11740 | 117649 | 68.50 | 1.208 |
| 41 | 64 | 32 | 8 | patched_fit | 0.085208 | 4.365e-06 | 2 | 41 | 1 | 11740 | 183635 | 151.13 | 0.498 |
| 64 | 99 | 128 | 9 | global_aci | 0.148520 | 2.705e-06 | 2 | 64 | 1 | 317712 | 141977 | 516.23 | 1.000 |
| 64 | 99 | 128 | 9 | global_fit | 0.098042 | 3.993e-06 | 2 | 64 | 1 | 25476 | 140014 | 175.07 | 1.000 |
| 64 | 99 | 128 | 9 | patched_aci | 0.073028 | 3.187e-06 | 2 | 64 | 1 | 28028 | 150487 | 134.98 | 2.034 |
| 64 | 99 | 128 | 9 | patched_fit | 0.248354 | 3.136e-06 | 2 | 64 | 1 | 28028 | 266189 | 352.90 | 0.395 |

## Case 3: Gaussian MPO-MPO contraction, global versus patched

| input χ | raw χ | N | R | arm | time (s) | sampled relative L2 | input patches | input max patch χ | output patches | parameters | allocations | allocated (MiB) | speedup |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 32 | 2 | 6 | global_fit | 0.008389 | 5.943e-07 | 2 | 16 | 1 | 2008 | 38651 | 111.35 | 1.000 |
| 16 | 32 | 2 | 6 | patched_fit | 0.010586 | 5.943e-07 | 2 | 16 | 1 | 2008 | 39559 | 10.78 | 0.793 |
| 24 | 50 | 8 | 7 | global_fit | 0.015479 | 7.318e-07 | 2 | 24 | 1 | 3652 | 45506 | 29.82 | 1.000 |
| 24 | 50 | 8 | 7 | patched_fit | 0.018397 | 7.318e-07 | 2 | 24 | 1 | 3652 | 51035 | 24.54 | 0.841 |
| 41 | 64 | 32 | 8 | global_fit | 0.059542 | 1.343e-06 | 2 | 41 | 1 | 7340 | 60258 | 61.71 | 1.000 |
| 41 | 64 | 32 | 8 | patched_fit | 0.054893 | 1.773e-06 | 2 | 41 | 1 | 7192 | 71251 | 59.75 | 1.085 |
| 64 | 99 | 128 | 9 | global_fit | 0.119545 | 1.525e-06 | 2 | 64 | 1 | 17604 | 77056 | 215.85 | 1.000 |
| 64 | 99 | 128 | 9 | patched_fit | 0.093247 | 2.108e-06 | 2 | 64 | 1 | 16764 | 96355 | 150.85 | 1.282 |
