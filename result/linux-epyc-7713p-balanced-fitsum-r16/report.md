# Tensor4all benchmark results

Profile: `linux-epyc-7713p-balanced-fitsum-r16`. CPU: AMD EPYC 7713P 64-Core Processor. Threads: 1. CPU affinity: 0. Source revision: `cfff5ff4e693ce44922115bcb3367a38dc989efc`. tensor4all-rs revision: `6926379a06689a206aed57f01857e905eb310366`.

The source revision identifies the clean code that was measured. The commit adding these generated records necessarily follows that revision.

Timed repetitions per arm: 1.

All timings exclude input construction, cache I/O, format conversion, patch preparation, output conversion and accuracy evaluation. Gaussian inputs use whole-mixture global TCI at fixed `R = 16`; balanced input patches use cap 128, while fit-sum output patches have no hard bond cap before the final adaptive truncation.

## Case 3: Gaussian MPO-MPO contraction, global versus patched

| input χ | raw χ | N | R | arm | time (s) | sampled relative L2 | input patches | input max patch χ | output patches | parameters | speedup |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 256 | 299 | 4096 | 16 | global_fit | 54.511854 | 2.116e-06 | 16 | 116 | 1 | 191816 | 1.000 |
| 256 | 299 | 4096 | 16 | patched_fit | 29.008589 | 3.143e-06 | 16 | 116 | 4 | 203764 | 1.879 |
| 418 | 898 | 20480 | 16 | global_fit | 432.907678 | 2.329e-06 | 64 | 128 | 1 | 586664 | 1.000 |
| 418 | 898 | 20480 | 16 | patched_fit | 281.049274 | 2.356e-05 | 64 | 128 | 16 | 898468 | 1.540 |

The balanced patched path prepartitions x, y, and z; each existing x/z output patch uses a cap-bounded initial sum followed by variational fit_sum over its shared-y contributions, with no recursive output splitting.
