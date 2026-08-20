# Tensor4all benchmark results

Profile: `linux-epyc-7713p-global-tci-r16`. CPU: AMD EPYC 7713P 64-Core Processor. Threads: 1. CPU affinity: 0. Source revision: `a82e2198c7e6db08b0843f8fb026fef953b6c9f6`. tensor4all-rs revision: `9e9aedaebe0d3918b34dd399ff0981e337f3835b`.

The source revision identifies the clean code that was measured. The commit adding these generated records necessarily follows that revision.

Timed repetitions per arm: 1.

All timings exclude input construction, cache I/O, format conversion, patch preparation, output conversion and accuracy evaluation. Gaussian inputs use whole-mixture global TCI at fixed `R = 16`, final relative-L2/SVD tolerance `1e-6`, and fixed patch cap 128.

## Case 3: Gaussian MPO-MPO contraction, global versus patched

| input χ | raw χ | N | R | arm | time (s) | sampled relative L2 | input patches | input max patch χ | output patches | parameters | speedup |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 84 | 193 | 512 | 16 | global_fit | 1.962863 | 2.102e-06 | 2 | 84 | 1 | 37484 | 1.000 |
| 84 | 193 | 512 | 16 | patched_fit | 0.871944 | 2.831e-06 | 2 | 84 | 1 | 35820 | 2.251 |
| 256 | 299 | 4096 | 16 | global_fit | 50.067640 | 2.116e-06 | 16 | 116 | 1 | 191816 | 1.000 |
| 256 | 299 | 4096 | 16 | patched_fit | 73.926512 | 3.914e-06 | 16 | 116 | 4 | 192428 | 0.677 |
| 418 | 898 | 20480 | 16 | global_fit | 402.559843 | 2.329e-06 | 60 | 121 | 1 | 586664 | 1.000 |
| 418 | 898 | 20480 | 16 | patched_fit | 1771.027045 | 1.254e-05 | 60 | 121 | 30 | 1067216 | 0.227 |
