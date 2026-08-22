# Fully correlated 3D Gaussian input rank scaling

This input-only Case 3 probe constructs `A(b,x,y)` and embeds it as the batch-diagonal MPO `A(b,x;b',y) = delta(b,b') A(b,x,y)`. It performs no patching and no contraction.

| N | raw TCI χ | compressed χ | diagonal MPO χ | raw parameters | compressed parameters | diagonal MPO parameters | principal-axis error | build (s) | compression (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 158 | 67 | 67 | 288,760 | 73,032 | 146,064 | 3.213e-07 | 24.241 | 0.034 |
| 2 | 296 | 125 | 125 | 727,824 | 170,488 | 340,976 | 6.516e-07 | 81.492 | 0.118 |
| 4 | 375 | 176 | 176 | 1,371,216 | 290,704 | 581,408 | 7.302e-07 | 114.765 | 0.237 |
| 8 | 364 | 201 | 201 | 1,788,960 | 397,184 | 794,368 | 8.016e-07 | 214.319 | 0.323 |
| 16 | 423 | 223 | 223 | 2,568,744 | 561,160 | 1,122,320 | 8.115e-07 | 225.672 | 0.487 |
| 32 | 522 | 251 | 251 | 3,649,320 | 784,744 | 1,569,488 | 6.922e-07 | 388.667 | 0.767 |

The local batch-diagonal embedding preserves every QTT bond dimension exactly. At N=32 the compressed input reaches χ251 while the raw Global TCI reaches χ522. Every recorded off-pivot principal-axis error is below the requested 1e-6 input target. The bounded N=64 command timed out after 570 seconds during input construction, before producing a record.
