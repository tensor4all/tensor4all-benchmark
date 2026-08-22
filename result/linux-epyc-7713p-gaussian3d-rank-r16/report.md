# Fully correlated 3D Gaussian input rank scaling

This input-only Case 3 probe constructs `A(b,x,y)` and embeds it as the batch-diagonal MPO `A(b,x;b',y) = delta(b,b') A(b,x,y)`. It performs no patching and no contraction.

| N | raw TCI χ | compressed χ | diagonal MPO χ | raw parameters | compressed parameters | diagonal MPO parameters | build (s) | compression (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 158 | 56 | 56 | 288,760 | 56,944 | 113,888 | 26.075 | 0.037 |
| 2 | 296 | 103 | 103 | 727,824 | 125,096 | 250,192 | 85.633 | 0.120 |
| 4 | 375 | 144 | 144 | 1,371,216 | 209,776 | 419,552 | 121.394 | 0.253 |
| 8 | 364 | 169 | 169 | 1,788,960 | 286,432 | 572,864 | 225.775 | 0.320 |
| 16 | 423 | 187 | 187 | 2,568,744 | 396,120 | 792,240 | 242.508 | 0.477 |
| 32 | 522 | 217 | 217 | 3,649,320 | 555,392 | 1,110,784 | 416.136 | 0.789 |

The local batch-diagonal embedding preserves every QTT bond dimension exactly. At N=32 the compressed input reaches χ217 while the raw Global TCI reaches χ522. The bounded N=64 command timed out after 570 seconds during input construction, before producing a record.
