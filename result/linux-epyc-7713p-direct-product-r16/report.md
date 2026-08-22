# Doubled-space direct-product input rank scaling

This input-only Case 3 probe forms `F(x,x';y,y') = f(x,y) tensor_product f(x',y')` and the analogous `G`. It performs no patching and no contraction.

| N | factor rtol | factor χ L/R | product χ L/R | max χ² identity | product memory (MiB) | product build (s) | max sampled factor error |
|---:|---:|---:|---:|:---:|---:|---:|---:|
| 1 | 1e-06 | 30/25 | 900/625 | yes | 137.8 | 0.408 | 3.477e-06 |
| 1 | 1e-05 | 23/20 | 529/400 | yes | 63.2 | 0.193 | 2.245e-05 |
| 1 | 3e-05 | 21/17 | 441/289 | yes | 41.0 | 0.125 | 6.011e-05 |
| 2 | 1e-06 | 32/30 | 1024/900 | yes | 245.4 | 0.744 | 2.850e-06 |
| 2 | 1e-05 | 27/24 | 729/576 | yes | 108.4 | 0.335 | 2.971e-05 |
| 2 | 3e-05 | 24/21 | 576/441 | yes | 64.7 | 0.191 | 8.342e-05 |
| 4 | 1e-06 | 35/46 | 1225/2116 | yes | 596.9 | 1.768 | 3.514e-06 |
| 4 | 1e-05 | 31/37 | 961/1369 | yes | 247.1 | 0.735 | 3.261e-05 |
| 4 | 3e-05 | 27/32 | 729/1024 | yes | 150.2 | 0.444 | 9.458e-05 |
| 8 | 1e-06 | 40/48 | 1600/2304 | yes | 1104.5 | 3.284 | 3.997e-06 |
| 8 | 1e-05 | 32/38 | 1024/1444 | yes | 408.7 | 1.201 | 4.525e-05 |

Every exact direct product satisfies χ_product = χ_factor² bond by bond. The largest bounded point is N=8, factor rtol=1e-06, with product χ=2304 and 1104.5 MiB for both product MPOs. These are materialized input ranks, not contraction timings.
