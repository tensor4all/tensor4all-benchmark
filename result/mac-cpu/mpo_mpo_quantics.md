# mpo_mpo_quantics

| algorithm | points | fitted time exponent | worst max relative error |
|---|---|---|---|
| fit_treetn | 3 | 5.16 | 2.88e-08 |
| naive | 3 | 10.54 | 1.05e-04 |
| zipup_simplett | 3 | 10.47 | 1.05e-04 |
| zipup_treetn | 3 | 5.72 | 1.05e-04 |

Note: every algorithm contracts at the same output budget, its maximum bond dimension capped at the input rank chi, so the error column is the discriminator. naive and zipup_simplett use the simplett engine with an absolute singular value cutoff; zipup_treetn and fit_treetn use the treetn engine with a relative cutoff, so which directions they keep inside that budget still differs by engine. The two zipup arms are the same algorithm on the two engines, so their difference isolates the engine. The fitted time exponent is measured against input chi along a sweep of r, where the site count also grows, so it is not a pure chi power law.

![time](./mpo_mpo_quantics-time.svg)

![error](./mpo_mpo_quantics-error.svg)
