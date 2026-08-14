# mpo_mpo_quantics

| algorithm | points | fitted time exponent | worst max relative error |
|---|---|---|---|
| fit_treetn | 5 | 8.36 | 4.17e-09 |
| naive | 5 | 19.07 | 4.17e-09 |
| zipup_simplett | 5 | 7.76 | 1.05e-04 |
| zipup_treetn | 5 | 7.14 | 1.05e-04 |

Note: every algorithm contracts at the same output budget, its maximum bond dimension capped at the input rank chi, and the cap is the only truncation control: the contraction tolerance is pinned inert at contract_tol, so every arm spends the whole budget unless its exact rank is smaller and the error column is the discriminator. naive and zipup_simplett run on the simplett engine, zipup_treetn and fit_treetn on treetn; both engines truncate relative to the largest singular value at the pinned revision. The two zipup arms are the same algorithm on the two engines, so their difference isolates the engine, and it is now confined to wall time. The fitted time exponent is measured against input chi along a sweep of r, where the site count also grows, so it is not a pure chi power law.

![time](./mpo_mpo_quantics-time.svg)

![error](./mpo_mpo_quantics-error.svg)
