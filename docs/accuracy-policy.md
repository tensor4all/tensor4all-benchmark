# Accuracy policy

This repository distinguishes approximation metrics instead of treating every numeric tolerance as interchangeable.

## Fit on disjoint patches

Fit uses a relative L2 SVD policy:

```rust
SvdTruncationPolicy::new(rtol * rtol)
    .with_relative()
    .with_squared_values()
    .with_discarded_tail_sum()
```

The same `rtol` applies independently to every disjoint output patch. If the exact patch outputs are `h_p` and their approximations are `h_p_tilde`, then

```text
sum_p ||h_p - h_p_tilde||^2
    <= rtol^2 sum_p ||h_p||^2
    =  rtol^2 ||h||^2.
```

The number of patches therefore does not require an `rtol / sqrt(n_patches)` adjustment.

Fit spends this error budget once. A fit result is not followed by another lossy truncation with the same `rtol`, because two independent approximation stages would not provide one `rtol` bound. Patch assembly may validate projectors and metadata, but it must not consume another numerical error budget.

A rank cap is a partitioning threshold, not permission to silently violate the tolerance. If a patch reaches the cap before satisfying `rtol`, the domain must be split and the children recomputed.

## ACI on patches

ACI's stopping value is an interpolation residual, not a relative L2 tolerance. It is named and recorded separately as `aci_residual_tolerance`.

ACI output may receive a separate relative-L2 SVD compression. Records keep both metrics:

- the ACI residual tolerance used to construct the product;
- the achieved deterministic sampled relative-L2 error against the common reference.

No claim equates these two values. The sampled error is the comparison metric across fit and ACI arms.

## Input and operation errors

Whole-mixture global TCI and final input compression happen before the timed operation. The TCI residual, localized-evaluator absolute tail bound, final relative-L2 SVD tolerance, random holdout error, and principal-axis holdout error are recorded separately. The per-Gaussian two-dimensional interpolative builder remains a focused reference test and is not the production generator.

Operation tolerances describe only the elementwise product or contraction. End-to-end sampled error includes both input and operation approximation.

## Timing boundary

Timed regions include only the requested elementwise product or contraction. Input generation, cache I/O, input compression, format conversion, patch preparation, output conversion, and accuracy evaluation remain outside timing. Any numerical postprocessing required to produce the benchmarked output remains inside timing.
