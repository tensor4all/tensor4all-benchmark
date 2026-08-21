# Bounded larger-N attempt

A padded default-tolerance `N=16,384` patch-only run was attempted with both layouts, one pinned CPU core, single-threaded Rayon/BLAS, and no integrated-output reference count:

```bash
timeout 570s env BENCH_NS=16384 BENCH_PATCH_ONLY=1 \
  BENCH_PATCH_LAYOUTS=balanced_xyz,shared_y_only \
  BENCH_SKIP_REFERENCE_COUNT=1 taskset -c 0 \
  target/release/mpo_mpo_aniso_patched
```

It exited with status 124 during global-TCI input construction, before creating an input cache or patch record. No partial result is reported. Consequently, this bounded single-core study locates no further N-driven patch staircase beyond the unchanged N=12,000 layouts.

After correcting the local-SVD tolerance allocation, separate N=12,000 balanced probes with input compression tolerances `1e-7` (χ above 282) and `1e-8` (χ423) also exited with status 124 before producing a record. The stricter patch cutoff makes these higher-rank constructions exceed the same 570-second bound. N=8,192 χ418 completed for both layouts and is retained as the bounded high-χ comparison.
