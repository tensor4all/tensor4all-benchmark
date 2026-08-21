# Bounded χ418 attempt

The contraction-free probe `patch-scaling-chi418.json` confirms that the padded `N=8192` raw cache compresses to left/right χ `417/418` at input relative-L2 tolerance `3e-10`.

A global-only profiled contraction was then run with the same input settings, one pinned core, single-threaded Rayon/BLAS, and the profile's hard command limit:

```bash
timeout 570s env T4A_PROFILE_FIT=1 BENCH_NS=8192 \
  BENCH_INPUT_L2_RTOL=3e-10 BENCH_ARM=global BENCH_RUNS=1 BENCH_WARMUPS=0 \
  taskset -c 0 target/release/mpo_mpo_aniso_patched
```

It exited with status 124 before completing the first `contract_fit` profile block, so it produced no valid timing record. The completed χ381 point uses `1e-9`; its global and patched arms each finish under 570 seconds and are the current bounded near-χ418 measurement. A timeout is not reported as a benchmark timing.
