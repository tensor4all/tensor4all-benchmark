# tensor4all-benchmark

Reproducible benchmarks for tensor network algorithms in [tensor4all-rs](https://github.com/tensor4all/tensor4all-rs). Inputs use fixed seeds, timed regions exclude preparation, every arm writes JSON, and reports are generated from those records.

## Maintained cases

| Case | Input and operation | Arms | Scaling axis |
| --- | --- | --- | --- |
| `elementwise_fourier` | Elementwise product of two random one-dimensional Fourier QTTs | naive, zip-up, fit, ACI | measured input χ |
| `gaussian_elementwise` | Elementwise product of randomly rotated anisotropic Gaussian QTTs | global fit, patched fit, global ACI, patched ACI | compressed input χ |
| `gaussian_mpo_contraction` | MPO-MPO contraction of the same Gaussian input family | global fit, patched fit | compressed input χ |

There is no independent `R` sweep. `R` is the smallest value that gives at least eight grid cells across the fixed minor-axis standard deviation. Gaussian count `N` and `R` are construction metadata, not analysis axes.

## Gaussian input

Every Gaussian has an independent positive weight, center, log-uniform aspect ratio, and orientation uniform in `[0, pi)`. Each term is independently constructed by two-dimensional `interpolate_multi_scale_nd`. Principal-axis unsafe points spaced by half the minor-axis standard deviation keep the narrow ridge refined.

The terms are combined by deterministic balanced pairwise reduction. Every pair is added and immediately compressed with the recorded intermediate relative-L2/SVD tolerance. This gives logarithmic reduction depth and avoids an exact intermediate sum whose bond dimension grows with the number of terms. A final global relative-L2/SVD truncation uses `1e-6`.

The expensive raw input pair is cached atomically in `.cache/inputs/`. The cache key includes the schema version, pinned tensor4all-rs revision, `N`, `R`, width, aspect range, spacing, polynomial degree, interpolation tolerance, addition tolerance, and seed. Cases 2 and 3 share the same cache entry. Set `BENCH_INPUT_CACHE_REFRESH=1` to rebuild it or `BENCH_INPUT_CACHE_DIR` to move the cache.

## Allocation counters

Every record carries the allocation calls and the requested bytes of the timed
operation, median over the timed runs. The counters come from a process-wide
`GlobalAlloc` wrapper in `src/alloc.rs`; the benchmark binaries are single
threaded and pinned to one CPU core, so the counts are attributable to the timed
arm. They expose the per-call overhead of the contraction engine, which the wall
time alone hides: at small bond dimensions an arm can allocate two orders of
magnitude more than its tensors hold, and that cost tracks the number of local
updates rather than the tensor size. Records written before the counters existed
omit both fields, and the report generator leaves their columns blank.

## Patching and accuracy

The patch cap is fixed at 128 and has no runtime setting. Gaussian fit arms use relative-L2/SVD truncation. ACI uses its own interpolation residual, which is not identified with an L2 tolerance. Records therefore carry both the internal tolerance metric and a common deterministic holdout sampled relative-L2 error.

Case 2 compares:

- global fit
- patched fit
- global ACI with scale-relative residual
- patched ACI with patch-local absolute residual followed by global L2 budgeting

Case 3 compares global TreeTN fit with patched chain-TreeTN fit. Its reference is the finite left-endpoint quantics grid contraction, evaluated analytically with endpoint corrections.

Input construction, cache I/O, global input compression, format conversion, patch preparation, output conversion, and accuracy evaluation are outside timed regions.

## Running

Requirements are Rust, HDF5, and a BLAS/LAPACK implementation. A complete single-core run is:

```bash
BENCH_CPU_CORE=0 scripts/run_all.sh linux-epyc-7713p
```

The script pins Rayon and common BLAS implementations and the tenferro thread count to one thread, runs the three maintained binaries in release mode, writes raw records under `result/<profile>/raw/`, records machine metadata in `run.yaml`, and generates `report.md`.

Set `BENCH_THREADS=n` for the optional scaling sweep. It uses `n` threads on the contiguous core range starting at `BENCH_CPU_CORE`, and belongs in its own profile named `<machine>-<n>t`. The single-core profile stays the source of truth.

```bash
BENCH_CPU_CORE=0 BENCH_THREADS=4 scripts/run_all.sh linux-epyc-7713p-4t
```

For a probe run:

```bash
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
BENCH_NS=2 OUT_DIR=/tmp/t4a-probe/raw \
taskset -c 0 cargo run --release --locked --bin elementwise_gauss2d_patched
```

Gaussian knobs shared by Cases 2 and 3:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `BENCH_NS` | `2,8,32,128` | Gaussian counts used to produce measured χ points |
| `BENCH_SEED` | `0` | deterministic instance seed |
| `BENCH_RUNS` | `1` | timed repetitions |
| `BENCH_WARMUPS` | `0` | untimed repetitions |
| `BENCH_INPUT_CACHE_DIR` | `.cache/inputs` | shared input cache |
| `BENCH_INPUT_CACHE_REFRESH` | `0` | rebuild cache entry when nonzero |
| `OUT_DIR` | `result/dev/raw` | JSON output directory |

Case 2 additionally accepts `BENCH_ACI_TOL`, default `1e-8`. This is an ACI residual threshold, not an L2 truncation tolerance.

Case 1 accepts `BENCH_KS`, `BENCH_R`, `BENCH_TOL`, `BENCH_MAX_BOND`, `BENCH_RUNS`, `BENCH_WARMUPS`, `BENCH_SEED`, and `BENCH_ALGOS`. Its report is sorted by measured input χ even though mode count generates the instances.

## Validation

```bash
cargo fmt --all -- --check
cargo test --release
cargo clippy --locked --all-targets -- -D warnings
cargo check --locked --all-targets
python3 scripts/report.py result/<profile>
git diff --check
```

The focused tests cover difficult rotated interpolation, balanced pairwise sums, cache round trips, index identity, global and patched fit/ACI accuracy, contraction accuracy, patch caps, parameter accounting, and JSON serialization.
