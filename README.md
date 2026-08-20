# tensor4all-benchmark

Reproducible benchmarks for tensor network algorithms in [tensor4all-rs](https://github.com/tensor4all/tensor4all-rs). Inputs use fixed seeds, timed regions exclude preparation, every arm writes JSON, and reports are generated from those records.

## Maintained cases

| Case | Input and operation | Arms | Scaling axis |
| --- | --- | --- | --- |
| `elementwise_fourier` | Elementwise product of two random one-dimensional Fourier QTTs | naive, zip-up, fit, ACI | measured input χ |
| `gaussian_elementwise` | Elementwise product of randomly rotated anisotropic Gaussian QTTs | global fit, patched fit, global ACI, patched ACI | compressed input χ |
| `gaussian_mpo_contraction` | MPO-MPO contraction of the same Gaussian input family | global fit, patched fit | compressed input χ |

There is no independent `R` sweep. Gaussian inputs use fixed `R = 16`, meaning 65,536 grid points per physical axis. Gaussian count `N` and `R` are construction metadata, not analysis axes.

## Gaussian input

Every Gaussian has an independent positive weight, center, log-uniform aspect ratio, and orientation uniform in `[0, pi)`. The production generator applies global TCI directly to the whole mixture. A spatially indexed evaluator omits Gaussian tails only under a rigorous global pointwise absolute bound. Deterministic centers and principal-axis points seed TCI so narrow rotated ridges are represented.

The raw global-TCI result is compressed once with relative-L2/SVD tolerance `1e-6`. The independent two-dimensional `interpolate_multi_scale_nd` builder and deterministic balanced pairwise reduction remain in the test suite as a reference for individual Gaussian and small-mixture accuracy.

The expensive raw input pair is cached atomically in `.cache/inputs/`. The cache key includes the schema version, pinned tensor4all-rs revision, `N`, fixed `R`, width, aspect range, spacing, TCI tolerance and cap, localized evaluator bound, pivot count, and seed. Cases 2 and 3 share the same cache entry. Set `BENCH_INPUT_CACHE_REFRESH=1` to rebuild it or `BENCH_INPUT_CACHE_DIR` to move the cache.

## Patching and accuracy

The [accuracy policy](docs/accuracy-policy.md) defines how disjoint patch errors combine and keeps fit relative-L2 tolerances separate from ACI residual tolerances.

The patch cap is fixed at 128 and has no runtime setting. Gaussian fit arms apply the same relative-L2/SVD tolerance once on each disjoint output patch. ACI uses its own interpolation residual, which is not identified with an L2 tolerance. Records therefore carry both the internal tolerance metric and a common deterministic holdout sampled relative-L2 error.

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

The script pins Rayon and common BLAS implementations to one thread, runs the three maintained binaries in release mode, writes raw records under `result/<profile>/raw/`, records machine metadata in `run.yaml`, and generates `report.md`.

For a probe run:

```bash
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
BENCH_NS=2 OUT_DIR=/tmp/t4a-probe/raw \
taskset -c 0 cargo run --release --locked --bin elementwise_gauss2d_patched
```

A large Gaussian cache can be prepared and validated without running the operations:

```bash
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 BENCH_INPUT_ONLY=1 BENCH_NS=512 \
cargo run --release --locked --bin elementwise_gauss2d_patched
```

Input construction is outside the timed operation.

Gaussian knobs shared by Cases 2 and 3:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `BENCH_NS` | `2,8,32,128` | Gaussian counts used to produce measured χ points |
| `BENCH_SEED` | `0` | deterministic instance seed |
| `BENCH_RUNS` | `1` | timed repetitions |
| `BENCH_WARMUPS` | `0` | untimed repetitions |
| `BENCH_INPUT_CACHE_DIR` | `.cache/inputs` | shared input cache |
| `BENCH_INPUT_CACHE_REFRESH` | `0` | rebuild cache entry when nonzero |
| `BENCH_INPUT_ONLY` | `0` | Case 2 only: prepare and validate caches without timing operations when nonzero |
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
