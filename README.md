# tensor4all-benchmark

An open experimentation ground for comparing tensor network contraction algorithms in
[tensor4all-rs](https://github.com/tensor4all/tensor4all-rs) on reproducible problem
instances. Every input is generated in Rust from a fixed seed, every run is recorded as a
JSON `RunRecord` (timings, accuracy, bond dimensions), and the reports are rendered from
those records, so a result can always be traced back to the instance and the upstream
revision that produced it. The pinned `tensor4all-rs` revision lives in `Cargo.toml`.

## Benchmark cases

**Case 1: elementwise (Hadamard) product of quantics tensor trains.**
Two random Fourier series of `K+1` modes each are built as exact rank-`(K+1)` QTTs on `R`
bits, SVD-compressed to the working tolerance, and multiplied elementwise. The exact
product series is known analytically (a Fourier series of `2K+1` modes), so accuracy is
measured pointwise against it rather than against another tensor network. The setup
follows the elementwise product benchmark of arXiv:2604.00037. Algorithms: `naive`
(full product then truncate), `zipup` (single-pass zip-up truncation), `fit`
(variational sweeps), `aci` (adaptive cross interpolation).
Runner: `src/bin/elementwise_fourier.rs`, sweep over `K`.

**Case 2: MPO-MPO contraction of 2D quantics Gaussian mixtures.**
Two random mixtures of `n` isotropic 2D Gaussians are represented as quantics MPOs on
`R` bits per variable over the box `[-L, L]^2`, with `x` as the row index and `y` as the
column index. Contracting the two MPOs over `y` and scaling by the grid spacing
approximates the integral `int dy f(x, y) g(y, z)`, which has a closed-form Gaussian
answer, so again the reference is analytic. Algorithms: `naive` (simplett),
`zipup_simplett`, `zipup_treetn`, `fit_treetn`. Where both upstream engines implement an
algorithm, both are benchmarked as separate arms, so the engine is a visible variable
rather than a hidden one; the two zipup arms are the same algorithm on the two engines and
their difference isolates the engine. Each record carries the engine that ran it as
`engine`. The one missing pair is simplett fit, excluded because its variational update is
a stub upstream (known issue 1).
The contraction output bond dimension is pinned to the input rank: every algorithm runs
with its maximum bond dimension capped at `chi_in`, the larger of the two input MPO ranks,
so all arms are compared at the same output budget. The reported error is then the
discriminator, namely the residual of the contracted MPO against the analytic Gaussian
integral. `BENCH_MAX_BOND` caps only the input TCI construction. As measured at `R` = 6
and `R` = 8, the two simplett arms `naive` and `zipup_simplett` return the same truncated
result and the same error, so the metric does not separate them and only their wall times
differ. `zipup_treetn` reaches the same error as `zipup_simplett` to within a percent while
running two to three orders of magnitude faster (0.14 s against 107 s at `R` = 8), so the
accuracy of zip-up is a property of the algorithm and its cost here is a property of the
engine. Only `fit_treetn` reaches an error several orders of magnitude lower, at a
`chi_out` below the budget it was allowed, which suggests zip-up truncation on either
engine is far from the best fixed-rank approximation available at that rank. Both gaps are
observed rather than diagnosed, and no upstream issue has been filed for either yet.
Runner: `src/bin/mpo_mpo_quantics.rs`, sweep over `R`.

## Latest results

- [`result/mac-cpu/elementwise_fourier.md`](result/mac-cpu/elementwise_fourier.md)
- [`result/mac-cpu/mpo_mpo_quantics.md`](result/mac-cpu/mpo_mpo_quantics.md)

The scaling plots sit next to these files, and `result/mac-cpu/run.yaml` records the
machine, the repository revision, and the pinned tensor4all-rs revision that produced
them.

## Running

Prerequisites:

- Rust (stable toolchain).
- HDF5: `brew install hdf5` on macOS, `sudo apt-get install -y libhdf5-dev` on Debian or
  Ubuntu. LAPACK is also linked through `tenferro-linalg`: on macOS the system Accelerate
  framework covers it, on Linux install `liblapack-dev` if the build cannot find it.
- [uv](https://docs.astral.sh/uv/) for the report generator (matplotlib, numpy).
- Julia (optional), only for the independent ITensors.jl correctness checks below.

Full run and report for a machine profile:

```bash
scripts/run_all.sh mac-cpu
```

This builds in release mode, runs both cases with their default sweeps into
`result/mac-cpu/raw/`, writes `result/mac-cpu/run.yaml`, and renders the Markdown reports
and SVG plots.

Smoke run (small, fast, useful for checking the toolchain):

```bash
BENCH_KS=4 BENCH_R=10 BENCH_RUNS=1 BENCH_WARMUPS=0 OUT_DIR=/tmp/smoke \
  EXPORT_HDF5=/tmp/smoke cargo run --release --bin elementwise_fourier
BENCH_RS=8 BENCH_NGAUSS=3 BENCH_RUNS=1 BENCH_WARMUPS=0 BENCH_SANITY=1e-1 \
  OUT_DIR=/tmp/smoke EXPORT_HDF5=/tmp/smoke cargo run --release --bin mpo_mpo_quantics
```

Sanity gates: both runners are self-checking. A runner exits nonzero if any algorithm's
measured error exceeds its gate. Case 1 uses `1e3 * BENCH_TOL` for `naive`, `zipup` and
`aci`, and a looser `1e-2` for `fit` (truncation is norm-relative and the TT norm grows
like `2^(R/2)`, so the pointwise error is not bounded by the tolerance itself). Case 2
uses `BENCH_SANITY`, default `1e-2`, for every algorithm: with the output budget fixed at
`chi_in` the truncation error is the quantity the case measures, so the gate only screens
order-unity wrongness. The gates are there to catch wrong results, not to certify
precision.

Cost note: the quantics rank of the default case-2 mixture saturates around chi = 70 to 80,
and `naive` and `zipup_simplett` both build the full contracted bond of size chi squared
before truncating, which costs tens of seconds to minutes per contraction at that rank. Every
algorithm truncates back to the same output budget `chi_out <= chi_in`, so the arms differ
in accuracy at equal cost rather than in how far their ranks are allowed to grow. The
default sweep (`R` = 6, 8, 10 with 1 timed run) therefore stays under about ten minutes
on a laptop. These kernels are deterministic, so a single run is a usable timing; raise
`BENCH_RUNS` when a median over repetitions is wanted. For the heavy tail, extend
explicitly, for example
`BENCH_RS=6,8,10,12,14,16 BENCH_RUNS=5`; cost grows roughly linearly in `R` once the rank
has saturated. Restrict `BENCH_ALGOS` when you only want a quick signal.

Environment knobs:

| Variable | Applies to | Default | Meaning |
| --- | --- | --- | --- |
| `BENCH_KS` | case 1 | `4,8,16,32,64` | comma-separated Fourier mode counts `K` to sweep |
| `BENCH_R` | case 1 | `20` | number of quantics bits |
| `BENCH_RS` | case 2 | `6,8,10` | comma-separated bits per variable `R` to sweep |
| `BENCH_NGAUSS` | case 2 | `8` | number of Gaussians per mixture |
| `BENCH_BOX_L` | case 2 | `6.0` | half-width `L` of the box `[-L, L]` |
| `BENCH_ALPHA_LO` | case 2 | `0.5` | lower bound of the Gaussian width parameter |
| `BENCH_ALPHA_HI` | case 2 | `8.0` | upper bound of the Gaussian width parameter |
| `BENCH_SANITY` | case 2 | `1e-2` | relative error gate for every algorithm |
| `BENCH_TOL` | both | `1e-8` | truncation tolerance passed to every algorithm |
| `BENCH_MAX_BOND` | both | `4096` (case 1), `512` (case 2) | bond dimension cap |
| `BENCH_RUNS` | both | `5` (case 1), `1` (case 2) | timed repetitions, the median is reported |
| `BENCH_WARMUPS` | both | `1` (case 1), `0` (case 2) | untimed warmup repetitions |
| `BENCH_SEED` | both | `0` | base seed for instance generation |
| `BENCH_ALGOS` | both | `naive,zipup,fit,aci` (case 1), `naive,zipup_simplett,zipup_treetn,fit_treetn` (case 2) | comma-separated algorithms to run |
| `OUT_DIR` | both | `result/dev/raw` | directory for the `RunRecord` JSON files |
| `EXPORT_HDF5` | both | unset | directory for ITensors-compatible HDF5 instance dumps, plus their JSON metadata. Set it to enable the Julia checks. An empty value counts as unset. |

## Julia correctness checks

The exported instances are read back by ITensors.jl and evaluated against the same
analytic formulas the Rust side uses, which is an engine-independent check that the
inputs really represent the intended functions. First instantiate the environment, then
run a check per instance (the trailing number is `K` for case 1 and `R` for case 2, and
the instance must have been exported with `EXPORT_HDF5`). Full profile runs through
`scripts/run_all.sh` do not export HDF5, so to produce instances for the checks set
`EXPORT_HDF5` on a runner invocation of your own, for example the case-1 smoke run above
with `EXPORT_HDF5=/tmp/smoke`:

```bash
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
julia --project=julia julia/check_elementwise.jl /tmp/smoke 4
julia --project=julia julia/check_mpo_mpo.jl /tmp/smoke 8
```

Case 1 exports the exact tensor trains that were benchmarked. Case-2 exported instances,
by contrast, are re-generated by TCI at export time, so they can differ slightly from the
exact tensors used in the timed runs (see known issue 5); the function-level check remains
valid, since both the exported instance and the benchmarked one approximate the same
analytic mixture to the working tolerance.

## Known issues

1. **`tensor4all_simplett::mpo::contract_fit` is a silent placeholder at the pinned
   revision.** Its two-site local update leaves the core untouched, so the routine
   returns the naive contraction and the sweeps are dead work, with environments built by
   impractical scalar loops. It fails no test and prints no warning. Upstream issue:
   [tensor4all-rs#571](https://github.com/tensor4all/tensor4all-rs/issues/571). This
   benchmark therefore has no simplett fit arm: the case-2 fit is `fit_treetn`, run on the
   `tensor4all-treetn` engine bridged via `tensor4all-itensorlike`, which has a complete
   fit implementation.
2. **Case 2 mixes engines and truncation semantics.** `naive` and `zipup_simplett` run on
   `simplett` with an absolute singular value cutoff, `zipup_treetn` and `fit_treetn` run
   on `treetn` with a relative cutoff, so the two engines discard different singular values
   at the same nominal tolerance. The rank cap now binds for all of them at
   `chi_out <= chi_in`, so output bond dimensions no longer diverge by engine and timings
   are compared at the same budget; what remains engine-dependent is which directions
   inside that budget get kept. Running zipup on both engines is what makes that
   difference measurable instead of confounded with the algorithm. The generated report
   repeats this note under its summary table. The `fit_treetn` arm also runs a single full
   sweep, pinned as part of the benchmark definition and recorded as `fit_nsweeps` in every
   JSON record, so its wall time is only comparable at that stated sweep count.
3. **The case-1 `fit` arm is pinned to two full sweeps.** The sweep count is part of the
   benchmark definition, since fit cost is linear in it. The upstream elementwise fit
   accuracy problem recorded in `tensor4all-itensorlike/tests/bug_fit_elementwise.rs` did
   not reproduce on the instances used here, so the arm is kept in the defaults, with a
   loosened sanity gate as a guard.
4. **Case 2 has a reference error floor near `1e-8`.** The analytic reference integrates
   `y` over the whole real line, while the MPO contraction sums only over the box, so the
   two differ by the tail outside the box. At the default box size that mismatch is a
   relative error around `1e-8`. Error curves that plateau at that level are hitting the
   reference, not a tensor network artifact.
5. **Case-2 quantics TCI construction is not bit-reproducible across runs** even at a
   fixed seed: the input bond dimension can vary by one between runs of the same
   instance. The recorded `input_max_bond_dim` always reflects the actual run, so the
   plots stay self-consistent, but two runs of the same configuration can differ slightly
   on the x axis.

## License

MIT, see [`LICENSE`](LICENSE).
