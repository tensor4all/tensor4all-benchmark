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
integral. `BENCH_MAX_BOND` caps only the input TCI construction. As measured at `R` = 6, 8
and 10 with the pinned revision, `naive` and `fit_treetn` land on the same error, around
`1e-8`, which is the reference floor of the case (known issue 4), at the same `chi_out` of
59 to 61, well below the budget they were allowed. The two zipup arms, `zipup_simplett` and
`zipup_treetn`, agree with each other to the last reported digit and sit three to four
orders of magnitude higher, around `1e-5` to `1e-4`, at the full budget. The split is
therefore algorithmic rather than engine-driven: single-pass zip-up truncation is what
costs accuracy, and both engines running it produce the same answer. What zip-up buys is
speed, since it is the fastest arm at every `R` and stays flat near 0.2 s, while `naive`
grows steeply (0.64 s at `R` = 8, 5.3 s at `R` = 10) because it forms the full contracted
bond before truncating. `fit_treetn` reaches naive accuracy at a fraction of the naive cost.
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

Cost note: the quantics rank of the default case-2 mixture saturates around chi = 70 to 80.
`naive` builds the full contracted bond of size chi squared before truncating and is the
only expensive arm: 0.03 s at `R` = 6, 0.64 s at `R` = 8, 5.3 s at `R` = 10. Every other arm
stays under half a second across that range. Every algorithm truncates back to the same
output budget `chi_out <= chi_in`, so the arms differ in accuracy at equal budget rather
than in how far their ranks are allowed to grow. The default sweep (`R` = 6, 8, 10 with 3
timed runs) therefore takes well under a minute on a laptop, and `scripts/run_all.sh`
finishes in about seventy seconds including case 1 and the reports. `R` = 12 is left out of the defaults
because naive costs about 12.6 s there. For the heavy tail, extend explicitly, for example
`BENCH_RS=6,8,10,12,14,16 BENCH_RUNS=5`. Restrict `BENCH_ALGOS`, dropping `naive`, when you
only want a quick signal.

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
| `BENCH_RUNS` | both | `5` (case 1), `3` (case 2) | timed repetitions, the median is reported |
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
2. **Case 2 mixes engines.** `naive` and `zipup_simplett` run on `simplett`, `zipup_treetn`
   and `fit_treetn` on `treetn`. Both engines truncate relative to the largest singular
   value at the pinned revision, and the rank cap binds for all of them at
   `chi_out <= chi_in`, so the two zipup arms now return the same result and their
   remaining difference is wall time. Running zipup on both engines is what makes that
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
6. **Resolved upstream, included in the pinned revision.**
   [tensor4all-rs#574](https://github.com/tensor4all/tensor4all-rs/pull/574) fixed three
   simplett defects that this benchmark had recorded as case-2 anomalies: MPO factorize
   truncated against an absolute singular value threshold and now truncates relative to
   the largest singular value, matching treetn; `contract_zipup` ran an eight-deep scalar
   loop and now uses einsum, about 800 times faster, which removes the two to three orders
   of magnitude engine gap the earlier results showed on the zipup arms; and
   `contract_naive`'s compression sweep now establishes a right-to-left QR gauge before
   truncating, which dropped its error by about three orders of magnitude so that naive
   matches the variational fit. The pinned rev is
   `7cfec2270700d4b218e02f451a340518b84016fb`, which contains all three. Earlier numbers in
   this repository's git history predate them and are not comparable.

## License

MIT, see [`LICENSE`](LICENSE).
