# tensor4all-benchmark

An open experimentation ground for comparing tensor network contraction algorithms in
[tensor4all-rs](https://github.com/tensor4all/tensor4all-rs) on reproducible problem
instances. Every input is generated in Rust from a fixed seed, every run is recorded as a
JSON `RunRecord` (timings, accuracy, bond dimensions), and the reports are rendered from
those records, so a result can always be traced back to the instance and the upstream
revision that produced it. The pinned `tensor4all-rs` revision lives in `Cargo.toml`.

## Cases at a glance

| Case | What it measures | Details | Runner | Latest report | Plots |
| --- | --- | --- | --- | --- | --- |
| 1. `elementwise_fourier` | Elementwise product of two 1D quantics Fourier series, swept over the mode count `K`, against the exact product series | [description](#case-1-elementwise-hadamard-product-of-quantics-tensor-trains) | [`src/bin/elementwise_fourier.rs`](src/bin/elementwise_fourier.rs) | [`result/mac-cpu/elementwise_fourier.md`](result/mac-cpu/elementwise_fourier.md) | [time](result/mac-cpu/elementwise_fourier-time.svg), [error](result/mac-cpu/elementwise_fourier-error.svg) |
| 2. `mpo_mpo_quantics` | Contraction of two 2D quantics Gaussian-mixture MPOs over their shared variable, swept over bits per variable `R`, against the closed-form Gaussian integral | [description](#case-2-mpo-mpo-contraction-of-2d-quantics-gaussian-mixtures) | [`src/bin/mpo_mpo_quantics.rs`](src/bin/mpo_mpo_quantics.rs) | [`result/mac-cpu/mpo_mpo_quantics.md`](result/mac-cpu/mpo_mpo_quantics.md) | [time](result/mac-cpu/mpo_mpo_quantics-time.svg), [error](result/mac-cpu/mpo_mpo_quantics-error.svg) |
| 3. `elementwise_gauss2d` | Elementwise product of two 2D quantics Gaussian mixtures at a fixed output budget, swept over bits per variable `R`, against the exact pointwise product | [description](#case-3-elementwise-product-of-2d-quantics-gaussian-mixtures) | [`src/bin/elementwise_gauss2d.rs`](src/bin/elementwise_gauss2d.rs) | [`result/mac-cpu/elementwise_gauss2d.md`](result/mac-cpu/elementwise_gauss2d.md) | [time](result/mac-cpu/elementwise_gauss2d-time.svg), [error](result/mac-cpu/elementwise_gauss2d-error.svg) |
| 4. `elementwise_gauss2d_scaling` | Density-constant scaling study of case 3: how the quantics input rank `chi_in` grows with the number of Gaussians `N` when the box area grows proportionally to `N` | [description](#case-4-density-constant-scaling-of-the-quantics-rank) | [`src/bin/elementwise_gauss2d_scaling.rs`](src/bin/elementwise_gauss2d_scaling.rs) | [`result/mac-cpu/elementwise_gauss2d_scaling.md`](result/mac-cpu/elementwise_gauss2d_scaling.md) | [chi](result/mac-cpu/elementwise_gauss2d_scaling-chi.svg), [time](result/mac-cpu/elementwise_gauss2d_scaling-time.svg), [error](result/mac-cpu/elementwise_gauss2d_scaling-error.svg) |

## Benchmark cases

### Case 1: elementwise (Hadamard) product of quantics tensor trains

Two random Fourier series of `K+1` modes each are built as exact rank-`(K+1)` QTTs on `R`
bits, SVD-compressed to the working tolerance, and multiplied elementwise. The exact
product series is known analytically (a Fourier series of `2K+1` modes), so accuracy is
measured pointwise against it rather than against another tensor network. The setup
follows the elementwise product benchmark of arXiv:2604.00037. Algorithms: `naive`
(full product then truncate), `zipup` (single-pass zip-up truncation), `fit`
(variational sweeps), `aci` (adaptive cross interpolation).
Runner: [`src/bin/elementwise_fourier.rs`](src/bin/elementwise_fourier.rs), sweep over `K`.

### Case 2: MPO-MPO contraction of 2D quantics Gaussian mixtures

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
The contraction output bond dimension is pinned to the input rank, and the rank cap is the
only thing that decides it: every algorithm runs with its maximum bond dimension capped at
`chi_in`, the larger of the two input MPO ranks, and with a truncation tolerance pinned
inert, so all arms are compared at the same output budget and each one genuinely exhausts
it unless its exact rank is smaller. `BENCH_TOL` therefore scopes only the input TCI
construction, where it fixes `chi_in` and thus the instance, while `BENCH_CONTRACT_TOL`,
default `1e-15`, is what the arms receive and is recorded as `contract_tol` in the params of
every record. This removes the ambiguity of which constraint binds: without it a
tolerance-driven arm could stop early and report a `chi_out` below the budget, so the error
column would compare arms at different effective ranks. `BENCH_MAX_BOND` also caps only the
input TCI construction. As measured over the default sweep `R` = 6, 8, 10, 12 and 14 with
the pinned revision, `naive` and `fit_treetn` land on the same error, `5.7e-10` to
`2.9e-9`, and the two zipup arms,
`zipup_simplett` and `zipup_treetn`, agree with each other to the last reported digit and
sit four to five orders of magnitude higher, `2.2e-5` to `1.1e-4`. Every arm returns
`chi_out` equal to the budget. The split is therefore algorithmic rather than engine-driven:
single-pass zip-up truncation is what costs accuracy, and both engines running it produce
the same answer. What zip-up buys is speed, since it is the fastest arm at every `R` and
stays between 0.015 s and 0.31 s, while `naive` grows steeply (0.4 s at `R` = 8, 5.3 s at
`R` = 10, about 13 s at `R` = 12 and 14) because it forms the full
contracted bond before truncating; at `R` >= 10 that arm is memory bound on a machine with
less headroom, where the same points cost several times more (known issue 10).
`fit_treetn` reaches naive accuracy at a fraction of the
naive cost, under 0.75 s at every `R`.
Runner: [`src/bin/mpo_mpo_quantics.rs`](src/bin/mpo_mpo_quantics.rs), sweep over `R`.

### Case 3: elementwise product of 2D quantics Gaussian mixtures

Two independent random mixtures of `n` isotropic 2D Gaussians are cross-interpolated into
fused 2D quantics tensor trains on the box `[-L, L)^2`, `R` sites of site dimension 4 whose
local index is `x_bit + 2 * y_bit` with the most significant bit first. The benchmarked
operation is the elementwise product `h = f * g` on the `2^R` by `2^R` grid. The reference
is exact and pointwise, `h(x, y) = f(x, y) * g(x, y)`, with no quadrature and no tail, so
unlike case 2 this case has no reference error floor of its own. Algorithms: `naive`
(core-wise bond Kronecker product then an SVD sweep, written in this repository on simplett
primitives, recorded with `engine` = `local`), `zipup_treetn` and `fit_treetn` (both
`tensor4all_treetn::hadamard`, `engine` = `treetn`), and `aci`
(`tensor4all_aci::elementwise`, adaptive cross interpolation of the pointwise product,
`engine` = `aci`). There is no simplett arm, because simplett exposes no elementwise product
for tensor trains at the pinned revision (known issue 7), so unlike case 2 this case cannot
compare two engines on one algorithm.
Like case 2, the output bond dimension is pinned to the input rank and the rank cap alone
decides it: every algorithm runs capped at `chi_in`, the larger of the two input ranks, with
the truncation tolerance pinned inert at `BENCH_CONTRACT_TOL`, default `1e-15`, and recorded
as `contract_tol`. `BENCH_TOL` scopes only the input TCI construction, where it fixes
`chi_in` and the instance, as does `BENCH_MAX_BOND`. All arms are therefore compared at the
same output budget, each one exhausts it unless its exact rank is smaller, and the error is
the discriminator. The `aci` arm additionally runs with `scale_tolerance` enabled, recorded
as `aci_scale_tolerance`, so that its pivot criterion is scale-relative and equally
unreachable and the cap is what decides for it too. As measured over the default sweep
`R` = 6, 8, 10, 12 and 14 with the pinned revision (`chi_in` of 53, 77 and then 78 once the
quantics rank saturates), `naive`, `fit_treetn` and `aci` agree to the last
reported digit or close to it, from `3.6e-11` at `R` = 6 to about `1.5e-8` at `R` = 12, all
at the full `chi_out`. `zipup_treetn`
collapses: it spends the same budget and returns `3.3e-1` to `6.2e-1` across the sweep, an
answer with no
correct digits, and that number swings by a factor of two between runs of the same
configuration, so read it as order one rather than as a measurement. The separation is much
sharper than in case 2, where the same single-pass
truncation cost only four to five orders of magnitude, because the exact elementwise
product has rank up to `chi_in` squared and a budget of `chi_in` discards nearly all of it,
while naive and fit find a near-optimal basis for the same budget. Raising the budget
recovers zipup smoothly, to `1.8e-7` at 8 `chi_in` and `3.9e-8` unconstrained, so this is
the price of the fixed budget rather than a broken arm (known issue 8). On cost, `naive` is
again the expensive one, forming the full `chi_in`-squared bond before truncating: 0.05 s at
`R` = 6, 3.2 s at `R` = 8 and 4 to 6 s at `R` = 10 to 14, against 1.3 s for `fit_treetn`
and 0.38 s for `zipup_treetn` at `R` = 14. `aci` is the cheapest arm at every point of the
sweep, 1 ms at `R` = 6 rising only to 73 ms at `R` = 14, five to
fifteen times below `zipup_treetn`, because the pinned revision carries its early exit on
rank saturation from
[tensor4all-rs#591](https://github.com/tensor4all/tensor4all-rs/pull/591), so its sweep stops
once the pivots stop improving instead of running to its iteration limit under the unreachable
stopping criterion.

One pitfall is worth naming, because the metric hides it. The relative error is normalized
by the largest sampled `|f * g|`, and `f` and `g` are drawn independently, so if the
Gaussians are made too narrow (a high `BENCH_ALPHA_HI`) or the density too low, the two
mixtures stop overlapping, the product is numerically zero everywhere on the grid, and the
normalization collapses exponentially while the error it divides stays finite. The reported
number would then be noise dressed as an accuracy. Both runners therefore measure, over the
same sampled points the error uses, the reference scale `ref_scale` = max `|f * g|` and the
input scales `input_scale_f` and `input_scale_g`, and refuse to benchmark the instance
before any timing if `ref_scale` falls below `1e-6 * input_scale_f * input_scale_g`, with a
message pointing at `BENCH_ALPHA_HI` and at the density. All three scales are recorded in
the params of every case-3 and case-4 record, so the health of an instance can be checked
after the fact. The default instances are nowhere near the guard: `ref_scale` is `5.3e-1`
against input scales of `1.2` and `0.87` at `R` = 6, a ratio of `0.49`, roughly five orders
of magnitude above the threshold.

Runner: [`src/bin/elementwise_gauss2d.rs`](src/bin/elementwise_gauss2d.rs), sweep over `R`.

### Case 4: density-constant scaling of the quantics rank

Case 3 holds the mixture fixed and sweeps the bit count, so its `chi_in` saturates and says
nothing about how hard a bigger problem is. Case 4 asks the complementary question: how does
the quantics input rank `chi_in` of a 2D Gaussian mixture grow with the number of Gaussians
`N` when the DENSITY of Gaussians is held constant? Two hypotheses are worth separating,
`chi ~ sqrt(N)`, which is what a boundary-law or one-dimensional-cut picture predicts for a
fused 2D quantics train, and `chi ~ N`, which is what a naive sum-of-terms picture predicts.

The construction keeps two things fixed while `N` grows. Density first: the box half-width
is `L = L0 * sqrt(N / N0)`, so the box area `(2L)^2` grows proportionally to `N` and the
number of Gaussians per unit area is the same at every point of the sweep. Resolution
second: growing the box at a fixed bit count would coarsen the grid and under-resolve each
Gaussian, which would confound rank growth with a loss of resolution, so the bit count grows
with the box as `R = R0 + round(log2(L / L0))`, one extra bit per doubling of `L`. That keeps
the grid spacing `2L / 2^R` roughly constant, so every Gaussian is resolved by roughly the
same number of grid points at every `N`. Because `R` moves in integer steps while `L` moves
continuously, the spacing is constant only up to a factor of at most `sqrt(2)`. The defaults
`L0` = 6.0, `N0` = 8, `R0` = 10 make `N` = 8 exactly the case-3 instance.

Everything else mirrors case 3: two independent mixtures, fused 2D quantics trains, and the
elementwise product `h = f * g` at the fixed output budget `chi_out <= chi_in`, decided by
the rank cap alone with the truncation tolerance pinned inert at `BENCH_CONTRACT_TOL` and the
`aci` arm running scale-relative, judged by the
sampled max relative error against the exact pointwise product. The arms are `zipup_treetn`,
`fit_treetn` and `aci` only. The case-3 `naive` arm is excluded from the defaults: it forms
the full `chi_in`-squared bond before truncating, and this case deliberately pushes `chi_in`
to roughly twice the case-3 value, where that arm would dominate the wall time of the whole
sweep without adding a separate conclusion, since it tracks `fit_treetn` to the last reported
digit in case 3. Pass it through `BENCH_ALGOS` if you want it. The overlap guard of case 3
applies here unchanged, and matters more, since the box grows with `N`: the runner records
`ref_scale`, `input_scale_f` and `input_scale_g` and refuses a degenerate instance before
timing. At the default `N` = 8 point `ref_scale` is `4.6e-1` against input scales of `1.0`
and `0.85`, a ratio of `0.54`.

As measured over the default sweep `N` = 8, 16, 32, 64 with the pinned revision, `chi_in` is
77, 103, 117 and 141 at `L` = 6.000, 8.485, 12.000 and 16.971 and `R` = 10, 11, 11 and 12. A
least-squares fit of `log(chi_in)` against `log(N)` gives `chi_in ~ N^0.28`, so over this
range the growth is sublinear by a wide margin and even slower than `sqrt(N)`: an eightfold
increase in the number of Gaussians costs less than a doubling of the rank. The linear
hypothesis is comfortably excluded, and the `sqrt(N)` hypothesis is the closer of the two
without being reached, which is consistent with `sqrt(N)` acting as an upper bound that
finite-size effects have not yet saturated. Read the exponent as a measurement over a factor
of 8 in `N` on one seed rather than as an asymptotic law. It is robust against the quantics
construction wobble of known issue 5: perturbing all four `chi_in` values by plus or minus 2
moves the fitted slope only within 0.26 to 0.30, and an independent rerun that landed on
`chi_in` of 79, 104, 117 and 143 fits to 0.27. The scaling conclusion is untouched by the
`chi_out`-driven change, since `chi_in` comes from the input TCI construction, which
`BENCH_CONTRACT_TOL` does not reach. On the arms themselves the case-3
verdict survives at twice the rank. Measured at `N` = 8 and 16, `fit_treetn` and `aci` agree
at about `9e-9` and `8e-9` while `zipup_treetn` returns `5e-1` and `8e-1`, all three at the
full `chi_out` of 77 and 103. `aci` is the cheap arm here too, five times below `zipup_treetn`
and an order of magnitude below `fit_treetn`: 0.05 s at `N` = 8 and 0.09 s at `N` = 16,
against 0.8 s and 1.6 s for `fit_treetn`, since the pinned revision stops its sweep once the
pivots saturate. Everything quoted above comes from the committed `mac-cpu` sweep.
Runner: [`src/bin/elementwise_gauss2d_scaling.rs`](src/bin/elementwise_gauss2d_scaling.rs),
sweep over `N`.

## Latest results

One profile per physical machine, so numbers from different hardware never overwrite
each other. Each profile was produced on its own machine, at the revision its own
`run.yaml` records, together with the machine label, the chip, the memory and the pinned
tensor4all-rs revision. A profile is therefore self-describing and is never regenerated on
somebody else's hardware: the report rendering of each profile follows the files committed
with it, so a profile taken before a change to `scripts/report.py` keeps the rendering it
was published with until its own machine runs the sweep again. Compare wall times only
within a profile, and read cross-profile time ratios as statements about hardware
(known issue 10).

`mac-cpu`, the maintainer's Mac, at the current pin and the full default sweeps, and the
only profile that carries case 4. The quoted numbers in the case descriptions above come
from this profile:

- [`result/mac-cpu/elementwise_fourier.md`](result/mac-cpu/elementwise_fourier.md)
- [`result/mac-cpu/mpo_mpo_quantics.md`](result/mac-cpu/mpo_mpo_quantics.md)
- [`result/mac-cpu/elementwise_gauss2d.md`](result/mac-cpu/elementwise_gauss2d.md)
- [`result/mac-cpu/elementwise_gauss2d_scaling.md`](result/mac-cpu/elementwise_gauss2d_scaling.md)

The scaling plots sit next to these files. This sweep is current with both the
`chi_out`-driven truncation semantics of the fixed-budget cases and the pinned upstream
revision, so every `chi_out` column sits at the budget and every arm column is directly
comparable.

`mac-m1-8gb`, an 8 GB Apple M1 MacBook Pro, contributed by a collaborator and kept exactly
as it was committed, at the revision recorded in its own `run.yaml`. It predates the
`chi_out`-driven truncation semantics and case 4, so its fixed-budget arms were still
tolerance-driven and some report a `chi_out` below the budget. Read it as that machine's
record of the sweep at that revision, and as the reference point for the memory-bound
naive timings of known issue 10:

- [`result/mac-m1-8gb/elementwise_fourier.md`](result/mac-m1-8gb/elementwise_fourier.md)
- [`result/mac-m1-8gb/mpo_mpo_quantics.md`](result/mac-m1-8gb/mpo_mpo_quantics.md)
- [`result/mac-m1-8gb/elementwise_gauss2d.md`](result/mac-m1-8gb/elementwise_gauss2d.md)

## Running

Prerequisites:

- Rust (stable toolchain).
- HDF5: `brew install hdf5` on macOS, `sudo apt-get install -y libhdf5-dev` on Debian or
  Ubuntu. LAPACK is also linked through `tenferro-linalg`: on macOS the system Accelerate
  framework covers it, on Linux install `liblapack-dev` if the build cannot find it.
- [uv](https://docs.astral.sh/uv/) for the report generator (matplotlib, numpy).
- Julia (optional), only for the independent ITensors.jl correctness checks below.

Full run and report for a machine profile. Name the profile after the machine, one
profile per physical machine, for example:

```bash
scripts/run_all.sh mac-m1-8gb
```

This builds in release mode, runs all four cases with their default sweeps into
`result/<profile>/raw/`, writes `result/<profile>/run.yaml`, and renders the Markdown
reports and SVG plots. `run.yaml` deliberately records no hostname, only a machine
label (`BENCH_MACHINE`, defaulting to the profile name) plus the chip and memory size,
since a hostname on a public repository can leak the operator's institution and
location. It also stamps `repo_rev` with a `-dirty` suffix when the source tree carries
uncommitted changes, so a sweep can never claim a clean revision it did not come from;
everything under `result/` is excluded from that check, since the script has just
rewritten it. On a machine without `uv`, point `REPORT_PYTHON` at any python that has
matplotlib and numpy.

Smoke run (small, fast, useful for checking the toolchain). Cases 2 and 3 write the same
`instance-r<R>` file names, so give them different `EXPORT_HDF5` directories when both are
exported; the Julia checks assert the case name and will refuse a mismatched pair:

```bash
BENCH_KS=4 BENCH_R=10 BENCH_RUNS=1 BENCH_WARMUPS=0 OUT_DIR=/tmp/smoke \
  EXPORT_HDF5=/tmp/smoke cargo run --release --bin elementwise_fourier
BENCH_RS=8 BENCH_NGAUSS=3 BENCH_RUNS=1 BENCH_WARMUPS=0 BENCH_SANITY=1e-1 \
  OUT_DIR=/tmp/smoke EXPORT_HDF5=/tmp/smoke cargo run --release --bin mpo_mpo_quantics
BENCH_RS=8 BENCH_NGAUSS=3 BENCH_RUNS=1 BENCH_WARMUPS=0 BENCH_SANITY=1e-1 \
  OUT_DIR=/tmp/smoke EXPORT_HDF5=/tmp/smoke-gauss2d \
  cargo run --release --bin elementwise_gauss2d
BENCH_NS=8 BENCH_R0=8 BENCH_RUNS=1 BENCH_WARMUPS=0 BENCH_SANITY=1e-1 \
  OUT_DIR=/tmp/smoke cargo run --release --bin elementwise_gauss2d_scaling
```

Case 4 exports no HDF5 instances and has no Julia check of its own. Its instances are the
same kind of object as case 3's, a pair of fused site-dimension-4 quantics tensor trains of
Gaussian mixtures, so what an export would verify is already verified by the case-3 check at
`N` = 8, which is the same instance by construction.

Sanity gates: every runner is self-checking. A runner exits nonzero if any algorithm's
measured error exceeds its gate. Case 1 uses `1e3 * BENCH_TOL` for `naive`, `zipup` and
`aci`, and a looser `1e-2` for `fit` (truncation is norm-relative and the TT norm grows
like `2^(R/2)`, so the pointwise error is not bounded by the tolerance itself). Case 2
uses `BENCH_SANITY`, default `1e-2`, for every algorithm: with the output budget fixed at
`chi_in` the truncation error is the quantity the case measures, so the gate only screens
order-unity wrongness. Case 3 uses `BENCH_SANITY` in the same way for `naive`, `fit_treetn`
and `aci`, and a hardcoded `5.0` for `zipup_treetn`, whose fixed-budget error is itself of
order one (known issue 8), so for that arm the gate can only catch a gross scale blow-up or
a non-finite result. Case 4 applies the case-3 rule unchanged, `BENCH_SANITY` for
`fit_treetn` and `aci` and a hardcoded `5.0` for `zipup_treetn`, and additionally fails if
any instance's `chi_in` reaches `BENCH_MAX_BOND`, since a rank pinned at the construction cap
would measure the cap rather than the function. The gates are there to catch wrong results,
not to certify precision. All the gates are absolute and unchanged by the `chi_out`-driven
truncation semantics; only the errors they screen moved.

Cost note: the quantics rank of the default case-2 mixture saturates around chi = 70 to 80.
`naive` builds the full contracted bond of size chi squared before truncating and is the
only expensive arm: 0.02 s at `R` = 6, 0.4 s at `R` = 8, 5.3 s at `R` = 10 and about 13 s
at `R` = 12 and 14. Every other arm
stays under a second across that range. At `R` >= 10 the naive intermediates are large
enough that on a machine with less memory headroom the same points cost several times more
and vary with ambient memory pressure (known issue 10). Every algorithm truncates back to
the same
output budget `chi_out <= chi_in`, so the arms differ in accuracy at equal budget rather
than in how far their ranks are allowed to grow. The default sweep (`R` = 6, 8, 10, 12, 14
with 3 timed runs) is dominated by the naive runs at `R` = 12 and 14 and takes about two
minutes on the maintainer's Mac, six on an 8 GB M1. For the heavy tail, extend explicitly,
for
example `BENCH_RS=6,8,10,12,14,16 BENCH_RUNS=5`. Restrict `BENCH_ALGOS`, dropping `naive`,
when you only want a quick signal.

Case 3 has the same shape and the same expensive arm, at its own scale: naive costs 0.05 s
at `R` = 6, 3.2 s at `R` = 8 and 4 to 6 s at `R` = 10 to 14, and every other arm stays
under one and a half seconds. Its default sweep takes about a minute. Its `aci` arm is the
cheapest of the
four at every `R`, 1 ms at `R` = 6 rising only to 73 ms at `R` = 14, because the pinned
revision includes the ACI rank-saturation early
exit of [tensor4all-rs#591](https://github.com/tensor4all/tensor4all-rs/pull/591), so the
unreachable stopping criterion of the fixed-budget cases no longer makes it run to its
iteration limit.

Case 4 costs about as much as case 3, but at the pinned revision the quantics TCI
construction of
its inputs is no longer what dominates: at `N` = 64 the two input trains build in a few
seconds, against about 5.5 s for one timed pass of the three arms (1.4 s for `zipup_treetn`,
3.7 s for `fit_treetn`, 0.2 s for `aci`). Arm cost therefore scales with `BENCH_RUNS` while
the construction runs once per `N`. The default sweep `N` = 8, 16, 32, 64 takes about a minute
(measured 48 s); `N` = 128 is left out because the cost roughly doubles again per step and the
fitted exponent is already stable over the factor of 8 in the defaults. All four cases plus
the reports finish in about four and a half minutes on the maintainer's Mac, on top of the
release build. On a slower or more memory-constrained machine the same defaults cost
substantially more, most of the difference sitting in the two naive arms at `R` = 10 to 14.

Environment knobs:

| Variable | Applies to | Default | Meaning |
| --- | --- | --- | --- |
| `BENCH_KS` | case 1 | `4,8,16,32,64,128` | comma-separated Fourier mode counts `K` to sweep |
| `BENCH_R` | case 1 | `20` | number of quantics bits |
| `BENCH_RS` | cases 2 and 3 | `6,8,10,12,14` | comma-separated bits per variable `R` to sweep |
| `BENCH_NGAUSS` | cases 2 and 3 | `8` | number of Gaussians per mixture |
| `BENCH_BOX_L` | cases 2 and 3 | `6.0` | half-width `L` of the box `[-L, L]` |
| `BENCH_NS` | case 4 | `8,16,32,64` | comma-separated Gaussian counts `N` to sweep. Case 4 derives `L` and `R` from `N`, so it ignores `BENCH_NGAUSS`, `BENCH_BOX_L` and `BENCH_RS` |
| `BENCH_L0` | case 4 | `6.0` | reference box half-width, the `L` at `N` = `BENCH_N0` |
| `BENCH_N0` | case 4 | `8` | reference Gaussian count, the `N` at which `L` = `BENCH_L0` and `R` = `BENCH_R0` |
| `BENCH_R0` | case 4 | `10` | reference bits per variable, the `R` at `N` = `BENCH_N0`. Lower it for a cheap probe of the whole sweep |
| `BENCH_ALPHA_LO` | cases 2, 3 and 4 | `0.5` | lower bound of the Gaussian width parameter |
| `BENCH_ALPHA_HI` | cases 2, 3 and 4 | `8.0` | upper bound of the Gaussian width parameter |
| `BENCH_SANITY` | cases 2, 3 and 4 | `1e-2` | relative error gate. Case 2 applies it to every algorithm; cases 3 and 4 apply it to all but `zipup_treetn`, which is gated at a hardcoded `5.0` |
| `BENCH_TOL` | all | `1e-8` | instance tolerance. In case 1 it is the working tolerance of the whole case, both the input compression and the product. In cases 2, 3 and 4 it is scoped to the input TCI construction only, where it fixes `chi_in` and therefore the instance; the arms take `BENCH_CONTRACT_TOL` instead. It is what each record's top-level `tolerance` field and the Julia-check metadata report, since both describe the inputs |
| `BENCH_CONTRACT_TOL` | cases 2, 3 and 4 | `1e-15` | truncation tolerance handed to every contraction or product arm, recorded as `contract_tol` in the params of each record. At the default it never fires, so the rank cap `chi_in` is the only binding truncation control and the fixed-budget cases are `chi_out`-driven by construction. Raise it if you want a tolerance-driven variant of those cases |
| `BENCH_MAX_BOND` | all | `4096` (case 1), `512` (cases 2, 3 and 4) | bond dimension cap. In cases 2, 3 and 4 it caps only the input TCI construction, since the arms themselves run at the fixed output budget `chi_in`. Case 4 fails rather than reports if an instance reaches it, since `chi_in` is what that case measures |
| `BENCH_RUNS` | all | `5` (case 1), `3` (cases 2, 3 and 4) | timed repetitions, the median is reported |
| `BENCH_WARMUPS` | all | `1` (case 1), `0` (cases 2, 3 and 4) | untimed warmup repetitions |
| `BENCH_SEED` | all | `0` | base seed for instance generation |
| `BENCH_ALGOS` | all | `naive,zipup,fit,aci` (case 1), `naive,zipup_simplett,zipup_treetn,fit_treetn` (case 2), `naive,zipup_treetn,fit_treetn,aci` (case 3), `zipup_treetn,fit_treetn,aci` (case 4) | comma-separated algorithms to run |
| `OUT_DIR` | all | `result/dev/raw` | directory for the `RunRecord` JSON files |
| `EXPORT_HDF5` | cases 1, 2 and 3 | unset | directory for ITensors-compatible HDF5 instance dumps, plus their JSON metadata. Set it to enable the Julia checks. An empty value counts as unset. Cases 2 and 3 use the same file names, so give them separate directories. Case 4 exports nothing and ignores it |

## Julia correctness checks

The exported instances are read back by ITensors.jl and evaluated against the same
analytic formulas the Rust side uses, which is an engine-independent check that the
inputs really represent the intended functions. First instantiate the environment, then
run a check per instance (the trailing number is `K` for case 1 and `R` for cases 2 and 3, and
the instance must have been exported with `EXPORT_HDF5`). Full profile runs through
`scripts/run_all.sh` do not export HDF5, so to produce instances for the checks set
`EXPORT_HDF5` on a runner invocation of your own, for example the case-1 smoke run above
with `EXPORT_HDF5=/tmp/smoke`:

```bash
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
julia --project=julia julia/check_elementwise.jl /tmp/smoke 4
julia --project=julia julia/check_mpo_mpo.jl /tmp/smoke 8
julia --project=julia julia/check_elementwise_gauss2d.jl /tmp/smoke-gauss2d 8
```

Cases 1 and 3 export the exact tensor trains that were benchmarked. Case-2 exported
instances, by contrast, are re-generated by TCI at export time, so they can differ slightly
from the exact tensors used in the timed runs (see known issue 5); the function-level check
remains valid, since both the exported instance and the benchmarked one approximate the same
analytic mixture to the working tolerance.

`check_mpo_mpo.jl` and `check_elementwise_gauss2d.jl` are near identical, because the two
cases export the same kind of instance, a pair of fused site-dimension-4 quantics tensor
trains of Gaussian mixtures. Each asserts its own case name in the instance JSON, so
pointing one at the other case's export directory fails loudly instead of silently checking
the wrong instance.

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
4. **Case 2 has a reference error floor.** The analytic reference integrates
   `y` over the whole real line, while the MPO contraction sums only over the box, so the
   two differ by the tail outside the box. Error curves that plateau at a level independent
   of the algorithm are hitting the reference, not a tensor network artifact. The level was
   quoted here as around `1e-8` from the tolerance-driven results; with the contraction now
   `chi_out`-driven, `naive` and `fit_treetn` reach `5.7e-10` at `R` = 6 and `2.3e-9` at
   `R` = 8, so at the default box size the floor sits below `1e-8` and the earlier figure was
   the truncation error of those arms rather than the reference.
5. **Quantics TCI construction is not bit-reproducible across runs** in cases 2, 3 and 4,
   even at a fixed seed: the input bond dimension can vary by one or two between runs of
   the same instance. Consecutive case-3 sweeps at identical code and seeds gave `chi_in`
   of 53, 75, 77 and then 53, 76, 79. The recorded `input_max_bond_dim` always reflects
   the actual run, so the plots stay self-consistent, but two runs of the same
   configuration can differ slightly on the x axis.
6. **Resolved upstream, included in the pinned revision.**
   [tensor4all-rs#574](https://github.com/tensor4all/tensor4all-rs/pull/574) fixed three
   simplett defects that this benchmark had recorded as case-2 anomalies: MPO factorize
   truncated against an absolute singular value threshold and now truncates relative to
   the largest singular value, matching treetn; `contract_zipup` ran an eight-deep scalar
   loop and now uses einsum, about 800 times faster, which removes the two to three orders
   of magnitude engine gap the earlier results showed on the zipup arms; and
   `contract_naive`'s compression sweep now establishes a right-to-left QR gauge before
   truncating, which dropped its error by about three orders of magnitude so that naive
   matches the variational fit. All three are contained in the current pin `1b9a517`, as they
   were in the earlier pin `7cfec22` where they first landed, and the current pin adds
   [tensor4all-rs#575](https://github.com/tensor4all/tensor4all-rs/pull/575), a treetci
   convergence fix that stops input TCI construction early once the rank saturates at
   `max_bond_dim`. Earlier numbers in this repository's git history predate these fixes and
   are not comparable.
7. **simplett has no elementwise product for tensor trains at the pinned revision.** It
   offers MPO-MPO contraction (`contract_naive`, `contract_zipup`, the stubbed
   `contract_fit`) but nothing that forms a Hadamard product of two tensor trains, so cases
   1 and 3 have no simplett arm and case 3 cannot put two engines on one algorithm the way
   case 2 does with its pair of zipup arms. Its `naive` arm is therefore written in this
   repository, as a core-wise bond Kronecker product plus an SVD sweep on simplett
   primitives, and is recorded with `engine` = `local` to keep that visible.
8. **Case-3 `zipup_treetn` has no correct digits at the fixed output budget.** It returns a
   relative error of order one, between `3e-1` and `9e-1` depending on `R`, on `N` and on the
   run, across the default sweeps of cases 3 and 4, having spent the whole
   `chi_in` budget. This is a property of the case, not a defect of the arm: the exact
   elementwise product has rank up to `chi_in` squared, and given more room the same arm
   converges normally, to `1.8e-7` at 8 `chi_in` and `3.9e-8` unconstrained. Because the
   error is of order one, the sanity gate cannot screen order-unity wrongness for this arm,
   so it is gated at a hardcoded `5.0` that only catches a scale blow-up or a non-finite
   result. Read the case-3 zipup error column as a verdict on the budget rather than as a
   precision measurement. Case 4 inherits all of this: it runs the same arm at the same fixed
   budget on the same kind of instance, at roughly twice the rank, and sees the same
   order-unity error.
9. **Resolved upstream: ACI no longer runs to its iteration limit when the cap binds.** The
   fixed-budget cases hand every arm an unreachable tolerance so that the rank cap alone
   decides the truncation, and for
   `aci` that used to mean the stopping criterion never fired and the sweep ran to
   `AciOptions::max_iters` even after the pivots had saturated, which made the `aci` arm of
   cases 3 and 4 cost far more wall time than the algorithm needed. Its accuracy was never
   affected. The early exit on rank saturation landed in
   [tensor4all-rs#591](https://github.com/tensor4all/tensor4all-rs/pull/591) and is included
   at the current pin `1b9a517`, and the fresh sweep confirms the fix: `aci` is now the
   cheapest arm at every point of cases 3 and 4, five to fifteen times below `zipup_treetn`,
   with the same errors as before. Numbers quoted in this repository's git history from before
   the bump are not comparable on the `aci` column. The `mac-m1-8gb` profile predates this
   pin, so its `aci` column still shows the pre-fix cost.
10. **`naive` wall times at `R` >= 10 are machine bound.** The naive arms form intermediates
    of bond `chi_in` squared, which at chi around 80 press against the free memory of a
    smaller machine, so their wall time depends on the hardware and on ambient memory
    pressure rather than on the algorithm alone. On the 8 GB M1 of the `mac-m1-8gb` profile
    the case-2 point at `R` = 10 measured about 28 s per run on a session with swap nearly
    full and 16 s on the same machine right after a reboot, same code, same errors, same
    `chi_out`; the committed sweep is the post-reboot one, taken on an otherwise idle
    machine, where the spread across the three timed runs stays within about 5 percent at
    `R` = 12 and 14 (22 percent at `R` = 10, whose first run pays the page-in). The
    `mac-cpu` profile measures the same `R` = 10 point at 5.3 s. The cheap arms differ far
    less. Errors and bond dimensions are unaffected everywhere, since the computation is
    the same arithmetic either way. So compare naive timings only within one profile, run
    official sweeps on an idle machine, and read cross-profile time ratios as hardware
    statements. This is also why profiles are per machine and why `run.yaml` records the
    chip and memory.

## License

MIT, see [`LICENSE`](LICENSE).
