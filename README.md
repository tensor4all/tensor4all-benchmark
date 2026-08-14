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
| 5. `elementwise_gauss2d_patched` | Patched (domain decomposed) elementwise product, controlled by a global relative tolerance instead of a fixed output budget, comparing patched and global representations at equal accuracy on size and time, over two instance families: anisotropic narrow spikes by default and case 4's smooth Gaussians on request | [description](#case-5-patched-elementwise-product-at-equal-accuracy) | [`src/bin/elementwise_gauss2d_patched.rs`](src/bin/elementwise_gauss2d_patched.rs) | not yet recorded in any profile | not yet recorded in any profile |

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
`3.5e-9`, and the two zipup arms,
`zipup_simplett` and `zipup_treetn`, agree with each other to the last reported digit and
sit four to five orders of magnitude higher, `2.1e-5` to `1.1e-4`. Every arm returns
`chi_out` equal to the budget. The split is therefore algorithmic rather than engine-driven:
single-pass zip-up truncation is what costs accuracy, and both engines running it produce
the same answer. What zip-up buys is speed, since it is the fastest arm at every `R` and
stays between 0.009 s and 0.25 s, while `naive` grows steeply (1.1 s at `R` = 8, 32 s at
`R` = 10, 89 to 91 s at `R` = 12 and 14) because it forms the full
contracted bond before truncating; at `R` >= 10 that arm's intermediates are large enough
that its wall time tracks the machine and the ambient memory pressure rather than the
algorithm alone, so it is not comparable across profiles (known issue 10).
`fit_treetn` reaches naive accuracy at a fraction of the
naive cost, under 0.8 s at every `R`.
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
`R` = 6, 8, 10, 12 and 14 with the pinned revision (`chi_in` of 53 and then 76 to 80 once
the quantics rank saturates), `naive`, `fit_treetn` and `aci` agree to the last
reported digit or close to it, from `3.6e-11` at `R` = 6 to about `1.5e-8` at `R` = 10, all
at the full `chi_out`. `zipup_treetn`
collapses: it spends the same budget and returns `1.3e-1` to `1.1` across the sweep, an
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
`R` = 6, 3.9 s at `R` = 8 and 6 to 7.6 s at `R` = 10 to 14, against 1.1 s for `fit_treetn`
and 0.35 s for `zipup_treetn` at `R` = 14. `aci` is the cheapest arm at every point of the
sweep, 1 ms at `R` = 6 rising only to 61 ms at `R` = 14, six to
fourteen times below `zipup_treetn`, because the pinned revision carries its early exit on
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
78, 103, 117 and 145 at `L` = 6.000, 8.485, 12.000 and 16.971 and `R` = 10, 11, 11 and 12. A
least-squares fit of `log(chi_in)` against `log(N)` gives `chi_in ~ N^0.29`, so over this
range the growth is sublinear by a wide margin and even slower than `sqrt(N)`: an eightfold
increase in the number of Gaussians costs less than a doubling of the rank. The linear
hypothesis is comfortably excluded, and the `sqrt(N)` hypothesis is the closer of the two
without being reached, which is consistent with `sqrt(N)` acting as an upper bound that
finite-size effects have not yet saturated. Read the exponent as a measurement over a factor
of 8 in `N` on one seed rather than as an asymptotic law. It is robust against the quantics
construction wobble of known issue 5: perturbing all four `chi_in` values by plus or minus 2
moves the fitted slope only within 0.27 to 0.31, and independent reruns that landed on
`chi_in` of 77, 103, 117, 141 and on 79, 104, 117, 143 fit to 0.28 and 0.27. The scaling conclusion is untouched by the
`chi_out`-driven change, since `chi_in` comes from the input TCI construction, which
`BENCH_CONTRACT_TOL` does not reach. On the arms themselves the case-3
verdict survives at twice the rank. Measured at `N` = 8 and 16, `fit_treetn` and `aci` agree
at about `1.3e-8` and `1.0e-8` while `zipup_treetn` returns `1.9e-1` and `3.5e-1`, all three at
the full `chi_out` of 78 and 103. Across the whole sweep that arm stays of order one, between
`1.7e-1` and `5.7e-1`, with no monotone trend in `N`, so the budget costs it every correct
digit at any rank in this range. `aci` is the cheap arm here too, eight to twelve
times below `zipup_treetn`
and more than an order of magnitude below `fit_treetn`: 0.03 s at `N` = 8 and 0.08 s at `N` = 16,
against 0.8 s and 2.1 s for `fit_treetn`, since the pinned revision stops its sweep once the
pivots saturate. Everything quoted above comes from the committed `mac-cpu` sweep.
Runner: [`src/bin/elementwise_gauss2d_scaling.rs`](src/bin/elementwise_gauss2d_scaling.rs),
sweep over `N`.

### Case 5: patched elementwise product at equal accuracy

Cases 3 and 4 both form one global tensor train for each input and one for the product, so
their rank is set by the hardest region of the box, and their comparison is made at a fixed
output budget. Case 5 changes both of those, and it runs on two instance families rather than
one. The representation becomes patched: each input
is a `PartitionedTT` from `tensor4all-partitionedtt`, a set of tensor trains over disjoint
subdomains, where a subdomain is a set of quantics digits held fixed. Fixing the leading fused
site picks one quadrant of the box, fixing the next one a sub-quadrant, so patches are regions
of the box rather than arbitrary slices: a contiguous quadrant when the digits fixed are the
leading ones, and a periodic union of them when they are not. The stopping rule is the
per-patch rank cap `BENCH_PATCH_MAX_BOND`: split until every patch fits under it. A hard region of the box therefore costs patches rather
than global rank.

Instance families, `BENCH_FAMILY`. The default is `aniso`: `N` anisotropic narrow spikes, of a
fixed minor width `BENCH_ANISO_SIGMA` = 0.05, each one stretched by its own aspect ratio, drawn
log-uniform in `[1, BENCH_ANISO_RHO_MAX]` = `[1, 8]`, along its own orientation, drawn uniform
in `[0, pi)`. Weights are `U[0.5, 1.5]` and centers uniform in nine tenths of the box, and the
draw order per spike is weight, aspect, orientation, center. The mean spacing between spikes is
held at `BENCH_ANISO_SPACING` = 3 minor widths, so `N` spikes fill a box of half-width
`L = 3 sigma sqrt(N) / 2`, and `R` is the smallest bit count whose grid step resolves the minor
width to a quarter, `2L / 2^R <= sigma / 4`. Fixing the ratio of spacing to width is what keeps
the elementwise product non degenerate as `N` grows: two independent draws overlap by a
constant fraction at every `N`. Over the default sweep the sampled `ref_scale` sits between
0.38 and 0.83 of `max|f| max|g|`, six orders of magnitude above the guard described above, with
no trend in `N`. The other family,
`smooth`, is case 4's and is unchanged: isotropic Gaussians of log-uniform inverse width at
constant density, `L = L0 sqrt(N / N0)` and `R = R0 + round(log2(L / L0))`, so the two cases can
still be read against each other point for point.

The default is `aniso` because the smooth family gives the patching nothing to isolate. A
smooth 2D mixture at constant density is easy everywhere, so subdividing it only pays the
per-patch overhead, which is what the case-5 verdict on it says. Narrow spikes at a fixed
spacing-to-width ratio are the opposite: the function is a field of small hard features whose
global rank climbs through the sweep while a patched representation is held at its per-patch
cap by construction. This is a contest of ranks rather than a wall: the geometric bound
`4^(R/2)` of the bit count grows like `sqrt(N)` as the resolution rule raises `R` with the
box, and the measured global rank does not even follow that, it decelerates toward a
density-set plateau. The global representation never runs out of room, it pays a bounded
but large rank where the patched one pays patch count. Measured single-threaded
at the pinned revision and `rtol` = 1e-8, the global input rank `chi_in` of the aniso
family is

| `N` | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|
| `R` | 6 | 6 | 7 | 7 | 8 | 8 | 9 | 9 |
| `chi_in`, aniso | 45 | 53 | 64 | 64 | 89 | 120 | 185 | 256 |
| `chi_in`, isotropic control | 49 | | | 64 | 94 | 126 | 196 | 256 |

which grows like `N^0.5` over the middle of the range, `N` = 64 to 512, and then decelerates
toward saturation: 256 at `N` = 1024 and 2048 and 289 at 4096, each verified converged (the
TCI reaches the tolerance in 4 to 6 of its 200 allowed iterations, and raising the
resolution rule from sigma/4 to sigma/16, hence `R` by two bits and the geometric bound to
4096, leaves the ranks unchanged). That the `N` = 1024 value equals the `R` = 9 bound is a
coincidence, not censoring. The saturation is what statistical homogeneity predicts: at
constant density a larger box adds more of the same landscape, so the local structure
diversity that sets the maximal bond is fixed by the density and the tolerance rather than
by `N`. The first row up to
`N` = 512 is the default sweep, the `N` = 1024 column and the second row are separate probe runs
at the same settings; a rank here moves by a unit or two between two constructions of the same
instance, since the input TCI is not bit-reproducible (88 against 89 at `N` = 128, 182 against
185 at 512 in two runs). The smooth family for
comparison grows like `N^0.33` over its own sweep (case 4's measurement), and its `R` grows
with the box, so it reaches a rank of 142 at `N` = 64 on a much finer grid.

The last row of that table is the control, `BENCH_ANISO_RHO_MAX=1`, which draws circular
spikes of one common shape and leaves everything else alone. It grows the same way. So at this
spacing-to-width ratio the rank comes from the density of narrow features and not from the
anisotropy, which is worth recording because the opposite is easy to assume: a family of one
common shape is a pure shift family, a low dimensional manifold, and one expects a quantics
TCI to exploit it. At `R` = 8 and up it does not, or not enough to matter. The per-spike aspect
and orientation are kept in the default family anyway, since a study of compressibility should
not rest on a shift family, but the measurement does not credit them with the growth.

Two constructions produce that patched input, and `BENCH_PATCH_INPUT` chooses between them.
The default, `norm`, builds one global train per input exactly as case 4 does and hands it to
`add_with_patching`, which truncates each subdomain against its volume share of the global
squared budget and splits whatever still exceeds the cap, choosing the split site by
`BENCH_PATCH_SPLIT`. Every decision it makes comes from Frobenius norms of an already-built
train, and no TCI runs inside the splitting loop. The alternative, `tci`, is
`adaptiveinterpolate`, which never forms a global train at all: it runs a TCI2 per patch on
the function itself and splits a patch whose own TCI does not converge under the cap. That is
the construction this case is eventually written for, since it is the one whose cost never
passes through a global rank. It used to be blocked by an upstream TCI2 defect,
which stopped it from `N` = 32 up on the smooth family's default instance; that is fixed at
the current pin `c9ecb7f` (known issue 11), and the path now completes at smooth `N` = 32 and 64 and at aniso
`N` = 64. `norm` stays the default anyway, and now on the measurement rather than on the
defect: for the same cap the `tci` path splits far harder, 514 and 622 input patches at smooth
`N` = 32 against 6 and 7, so it holds 1.04e6 input parameters against 234952 and returns a
product of 572436 parameters against 129276 at the same accuracy. Making it the default is a
separate decision and is not made here. The price of defaulting to `norm` is exactly the
property `tci` has and it does
not: a global `chi_in` train is built first, so the patched arms of the default sweep pay for
one, and their `input_build_secs` includes it. What the case still measures at equal accuracy
is the size and the cost of the two representations, which is what its conclusion is about.

The control changes from a budget to an accuracy. A fixed `chi_out` has no meaning for a
patched representation, which has no single rank to cap: its size is a parameter count spread
over patches. So every arm is instead asked for the same global relative tolerance
`BENCH_RTOL`, default `1e-8`, and what the case measures is the size and the wall time each
arm needs to reach it. The error column becomes a check that an arm got there rather than the
discriminator, and `n_params`, the total number of stored core entries, becomes the size
metric, since it is the one number that means the same thing for a single global train and
for a set of patch trains. For a patched arm it counts the free sites of each patch only: the
cores at the projected sites are one-hot copy selectors, structure rather than data, and an
implementation that stored each patch over its free sites would not hold them at all.

Two details of the tolerance handling are deliberate and are what make the numbers
trustworthy. First, no patch is ever held to a tolerance relative to its own norm, since a
patch sitting in a near-empty corner of the box must be allowed to converge at rank one
instead of being asked for eight relative digits of a quantity that contributes nothing to the
global norm. The `norm` path gets that for free: `add_with_patching` spends `rtol^2 ||F||^2`
over the patches by volume, so the budget of a patch is set by the norm of the whole function.
The `tci` path has to arrange it explicitly, and its per-patch TCI tolerance is therefore
absolute, `rtol` times the sampled scale of the function. Second, the
product is budgeted exactly once, at the end, by `truncate_adaptive`, which spends
`rtol^2 ||F||^2` over the patches proportionally to patch volume and drops a patch whose norm
is below its share. That is the correct treatment of shrinking patch norms; a plain relative
tolerance applied per patch would be the wrong thing. The tolerance handed to the per-patch
engine is kept two orders of magnitude tighter than `rtol` so that it is the final budgeting,
not the engine, that decides where the output is truncated.

The same rule is why both `aci` paths of this case, the global baseline and the per-patch
engine, run on an ABSOLUTE pivot budget rather than on the upstream default, which since
[tensor4all-rs#619](https://github.com/tensor4all/tensor4all-rs/pull/619) is scale-relative:
the pivot error of a bond divided by the largest output magnitude sampled at that bond. That
default is the right contract for a case whose accuracy target is local, and it is the one
cases 3 and 4 use, but this case measures one global relative error, normalized by the largest
sampled `|f g|` of the whole box, and a per-bond or per-patch normalization is exactly the
per-region relative tolerance the paragraph above refuses. Measured, the difference is not
cosmetic. On the smooth family at `N` = 32, the global `aci` baseline at the same pin and the
same `rtol` returns `chi_out` = 138, 147972 parameters and `8.4e-9` on an absolute budget,
against `chi_out` = 473 pinned at `BENCH_MAX_BOND`, 1.7e6 parameters and `2.7e-1` on the
scale-relative one, which fails the sanity gate; the per-patch engine shows the same thing
one level down, `5.3e-4` for `patched_aci` against `3.5e-8` for the three other engines, and
absolute restores the agreement. Both are recorded as `aci_tolerance` in the params of every
case-5 record. Known issue 13 says what that choice costs the verdict, since a scale-relative
baseline is the smaller object on the aniso family.

The product itself is formed patch pair by patch pair. Two patches contribute only when their
projectors are compatible, and then the product lives on the intersection of the two
subdomains. Since both inputs cover the domain disjointly, those intersections are disjoint
too and each one is produced exactly once, so no tensor train addition is ever needed. Inside
a pair the product runs on the free sites only, the fixed sites being sliced out of both
inputs first and put back afterwards as one-hot cores. That keeps the work proportional to
the patch volume, and it is what makes the `aci` engine usable at all: on the embedded train
the product vanishes outside a `4^-k` fraction of the index space, so a pivot search seeded
at random points would find nothing but zeros.

The three default arms `patched_fit_treetn`, `patched_naive` and `patched_aci` are three of the
four engines of case 3 run on the projected patch trains, so a difference between a patched arm
and its global namesake is the patching and not the engine. The fourth, `patched_zipup_treetn`,
is excluded from the defaults on cost alone, the way case 4 excludes `naive`: on the `norm`
path a patch carries a rank near the global one, and a single-pass zip-up of two such patches
with no binding output cap has to form the full product bond before it truncates, which at
`N` = 8 of the smooth family costs 98 s per pass against 3.5 s for each of the other three and
returns the same
product to every reported digit. It is one `BENCH_ALGOS` away. The runner also
measures the two global arms `fit_treetn` and `aci` at the same `rtol` with no binding rank
cap, which is what lets the report put the two representations side by side at equal accuracy.
Each of them has its own `N` ceiling, since the uncapped variational fit and the interpolating
arm are two cost classes: on the smooth family both stop at `N` = 64, and on the aniso family
`aci` runs everywhere while the fit stops at `N` = 1024, the last point measured to keep it
under two minutes (25.1 s there when probed at the pin `1b9a517`, where `aci` cost 0.18 s);
on the `norm` path those two arms run on the very trains the patched inputs were split out of,
so the global construction is counted once and shared. Input construction is timed separately
and recorded as `input_build_secs`, never folded into the product time, since one build is
shared by every arm of an instance. A patched arm on the `norm` path therefore reports the
global build plus the splitting, and a global arm on the same instance reports the global build
alone.

Measured on the aniso family with the pinned revision on the maintainer's Mac, single-threaded,
at the defaults (`norm` path, cap 64, `rtol` = 1e-8) over the whole default sweep `N` = 8 to
512, which takes 178 s in total, the picture is about the two global arms separately and only
one half of it is a win. Against global `aci`, the interpolating arm, the patched product is
smaller at every point, but only just: 5536, 6560, 17848, 20960, 70128, 82560 and 294680
parameters against 7456, 8224, 22880, 24480, 92776, 113944 and 312528, all at a relative error
between `2.9e-9` and `4.7e-8`. That margin is 26 percent at `N` = 8 and 6 percent at 512, and
what is behind it is visible in the bond columns: global `aci` reaches `chi_out` = 173, 209 and
256 at `N` = 128, 256 and 512, and 256 is exactly the geometric bound of `R` = 9, while the
patched arms sit at 64 by construction. Read that six percent as a tie rather than as a result,
for two reasons. It is of the order of the run to run spread of the instance itself, whose input
TCI is not bit-reproducible, and it does not survive a different reading of the `aci` baseline's
tolerance: asked for the same `rtol` scale-relative rather than absolute, the same baseline
returns 205008 parameters at `N` = 512 at `5.4e-9`, a third below the patched count, and stops
at `chi_out` = 233 instead of at the geometric bound. That variant is not the default because it
breaks the smooth family (see the tolerance discussion above and known issue 13), but it is
enough to retire the claim that patching beats interpolation on size. Against global
`fit_treetn`, the variational arm, the patched product was never smaller and still is not:
40864 at `N` = 128, 57584 at 256 and 118568 at 512, a factor of 2.5 below the patched count at
the top of the sweep. So an uncapped variational fit remains the smallest representation of this
product, exactly as on the smooth family, and what patching has on size is at best a tie with
interpolation.

Wall time is the other half, and there the patched arms do win something. `patched_fit_treetn`
costs 0.039, 0.043,
0.218, 0.272, 0.820, 1.03 and 4.27 s, against 0.031, 0.036, 0.179, 0.191, 1.49, 1.85 and
12.48 s for global `fit_treetn`: so from `N` = 128 up the patched fit is cheaper than the
global fit that returns the smaller object, by a factor of 2.9 at `N` = 512, and `patched_aci`
was cheaper there too, by a factor of 2.5 (0.132, 0.102, 0.189, 0.216, 0.849, 0.865 and
4.91 s). Both `aci` arms of this sweep were measured at the pin `b160bb7`, where the ACI global
pivot guard still cost a few tenths of a second per pass, so their wall times are stale at the
current pin and can only have come down (known issue 12): the ordering above therefore stands
and its ratios understate it. Global `aci` cost 0.082 to 0.84 s and stayed roughly an order of
magnitude below everything from `N` = 128 up, as on the smooth family. The
third arm, `patched_naive`, is not competitive at these ranks: 2.17 s at `N` = 64 and 8.46 s at
256 with the same output to every reported digit, since a local bond-Kronecker product forms the
full product bond before it truncates. The three patched arms return identical sizes and errors
at every point, which is expected when nothing binds, and they differ only in wall time.

The recorded cost breakdown says where that time goes, and it moves with the family. On the
aniso family the per-patch products dominate every arm, 3.39 s of the 4.27 s of
`patched_fit_treetn` at `N` = 512 and 3.83 s of the 4.91 s of `patched_aci`, while the final
`truncate_adaptive` is 0.89 s and 1.08 s. On the smooth family the balance is the other way for
the interpolating arm: at `N` = 64 `truncate_adaptive` is 23.1 s of the 40.1 s of `patched_aci`,
against 4.7 s of the 36.0 s of `patched_fit_treetn`, since the `aci` patches carry larger ranks
into the budgeting than a variational fit leaves behind. The two `patched_aci` splits are the
`b160bb7` ones and share the staleness of that arm's totals; the shape of the split, products
against final budgeting, is what the paragraph is about and the guard does not move it.
Reading a case-5 total without that
split is therefore misleading, which is
why `n_pairs`, `pairs_secs` and `truncate_secs` sit in the params of every patched record.

What is expensive on the aniso family is neither arm: it is the input construction, and
specifically `add_with_patching`. The patched build costs 0.06, 0.06, 0.23, 0.30, 8.7, 20.5 and
95.0 s over the sweep against 0.02 to 2.31 s for the two global trains it starts from, so from
`N` = 128 up, once the cap really binds and the splitting runs, the splitting is between 95 and
98 percent of it. That is the wall of this case, and it is the reason `N` = 1024 is not in the
default sweep: probed at the defaults at the pin `1b9a517` that point cost 302 s, of which 242 s
is the patched build, against 3.94 s for `patched_fit_treetn`, 1.22 s for `patched_aci` and
25.1 s for the
global fit. Only the `aci` timings of that probe are pin-sensitive, since the
construction and the fit reproduce it to a few percent across the sweep. It is also where the
case gets interesting, since `chi_in` = 256 there is the
geometric bound, so it is one `BENCH_NS` away rather than out of reach. The patched inputs are
also larger than the global ones they were split out of, 116284 against 93856 parameters at
`N` = 128 and 455756 against 267984 at 512, which is the same per-patch overhead the smooth
family shows.

Measured on the smooth family with the pinned revision on the maintainer's Mac,
single-threaded, at the defaults
(`norm` path, cap 64, `rtol` = 1e-8), the verdict over `N` = 8 to 64 is that patching costs
rather than saves on this instance family. The patched product holds 35952 parameters at
`N` = 8, 107600 at 16, 129276 at 32 and 381332 at 64, against 34752, 50612, 74800 and 114024
for global `fit_treetn`, which is the smallest representation at every point. So the patched
size is level with the global one at `N` = 8, where the cap does not bind and the construction
returns a single patch, and grows to about three times it at `N` = 64. Against global `aci`,
which is interpolation-based and keeps more, the patched product is the smaller of the two at
`N` = 8 (35952 against 52832) and the larger at 64 (381332 against 292304). All of them sit at
a relative error between `3.8e-9` and `3.8e-8`, so the sizes are comparable at equal accuracy.
The patched inputs cost more still: 67368, 163644, 234952 and 432428 parameters against 103236,
162444, 223324 and 324856 for the two global trains they were split out of. That is a result
about the instance rather than about the implementation: a 2D Gaussian mixture at constant
density is smooth everywhere, so it has no hard region for the patching to isolate, and
subdividing a function that a single train already represents well only pays the per-patch
overhead. What patching buys is bounded rank, 64 here against a global `chi_in` that reaches
142, and the case shows the price of that bound. Wall times are the other half of the picture:
the patched arms cost 2.5, 12.8, 6.5 and 36.0 s for `patched_fit_treetn`, 2.9, 7.9, 8.9 and
22.2 s for `patched_naive` and 3.8, 9.7, 17.2 and 40.1 s for `patched_aci`, against 9.2, 16.4,
20.5 and 31.9 s for global `fit_treetn` and 0.47, 1.44, 1.50 and 2.00 s for global `aci`. So
two of the three patched arms are cheaper than an uncapped global fit at the top of the sweep
and all three lose to global `aci` by one to two orders of magnitude everywhere, and no engine
ordering among them is stable: which of the three is cheapest changes with `N`, since it
depends on how the splitting happened to size the patches. The three arms land on the same
error and the same parameter count to every
reported digit, which is expected here: nothing binds, so each engine computes the same product
to the same tolerance and the final budgeting truncates all of them identically. At equal
accuracy the arms differ in wall time, not in what they return.

Per-patch rank cap. On the `norm` path the cap decides the entire patch structure, and it
binds hard everywhere the splitting runs at all. Measured in one paired run at `N` = 32 with
the `patched_aci` arm, so that the two halves are comparable with each other rather than with
the sweep's own row, the default cap of 64 leaves 6 and 7 input patches whose largest bond is
62, holding 234952 parameters and built in 41 s, and gives a product of 6 patches and 129276
parameters
formed in 15.1 s, all of which the later `b160bb7` sweep reproduces (234952 parameters, 39.4 s,
6 patches, 129276 parameters, 17.2 s). Halving the cap to 32 leaves 59 and 67 input patches whose largest bond is
31, holding 540984 parameters and built in 68 s, and gives a product of 72 patches and 312840
parameters formed in 2.2 s, at the same accuracy, `4.1e-8` against `4.3e-8`. So the cap trades
representation size and construction time against the cost of the per-patch products: more and
smaller patches cost more parameters and a longer build, and make each product much cheaper,
since the work in a patch pair grows like its rank cubed. The default is 64 because this case
is written for large `N`, where a cap of 32 would multiply the patch count by ten at every
point. Where the cap does not bind at all the construction returns a single patch: at `N` = 8
the volume-budget truncation already brings the global train to bond 63, under the cap, so the
`norm` path splits nothing and the patched arms are the global product with the patch
bookkeeping around it, which is the honest lower end of the sweep rather than a failure.
`BENCH_PATCH_MAX_ITER` applies only to the `tci` path; raising it from the upstream 20 to 200
changed neither its patch counts nor its patch ranks, at six times the build cost, so the
iteration limit does not drive that splitting either.

The default sweep depends on the family. On `aniso` it is `N` = 8, 16, 32, 64, 128, 256 and 512,
which runs in 178 s: that family is far cheaper to build per point, since its grid is coarser at
equal `N`, `R` = 9 at `N` = 512 against `R` = 15 for the smooth family at the same count. On
`smooth` it is `N` = 8, 16, 32 and 64, the four points of case 4, so the two cases can be
read against each other over the same factor of eight in `N`. Either way the runner does one
timed pass per arm
rather than the median of three the other cases take: `BENCH_RUNS` defaults to 1 here because
the arms cost tens of seconds each at the top of the smooth sweep while their run to run spread
is
about three percent (2.67, 2.72 and 2.77 s for three passes of `patched_fit_treetn` at
`N` = 8), which is smaller than the spread between two constructions of the same instance,
whose input TCI is not bit-reproducible. On the smooth family `N` = 128 is left out on cost
alone and not because
anything blocks it: probed at the defaults at the pin `1b9a517` it completed in 379 s, of which
297 s is the input
construction, with 28 and 28 input patches, 40 output patches of 520588 parameters, arm times
of 22.4 s for `patched_fit_treetn`, 23.3 s for `patched_naive` and 36.1 s for `patched_aci`,
and a relative error of `1.1e-8`. Only the `patched_aci` time of that probe is pin-sensitive.
That one point costs as much as the whole default sweep, so
it stays one `BENCH_NS` away.
Runner: [`src/bin/elementwise_gauss2d_patched.rs`](src/bin/elementwise_gauss2d_patched.rs),
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
(known issue 10). `run.yaml` also records the thread count the sweep ran with, and profiles
taken before the runner pinned `RAYON_NUM_THREADS=1` were recorded multi-threaded, which is
one more reason not to compare timings across profiles.

`mac-cpu`, the maintainer's Mac, at the full default sweeps, run
single-threaded (`threads: 1`), and the
only profile that carries case 4. It was taken at the pin `1b9a517`, two bumps back, and so
predates the current one: it is kept as that machine's record of the sweep at that revision,
and it is
due a rerun. Spot-checked at the current pin, everything the two bumps changed for cases 1 to 4
is
in the `aci` wall time column and nothing else: the case-3 default sweep reproduces the
committed `chi_out` = `chi_in` at every `R` and the committed errors to a digit or two, the
case-1 arms reproduce their errors and `chi_out` including the `zipup` arm's `3.9e-6` at
`K` = 128 against its `1e-5` gate, and the case-1 `aci` arm is more than an order of magnitude
slower for the same answer, the global pivot guard's floor of known issue 12. It also predates case 5,
so no profile carries case-5
records
yet and the case-5 numbers quoted above come from runs on that same machine rather than from a
committed sweep: both families from one full default sweep each at `b160bb7`, whose `aci` arm
times are stale at the current pin (known issue 12), and the two
`N` beyond the defaults from probes at `1b9a517`. The quoted numbers in the case
descriptions above come
from this profile:

- [`result/mac-cpu/elementwise_fourier.md`](result/mac-cpu/elementwise_fourier.md)
- [`result/mac-cpu/mpo_mpo_quantics.md`](result/mac-cpu/mpo_mpo_quantics.md)
- [`result/mac-cpu/elementwise_gauss2d.md`](result/mac-cpu/elementwise_gauss2d.md)
- [`result/mac-cpu/elementwise_gauss2d_scaling.md`](result/mac-cpu/elementwise_gauss2d_scaling.md)

The scaling plots sit next to these files. This sweep is current with the `chi_out`-driven
truncation semantics of the fixed-budget cases, so every `chi_out` column sits at the budget
and every arm column is directly comparable at one revision, `1b9a517`.

`mac-m1-8gb`, an 8 GB Apple M1 MacBook Pro, contributed by a collaborator and kept exactly
as it was committed, at the revision recorded in its own `run.yaml`. It predates the
`chi_out`-driven truncation semantics and case 4, so its fixed-budget arms were still
tolerance-driven and some report a `chi_out` below the budget. It also predates the
single-threaded default and was recorded multi-threaded (`threads: default` in its
`run.yaml`), so none of its wall times are comparable with the single-threaded `mac-cpu`
ones. Read it as that machine's
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

This builds in release mode, runs all five cases with their default sweeps into
`result/<profile>/raw/`, writes `result/<profile>/run.yaml`, and renders the Markdown
reports and SVG plots. `run.yaml` deliberately records no hostname, only a machine
label (`BENCH_MACHINE`, defaulting to the profile name) plus the chip and memory size,
since a hostname on a public repository can leak the operator's institution and
location. It also stamps `repo_rev` with a `-dirty` suffix when the source tree carries
uncommitted changes, so a sweep can never claim a clean revision it did not come from;
everything under `result/` is excluded from that check, since the script has just
rewritten it. The script pins `RAYON_NUM_THREADS=1` by default so wall times do not
depend on the machine's core count or background load; export `RAYON_NUM_THREADS`
yourself to run multi-threaded, and the value used is recorded in `run.yaml`. On a
machine without `uv`, point `REPORT_PYTHON` at any python that has matplotlib and
numpy.

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
BENCH_NS=8 BENCH_RUNS=1 BENCH_WARMUPS=0 BENCH_SANITY=1e-1 \
  OUT_DIR=/tmp/smoke cargo run --release --bin elementwise_gauss2d_patched
BENCH_FAMILY=smooth BENCH_NS=8 BENCH_R0=8 BENCH_RUNS=1 BENCH_WARMUPS=0 BENCH_SANITY=1e-1 \
  OUT_DIR=/tmp/smoke cargo run --release --bin elementwise_gauss2d_patched
```

Cases 4 and 5 export no HDF5 instances and have no Julia check of their own. Their instances
are the same kind of object as case 3's, a pair of fused site-dimension-4 quantics tensor
trains of Gaussian mixtures, so what an export would verify is already verified by the
case-3 check at `N` = 8, which is the same instance by construction for case 5's `smooth`
family. Case 5 additionally
checks itself against the global representation from inside the crate: a unit test forms the
same product both ways at `R` = 6 and requires the two to agree, which is the statement that
the patching changes the representation and not the function. Its `aniso` family has no case-3
counterpart and is covered by its own unit tests instead: the drawn quadratic forms against a
hand-computed rotated one, and a four-engine patched product at `R` = 6 against the exact
pointwise product.

Sanity gates: every runner is self-checking. A runner exits nonzero if any algorithm's
measured error exceeds its gate. Case 1 uses `1e3 * BENCH_TOL` for `naive`, `zipup` and
`aci`, and a looser `1e-2` for `fit` (truncation is norm-relative and the TT norm grows
like `2^(R/2)`, so the pointwise error is not bounded by the tolerance itself). At the top
of the case-1 sweep the `zipup` arm comes within a factor of three of its gate, `3.9e-6`
against `1e-5` at `K` = 128, so extending `BENCH_KS` beyond 128 is likely to trip it; that
would be the gate reporting the growth of single-pass truncation error with the mode count,
not a regression. Case 2
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
only expensive arm: 0.02 s at `R` = 6, 1.1 s at `R` = 8, 32 s at `R` = 10 and 89 to 91 s
at `R` = 12 and 14. Every other arm
stays under a second across that range. At `R` >= 10 the naive intermediates are large
enough that the same points vary with the machine and with ambient memory pressure
(known issue 10). Every algorithm truncates back to
the same
output budget `chi_out <= chi_in`, so the arms differ in accuracy at equal budget rather
than in how far their ranks are allowed to grow. The default sweep (`R` = 6, 8, 10, 12, 14
with 3 timed runs) is dominated by the naive runs at `R` = 12 and 14 and takes about eleven
minutes single-threaded on the maintainer's Mac, and is by far the most expensive of the
four cases. For the heavy tail, extend explicitly,
for
example `BENCH_RS=6,8,10,12,14,16 BENCH_RUNS=5`. Restrict `BENCH_ALGOS`, dropping `naive`,
when you only want a quick signal.

Case 3 has the same shape and the same expensive arm, at its own scale: naive costs 0.05 s
at `R` = 6, 3.9 s at `R` = 8 and 6 to 7.6 s at `R` = 10 to 14, and every other arm stays
under one and a half seconds. Its default sweep takes about a minute and a half. Its `aci`
arm is the cheapest of the
four at every `R`, 1 ms at `R` = 6 rising only to 61 ms at `R` = 14, because the pinned
revision includes the ACI rank-saturation early
exit of [tensor4all-rs#591](https://github.com/tensor4all/tensor4all-rs/pull/591), so the
unreachable stopping criterion of the fixed-budget cases no longer makes it run to its
iteration limit.

Case 5's cost sits in two different places, and on both families the input construction is the
larger one. The default `norm` path builds one global train per input and then splits it, and the
splitting is the expensive half. On the default aniso family the whole construction costs 0.06 s
at `N` = 8, 8.7 s at 128, 20.5 s at 256 and 95.0 s at 512, of which the two global trains are
0.03 s and 2.3 s at the ends, so from `N` = 128 up the splitting is over 95 percent of it. On the
smooth family it is worse per point: at `N` = 64 the two global trains build in 3.5 s and the
whole construction takes 106 s, against 1.3 s at `N` = 8, 17.7 s at 16 and 39.4 s at 32. That
happens once per `N` rather than once per timed run. The patched arms themselves are cheap on the
aniso family, 0.04 to 0.13 s per pass at `N` = 8 and 4.3 to 7.0 s at 512, and expensive on the
smooth one, 2.5 to 3.8 s at `N` = 8 and 22.2 to 40.1 s at 64, so arm cost scales with
`BENCH_RUNS` while the construction does not. The global `fit_treetn` baseline is the most
expensive arm of the smooth sweep, 9.2 s per pass at `N` = 8 rising to 31.9 s at 64, and a
moderate one on the aniso sweep, 0.03 s at `N` = 8 rising to 12.5 s at 512 and 25.1 s at 1024,
since a tolerance-driven fit has no cap to stop it early; both stay under the 120 s per point
that sets the baseline ceilings. The excluded `patched_zipup_treetn` arm is in a class of its own
at 98 s per pass at `N` = 8 of the smooth family. The default aniso sweep takes 178 s with one
timed pass per arm, and the default smooth sweep about seven minutes (measured 418 s). Adding the
next point doubles either: `N` = 1024 of the aniso family costs 302 s, 242 s of it construction,
and `N` = 128 of the smooth family 379 s, 297 s of it construction, so extend explicitly rather
than by default.

Case 4 costs about as much as case 3, but at the pinned revision the quantics TCI
construction of
its inputs is no longer what dominates: at `N` = 64 the two input trains build in a few
seconds, against about 10 s for one timed pass of the three arms (2.5 s for `zipup_treetn`,
7.2 s for `fit_treetn`, 0.2 s for `aci`). Arm cost therefore scales with `BENCH_RUNS` while
the construction runs once per `N`. The default sweep `N` = 8, 16, 32, 64 takes about a minute;
`N` = 128 is left out because the cost roughly doubles again per step and the
fitted exponent is already stable over the factor of 8 in the defaults. All five cases plus
the reports finish in about a quarter of an hour on the maintainer's Mac (measured 823 s for the
first four plus 178 s for case 5 at its default aniso family, or 418 s if case 5 is run on the
smooth family instead),
on top of the
release build, with case 2 accounting for most of it. That figure is for the
single-threaded default, `RAYON_NUM_THREADS=1`, which the runner pins so that wall times do
not depend on the core count or on background load; export `RAYON_NUM_THREADS` yourself to
run multi-threaded and the sweep is correspondingly faster. On a slower or more
memory-constrained machine the same defaults cost
substantially more, most of the difference sitting in the two naive arms at `R` = 10 to 14.

Environment knobs:

| Variable | Applies to | Default | Meaning |
| --- | --- | --- | --- |
| `BENCH_KS` | case 1 | `4,8,16,32,64,128` | comma-separated Fourier mode counts `K` to sweep |
| `BENCH_R` | case 1 | `20` | number of quantics bits |
| `BENCH_RS` | cases 2 and 3 | `6,8,10,12,14` | comma-separated bits per variable `R` to sweep |
| `BENCH_NGAUSS` | cases 2 and 3 | `8` | number of Gaussians per mixture |
| `BENCH_BOX_L` | cases 2 and 3 | `6.0` | half-width `L` of the box `[-L, L]` |
| `BENCH_NS` | cases 4 and 5 | `8,16,32,64` (case 4, and case 5 on the `smooth` family), `8,16,32,64,128,256,512` (case 5 on the default `aniso` family) | comma-separated Gaussian or spike counts `N` to sweep. Both cases derive `L` and `R` from `N`, so they ignore `BENCH_NGAUSS`, `BENCH_BOX_L` and `BENCH_RS` |
| `BENCH_FAMILY` | case 5 | `aniso` | which instance family to sweep. `aniso` is `N` anisotropic narrow spikes at a fixed spacing-to-width ratio, the family the case is written for; `smooth` is case 4's isotropic Gaussians at constant density, unchanged, and it also switches `BENCH_NS` to case 4's four points. Recorded in the params of every record as `family`, and part of every record's filename, so one profile can hold both |
| `BENCH_ANISO_SIGMA` | case 5, `aniso` family | `0.05` | minor width of every spike, in box units. This is the length the grid has to resolve: `R` is chosen so a grid step is at most a quarter of it |
| `BENCH_ANISO_RHO_MAX` | case 5, `aniso` family | `8.0` | upper end of the aspect ratio, drawn log-uniform in `[1, BENCH_ANISO_RHO_MAX]` per spike. Setting it to `1` is legal and is the isotropic control: circular spikes of one common shape, everything else unchanged |
| `BENCH_ANISO_SPACING` | case 5, `aniso` family | `3.0` | mean spacing between spikes, in minor widths, held fixed as `N` grows. It fixes both the box, `L = BENCH_ANISO_SPACING * sigma * sqrt(N) / 2`, and the overlap of the two mixtures, which is what keeps the product non degenerate |
| `BENCH_L0` | cases 4 and 5 (`smooth` family) | `6.0` | reference box half-width, the `L` at `N` = `BENCH_N0` |
| `BENCH_N0` | cases 4 and 5 (`smooth` family) | `8` | reference Gaussian count, the `N` at which `L` = `BENCH_L0` and `R` = `BENCH_R0` |
| `BENCH_R0` | cases 4 and 5 (`smooth` family) | `10` | reference bits per variable, the `R` at `N` = `BENCH_N0`. Lower it for a cheap probe of the whole sweep |
| `BENCH_RTOL` | case 5 | `1e-8` | the one accuracy knob of case 5: the absolute-per-patch input TCI tolerance of the patched arms, the tolerance of the global baseline inputs, the absolute pivot budget of both `aci` arms, and the global output budget of both products. Every arm is compared at this one value |
| `BENCH_PATCH_MAX_BOND` | case 5 | `64` | per-patch rank cap of the patched input construction. A subdomain that does not fit under it is split again, so this is what decides how deep the patch tree goes: a smaller value means more, smaller patches |
| `BENCH_PATCH_INPUT` | case 5 | `norm` | which construction produces the patched inputs. `norm` builds one global train per input, exactly as case 4 does, and splits it with `partitionedtt::add_with_patching`, whose decisions come from Frobenius norms alone. `tci` runs `partitionedtt::adaptiveinterpolate` instead, which never forms a global train and splits a patch whose own TCI does not converge under the cap; it is the construction the case is eventually written for, it works at the current pin (known issue 11 records the upstream fix), and it is not the default because for the same cap it splits far harder and returns a much larger representation. Recorded as `input_path` |
| `BENCH_PATCH_SPLIT` | case 5 | `gain` | how the `norm` path picks the site to split. `gain` is the upstream `ExactParameterGain`: it forms and budget-truncates the children of every candidate site and keeps the cheapest. `sequential` takes the first unprojected site of the patch order instead, so the splitting runs strictly coarse to fine and a patch is a single quadrant. Ignored by the `tci` path, whose splitting is sequential by construction. Recorded as `split_strategy` |
| `BENCH_PATCH_MAX_ITER` | case 5, `tci` path | `20` | half-sweep limit of each patch's own TCI run on the `tci` path, the upstream default. It exists as a knob because a run that stops at its limit is not converged and its patch is split whatever its rank was, which would make the patch tree a measurement of the limit; measured, it does not bind, and raising it to 200 changed neither the patch counts nor the patch ranks at six times the build cost. Ignored by the default `norm` path, which runs no TCI, and recorded as `patch_max_iter` either way |
| `BENCH_PATCH_TRACE` | case 5 | unset | set it to any value to print one line per contracted patch pair and one for the final budgeting, for cost-breakdown sessions. It changes nothing that is measured and appears in no record; the per-record breakdown is `n_pairs`, `pairs_secs` and `truncate_secs` |
| `BENCH_PATCH_OUT_MAX_BOND` | case 5 | `BENCH_PATCH_MAX_BOND` squared | rank cap of the per-patch product and of the budgeted output. Left non-binding by default, at the exact product rank of two capped patches, so that `BENCH_RTOL` is the only thing that truncates; the runner fails rather than reports if it binds |
| `BENCH_BASELINES` | case 5 | `1` | whether to also measure the two global arms at the same `rtol`. Each has its own `N` ceiling whatever this says, since they are two cost classes: on the `smooth` family both stop at `N` = 64, and on `aniso` the interpolating `aci` arm runs everywhere while the uncapped global fit stops at `N` = 1024, the last point measured to stay under two minutes for it |
| `BENCH_ALPHA_LO` | cases 2, 3, 4 and 5 (`smooth` family) | `0.5` | lower bound of the Gaussian width parameter |
| `BENCH_ALPHA_HI` | cases 2, 3, 4 and 5 (`smooth` family) | `8.0` | upper bound of the Gaussian width parameter |
| `BENCH_SANITY` | cases 2, 3, 4 and 5 | `1e-2` | relative error gate. Cases 2 and 5 apply it to every algorithm; cases 3 and 4 apply it to all but `zipup_treetn`, which is gated at a hardcoded `5.0` |
| `BENCH_TOL` | cases 1, 2, 3 and 4 | `1e-8` | instance tolerance. In case 1 it is the working tolerance of the whole case, both the input compression and the product. In cases 2, 3 and 4 it is scoped to the input TCI construction only, where it fixes `chi_in` and therefore the instance; the arms take `BENCH_CONTRACT_TOL` instead. It is what each record's top-level `tolerance` field and the Julia-check metadata report, since both describe the inputs. Case 5 has no separate instance tolerance: `BENCH_RTOL` is its one knob and is what its records report |
| `BENCH_CONTRACT_TOL` | cases 2, 3 and 4 | `1e-15` | truncation tolerance handed to every contraction or product arm, recorded as `contract_tol` in the params of each record. At the default it never fires, so the rank cap `chi_in` is the only binding truncation control and the fixed-budget cases are `chi_out`-driven by construction. Raise it if you want a tolerance-driven variant of those cases |
| `BENCH_MAX_BOND` | all | `4096` (case 1), `512` (cases 2, 3, 4 and 5) | bond dimension cap. In cases 2, 3 and 4 it caps only the input TCI construction, since the arms themselves run at the fixed output budget `chi_in`. Case 4 fails rather than reports if an instance reaches it, since `chi_in` is what that case measures. In case 5 it caps the global input trains, which are the global baselines' inputs and, on the default `norm` path, also the trains the patches are split out of; the patched counterpart is `BENCH_PATCH_MAX_BOND`. Reaching it is likewise a failure, since a cap-limited train is no longer tolerance-driven |
| `BENCH_RUNS` | all | `5` (case 1), `3` (cases 2, 3 and 4), `1` (case 5) | timed repetitions, the median is reported. Case 5 defaults to a single pass because it is the most expensive case per point while its run to run spread is a few percent, smaller than the spread between two constructions of the same instance |
| `BENCH_WARMUPS` | all | `1` (case 1), `0` (cases 2, 3, 4 and 5) | untimed warmup repetitions |
| `BENCH_SEED` | all | `0` | base seed for instance generation |
| `BENCH_ALGOS` | all | `naive,zipup,fit,aci` (case 1), `naive,zipup_simplett,zipup_treetn,fit_treetn` (case 2), `naive,zipup_treetn,fit_treetn,aci` (case 3), `zipup_treetn,fit_treetn,aci` (case 4), `patched_fit_treetn,patched_naive,patched_aci` (case 5, whose fourth arm `patched_zipup_treetn` is excluded from the defaults on cost and the global baselines are controlled by `BENCH_BASELINES` instead) | comma-separated algorithms to run |
| `OUT_DIR` | all | `result/dev/raw` | directory for the `RunRecord` JSON files |
| `EXPORT_HDF5` | cases 1, 2 and 3 | unset | directory for ITensors-compatible HDF5 instance dumps, plus their JSON metadata. Set it to enable the Julia checks. An empty value counts as unset. Cases 2 and 3 use the same file names, so give them separate directories. Cases 4 and 5 export nothing and ignore it |

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
   matches the variational fit. All three are contained in the current pin `c9ecb7f`, as they
   were in the earlier pins `7cfec22`, where they first landed, `1b9a517`, which added
   [tensor4all-rs#575](https://github.com/tensor4all/tensor4all-rs/pull/575), a treetci
   convergence fix that stops input TCI construction early once the rank saturates at
   `max_bond_dim`, and `b160bb7`. Earlier numbers in this repository's git history predate these fixes and
   are not comparable.
7. **simplett has no elementwise product for tensor trains at the pinned revision.** It
   offers MPO-MPO contraction (`contract_naive`, `contract_zipup`, the stubbed
   `contract_fit`) but nothing that forms a Hadamard product of two tensor trains, so cases
   1 and 3 have no simplett arm and case 3 cannot put two engines on one algorithm the way
   case 2 does with its pair of zipup arms. Its `naive` arm is therefore written in this
   repository, as a core-wise bond Kronecker product plus an SVD sweep on simplett
   primitives, and is recorded with `engine` = `local` to keep that visible.
8. **Case-3 `zipup_treetn` has no correct digits at the fixed output budget.** It returns a
   relative error of order one, between `1.3e-1` and `1.1` depending on `R`, on `N` and on the
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
   at the current pin `c9ecb7f`, and the fresh sweep confirms the fix: `aci` is now the
   cheapest arm at every point of cases 3 and 4, six to fourteen times below `zipup_treetn`,
   with the same errors as before. Numbers quoted in this repository's git history from before
   the bump are not comparable on the `aci` column. The `mac-m1-8gb` profile predates this
   pin, so its `aci` column still shows the pre-fix cost.
10. **`naive` wall times at `R` >= 10 are machine bound.** The naive arms form intermediates
    of bond `chi_in` squared, which at chi around 80 press against the free memory of a
    smaller machine, so their wall time depends on the hardware and on ambient memory
    pressure rather than on the algorithm alone. The clearest evidence is a comparison on one
    machine: on the 8 GB M1 of the `mac-m1-8gb` profile, both runs multi-threaded, the case-2
    point at `R` = 10 measured about 28 s per run on a session with swap nearly
    full and 16 s on the same machine right after a reboot, same code, same errors, same
    `chi_out`; the committed sweep is the post-reboot one, taken on an otherwise idle
    machine, where the spread across the three timed runs stays within about 5 percent at
    `R` = 12 and 14 (22 percent at `R` = 10, whose first run pays the page-in). The
    `mac-cpu` profile measures the same `R` = 10 point at about 32 s, but that number is
    single-threaded on different hardware, so it says nothing about the two figures above.
    The cheap arms differ far
    less. Errors and bond dimensions are unaffected everywhere, since the computation is
    the same arithmetic either way. So compare naive timings only within one profile, run
    official sweeps on an idle machine, and read cross-profile time ratios as hardware
    statements. This is also why profiles are per machine and why `run.yaml` records the
    chip and memory.

11. **Resolved upstream: the TCI-driven input construction of case 5 no longer trips a
    bond mismatch.** With `BENCH_PATCH_INPUT=tci` the construction of a patched input used
    to fail with `Tensor train error: Dimension mismatch: tensor at site k has incompatible
    dimensions`, raised where `adaptiveinterpolate` turns an accepted patch's TCI into a
    tensor train, so the TCI2 site tensors of that patch did not agree on their shared bond.
    It was deterministic and instance specific, and it stopped the `tci` path from `N` = 32
    up on the default smooth instance, which is why `norm` became the default construction.
    [tensor4all-rs#602](https://github.com/tensor4all/tensor4all-rs/pull/602) fixed it
    upstream (issue
    [#598](https://github.com/tensor4all/tensor4all-rs/issues/598)) and the fix is included
    at the current pin `c9ecb7f`. Verified at the previously failing point and beyond: the
    `tci` path now completes at `N` = 32 and 64 of the smooth family and at `N` = 64 of the
    aniso family. `norm` nonetheless stays the default, on the measurement rather than on the
    defect: the `tci` path splits far harder for the same cap, 514 and 622 input patches at
    smooth `N` = 32 against 6 and 7 for `norm`, so it holds 1.04e6 input parameters against
    234952 and returns a product of 572436 parameters against 129276 at the same accuracy.
    Which of the two constructions should be the default is therefore a separate decision
    from this fix, and it is not made here. One caveat if you run it: the per-patch `aci`
    engine loses accuracy on the many tiny patches that path produces (`2.9e-5` at smooth
    `N` = 32 and `4.3e-4` at 64, against `1.7e-8` and `2.2e-8` for the other three engines),
    so read `patched_aci` on the `tci` path with suspicion.

12. **Resolved upstream: the ACI global pivot guard no longer dominates the wall time of a
    many-site instance.** `AciOptions` gained a global pivot guard in
    [tensor4all-rs#610](https://github.com/tensor4all/tensor4all-rs/pull/610), on by default,
    which runs several floating-zone walks over the index space after every sweep and only
    accepts convergence once they find nothing. It buys robustness against a feature that no
    bond-local error estimate can see, and its cost grows with the site count rather than with
    the rank. As first written the guard rebuilt its evaluation cache and recompiled an einsum
    per call, which at the pin `b160bb7` cost the case-1 `aci` arm, on `R` = 20 sites, 0.9 to
    1.3 s per pass against 0.4 to 1.2 ms at the pre-guard pin `1b9a517`, three orders of
    magnitude for the same answer.
    [tensor4all-rs#621](https://github.com/tensor4all/tensor4all-rs/pull/621) (issue
    [#620](https://github.com/tensor4all/tensor4all-rs/issues/620)) fixed that at the current
    pin `c9ecb7f`: the guard now reuses one `TTCache` across sweeps, keeps a single solution
    cache per invocation, and applies its mat-vec inline instead of recompiling an einsum per
    call. Re-measured at `c9ecb7f`, the case-1 `aci` arm costs 14 to 21 ms per pass over the
    default `K` sweep, with ranks and errors unchanged and `chi_out` of 9, 15, 16, 22, 29 and
    32. What remains is a floor of roughly 15 ms per pass on a many-site instance, which is
    the guard doing the work it exists for and is the safety contract of #610 rather than a
    defect: the case-1 arm is still an order of magnitude above its pre-guard cost and will
    stay there while the guard is on. Cases 3 and 4, on 6 to 14 fused sites, never moved
    measurably. The `aci` columns of the committed `mac-cpu` sweep were taken at the pre-guard
    pin `1b9a517`, so case 1's is low by that floor until that machine reruns; anything quoted
    in this repository's git history from the `b160bb7` window is stale on the `aci` column in
    the other direction.

13. **The case-5 size verdict against global `aci` depends on how that baseline's tolerance is
    read.** Case 5 runs both its `aci` paths on an absolute pivot budget, `BENCH_RTOL` against
    an output of order one, because its accuracy target is one global relative error and the
    upstream default since
    [tensor4all-rs#619](https://github.com/tensor4all/tensor4all-rs/pull/619) normalizes per
    bond instead. That choice is forced: measured on the smooth family, the scale-relative
    variant of the same baseline runs its rank to `BENCH_MAX_BOND` and returns `2.7e-1` at
    `N` = 32 and `2.8e-1` at 64, which fails the sanity gate, and the per-patch `aci` engine
    loses four to six orders of magnitude of accuracy against the other three engines on the
    same instances. But on the aniso family the scale-relative variant is both accurate and
    smaller than the absolute one, 205008 parameters against 312528 at `N` = 512 at `5.4e-9`,
    below the patched product's 294680. So the six percent size margin the default sweep
    reports for patching over global `aci` at the top of the aniso sweep is not robust to this
    knob, and the README verdict says so. What does not depend on it is the comparison against
    the global variational fit, which is the smallest representation on both families, and the
    wall time ordering, where the patched arms beat that fit from `N` = 128 up. A tolerance
    contract that is per-region for the pivot search and global for the reported error is the
    real gap, and closing it upstream is what would make this margin a measurement.

## License

MIT, see [`LICENSE`](LICENSE).
