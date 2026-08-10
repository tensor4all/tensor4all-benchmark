//! Case 4 runner: density-constant scaling study of case 3.
//!
//! Case 3 sweeps the bit count `R` at a fixed mixture and watches the quantics
//! rank saturate. This case asks the complementary question: how does the input
//! rank `chi_in` of a 2D Gaussian mixture grow with the number of Gaussians `N`
//! when the DENSITY of Gaussians is held constant? The two hypotheses worth
//! separating are `chi ~ sqrt(N)`, which is what a boundary-law or
//! one-dimensional-cut picture predicts for a fused 2D quantics train, and
//! `chi ~ N`, which is what a naive sum-of-terms picture predicts.
//!
//! Density-constant construction. For each `N` in `BENCH_NS` the box half-width
//! grows as
//!
//! ```text
//! L = L0 * sqrt(N / N0)
//! ```
//!
//! so the box area `(2L)^2` grows proportionally to `N` and the number of
//! Gaussians per unit area is the same at every point of the sweep. Growing the
//! box while keeping `R` fixed would coarsen the grid and under-resolve each
//! Gaussian, which would confound rank growth with a loss of resolution, so the
//! bit count grows with the box:
//!
//! ```text
//! R = R0 + round(log2(L / L0))
//! ```
//!
//! that is, one extra bit per doubling of the box, which keeps the grid spacing
//! `2L / 2^R` roughly constant, so every Gaussian is resolved by roughly the
//! same number of grid points at every `N`. The rounding means `R` moves in
//! integer steps while `L` moves continuously, so the spacing is only constant
//! up to a factor of at most `sqrt(2)`.
//!
//! Everything else mirrors case 3: two independent mixtures from
//! `GaussianMixture2D::random` at seeds `seed + 1` and `seed + 2`, fused 2D
//! quantics trains via `to_quantics_fused_tt`, and the elementwise product
//! `h = f * g` at the fixed output budget `chi_out <= chi_in`, judged by the
//! sampled max relative error against the exact pointwise product.
//!
//! As in cases 2 and 3, that budget is decided by the rank cap alone.
//! `BENCH_TOL` (default 1e-8) scopes only the input TCI construction, so it
//! defines `chi_in` and the instance, while `BENCH_CONTRACT_TOL` (default 1e-15)
//! is the truncation tolerance every arm receives and is recorded as
//! `contract_tol`. At 1e-15 it never fires, so the arms exhaust the budget unless
//! their exact rank is smaller. The `aci` arm additionally runs with
//! `scale_tolerance` enabled (`AciTolerance::ScaleRelative`, recorded as
//! `aci_scale_tolerance`) so that its pivot criterion is scale-relative and
//! equally unreachable; at the pinned rev that means it always runs to
//! `max_iters` when capped, since the saturation early exit lands in
//! tensor4all-rs#591, which is not merged there, so the arm is slow until that
//! change is picked up.
//!
//! Arms: `zipup_treetn`, `fit_treetn` and `aci` only. The `naive` arm of case 3
//! is excluded here: it forms the full `chi_in`-squared bond before truncating,
//! and this case deliberately pushes `chi_in` to roughly twice the case-3 value,
//! where that arm dominates the wall time of the whole sweep without adding a
//! separate conclusion (it tracks `fit_treetn` to the last reported digit in
//! case 3). Run it explicitly through `BENCH_ALGOS` if you want it.
//!
//! As in case 3, `zipup_treetn` is expected to return an order-one relative
//! error at this budget and is gated only against a scale blow-up.

use std::path::PathBuf;
use t4a_bench::elementwise::{
    check_mixture_product_not_degenerate, elementwise_product, max_rel_error_vs_mixture_product,
    AciTolerance, ElementwiseAlgo, FIT_NFULLSWEEPS,
};
use t4a_bench::gaussian::{to_quantics_fused_tt, GaussianMixture2D};
use t4a_bench::harness::time_median;
use t4a_bench::record::{write_record, RunRecord, SCHEMA_VERSION};
use tensor4all_simplett::AbstractTensorTrain;

const CASE: &str = "elementwise_gauss2d_scaling";

/// Sanity gate for the `zipup_treetn` arm, which cannot share `BENCH_SANITY`.
///
/// Same reasoning as case 3: at an output budget of `chi_in` the single-pass
/// zip-up truncation of an elementwise product returns a relative error of
/// order one, so a gate that screens order-unity wrongness has nothing to grip.
/// This arm is therefore gated only against a gross scale blow-up or a
/// non-finite result.
const ZIPUP_SANITY: f64 = 5.0;

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn parse_algo(s: &str) -> ElementwiseAlgo {
    match s {
        "naive" => ElementwiseAlgo::Naive,
        "zipup_treetn" => ElementwiseAlgo::Zipup,
        "fit_treetn" => ElementwiseAlgo::Fit,
        "aci" => ElementwiseAlgo::Aci,
        other => panic!("unknown algorithm {other}"),
    }
}

/// Box half-width and bit count for `n` Gaussians at constant density.
///
/// `L` grows like `sqrt(N)` so the area grows like `N`, and `R` gains one bit
/// per doubling of `L` so the grid spacing stays roughly constant. Exposed as a
/// function so the test and the runner cannot drift apart.
pub fn box_and_bits(n: usize, n0: usize, l0: f64, r0: usize) -> (f64, usize) {
    let l = l0 * (n as f64 / n0 as f64).sqrt();
    let r = r0 as f64 + (l / l0).log2().round();
    (l, r.max(1.0) as usize)
}

fn main() -> anyhow::Result<()> {
    let ns: Vec<usize> = std::env::var("BENCH_NS")
        .unwrap_or_else(|_| "8,16,32,64".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let l0: f64 = env_or("BENCH_L0", 6.0);
    let n0: usize = env_or("BENCH_N0", 8);
    let r0: usize = env_or("BENCH_R0", 10);
    let alpha_lo: f64 = env_or("BENCH_ALPHA_LO", 0.5);
    let alpha_hi: f64 = env_or("BENCH_ALPHA_HI", 8.0);
    // Instance tolerance: this defines chi_in through the input TCI, and nothing
    // else. It is what `RunRecord::tolerance` reports, since that describes the
    // inputs rather than the product.
    let tol: f64 = env_or("BENCH_TOL", 1e-8);
    // Product tolerance, pinned inert so the rank cap chi_in is the only binding
    // truncation control for every arm.
    let contract_tol: f64 = env_or("BENCH_CONTRACT_TOL", 1e-15);
    let max_bond: usize = env_or("BENCH_MAX_BOND", 512);
    let runs: usize = env_or("BENCH_RUNS", 3);
    let warmups: usize = env_or("BENCH_WARMUPS", 0);
    let seed: u64 = env_or("BENCH_SEED", 0);
    // As in case 3, the output budget is fixed at chi_in, so truncation error is
    // the quantity being measured and the gate only screens order-unity
    // wrongness.
    let sanity: f64 = env_or("BENCH_SANITY", 1e-2);
    let algos: Vec<String> = std::env::var("BENCH_ALGOS")
        .unwrap_or_else(|_| "zipup_treetn,fit_treetn,aci".into())
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    let out_dir =
        PathBuf::from(std::env::var("OUT_DIR").unwrap_or_else(|_| "result/dev/raw".into()));

    // Accuracy check sampling, recorded in every record's params.
    let n_error_samples: usize = 128;
    let error_seed: u64 = seed.wrapping_add(99);

    let mut failures = Vec::new();
    for &n in &ns {
        let (box_l, r) = box_and_bits(n, n0, l0, r0);
        let f = GaussianMixture2D::random(n, box_l, (alpha_lo, alpha_hi), seed.wrapping_add(1));
        let g = GaussianMixture2D::random(n, box_l, (alpha_lo, alpha_hi), seed.wrapping_add(2));

        let (fa, _step) = to_quantics_fused_tt(&f, r, box_l, tol, max_bond)?;
        let (gb, _) = to_quantics_fused_tt(&g, r, box_l, tol, max_bond)?;
        let input_chi = fa.rank().max(gb.rank());
        // Fail fast, before any timing: at constant density the box grows with
        // N, so this is where too narrow a Gaussian would empty the product.
        let scales =
            check_mixture_product_not_degenerate(&f, &g, r, box_l, n_error_samples, error_seed)?;
        eprintln!(
            "n={n} box_l={box_l:.3} r={r} input_chi={input_chi} ref_scale={:.3e}",
            scales.ref_scale
        );
        // The point of the case is how chi_in grows, so a chi_in pinned at the
        // construction cap would be a measurement of the cap instead.
        if input_chi >= max_bond {
            failures.push(format!(
                "n={n}: input chi {input_chi} reached the construction cap \
                 BENCH_MAX_BOND={max_bond}, so the measured rank is the cap, \
                 not the rank of the function"
            ));
        }

        for algo_name in &algos {
            let algo = parse_algo(algo_name);
            let (h, timing) = time_median(warmups, runs, || {
                elementwise_product(
                    algo,
                    &fa,
                    &gb,
                    contract_tol,
                    input_chi,
                    AciTolerance::ScaleRelative,
                )
                .expect("product failed")
            });
            let max_error =
                max_rel_error_vs_mixture_product(&h, &f, &g, r, box_l, n_error_samples, error_seed);
            let rec = RunRecord {
                schema_version: SCHEMA_VERSION,
                case: CASE.into(),
                algorithm: algo_name.clone(),
                params: serde_json::json!({
                    "n_gauss": n, "box_l": box_l, "r": r,
                    "l0": l0, "n0": n0, "r0": r0,
                    "alpha_range": [alpha_lo, alpha_hi], "max_bond": max_bond,
                    // Output budget shared by every algorithm: the input rank.
                    "contract_max_bond": input_chi,
                    // Truncation tolerance the arms actually ran with, pinned
                    // inert so the cap above is what decides. The top-level
                    // `tolerance` field is the instance tolerance instead.
                    "contract_tol": contract_tol,
                    // True on the aci arm, whose stopping criterion is then
                    // scale-relative, so the inert tolerance above is
                    // unreachable for it too. The other arms are SVD-based and
                    // have no such switch.
                    "aci_scale_tolerance": matches!(algo, ElementwiseAlgo::Aci),
                    "runs": runs, "warmups": warmups,
                    "n_error_samples": n_error_samples, "error_seed": error_seed,
                    "error_metric": "max_rel_vs_analytic",
                    // Scales of the reference and of the two inputs at the same
                    // sampled points, so a degenerate instance is visible in the
                    // record rather than hidden inside the relative error.
                    "ref_scale": scales.ref_scale,
                    "input_scale_f": scales.input_scale_f,
                    "input_scale_g": scales.input_scale_g,
                    // Upstream engine that actually ran this arm.
                    "engine": algo.engine(),
                    // Part of the benchmark definition for the fit arm.
                    "fit_nsweeps": FIT_NFULLSWEEPS,
                }),
                seed,
                tolerance: tol,
                wall_time_median_secs: timing.median_secs,
                wall_times_secs: timing.runs_secs,
                max_error,
                input_max_bond_dim: input_chi,
                output_max_bond_dim: h.rank(),
                output_bond_dims: h.link_dims(),
            };
            write_record(&out_dir, &format!("{CASE}-{algo_name}-n{n}"), &rec)?;
            eprintln!(
                "  {algo_name}: t={:.4}s rel_err={max_error:.2e} chi_out={}",
                timing.median_secs,
                h.rank()
            );
            let gate = if matches!(algo, ElementwiseAlgo::Zipup) {
                ZIPUP_SANITY
            } else {
                sanity
            };
            if !max_error.is_finite() || max_error > gate {
                failures.push(format!(
                    "{algo_name} n={n}: rel err {max_error:.2e} > {gate:.2e}"
                ));
            }
        }
    }
    if !failures.is_empty() {
        anyhow::bail!("sanity failures:\n{}", failures.join("\n"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{box_and_bits, ZIPUP_SANITY};
    use t4a_bench::elementwise::{
        elementwise_product, max_rel_error_vs_mixture_product, AciTolerance, ElementwiseAlgo,
    };
    use t4a_bench::gaussian::{to_quantics_fused_tt, GaussianMixture2D};
    use tensor4all_simplett::AbstractTensorTrain;

    /// The density-constant construction: area proportional to N, grid spacing
    /// constant up to the integer rounding of R.
    #[test]
    fn box_and_bits_hold_density_and_resolution() {
        let (l0, n0, r0) = (6.0, 8, 10);
        let (l8, r8) = box_and_bits(8, n0, l0, r0);
        assert!((l8 - 6.0).abs() < 1e-12);
        assert_eq!(r8, 10);
        // Area per Gaussian is exactly constant, by construction.
        for &n in &[8usize, 16, 32, 64, 128] {
            let (l, r) = box_and_bits(n, n0, l0, r0);
            let area_per = (2.0 * l).powi(2) / n as f64;
            let area_per0 = (2.0 * l8).powi(2) / 8.0;
            assert!(
                (area_per / area_per0 - 1.0).abs() < 1e-12,
                "n={n}: density drifted"
            );
            // Grid spacing constant to within the sqrt(2) the rounding allows.
            let step = 2.0 * l / (1u64 << r) as f64;
            let step0 = 2.0 * l8 / (1u64 << r8) as f64;
            let ratio: f64 = step / step0;
            assert!(
                ratio > 0.7 && ratio < 1.42,
                "n={n}: grid spacing ratio {ratio} outside the rounding band"
            );
        }
    }

    /// The smallest point of the default sweep, at a reduced `R0` to keep the
    /// test cheap: the three default arms must build, honour the fixed output
    /// budget and land inside the gates the runner would apply to them.
    ///
    /// The bounds are the runner's own: `BENCH_SANITY` (1e-2) for `fit_treetn`
    /// and `aci`, and the hardcoded `ZIPUP_SANITY` for `zipup_treetn`, whose
    /// error at this budget is genuinely of order one (README known issue 8).
    /// The tolerance handed to the arms is the runner's inert 1e-15 and the aci
    /// arm runs scale-relative, so the cap is what decides and the SVD-based arms
    /// are expected to spend the whole budget. Measured at the pinned revision on
    /// this instance (chi_in 76): zipup 1.8e-1, fit 6.7e-9 and aci 6.7e-9, all at
    /// chi_out 76.
    /// This is a smoke test of the case-4 wiring, not a precision claim: the
    /// scaling answer comes from the runner, not from here.
    #[test]
    fn scaling_arms_meet_their_gates_at_the_smallest_point() {
        let (l, r) = box_and_bits(8, 8, 6.0, 8);
        assert_eq!((l, r), (6.0, 8));
        let f = GaussianMixture2D::random(8, l, (0.5, 8.0), 1);
        let g = GaussianMixture2D::random(8, l, (0.5, 8.0), 2);
        let (fa, _) = to_quantics_fused_tt(&f, r, l, 1e-8, 512).unwrap();
        let (gb, _) = to_quantics_fused_tt(&g, r, l, 1e-8, 512).unwrap();
        let chi_in = fa.rank().max(gb.rank());

        // The third element says whether the arm is expected to spend the whole
        // budget. The two SVD-based arms keep everything the cap allows, since the
        // tolerance can no longer stop them; aci is interpolation-based and may
        // settle below the cap, so it is only held to the cap as an upper bound.
        for (algo, bound, exhausts_budget) in [
            (ElementwiseAlgo::Zipup, ZIPUP_SANITY, true),
            (ElementwiseAlgo::Fit, 1e-2, true),
            (ElementwiseAlgo::Aci, 1e-2, false),
        ] {
            // The budget is the cap, so the tolerance is pinned inert, exactly as
            // the runner does it.
            let out =
                elementwise_product(algo, &fa, &gb, 1e-15, chi_in, AciTolerance::ScaleRelative)
                    .unwrap();
            assert!(
                out.rank() <= chi_in,
                "{algo:?}: chi_out {} exceeds the budget {chi_in}",
                out.rank()
            );
            if exhausts_budget {
                assert_eq!(
                    out.rank(),
                    chi_in,
                    "{algo:?}: chi_out {} fell short of the budget {chi_in}, so something \
                     other than the cap truncated it",
                    out.rank()
                );
            }
            let err = max_rel_error_vs_mixture_product(&out, &f, &g, r, l, 128, 99);
            println!(
                "{algo:?}: rel err {err:.3e} (bound {bound:.0e}), chi_out {} of {chi_in}",
                out.rank()
            );
            assert!(
                err.is_finite() && err < bound,
                "{algo:?}: rel err {err} exceeds {bound}"
            );
        }
    }
}
