//! Case 3 runner: elementwise (Hadamard) product of two 2D quantics Gaussian
//! mixtures, swept over the number of bits per variable for each algorithm.
//!
//! Two independent mixtures are cross-interpolated into fused 2D quantics
//! tensor trains on `[-L, L)^2`, `r` sites of dimension 4, local index
//! `x_bit + 2 * y_bit` with the most significant bit first. The benchmarked
//! operation is `h = f * g` on the `2^r` by `2^r` grid, whose reference is
//! exact and pointwise: `h(x, y) = f(x, y) * g(x, y)`, with no quadrature and
//! no tail, so unlike case 2 this case has no reference error floor of its own.
//!
//! Fixed output budget, the same philosophy as case 2: every arm runs with its
//! maximum bond dimension capped at `chi_in`, the larger of the two input
//! ranks, so all arms pay the same output budget and the error column is the
//! discriminator. `BENCH_MAX_BOND` caps only the input TCI construction.
//!
//! Arms, selectable with `BENCH_ALGOS`: `naive` (core-wise bond Kronecker
//! product then an SVD sweep, written locally on simplett primitives, so the
//! recorded engine is `local`), `zipup_treetn` and `fit_treetn` (both
//! `tensor4all_treetn::hadamard` through the bridge in `elementwise`), and
//! `aci` (`tensor4all_aci::elementwise`, adaptive cross interpolation of the
//! pointwise product). There is no simplett arm: `tensor4all-simplett` exposes
//! no elementwise product for tensor trains at the pinned revision, so unlike
//! case 2 this case cannot compare the two engines on the same algorithm.
//!
//! `fit_treetn` runs at a fixed sweep count (`elementwise::FIT_NFULLSWEEPS`),
//! shared with case 1 and recorded as `fit_nsweeps`: the fit cost is linear in
//! the sweep count, so its wall time is only comparable at a stated count.
//!
//! What the fixed budget measures, as observed at r = 6, 8, 10 with the pinned
//! revision (chi_in of 53, 76, 77): `naive` and `fit_treetn` agree to the last
//! reported digit at 8.5e-9, 2.3e-8 and 2.5e-8, at the same chi_out of 39, 60
//! and 61, well inside the budget. `aci` matches them or beats them (3.6e-11,
//! 8.4e-9, 1.1e-8) and is by far the cheapest arm, 1 ms to 43 ms, because it
//! never forms the product it is approximating. `zipup_treetn` collapses: it
//! spends the whole budget and still returns 6.2e-1, 4.2e-1 and 5.6e-1, that
//! is, an answer with no correct digits. Its error also swings by a factor of
//! two between runs of the same configuration, since chi_in moves by one and
//! the truncation it forces is severe, so read it as order one rather than as
//! a number. The separation is much sharper than in case 2, where the same
//! single-pass truncation cost only three to four orders of magnitude, because
//! the exact elementwise product has rank up to chi_in squared and a budget of
//! chi_in discards almost all of it, whereas naive and fit reach a
//! near-optimal basis for the same budget. Raising the budget recovers zipup
//! smoothly (1.8e-7 at 8 chi_in, 3.9e-8 unconstrained at chi_out = 837), so
//! this is the price of the budget, not a broken arm.
//! On cost, `naive` is again the expensive one, forming the full chi_in-squared
//! bond before truncating: it grows from well under a second at r = 6 to about
//! 10 s per run at r = 12 and 14, while every other arm stays under two
//! seconds across the default sweep (r = 6, 8, 10, 12, 14).

use std::path::PathBuf;
use t4a_bench::elementwise::{
    elementwise_product, max_rel_error_vs_mixture_product, ElementwiseAlgo, FIT_NFULLSWEEPS,
};
use t4a_bench::gaussian::{to_quantics_fused_tt, GaussianMixture2D};
use t4a_bench::harness::time_median;
use t4a_bench::record::{write_record, RunRecord, SCHEMA_VERSION};
use tensor4all_simplett::AbstractTensorTrain;

const CASE: &str = "elementwise_gauss2d";

/// Sanity gate for the `zipup_treetn` arm, which cannot share `BENCH_SANITY`.
///
/// At the fixed output budget the single-pass zip-up truncation of an
/// elementwise product is not merely less accurate, it fails outright: the
/// measured relative error at the pinned revision is of order one across the
/// default sweep, between 4e-1 and 9e-1 depending on r and on the run, against
/// 1e-8 or better for every other arm. That
/// is the headline result of this case, not a defect: given the same two inputs
/// and a budget of 8 chi_in the same arm reaches 1.8e-7, and unconstrained it
/// reaches 3.9e-8, so the algorithm is sound and the budget is what breaks it.
/// A relative error near one leaves no room for a gate that screens order-unity
/// wrongness, so this arm is gated only against a gross scale blow-up or a
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

fn main() -> anyhow::Result<()> {
    let rs: Vec<usize> = std::env::var("BENCH_RS")
        .unwrap_or_else(|_| "6,8,10,12,14".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let ngauss: usize = env_or("BENCH_NGAUSS", 8);
    let box_l: f64 = env_or("BENCH_BOX_L", 6.0);
    let alpha_lo: f64 = env_or("BENCH_ALPHA_LO", 0.5);
    let alpha_hi: f64 = env_or("BENCH_ALPHA_HI", 8.0);
    let tol: f64 = env_or("BENCH_TOL", 1e-8);
    let max_bond: usize = env_or("BENCH_MAX_BOND", 512);
    let runs: usize = env_or("BENCH_RUNS", 3);
    let warmups: usize = env_or("BENCH_WARMUPS", 0);
    let seed: u64 = env_or("BENCH_SEED", 0);
    // With the output budget fixed at chi_in, truncation error is the quantity
    // being measured, not a defect, so the gate only screens order-unity
    // wrongness rather than certifying precision.
    let sanity: f64 = env_or("BENCH_SANITY", 1e-2);
    let algos: Vec<String> = std::env::var("BENCH_ALGOS")
        .unwrap_or_else(|_| "naive,zipup_treetn,fit_treetn,aci".into())
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    let out_dir =
        PathBuf::from(std::env::var("OUT_DIR").unwrap_or_else(|_| "result/dev/raw".into()));

    // Accuracy check sampling, recorded in every record's params.
    let n_error_samples: usize = 128;
    let error_seed: u64 = seed.wrapping_add(99);

    let f = GaussianMixture2D::random(ngauss, box_l, (alpha_lo, alpha_hi), seed.wrapping_add(1));
    let g = GaussianMixture2D::random(ngauss, box_l, (alpha_lo, alpha_hi), seed.wrapping_add(2));

    let mut failures = Vec::new();
    for &r in &rs {
        let (fa, _step) = to_quantics_fused_tt(&f, r, box_l, tol, max_bond)?;
        let (gb, _) = to_quantics_fused_tt(&g, r, box_l, tol, max_bond)?;
        let input_chi = fa.rank().max(gb.rank());
        eprintln!("r={r} input_chi={input_chi}");

        // An empty EXPORT_HDF5 counts as unset, so `EXPORT_HDF5=` disables export.
        if let Some(dir) = std::env::var("EXPORT_HDF5").ok().filter(|d| !d.is_empty()) {
            std::fs::create_dir_all(&dir)?;
            let h5 = format!("{dir}/instance-r{r}.h5");
            // These are the exact trains the arms below consume, so the export
            // is the benchmarked instance rather than a fresh TCI run.
            t4a_bench::hdf5_export::save_tt_as_mps(&h5, "f", &fa, false)?;
            t4a_bench::hdf5_export::save_tt_as_mps(&h5, "g", &gb, true)?;
            let meta = serde_json::json!({
                "schema_version": 1,
                "case": CASE,
                "r": r,
                "box_l": box_l,
                "tolerance": tol,
                "f": {"weights": f.weights, "alphas": f.alphas, "centers": f.centers},
                "g": {"weights": g.weights, "alphas": g.alphas, "centers": g.centers},
            });
            std::fs::write(
                format!("{dir}/instance-r{r}.json"),
                serde_json::to_string_pretty(&meta)?,
            )?;
        }

        for algo_name in &algos {
            let algo = parse_algo(algo_name);
            let (h, timing) = time_median(warmups, runs, || {
                elementwise_product(algo, &fa, &gb, tol, input_chi).expect("product failed")
            });
            let max_error =
                max_rel_error_vs_mixture_product(&h, &f, &g, r, box_l, n_error_samples, error_seed);
            let rec = RunRecord {
                schema_version: SCHEMA_VERSION,
                case: CASE.into(),
                algorithm: algo_name.clone(),
                params: serde_json::json!({
                    "r": r, "n_gauss": ngauss, "box_l": box_l,
                    "alpha_range": [alpha_lo, alpha_hi], "max_bond": max_bond,
                    // Output budget shared by every algorithm: the input rank.
                    "contract_max_bond": input_chi,
                    "runs": runs, "warmups": warmups,
                    "n_error_samples": n_error_samples, "error_seed": error_seed,
                    "error_metric": "max_rel_vs_analytic",
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
            write_record(&out_dir, &format!("{CASE}-{algo_name}-r{r}"), &rec)?;
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
                    "{algo_name} r={r}: rel err {max_error:.2e} > {gate:.2e}"
                ));
            }
        }
    }
    if !failures.is_empty() {
        anyhow::bail!("sanity failures:\n{}", failures.join("\n"));
    }
    Ok(())
}
