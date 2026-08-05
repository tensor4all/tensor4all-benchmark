//! Case 2 runner: contraction of two quantics MPOs representing 2D Gaussian
//! mixtures, swept over the number of bits per variable for each algorithm.
//!
//! Default algorithms are `naive,zipup,fit`, selectable with `BENCH_ALGOS`.
//!
//! Fixed output budget: the contraction is run with the maximum bond dimension
//! capped at the input rank (`chi_out <= chi_in`, where `chi_in` is the larger
//! of the two input MPO ranks), identically for every algorithm. All arms
//! therefore pay the same output budget, and the error column, the residual
//! against the analytic Gaussian integral recorded as `max_error` with
//! `error_metric = "max_rel_vs_analytic"`, is the discriminator between them.
//! A side effect is that the differing truncation semantics of the engines
//! (absolute cutoff on simplett, relative on treetn) no longer let one arm's
//! rank explode at a loose cap, so the timings stay comparable.
//! `BENCH_MAX_BOND` keeps its role only as the cap for the input TCI
//! construction in `to_quantics_mpo`.
//!
//! What the fixed budget measures, as observed at r = 6 and r = 8: the two
//! simplett arms, naive and zipup, return the same truncated result and the
//! same error, so the metric does not separate them, only their wall times
//! differ. The treetn fit arm reaches an error several orders of magnitude
//! lower at a `chi_out` below the budget it was given. That gap suggests the
//! simplett truncation is far from the best fixed-rank approximation
//! available at that rank. This is an observed gap, not a diagnosed cause,
//! and no upstream issue has been filed for it yet.
//!
//! The `fit` column runs on the treetn variational engine, reached through
//! `tensor4all_itensorlike::TensorTrain::contract` with `ContractOptions::fit()`
//! (see `mpo_contract::contract_fit_treetn`). This is the same engine case 1
//! uses for its elementwise fit. `tensor4all_simplett::mpo::contract_fit` is
//! deliberately NOT used: at the pinned upstream rev (tensor4all-rs 69a24e7)
//! its local update `update_two_site_core` is a placeholder that leaves the
//! core untouched, so that path degenerates to naive plus dead sweeps.
//!
//! Fit is run at a fixed sweep count (`mpo_contract::FIT_NSWEEPS`), which is
//! part of the benchmark definition: its cost is linear in the sweep count, so
//! the timing column is only comparable against naive and zipup at a stated
//! number of sweeps.
//!
//! Default sweep size: the quantics rank of the default mixture saturates
//! around chi = 70 to 80, and the simplett naive and zipup arms then cost tens
//! of seconds to minutes per contraction (bond Kronecker product of size
//! chi^2 followed by SVDs). The defaults (r = 6, 8, 10 and 1 timed run, no
//! warmup) keep a full sweep under about ten minutes on a laptop. Extend with
//! for example `BENCH_RS=6,8,10,12,14,16 BENCH_RUNS=5` for the heavy tail;
//! cost grows roughly linearly in r once the rank has saturated.

use std::path::PathBuf;
use t4a_bench::gaussian::{to_quantics_mpo, GaussianMixture2D};
use t4a_bench::harness::time_median;
use t4a_bench::mpo_contract::{max_rel_error_vs_analytic, mpo_contract, MpoAlgo, FIT_NSWEEPS};
use t4a_bench::record::{write_record, RunRecord, SCHEMA_VERSION};

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn parse_algo(s: &str) -> MpoAlgo {
    match s {
        "naive" => MpoAlgo::Naive,
        "zipup" => MpoAlgo::Zipup,
        "fit" => MpoAlgo::Fit,
        other => panic!("unknown algorithm {other}"),
    }
}

fn main() -> anyhow::Result<()> {
    let rs: Vec<usize> = std::env::var("BENCH_RS")
        .unwrap_or_else(|_| "6,8,10".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let ngauss: usize = env_or("BENCH_NGAUSS", 8);
    let box_l: f64 = env_or("BENCH_BOX_L", 6.0);
    let alpha_lo: f64 = env_or("BENCH_ALPHA_LO", 0.5);
    let alpha_hi: f64 = env_or("BENCH_ALPHA_HI", 8.0);
    let tol: f64 = env_or("BENCH_TOL", 1e-8);
    let max_bond: usize = env_or("BENCH_MAX_BOND", 512);
    // The heavy simplett arms are multi-second deterministic kernels, so one
    // timed run keeps a default sweep under about ten minutes. Raise
    // `BENCH_RUNS` when a median over repetitions is wanted.
    let runs: usize = env_or("BENCH_RUNS", 1);
    let warmups: usize = env_or("BENCH_WARMUPS", 0);
    let seed: u64 = env_or("BENCH_SEED", 0);
    // With the output budget fixed at chi_in, truncation error is the quantity
    // being measured, not a defect, so the gate only screens order-unity
    // wrongness rather than certifying precision.
    let sanity: f64 = env_or("BENCH_SANITY", 1e-2);
    let algos: Vec<String> = std::env::var("BENCH_ALGOS")
        .unwrap_or_else(|_| "naive,zipup,fit".into())
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
        let (fa, dy) = to_quantics_mpo(&f, r, box_l, tol, max_bond)?;
        let (gb, _) = to_quantics_mpo(&g, r, box_l, tol, max_bond)?;
        let input_chi = fa.rank().max(gb.rank());
        eprintln!("r={r} input_chi={input_chi}");

        // An empty EXPORT_HDF5 counts as unset, so `EXPORT_HDF5=` disables export.
        if let Some(dir) = std::env::var("EXPORT_HDF5").ok().filter(|d| !d.is_empty()) {
            std::fs::create_dir_all(&dir)?;
            let h5 = format!("{dir}/instance-r{r}.h5");
            // Export the FUSED site-dim-4 TTs (before the Tensor4 conversion),
            // so `save_mps` applies directly and ITensorMPS.jl sees one site
            // index of dimension 4 per site.
            let (ftt, _) = t4a_bench::gaussian::to_quantics_fused_tt(&f, r, box_l, tol, max_bond)?;
            let (gtt, _) = t4a_bench::gaussian::to_quantics_fused_tt(&g, r, box_l, tol, max_bond)?;
            t4a_bench::hdf5_export::save_tt_as_mps(&h5, "f", &ftt, false)?;
            t4a_bench::hdf5_export::save_tt_as_mps(&h5, "g", &gtt, true)?;
            let meta = serde_json::json!({
                "schema_version": 1,
                "case": "mpo_mpo_quantics",
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
                mpo_contract(algo, &fa, &gb, tol, input_chi).expect("contraction failed")
            });
            let max_error =
                max_rel_error_vs_analytic(&h, dy, &f, &g, r, box_l, n_error_samples, error_seed);
            let rec = RunRecord {
                schema_version: SCHEMA_VERSION,
                case: "mpo_mpo_quantics".into(),
                algorithm: algo_name.clone(),
                params: serde_json::json!({
                    "r": r, "n_gauss": ngauss, "box_l": box_l,
                    "alpha_range": [alpha_lo, alpha_hi], "max_bond": max_bond,
                    // Output budget shared by every algorithm: the input rank.
                    "contract_max_bond": input_chi,
                    "runs": runs, "warmups": warmups,
                    "n_error_samples": n_error_samples, "error_seed": error_seed,
                    "error_metric": "max_rel_vs_analytic",
                    // Part of the benchmark definition for the fit arm.
                    "fit_nsweeps": FIT_NSWEEPS,
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
            write_record(
                &out_dir,
                &format!("mpo_mpo_quantics-{algo_name}-r{r}"),
                &rec,
            )?;
            eprintln!(
                "  {algo_name}: t={:.4}s rel_err={max_error:.2e} chi_out={}",
                timing.median_secs,
                h.rank()
            );
            if max_error > sanity {
                failures.push(format!(
                    "{algo_name} r={r}: rel err {max_error:.2e} > {sanity:.2e}"
                ));
            }
        }
    }
    if !failures.is_empty() {
        anyhow::bail!("sanity failures:\n{}", failures.join("\n"));
    }
    Ok(())
}
