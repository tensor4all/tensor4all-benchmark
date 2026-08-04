//! Case 2 runner: contraction of two quantics MPOs representing 2D Gaussian
//! mixtures, swept over the number of bits per variable for each algorithm.
//!
//! Cost warning: at the pinned upstream rev, `contract_fit` builds its
//! environments with scalar loops over all six bond indices, costing
//! O((chi_c chi_a chi_b)^2 d^3) per site per half-sweep. Measured here at
//! n_gauss=3, tol=1e-6: 1.1s at r=4, 29s at r=5, hours at r=6, and out of
//! reach at r>=8. Select `BENCH_ALGOS=naive,zipup` for the large-r sweep and
//! run `fit` only at small r until that is fixed upstream.

use std::path::PathBuf;
use t4a_bench::gaussian::{to_quantics_mpo, GaussianMixture2D};
use t4a_bench::harness::time_median;
use t4a_bench::mpo_contract::{max_rel_error_vs_analytic, mpo_contract, MpoAlgo};
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
        .unwrap_or_else(|_| "10,12,14,16".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let ngauss: usize = env_or("BENCH_NGAUSS", 8);
    let box_l: f64 = env_or("BENCH_BOX_L", 6.0);
    let alpha_lo: f64 = env_or("BENCH_ALPHA_LO", 0.5);
    let alpha_hi: f64 = env_or("BENCH_ALPHA_HI", 8.0);
    let tol: f64 = env_or("BENCH_TOL", 1e-8);
    let max_bond: usize = env_or("BENCH_MAX_BOND", 512);
    let runs: usize = env_or("BENCH_RUNS", 5);
    let warmups: usize = env_or("BENCH_WARMUPS", 1);
    let seed: u64 = env_or("BENCH_SEED", 0);
    let sanity: f64 = env_or("BENCH_SANITY", 1e-4);
    let algos: Vec<String> = std::env::var("BENCH_ALGOS")
        .unwrap_or_else(|_| "naive,zipup,fit".into())
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    let out_dir =
        PathBuf::from(std::env::var("OUT_DIR").unwrap_or_else(|_| "result/dev/raw".into()));

    let f = GaussianMixture2D::random(ngauss, box_l, (alpha_lo, alpha_hi), seed.wrapping_add(1));
    let g = GaussianMixture2D::random(ngauss, box_l, (alpha_lo, alpha_hi), seed.wrapping_add(2));

    let mut failures = Vec::new();
    for &r in &rs {
        let (fa, dy) = to_quantics_mpo(&f, r, box_l, tol, max_bond)?;
        let (gb, _) = to_quantics_mpo(&g, r, box_l, tol, max_bond)?;
        let input_chi = fa.rank().max(gb.rank());
        eprintln!("r={r} input_chi={input_chi}");

        for algo_name in &algos {
            let algo = parse_algo(algo_name);
            let (h, timing) = time_median(warmups, runs, || {
                mpo_contract(algo, &fa, &gb, tol, max_bond).expect("contraction failed")
            });
            let max_error =
                max_rel_error_vs_analytic(&h, dy, &f, &g, r, box_l, 128, seed.wrapping_add(99));
            let rec = RunRecord {
                schema_version: SCHEMA_VERSION,
                case: "mpo_mpo_quantics".into(),
                algorithm: algo_name.clone(),
                params: serde_json::json!({
                    "r": r, "n_gauss": ngauss, "box_l": box_l,
                    "alpha_range": [alpha_lo, alpha_hi], "max_bond": max_bond,
                    "runs": runs, "warmups": warmups,
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
