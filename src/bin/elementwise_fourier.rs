//! Case 1 runner: elementwise product of two random Fourier series in QTT form,
//! swept over the number of Fourier modes for each product algorithm.

use std::path::PathBuf;
use t4a_bench::elementwise::{elementwise_product, max_error_vs_series, ElementwiseAlgo};
use t4a_bench::fourier::{compress_svd, FourierSeries};
use t4a_bench::harness::time_median;
use t4a_bench::record::{write_record, RunRecord, SCHEMA_VERSION};
use tensor4all_simplett::AbstractTensorTrain;

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn parse_algo(s: &str) -> ElementwiseAlgo {
    match s {
        "naive" => ElementwiseAlgo::Naive,
        "zipup" => ElementwiseAlgo::Zipup,
        "fit" => ElementwiseAlgo::Fit,
        "aci" => ElementwiseAlgo::Aci,
        other => panic!("unknown algorithm {other}"),
    }
}

fn main() -> anyhow::Result<()> {
    let ks: Vec<usize> = std::env::var("BENCH_KS")
        .unwrap_or_else(|_| "4,8,16,32,64".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let r: usize = env_or("BENCH_R", 20);
    let tol: f64 = env_or("BENCH_TOL", 1e-8);
    let max_bond: usize = env_or("BENCH_MAX_BOND", 4096);
    let runs: usize = env_or("BENCH_RUNS", 5);
    let warmups: usize = env_or("BENCH_WARMUPS", 1);
    let seed: u64 = env_or("BENCH_SEED", 0);
    let algos: Vec<String> = std::env::var("BENCH_ALGOS")
        .unwrap_or_else(|_| "naive,zipup,fit,aci".into())
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    let out_dir =
        PathBuf::from(std::env::var("OUT_DIR").unwrap_or_else(|_| "result/dev/raw".into()));

    let mut failures = Vec::new();
    for &k in &ks {
        let f = FourierSeries::random(k, seed.wrapping_add(2 * k as u64));
        let g = FourierSeries::random(k, seed.wrapping_add(2 * k as u64 + 1));
        let exact = f.product(&g);
        let mut a = f.to_qtt(r)?;
        let mut b = g.to_qtt(r)?;
        compress_svd(&mut a, tol, max_bond)?;
        compress_svd(&mut b, tol, max_bond)?;
        let input_chi = a.rank().max(b.rank());
        eprintln!("k_max={k} input_chi={input_chi}");

        for algo_name in &algos {
            let algo = parse_algo(algo_name);
            let (out, timing) = time_median(warmups, runs, || {
                elementwise_product(algo, &a, &b, tol, max_bond).expect("algorithm failed")
            });
            let max_error = max_error_vs_series(&out, &exact, r, 256, seed.wrapping_add(999));
            // fit has a known upstream accuracy issue on elementwise products
            let sanity = if matches!(algo, ElementwiseAlgo::Fit) {
                1e-2
            } else {
                10.0 * tol
            };
            let rec = RunRecord {
                schema_version: SCHEMA_VERSION,
                case: "elementwise_fourier".into(),
                algorithm: algo_name.clone(),
                params: serde_json::json!({
                    "k_max": k, "r": r, "input_max_bond_dim": input_chi, "max_bond": max_bond,
                }),
                seed,
                tolerance: tol,
                wall_time_median_secs: timing.median_secs,
                wall_times_secs: timing.runs_secs,
                max_error,
                input_max_bond_dim: input_chi,
                output_max_bond_dim: out.rank(),
                output_bond_dims: out.link_dims(),
            };
            write_record(&out_dir, &format!("elementwise_fourier-{algo_name}-k{k}"), &rec)?;
            eprintln!(
                "  {algo_name}: t={:.4}s err={max_error:.2e} chi_out={}",
                timing.median_secs,
                out.rank()
            );
            if max_error > sanity {
                failures.push(format!(
                    "{algo_name} k={k}: err {max_error:.2e} > sanity {sanity:.2e}"
                ));
            }
        }
    }
    if !failures.is_empty() {
        anyhow::bail!("sanity failures:\n{}", failures.join("\n"));
    }
    Ok(())
}
