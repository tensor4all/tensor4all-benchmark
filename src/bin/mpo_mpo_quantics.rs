//! Case 2 runner: contraction of two quantics MPOs representing 2D Gaussian
//! mixtures, swept over the number of bits per variable for each algorithm.
//!
//! Default algorithms are `naive,zipup_simplett,zipup_treetn,fit_treetn`,
//! selectable with `BENCH_ALGOS`. Where both upstream engines implement an
//! algorithm, both are benchmarked as separate arms and the engine is recorded
//! as `engine` in every JSON record, so an engine difference shows up as its own
//! column instead of being folded into one. The only missing pair is simplett
//! fit, excluded for the upstream stub reason stated below.
//!
//! Fixed output budget, decided by the rank cap alone: the contraction is run
//! with the maximum bond dimension capped at the input rank (`chi_out <=
//! chi_in`, where `chi_in` is the larger of the two input MPO ranks),
//! identically for every algorithm, and the truncation tolerance handed to the
//! contraction is pinned inert. `BENCH_TOL` (default 1e-8) now scopes only the
//! input TCI construction in `to_quantics_mpo`, so it defines `chi_in` and thus
//! the instance. `BENCH_CONTRACT_TOL` (default 1e-15) is what every arm
//! receives as its truncation tolerance and is recorded as `contract_tol` in the
//! params of every record. At 1e-15 that tolerance never fires, so `chi_in` is
//! the only binding truncation control and the cases cannot be read ambiguously
//! as measuring whichever of the two constraints happened to bind first. All
//! arms therefore genuinely exhaust the same output budget unless their exact
//! rank is smaller, and the error column, the residual against the analytic
//! Gaussian integral recorded as `max_error` with
//! `error_metric = "max_rel_vs_analytic"`, is the discriminator between them.
//! Both engines truncate relative to the largest singular value, so at the same
//! nominal tolerance they discard the same singular values.
//!
//! What the fixed budget measures, as observed over the default sweep r = 6 to
//! 14 with the pinned rev: `naive` and `fit_treetn` land on the same error,
//! 5.7e-10 to 3.6e-9, which is below the reference floor the case was previously
//! credited with. The two zipup arms, `zipup_simplett` and `zipup_treetn`, agree
//! with each other to the last reported digit and sit four to five orders of
//! magnitude higher, 2.0e-5 to 1.1e-4. Every arm spends the whole budget, since
//! the tolerance no longer stops it early. So
//! the split is algorithmic rather than engine-driven: single-pass zip-up
//! truncation is what costs accuracy, and the two engines running it produce the
//! same answer. What zipup buys is speed: it is the fastest arm at every r and
//! stays between 0.015 s and 0.31 s across the sweep, while `naive` grows
//! steeply, from 0.02 s at r = 6 to 0.4 s at r = 8, 5.3 s at r = 10 and 16 to
//! 17 s at r = 12 and 14, because it forms the full
//! contracted bond before truncating. `fit_treetn` reaches naive accuracy at a
//! fraction of the naive cost, under 0.75 s at every r.
//!
//! The `zipup_treetn` and `fit_treetn` columns run on the treetn engine,
//! reached through `tensor4all_itensorlike::TensorTrain::contract` with
//! `ContractOptions::zipup()` and `ContractOptions::fit()` respectively (see
//! `mpo_contract`). They share the same bridge, max rank and SVD policy, so the
//! only difference between them is the contraction method. That is the same
//! engine case 1 uses for its elementwise fit.
//! `tensor4all_simplett::mpo::contract_fit` is deliberately NOT benchmarked: at
//! the pinned upstream rev (tensor4all-rs c9ecb7f) its local update
//! `update_two_site_core` is still a placeholder that leaves the core
//! untouched, so that path degenerates to naive plus dead sweeps
//! (tensor4all-rs#571).
//!
//! The pinned rev includes tensor4all-rs#574, which changed simplett in three
//! ways that this case sees directly: MPO factorize truncation became relative
//! to the largest singular value instead of absolute, `contract_zipup`'s scalar
//! loop became an einsum (about 800x faster), and `contract_naive`'s
//! compression sweep now establishes a right-to-left QR gauge before
//! truncating, which dropped its error by about three orders of magnitude onto
//! the variational fit result.
//!
//! `fit_treetn` is run at a fixed sweep count (`mpo_contract::FIT_NSWEEPS`),
//! which is part of the benchmark definition: its cost is linear in the sweep
//! count, so the timing column is only comparable against the naive and zipup
//! arms at a stated number of sweeps.
//!
//! Default sweep size: the quantics rank of the default mixture saturates
//! around chi = 70 to 80. `naive` is the only expensive arm, since it forms the
//! full contracted bond of size chi^2 before truncating; every other arm stays
//! around a second or less across the default range. The defaults (r = 6, 8,
//! 10, 12, 14 with 3 timed runs, no warmup) size the whole case at roughly two
//! minutes on the maintainer's Mac, nearly all of it naive at r = 10 to 14,
//! which costs 5 to 17 s per run there. That arm is memory bound on a machine
//! with less headroom, where the same points cost 16 to 48 s per run, so the
//! wall time of the whole case is a property of the machine as much as of the
//! sweep (README known issue 10). Extend with for example
//! `BENCH_RS=6,8,10,12,14,16 BENCH_RUNS=5` for the heavy tail, and restrict
//! `BENCH_ALGOS` to drop naive if only the cheap arms are wanted.

use std::path::PathBuf;
use t4a_bench::gaussian::{to_quantics_mpo, GaussianMixture2D};
use t4a_bench::harness::time_median;
use t4a_bench::mpo_contract::{
    max_rel_error_vs_analytic, mpo_contract, mpo_n_params, MpoAlgo, FIT_NSWEEPS,
};
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
        "zipup_simplett" => MpoAlgo::ZipupSimplett,
        "zipup_treetn" => MpoAlgo::ZipupTreetn,
        "fit_treetn" => MpoAlgo::FitTreetn,
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
    // Instance tolerance: this defines chi_in through the input TCI, and nothing
    // else. It is what `RunRecord::tolerance` and the Julia-check metadata
    // report, since both describe the inputs rather than the contraction.
    let tol: f64 = env_or("BENCH_TOL", 1e-8);
    // Contraction tolerance, pinned inert so the rank cap chi_in is the only
    // binding truncation control for every arm.
    let contract_tol: f64 = env_or("BENCH_CONTRACT_TOL", 1e-15);
    let max_bond: usize = env_or("BENCH_MAX_BOND", 512);
    // Since tensor4all-rs#574 the whole default sweep costs seconds, so three
    // timed runs are affordable and the reported median is more stable than a
    // single sample. Raise `BENCH_RUNS` further for the heavy tail.
    let runs: usize = env_or("BENCH_RUNS", 3);
    let warmups: usize = env_or("BENCH_WARMUPS", 0);
    let seed: u64 = env_or("BENCH_SEED", 0);
    // With the output budget fixed at chi_in, truncation error is the quantity
    // being measured, not a defect, so the gate only screens order-unity
    // wrongness rather than certifying precision.
    let sanity: f64 = env_or("BENCH_SANITY", 1e-2);
    let algos: Vec<String> = std::env::var("BENCH_ALGOS")
        .unwrap_or_else(|_| "naive,zipup_simplett,zipup_treetn,fit_treetn".into())
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
                mpo_contract(algo, &fa, &gb, contract_tol, input_chi).expect("contraction failed")
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
                    // Truncation tolerance the arms actually ran with, pinned
                    // inert so the cap above is what decides. The top-level
                    // `tolerance` field is the instance tolerance instead.
                    "contract_tol": contract_tol,
                    "runs": runs, "warmups": warmups,
                    "n_error_samples": n_error_samples, "error_seed": error_seed,
                    "error_metric": "max_rel_vs_analytic",
                    // Upstream engine that actually ran this arm.
                    "engine": algo.engine(),
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
                n_params: Some(mpo_n_params(&h)),
                n_patches: None,
                max_patch_bond: None,
                rtol: None,
                input_build_secs: None,
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
