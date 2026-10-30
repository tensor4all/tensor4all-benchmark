//! Case 3: global and patched fit contraction of interpolative Gaussian MPOs.
//!
//! The patched arm is paused at the pinned tensor4all-rs revision: it shares the
//! patch machinery the paused case-2 arms use, and its input patches are built
//! by `PatchedMpoPair`. It stays behind the `patched-partitionedtt` feature, which
//! is not buildable here, and the port is tracked in
//! https://github.com/tensor4all/tensor4all-benchmark/issues/11. The global arm
//! still records the chain TreeTN patch preparation of its inputs, which is
//! untimed and builds at the pin.

use std::path::{Path, PathBuf};
use std::time::Instant;

use tensor4all_itensorlike::TensorTrain;
use tensor4all_simplett::AbstractTensorTrain;

use t4a_bench::gaussian::{fused_qtt_to_mpo, AnisoMixture2D};
use t4a_bench::gaussian_input::{
    prepare_gaussian_pair, sampled_input_relative_l2, tensortrain_n_params as simple_n_params,
    GaussianInputConfig, GaussianInputPair, INPUT_L2_RTOL, PATCH_CAP,
};
use t4a_bench::harness::time_median;
use t4a_bench::mpo_contract::sampled_relative_l2_vs_aniso_grid;
#[cfg(feature = "patched-partitionedtt")]
use t4a_bench::patched_mpo::partitioned_treetn_n_params;
use t4a_bench::patched_mpo::{tensortrain_n_params, PatchedMpoPair};
use t4a_bench::record::{write_record, RunRecord, SCHEMA_VERSION};

const CASE: &str = "gaussian_mpo_contraction";
const GLOBAL_MAX_BOND: usize = 4096;
const ERROR_SAMPLES: usize = 256;
const ERROR_SANITY: f64 = 1e-4;
/// Arms this pin can build and measure.
const MEASURED_ARMS: [&str; 1] = ["global_fit"];
/// Arms that are paused, see [`PATCHED_ARMS_PAUSED`].
const PAUSED_ARMS: [&str; 1] = ["patched_fit"];

#[cfg(feature = "patched-partitionedtt")]
const DEFAULT_ARMS: &str = "global_fit,patched_fit";
#[cfg(not(feature = "patched-partitionedtt"))]
const DEFAULT_ARMS: &str = "global_fit";

/// Why the patched arm cannot run at the pinned tensor4all-rs revision.
const PATCHED_ARMS_PAUSED: &str = "the patched arms are paused: tensor4all-rs#1005 removed \
     tensor4all-partitionedtt, so they are not built at this pin. The port is tracked in \
     https://github.com/tensor4all/tensor4all-benchmark/issues/11";

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// One term of the closed form the interpolative input approximates.
fn mixture_json(mixture: &AnisoMixture2D) -> serde_json::Value {
    serde_json::json!({
        "weights": mixture.weights,
        "quad": mixture.quad.iter().map(|&q| [q.0, q.1, q.2]).collect::<Vec<_>>(),
        "centers": mixture.centers.iter().map(|&c| [c.0, c.1]).collect::<Vec<_>>(),
    })
}

/// Write the ITensors-compatible instance dump for the Julia cross-check.
///
/// The two groups `f` and `g` are the fused site-dimension-4 QTTs, one MPS site
/// per quantics bit pair, and the JSON carries the mixtures they approximate.
/// The case name is asserted on the Julia side, because cases 2 and 3 once
/// shared these file names.
fn export_instance(dir: &str, n: usize, input: &GaussianInputPair) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    let h5 = format!("{dir}/instance-n{n}.h5");
    t4a_bench::hdf5_export::save_tt_as_mps(&h5, "f", &input.left, false)?;
    t4a_bench::hdf5_export::save_tt_as_mps(&h5, "g", &input.right, true)?;
    let meta = serde_json::json!({
        "schema_version": 1,
        "case": CASE,
        "n_gauss": n,
        "r": input.r,
        "box_l": input.box_l,
        "interpolation_tolerance": 1e-10,
        "input_l2_rtol": INPUT_L2_RTOL,
        "left": mixture_json(&input.left_mixture),
        "right": mixture_json(&input.right_mixture),
    });
    std::fs::write(
        format!("{dir}/instance-n{n}.json"),
        serde_json::to_string_pretty(&meta)?,
    )?;
    eprintln!("exported instance-n{n} to {dir}");
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let ns = std::env::var("BENCH_NS").unwrap_or_else(|_| "2,8,32,128".into());
    let ns = ns
        .split(',')
        .map(|value| value.trim().parse::<usize>().map_err(anyhow::Error::from))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let out_dir =
        PathBuf::from(std::env::var("OUT_DIR").unwrap_or_else(|_| "result/dev/raw".into()));
    let cache_dir = PathBuf::from(
        std::env::var("BENCH_INPUT_CACHE_DIR").unwrap_or_else(|_| ".cache/inputs".into()),
    );
    let refresh = env_or("BENCH_INPUT_CACHE_REFRESH", 0usize) != 0;
    let runs = env_or("BENCH_RUNS", 1usize);
    let warmups = env_or("BENCH_WARMUPS", 0usize);
    let seed = env_or("BENCH_SEED", 0u64);
    let arms = select_arms()?;
    for n in ns {
        run_point(n, seed, runs, warmups, &cache_dir, refresh, &out_dir, &arms)?;
    }
    Ok(())
}

/// Read `BENCH_ARMS` and refuse arms that this pin cannot build.
fn select_arms() -> anyhow::Result<Vec<String>> {
    let requested = std::env::var("BENCH_ARMS").unwrap_or_else(|_| DEFAULT_ARMS.into());
    let arms: Vec<String> = requested
        .split(',')
        .map(|value| value.trim().to_string())
        .collect();
    for arm in &arms {
        if PAUSED_ARMS.contains(&arm.as_str()) {
            anyhow::ensure!(
                cfg!(feature = "patched-partitionedtt"),
                "arm {arm:?} is not available: {PATCHED_ARMS_PAUSED}"
            );
        } else {
            anyhow::ensure!(
                MEASURED_ARMS.contains(&arm.as_str()),
                "unknown arm {arm:?}; case 3 measured arms are {} and the paused arms are {}",
                MEASURED_ARMS.join(", "),
                PAUSED_ARMS.join(", ")
            );
        }
    }
    Ok(arms)
}

#[allow(clippy::too_many_arguments)]
fn run_point(
    n: usize,
    seed: u64,
    runs: usize,
    warmups: usize,
    cache_dir: &Path,
    refresh: bool,
    out_dir: &Path,
    arms: &[String],
) -> anyhow::Result<()> {
    let config = GaussianInputConfig {
        n,
        sigma_minor: 0.05,
        rho_max: 8.0,
        spacing: 3.0,
        polynomial_degree: 28,
        interpolation_tolerance: 1e-10,
        addition_tolerance: 1e-10,
        seed,
        cache_dir: cache_dir.to_path_buf(),
        refresh,
    };
    let input = prepare_gaussian_pair(&config)?;
    let (left_input_error, right_input_error) =
        sampled_input_relative_l2(&input, ERROR_SAMPLES, seed.wrapping_add(41))?;
    anyhow::ensure!(
        left_input_error <= ERROR_SANITY && right_input_error <= ERROR_SANITY,
        "input error exceeds sanity gate: ({left_input_error:.3e}, {right_input_error:.3e})"
    );
    let left_mpo = fused_qtt_to_mpo(&input.left)?;
    let right_mpo = fused_qtt_to_mpo(&input.right)?;
    let patch_start = Instant::now();
    let prepared = PatchedMpoPair::new(&left_mpo, &right_mpo, INPUT_L2_RTOL, PATCH_CAP)?;
    let patch_secs = patch_start.elapsed().as_secs_f64();
    let (left_patch_count, right_patch_count) = prepared.input_patch_counts();
    let (left_patch_chi, right_patch_chi) = prepared.input_patch_max_bonds();
    let (left_patch_params, right_patch_params) = prepared.input_patch_n_params();
    anyhow::ensure!(
        left_patch_chi.max(right_patch_chi) <= PATCH_CAP,
        "input patch cap exceeded"
    );
    let input_chi = input.left.rank().max(input.right.rank());
    let input_params = simple_n_params(&input.left) + simple_n_params(&input.right);

    // Independent ITensors.jl cross-check export: the two input QTTs and the
    // analytic mixtures they approximate. `julia/check_mpo_mpo.jl` reads both and
    // evaluates the tensors at sampled grid points against the closed form.
    if let Some(dir) = std::env::var("EXPORT_HDF5").ok().filter(|d| !d.is_empty()) {
        export_instance(&dir, n, &input)?;
    }

    let grid_step = 2.0 * input.box_l / (1usize << input.r) as f64;
    let common = common_params(
        &input,
        &config,
        input_params,
        left_patch_count,
        right_patch_count,
        left_patch_chi,
        right_patch_chi,
        left_patch_params,
        right_patch_params,
        patch_secs,
        left_input_error,
        right_input_error,
    );

    // Timing of the global arm, the reference of the patched speedup column.
    #[cfg_attr(not(feature = "patched-partitionedtt"), allow(unused_variables))]
    let global_fit_secs = if arms.iter().any(|arm| arm == "global_fit") {
        let (global_result, global_timing) = time_median(warmups, runs, || {
            prepared.contract_fit_global(INPUT_L2_RTOL, GLOBAL_MAX_BOND)
        });
        let global: TensorTrain = global_result?;
        let global_params = tensortrain_n_params(&global);
        let global_chi = global.max_bond_dim();
        let global_mpo = prepared.finish_global_output(&global)?;
        let global_error = sampled_relative_l2_vs_aniso_grid(
            &global_mpo,
            grid_step,
            &input.left_mixture,
            &input.right_mixture,
            input.r,
            input.box_l,
            ERROR_SAMPLES,
            seed.wrapping_add(99),
        )?;
        anyhow::ensure!(
            global_error <= ERROR_SANITY,
            "global contraction error {global_error:.3e}"
        );
        write_record(
            out_dir,
            &format!("{CASE}-global_fit-chi{input_chi}"),
            &RunRecord {
                schema_version: SCHEMA_VERSION,
                case: CASE.into(),
                algorithm: "global_fit".into(),
                params: common.clone(),
                seed,
                tolerance: INPUT_L2_RTOL,
                wall_time_median_secs: global_timing.median_secs,
                wall_times_secs: global_timing.runs_secs.clone(),
                max_error: global_error,
                input_max_bond_dim: input_chi,
                output_max_bond_dim: global_chi,
                output_bond_dims: global.bond_dims(),
                n_params: Some(global_params),
                n_patches: None,
                max_patch_bond: None,
                rtol: Some(INPUT_L2_RTOL),
                input_build_secs: Some(input.build.as_secs_f64()),
                n_allocations: Some(global_timing.alloc_calls_median),
                allocated_bytes: Some(global_timing.alloc_bytes_median),
            },
        )?;
        Some(global_timing.median_secs)
    } else {
        None
    };

    #[cfg(feature = "patched-partitionedtt")]
    if arms.iter().any(|arm| arm == "patched_fit") {
        let (patched_result, patched_timing) = time_median(warmups, runs, || {
            prepared.contract_fit_treetn_partitioned(INPUT_L2_RTOL, PATCH_CAP)
        });
        let patched = patched_result?;
        let patched_params = partitioned_treetn_n_params(&patched);
        let patched_finished = prepared.finish_treetn_output(patched)?;
        anyhow::ensure!(
            patched_finished.max_patch_bond <= PATCH_CAP,
            "output patch cap exceeded"
        );
        let patched_error = sampled_relative_l2_vs_aniso_grid(
            &patched_finished.mpo,
            grid_step,
            &input.left_mixture,
            &input.right_mixture,
            input.r,
            input.box_l,
            ERROR_SAMPLES,
            seed.wrapping_add(99),
        )?;
        anyhow::ensure!(
            patched_error <= ERROR_SANITY,
            "patched contraction error {patched_error:.3e}"
        );
        let mut patched_common = common;
        // Only a run that also measured the global arm has a speedup to report.
        if let Some(global_fit_secs) = global_fit_secs {
            patched_common["speedup_vs_global"] =
                serde_json::json!(global_fit_secs / patched_timing.median_secs);
        }
        write_record(
            out_dir,
            &format!("{CASE}-patched_fit-chi{input_chi}"),
            &RunRecord {
                schema_version: SCHEMA_VERSION,
                case: CASE.into(),
                algorithm: "patched_fit".into(),
                params: patched_common,
                seed,
                tolerance: INPUT_L2_RTOL,
                wall_time_median_secs: patched_timing.median_secs,
                n_allocations: Some(patched_timing.alloc_calls_median),
                allocated_bytes: Some(patched_timing.alloc_bytes_median),
                wall_times_secs: patched_timing.runs_secs,
                max_error: patched_error,
                input_max_bond_dim: input_chi,
                output_max_bond_dim: patched_finished.max_patch_bond,
                output_bond_dims: vec![patched_finished.max_patch_bond],
                n_params: Some(patched_params),
                n_patches: Some(patched_finished.n_patches),
                max_patch_bond: Some(patched_finished.max_patch_bond),
                rtol: Some(INPUT_L2_RTOL),
                input_build_secs: Some(input.build.as_secs_f64()),
            },
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn common_params(
    input: &t4a_bench::gaussian_input::GaussianInputPair,
    config: &GaussianInputConfig,
    input_params: usize,
    left_patch_count: usize,
    right_patch_count: usize,
    left_patch_chi: usize,
    right_patch_chi: usize,
    left_patch_params: usize,
    right_patch_params: usize,
    patch_secs: f64,
    left_input_error: f64,
    right_input_error: f64,
) -> serde_json::Value {
    serde_json::json!({
        "n_gauss": config.n, "r": input.r, "box_l": input.box_l,
        "sigma_minor": config.sigma_minor, "rho_max": config.rho_max,
        "spacing": config.spacing, "polynomial_degree": config.polynomial_degree,
        "interpolation_tolerance": config.interpolation_tolerance,
        "addition_tolerance": config.addition_tolerance,
        "input_l2_rtol": INPUT_L2_RTOL, "patch_cap": PATCH_CAP,
        "raw_left_chi": input.raw_left_chi, "raw_right_chi": input.raw_right_chi,
        "left_chi": input.left.rank(), "right_chi": input.right.rank(),
        "raw_left_params": input.raw_left_params, "raw_right_params": input.raw_right_params,
        "input_params": input_params,
        "left_input_patch_count": left_patch_count, "right_input_patch_count": right_patch_count,
        "left_input_max_patch_chi": left_patch_chi, "right_input_max_patch_chi": right_patch_chi,
        "left_input_patch_params": left_patch_params, "right_input_patch_params": right_patch_params,
        "cache_key": input.cache_key, "cache_hit": input.cache_hit,
        "cache_load_secs": input.cache_load.as_secs_f64(),
        "input_build_secs": input.build.as_secs_f64(),
        "input_compression_secs": input.compression.as_secs_f64(),
        "patch_build_secs": patch_secs,
        "left_input_sampled_relative_l2": left_input_error,
        "right_input_sampled_relative_l2": right_input_error,
        "error_samples": ERROR_SAMPLES,
        "internal_tolerance": INPUT_L2_RTOL,
        "internal_tolerance_metric": "relative_l2_svd",
        "external_error_metric": "sampled_relative_l2"
    })
}
