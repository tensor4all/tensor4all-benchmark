//! Case 2: global and patched elementwise products of interpolative Gaussian QTTs.

use std::path::{Path, PathBuf};
use std::time::Instant;

use tensor4all_partitionedtt::PatchSplitStrategy;
use tensor4all_simplett::AbstractTensorTrain;

use t4a_bench::elementwise::{
    elementwise_product, sampled_relative_l2_vs_product, tt_n_params, AciTolerance, ElementwiseAlgo,
};
use t4a_bench::gaussian_input::{
    prepare_gaussian_pair, sampled_input_relative_l2, tensortrain_n_params, GaussianInputConfig,
    INPUT_L2_RTOL, PATCH_CAP,
};
use t4a_bench::harness::time_median;
use t4a_bench::patched::{
    fused_site_indices, max_patch_bond, patched_elementwise_with_stats, patched_input_from_global,
    sampled_relative_l2_patched, total_params, NormPatchedInputOptions, PatchedEngine,
    PatchedProductOptions, PatchedProductStats,
};
use t4a_bench::record::{write_record, RunRecord, SCHEMA_VERSION};

const CASE: &str = "gaussian_elementwise";
const GLOBAL_MAX_BOND: usize = 4096;
const ERROR_SAMPLES: usize = 256;
const ERROR_SANITY: f64 = 1e-4;

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
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
    let aci_tolerance = env_or("BENCH_ACI_TOL", 1e-8_f64);
    for n in ns {
        run_point(
            n,
            seed,
            aci_tolerance,
            runs,
            warmups,
            &cache_dir,
            refresh,
            &out_dir,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_point(
    n: usize,
    seed: u64,
    aci_tolerance: f64,
    runs: usize,
    warmups: usize,
    cache_dir: &Path,
    refresh: bool,
    out_dir: &Path,
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
    if env_or("BENCH_INPUT_ONLY", 0usize) != 0 {
        println!(
            "N={} R={} raw_chi=({}, {}) compressed_chi=({}, {}) build_secs={:.3} cache_hit={}",
            config.n,
            input.r,
            input.raw_left_chi,
            input.raw_right_chi,
            input.left.rank(),
            input.right.rank(),
            input.build.as_secs_f64(),
            input.cache_hit
        );
        return Ok(());
    }
    let sites = fused_site_indices(input.r);
    let patch_start = Instant::now();
    let patch_options = NormPatchedInputOptions {
        rtol: INPUT_L2_RTOL,
        max_bond_dim: PATCH_CAP,
        strategy: PatchSplitStrategy::ExactParameterGain,
    };
    let patched_left = patched_input_from_global(&input.left, &sites, patch_options)?;
    let patched_right = patched_input_from_global(&input.right, &sites, patch_options)?;
    let patch_secs = patch_start.elapsed().as_secs_f64();
    let input_chi = input.left.rank().max(input.right.rank());
    let input_params = tensortrain_n_params(&input.left) + tensortrain_n_params(&input.right);
    let patch_params = total_params(&patched_left, &sites)? + total_params(&patched_right, &sites)?;
    let patch_count = patched_left.len() + patched_right.len();
    let input_max_patch = max_patch_bond(&patched_left).max(max_patch_bond(&patched_right));
    anyhow::ensure!(input_max_patch <= PATCH_CAP, "input patch cap exceeded");

    let (global_fit_result, global_fit_timing) = time_median(warmups, runs, || {
        elementwise_product(
            ElementwiseAlgo::Fit,
            &input.left,
            &input.right,
            INPUT_L2_RTOL,
            GLOBAL_MAX_BOND,
            AciTolerance::Absolute,
        )
    });
    let global_fit = global_fit_result?;
    let global_fit_error = sampled_relative_l2_vs_product(
        &global_fit,
        &input.left_mixture,
        &input.right_mixture,
        input.r,
        input.box_l,
        ERROR_SAMPLES,
        seed.wrapping_add(99),
    );
    anyhow::ensure!(
        global_fit_error <= ERROR_SANITY,
        "global fit error {global_fit_error:.3e}"
    );
    write_global_record(
        out_dir,
        "global_fit",
        &input,
        &config,
        input_chi,
        input_params,
        patch_count,
        input_max_patch,
        patch_params,
        patch_secs,
        left_input_error,
        right_input_error,
        INPUT_L2_RTOL,
        "relative_l2_svd",
        &global_fit,
        global_fit_error,
        &global_fit_timing,
    )?;

    let (patched_fit_result, patched_fit_timing) = time_median(warmups, runs, || {
        patched_elementwise_with_stats(
            &patched_left,
            &patched_right,
            &sites,
            PatchedProductOptions {
                engine: PatchedEngine::FitTreetn {
                    l2_rtol: INPUT_L2_RTOL,
                },
                max_bond_dim: PATCH_CAP,
            },
        )
    });
    let (patched_fit, patched_fit_stats) = patched_fit_result?;
    let patched_fit_error = sampled_relative_l2_patched(
        &patched_fit,
        &sites,
        &input.left_mixture,
        &input.right_mixture,
        input.box_l,
        ERROR_SAMPLES,
        seed.wrapping_add(99),
    )?;
    anyhow::ensure!(
        patched_fit_error <= ERROR_SANITY,
        "patched fit error {patched_fit_error:.3e}"
    );
    write_patched_record(
        out_dir,
        "patched_fit",
        &input,
        &config,
        input_chi,
        input_params,
        patch_count,
        input_max_patch,
        patch_params,
        patch_secs,
        left_input_error,
        right_input_error,
        INPUT_L2_RTOL,
        "relative_l2_svd",
        &patched_fit,
        patched_fit_error,
        &patched_fit_timing,
        global_fit_timing.median_secs,
        patched_fit_stats,
        &sites,
    )?;

    let (global_aci_result, global_aci_timing) = time_median(warmups, runs, || {
        elementwise_product(
            ElementwiseAlgo::Aci,
            &input.left,
            &input.right,
            aci_tolerance,
            GLOBAL_MAX_BOND,
            AciTolerance::ScaleRelative,
        )
    });
    let global_aci = global_aci_result?;
    let global_aci_error = sampled_relative_l2_vs_product(
        &global_aci,
        &input.left_mixture,
        &input.right_mixture,
        input.r,
        input.box_l,
        ERROR_SAMPLES,
        seed.wrapping_add(99),
    );
    anyhow::ensure!(
        global_aci_error <= ERROR_SANITY,
        "global ACI error {global_aci_error:.3e}"
    );
    write_global_record(
        out_dir,
        "global_aci",
        &input,
        &config,
        input_chi,
        input_params,
        patch_count,
        input_max_patch,
        patch_params,
        patch_secs,
        left_input_error,
        right_input_error,
        aci_tolerance,
        "aci_scale_relative_residual",
        &global_aci,
        global_aci_error,
        &global_aci_timing,
    )?;

    let (patched_aci_result, patched_aci_timing) = time_median(warmups, runs, || {
        patched_elementwise_with_stats(
            &patched_left,
            &patched_right,
            &sites,
            PatchedProductOptions {
                engine: PatchedEngine::Aci {
                    residual_tolerance: aci_tolerance,
                    output_l2_rtol: INPUT_L2_RTOL,
                },
                max_bond_dim: PATCH_CAP,
            },
        )
    });
    let (patched_aci, patched_aci_stats) = patched_aci_result?;
    let patched_aci_error = sampled_relative_l2_patched(
        &patched_aci,
        &sites,
        &input.left_mixture,
        &input.right_mixture,
        input.box_l,
        ERROR_SAMPLES,
        seed.wrapping_add(99),
    )?;
    anyhow::ensure!(
        patched_aci_error <= ERROR_SANITY,
        "patched ACI error {patched_aci_error:.3e}"
    );
    write_patched_record(
        out_dir,
        "patched_aci",
        &input,
        &config,
        input_chi,
        input_params,
        patch_count,
        input_max_patch,
        patch_params,
        patch_secs,
        left_input_error,
        right_input_error,
        aci_tolerance,
        "aci_absolute_patch_residual_then_global_l2",
        &patched_aci,
        patched_aci_error,
        &patched_aci_timing,
        global_aci_timing.median_secs,
        patched_aci_stats,
        &sites,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn common_params(
    input: &t4a_bench::gaussian_input::GaussianInputPair,
    config: &GaussianInputConfig,
    input_params: usize,
    patch_count: usize,
    input_max_patch: usize,
    patch_params: usize,
    patch_secs: f64,
    left_input_error: f64,
    right_input_error: f64,
    internal_tolerance: f64,
    tolerance_metric: &str,
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
        "input_params": input_params, "input_patch_count": patch_count,
        "input_max_patch_chi": input_max_patch, "input_patch_params": patch_params,
        "cache_key": input.cache_key, "cache_hit": input.cache_hit,
        "cache_load_secs": input.cache_load.as_secs_f64(),
        "input_build_secs": input.build.as_secs_f64(),
        "input_compression_secs": input.compression.as_secs_f64(),
        "patch_build_secs": patch_secs,
        "left_input_sampled_relative_l2": left_input_error,
        "right_input_sampled_relative_l2": right_input_error,
        "error_samples": ERROR_SAMPLES,
        "internal_tolerance": internal_tolerance,
        "internal_tolerance_metric": tolerance_metric,
        "external_error_metric": "sampled_relative_l2"
    })
}

#[allow(clippy::too_many_arguments)]
fn write_global_record(
    out_dir: &Path,
    arm: &str,
    input: &t4a_bench::gaussian_input::GaussianInputPair,
    config: &GaussianInputConfig,
    input_chi: usize,
    input_params: usize,
    patch_count: usize,
    input_max_patch: usize,
    patch_params: usize,
    patch_secs: f64,
    left_input_error: f64,
    right_input_error: f64,
    internal_tolerance: f64,
    tolerance_metric: &str,
    output: &tensor4all_simplett::SimpleTensorTrain<f64>,
    error: f64,
    timing: &t4a_bench::harness::Timing,
) -> anyhow::Result<()> {
    let params = common_params(
        input,
        config,
        input_params,
        patch_count,
        input_max_patch,
        patch_params,
        patch_secs,
        left_input_error,
        right_input_error,
        internal_tolerance,
        tolerance_metric,
    );
    write_record(
        out_dir,
        &format!("{CASE}-{arm}-chi{input_chi}"),
        &RunRecord {
            schema_version: SCHEMA_VERSION,
            case: CASE.into(),
            algorithm: arm.into(),
            params,
            seed: config.seed,
            tolerance: internal_tolerance,
            wall_time_median_secs: timing.median_secs,
            wall_times_secs: timing.runs_secs.clone(),
            max_error: error,
            input_max_bond_dim: input_chi,
            output_max_bond_dim: output.rank(),
            output_bond_dims: output.link_dims(),
            n_params: Some(tt_n_params(output)),
            n_patches: None,
            max_patch_bond: None,
            rtol: (tolerance_metric == "relative_l2_svd").then_some(INPUT_L2_RTOL),
            input_build_secs: Some(input.build.as_secs_f64()),
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn write_patched_record(
    out_dir: &Path,
    arm: &str,
    input: &t4a_bench::gaussian_input::GaussianInputPair,
    config: &GaussianInputConfig,
    input_chi: usize,
    input_params: usize,
    patch_count: usize,
    input_max_patch: usize,
    patch_params: usize,
    patch_secs: f64,
    left_input_error: f64,
    right_input_error: f64,
    internal_tolerance: f64,
    tolerance_metric: &str,
    output: &tensor4all_partitionedtt::PartitionedTT,
    error: f64,
    timing: &t4a_bench::harness::Timing,
    global_time: f64,
    stats: PatchedProductStats,
    sites: &[tensor4all_core::DynIndex],
) -> anyhow::Result<()> {
    let mut params = common_params(
        input,
        config,
        input_params,
        patch_count,
        input_max_patch,
        patch_params,
        patch_secs,
        left_input_error,
        right_input_error,
        internal_tolerance,
        tolerance_metric,
    );
    params["speedup_vs_global"] = serde_json::json!(global_time / timing.median_secs);
    params["last_run_pair_product_secs"] = serde_json::json!(stats.pairs_secs);
    params["last_run_postprocess_secs"] = serde_json::json!(stats.postprocess_secs);
    params["pre_compression_max_bond"] = serde_json::json!(stats.pre_compression_max_bond);
    write_record(
        out_dir,
        &format!("{CASE}-{arm}-chi{input_chi}"),
        &RunRecord {
            schema_version: SCHEMA_VERSION,
            case: CASE.into(),
            algorithm: arm.into(),
            params,
            seed: config.seed,
            tolerance: internal_tolerance,
            wall_time_median_secs: timing.median_secs,
            wall_times_secs: timing.runs_secs.clone(),
            max_error: error,
            input_max_bond_dim: input_chi,
            output_max_bond_dim: max_patch_bond(output),
            output_bond_dims: output.values().map(|patch| patch.max_bond_dim()).collect(),
            n_params: Some(total_params(output, sites)?),
            n_patches: Some(output.len()),
            max_patch_bond: Some(max_patch_bond(output)),
            rtol: Some(INPUT_L2_RTOL),
            input_build_secs: Some(input.build.as_secs_f64()),
        },
    )
}
