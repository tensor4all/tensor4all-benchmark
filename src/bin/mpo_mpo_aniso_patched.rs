//! Case 3: global and patched fit contraction of interpolative Gaussian MPOs.

#![recursion_limit = "256"]

use std::path::{Path, PathBuf};
use std::time::Instant;

use tensor4all_simplett::AbstractTensorTrain;

use t4a_bench::gaussian::fused_qtt_to_mpo;
use t4a_bench::gaussian_input::{
    prepare_gaussian_pair_with_l2_rtol, principal_axis_input_relative_l2,
    sampled_input_relative_l2, tensortrain_n_params as simple_n_params, GaussianInputConfig,
    INPUT_L2_RTOL, INPUT_LOCAL_ABS_TOLERANCE, INPUT_TCI_MAX_BOND, INPUT_TCI_PIVOT_COMPONENTS,
    INPUT_TCI_TOLERANCE, PATCH_CAP,
};
use t4a_bench::harness::time_median;
use t4a_bench::integrated_gaussian::{
    count_integrated_components, prepare_integrated_reference, OUTPUT_REFERENCE_ABS_TOLERANCE,
};
use t4a_bench::mpo_contract::center_errors_vs_integrated_gaussians;
use t4a_bench::patched_mpo::{partitioned_treetn_n_params, tensortrain_n_params, PatchedMpoPair};
use t4a_bench::record::{write_record, RunRecord, SCHEMA_VERSION};

const CASE: &str = "gaussian_mpo_contraction";
const GLOBAL_MAX_BOND: usize = 4096;
const ERROR_SAMPLES: usize = 256;
const ERROR_SANITY: f64 = 1e-4;

fn stable_key_hash(value: &str) -> u64 {
    value
        .as_bytes()
        .iter()
        .fold(0xcbf29ce484222325, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        })
}

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
    for n in ns {
        run_point(n, seed, runs, warmups, &cache_dir, refresh, &out_dir)?;
    }
    Ok(())
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
) -> anyhow::Result<()> {
    let config = GaussianInputConfig {
        n,
        sigma_minor: 0.05,
        rho_max: 8.0,
        spacing: 3.0,
        tci_tolerance: INPUT_TCI_TOLERANCE,
        tci_max_bond_dim: INPUT_TCI_MAX_BOND,
        localized_absolute_tolerance: INPUT_LOCAL_ABS_TOLERANCE,
        tci_pivot_components: INPUT_TCI_PIVOT_COMPONENTS,
        seed,
        cache_dir: cache_dir.to_path_buf(),
        refresh,
    };
    let patch_only = env_or("BENCH_PATCH_ONLY", 0usize) != 0;
    let input_l2_rtol = env_or("BENCH_INPUT_L2_RTOL", INPUT_L2_RTOL);
    let input = prepare_gaussian_pair_with_l2_rtol(&config, input_l2_rtol)?;
    let (left_input_error, right_input_error) = if patch_only {
        (0.0, 0.0)
    } else {
        sampled_input_relative_l2(&input, ERROR_SAMPLES, seed.wrapping_add(41))?
    };
    anyhow::ensure!(
        left_input_error <= ERROR_SANITY && right_input_error <= ERROR_SANITY,
        "input error exceeds sanity gate: ({left_input_error:.3e}, {right_input_error:.3e})"
    );
    let (left_axis_error, right_axis_error) =
        principal_axis_input_relative_l2(&input, INPUT_TCI_PIVOT_COMPONENTS)?;
    anyhow::ensure!(
        left_axis_error <= ERROR_SANITY && right_axis_error <= ERROR_SANITY,
        "principal-axis input error exceeds sanity gate: ({left_axis_error:.3e}, {right_axis_error:.3e})"
    );
    let left_mpo = fused_qtt_to_mpo(&input.left)?;
    let right_mpo = fused_qtt_to_mpo(&input.right)?;
    let patch_start = Instant::now();
    let prepared = PatchedMpoPair::new(&left_mpo, &right_mpo, INPUT_L2_RTOL, PATCH_CAP)?;
    let patch_secs = patch_start.elapsed().as_secs_f64();
    let (left_patch_count, right_patch_count) = prepared.input_patch_counts();
    let (x_patch_count, left_y_patch_count, right_y_patch_count, z_patch_count) =
        prepared.input_axis_patch_counts();
    let y_patch_count = left_y_patch_count.max(right_y_patch_count);
    let (compatible_pair_count, output_projector_count) =
        prepared.input_contraction_layout_counts();
    let cartesian_pair_count = x_patch_count
        .checked_mul(y_patch_count)
        .and_then(|count| count.checked_mul(z_patch_count))
        .ok_or_else(|| anyhow::anyhow!("Cartesian compatible-pair estimate overflow"))?;
    let cartesian_output_patch_count = x_patch_count
        .checked_mul(z_patch_count)
        .ok_or_else(|| anyhow::anyhow!("Cartesian output-patch estimate overflow"))?;
    let (left_patch_chi, right_patch_chi) = prepared.input_patch_max_bonds();
    let (left_patch_params, right_patch_params) = prepared.input_patch_n_params();
    anyhow::ensure!(
        left_patch_chi.max(right_patch_chi) <= PATCH_CAP,
        "input patch cap exceeded"
    );
    let input_chi = input.left.rank().max(input.right.rank());
    let input_params = simple_n_params(&input.left) + simple_n_params(&input.right);

    if patch_only {
        let reference_count_start = Instant::now();
        let integrated_stats = count_integrated_components(
            &input.left_mixture,
            &input.right_mixture,
            OUTPUT_REFERENCE_ABS_TOLERANCE,
        )?;
        let reference_count_secs = reference_count_start.elapsed().as_secs_f64();
        let estimated_component_bytes = integrated_stats
            .retained_pair_count
            .checked_mul(6 * std::mem::size_of::<f64>())
            .ok_or_else(|| anyhow::anyhow!("integrated Gaussian storage estimate overflow"))?;
        let mut params = common_params(
            &input,
            &config,
            input_params,
            input_l2_rtol,
            left_patch_count,
            right_patch_count,
            x_patch_count,
            y_patch_count,
            z_patch_count,
            left_y_patch_count,
            right_y_patch_count,
            compatible_pair_count,
            output_projector_count,
            cartesian_pair_count,
            cartesian_output_patch_count,
            left_patch_chi,
            right_patch_chi,
            left_patch_params,
            right_patch_params,
            patch_secs,
            left_input_error,
            right_input_error,
            left_axis_error,
            right_axis_error,
        );
        params["compatible_pair_count"] = serde_json::json!(compatible_pair_count);
        params["output_projector_count"] = serde_json::json!(output_projector_count);
        params["cartesian_pair_count"] = serde_json::json!(cartesian_pair_count);
        params["cartesian_output_patch_count"] = serde_json::json!(cartesian_output_patch_count);
        params["integrated_total_pair_count"] =
            serde_json::json!(integrated_stats.total_pair_count);
        params["integrated_candidate_pair_count"] =
            serde_json::json!(integrated_stats.candidate_pair_count);
        params["integrated_retained_pair_count"] =
            serde_json::json!(integrated_stats.retained_pair_count);
        params["integrated_count_secs"] = serde_json::json!(reference_count_secs);
        params["integrated_y_cell_width"] = serde_json::json!(integrated_stats.y_cell_width);
        params["integrated_estimated_component_bytes"] =
            serde_json::json!(estimated_component_bytes);
        params["integrated_omitted_absolute_bound"] =
            serde_json::json!(integrated_stats.omitted_absolute_bound);
        params["left_input_sampled_relative_l2"] = serde_json::Value::Null;
        params["right_input_sampled_relative_l2"] = serde_json::Value::Null;
        params["error_samples"] = serde_json::json!(0);
        params["external_error_metric"] = serde_json::json!("principal_axis_relative_l2");
        write_record(
            out_dir,
            &format!("gaussian_mpo_patch_scaling-n{n}-chi{input_chi}"),
            &RunRecord {
                schema_version: SCHEMA_VERSION,
                case: "gaussian_mpo_patch_scaling".into(),
                algorithm: "balanced_input_patching".into(),
                params,
                seed,
                tolerance: INPUT_L2_RTOL,
                wall_time_median_secs: patch_secs,
                wall_times_secs: vec![patch_secs],
                max_error: left_axis_error.max(right_axis_error),
                input_max_bond_dim: input_chi,
                output_max_bond_dim: left_patch_chi.max(right_patch_chi),
                output_bond_dims: vec![left_patch_chi, right_patch_chi],
                n_params: Some(left_patch_params + right_patch_params),
                n_patches: Some(left_patch_count + right_patch_count),
                max_patch_bond: Some(left_patch_chi.max(right_patch_chi)),
                rtol: Some(INPUT_L2_RTOL),
                input_build_secs: Some(input.build.as_secs_f64()),
            },
        )?;
        return Ok(());
    }

    let arm = std::env::var("BENCH_ARM").unwrap_or_else(|_| "both".into());
    anyhow::ensure!(
        matches!(arm.as_str(), "both" | "global" | "patched"),
        "BENCH_ARM must be 'both', 'global', or 'patched'"
    );
    let global_measurement = if arm != "patched" {
        let (result, timing) = time_median(warmups, runs, || {
            prepared.contract_fit_global(INPUT_L2_RTOL, GLOBAL_MAX_BOND)
        });
        Some((result?, timing))
    } else {
        None
    };
    let patched_measurement = if arm != "global" {
        let (result, timing) = time_median(warmups, runs, || {
            prepared.contract_fit_treetn_partitioned(INPUT_L2_RTOL, PATCH_CAP)
        });
        Some((result?, timing))
    } else {
        None
    };

    // Reference preparation and validation are deliberately after every timed
    // operation so their cost cannot delay or contaminate fit profiling.
    let reference_cache = cache_dir.join(format!(
        "integrated-v2-{:016x}-tol{:016x}.bin",
        stable_key_hash(&input.cache_key),
        OUTPUT_REFERENCE_ABS_TOLERANCE.to_bits()
    ));
    let reference = prepare_integrated_reference(
        &input.left_mixture,
        &input.right_mixture,
        OUTPUT_REFERENCE_ABS_TOLERANCE,
        &reference_cache,
        refresh,
    )?;
    let reference_cache_bytes = std::fs::metadata(&reference.cache_path)?.len();
    let grid_step = 2.0 * input.box_l / (1usize << input.r) as f64;
    let mut common = common_params(
        &input,
        &config,
        input_params,
        input_l2_rtol,
        left_patch_count,
        right_patch_count,
        x_patch_count,
        y_patch_count,
        z_patch_count,
        left_y_patch_count,
        right_y_patch_count,
        compatible_pair_count,
        output_projector_count,
        cartesian_pair_count,
        cartesian_output_patch_count,
        left_patch_chi,
        right_patch_chi,
        left_patch_params,
        right_patch_params,
        patch_secs,
        left_input_error,
        right_input_error,
        left_axis_error,
        right_axis_error,
    );
    common["integrated_reference_cache_hit"] = serde_json::json!(reference.cache_hit);
    common["integrated_reference_cache_load_secs"] =
        serde_json::json!(reference.cache_load.as_secs_f64());
    common["integrated_reference_build_secs"] = serde_json::json!(reference.build.as_secs_f64());
    common["integrated_reference_cache_bytes"] = serde_json::json!(reference_cache_bytes);
    common["integrated_reference_components"] =
        serde_json::json!(reference.stats.retained_pair_count);
    common["integrated_reference_candidate_pairs"] =
        serde_json::json!(reference.stats.candidate_pair_count);
    common["integrated_reference_spatial_bins"] = serde_json::json!(reference.field.bin_count());
    common["integrated_reference_absolute_tolerance"] =
        serde_json::json!(OUTPUT_REFERENCE_ABS_TOLERANCE);
    common["center_error_samples"] =
        serde_json::json!(ERROR_SAMPLES.min(reference.stats.retained_pair_count));
    common["external_error_metric"] = serde_json::json!("retained_center_relative_l2");

    let global_time = global_measurement
        .as_ref()
        .map(|(_, timing)| timing.median_secs);
    if let Some((global, global_timing)) = global_measurement {
        let global_params = tensortrain_n_params(&global);
        let global_chi = global.max_bond_dim();
        let global_mpo = prepared.finish_global_output(&global)?;
        let (global_error, global_max_scaled_error) = center_errors_vs_integrated_gaussians(
            &global_mpo,
            grid_step,
            &reference.field,
            input.r,
            input.box_l,
            ERROR_SAMPLES,
        )?;
        anyhow::ensure!(
            global_error <= ERROR_SANITY,
            "global contraction error {global_error:.3e}"
        );
        let mut global_common = common.clone();
        global_common["global_max_rms_scaled_error"] = serde_json::json!(global_max_scaled_error);
        write_record(
            out_dir,
            &format!("{CASE}-global_fit-chi{input_chi}"),
            &RunRecord {
                schema_version: SCHEMA_VERSION,
                case: CASE.into(),
                algorithm: "global_fit".into(),
                params: global_common,
                seed,
                tolerance: INPUT_L2_RTOL,
                wall_time_median_secs: global_timing.median_secs,
                wall_times_secs: global_timing.runs_secs,
                max_error: global_error,
                input_max_bond_dim: input_chi,
                output_max_bond_dim: global_chi,
                output_bond_dims: global.bond_dims(),
                n_params: Some(global_params),
                n_patches: None,
                max_patch_bond: None,
                rtol: Some(INPUT_L2_RTOL),
                input_build_secs: Some(input.build.as_secs_f64()),
            },
        )?;
    }

    if let Some((patched, patched_timing)) = patched_measurement {
        let patched_params = partitioned_treetn_n_params(&patched);
        let patched_finished = prepared.finish_treetn_output(patched)?;
        let (patched_error, patched_max_scaled_error) = center_errors_vs_integrated_gaussians(
            &patched_finished.mpo,
            grid_step,
            &reference.field,
            input.r,
            input.box_l,
            ERROR_SAMPLES,
        )?;
        anyhow::ensure!(
            patched_error <= ERROR_SANITY,
            "patched contraction error {patched_error:.3e}"
        );
        let mut patched_common = common;
        if let Some(global_time) = global_time {
            patched_common["speedup_vs_global"] =
                serde_json::json!(global_time / patched_timing.median_secs);
        }
        patched_common["patched_max_rms_scaled_error"] =
            serde_json::json!(patched_max_scaled_error);
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
    input_l2_rtol: f64,
    left_patch_count: usize,
    right_patch_count: usize,
    x_patch_count: usize,
    y_patch_count: usize,
    z_patch_count: usize,
    left_y_patch_count: usize,
    right_y_patch_count: usize,
    compatible_pair_count: usize,
    output_projector_count: usize,
    cartesian_pair_count: usize,
    cartesian_output_patch_count: usize,
    left_patch_chi: usize,
    right_patch_chi: usize,
    left_patch_params: usize,
    right_patch_params: usize,
    patch_secs: f64,
    left_input_error: f64,
    right_input_error: f64,
    left_axis_error: f64,
    right_axis_error: f64,
) -> serde_json::Value {
    serde_json::json!({
        "n_gauss": config.n, "r": input.r,
        "active_box_l": input.active_box_l, "box_l": input.box_l,
        "padding_factor": input.box_l / input.active_box_l,
        "sigma_minor": config.sigma_minor, "rho_max": config.rho_max,
        "spacing": config.spacing, "input_generator": "global_tci",
        "input_tci_tolerance": config.tci_tolerance,
        "input_tci_max_bond_dim": config.tci_max_bond_dim,
        "input_localized_absolute_tolerance": config.localized_absolute_tolerance,
        "input_tci_pivot_components": config.tci_pivot_components,
        "input_l2_rtol": input_l2_rtol,
        "patch_input_rtol": INPUT_L2_RTOL, "patch_cap": PATCH_CAP,
        "raw_left_chi": input.raw_left_chi, "raw_right_chi": input.raw_right_chi,
        "left_chi": input.left.rank(), "right_chi": input.right.rank(),
        "raw_left_params": input.raw_left_params, "raw_right_params": input.raw_right_params,
        "input_params": input_params,
        "left_input_patch_count": left_patch_count, "right_input_patch_count": right_patch_count,
        "x_patch_count": x_patch_count, "y_patch_count": y_patch_count,
        "left_y_patch_count": left_y_patch_count,
        "right_y_patch_count": right_y_patch_count,
        "y_patch_layouts_match": left_y_patch_count == right_y_patch_count,
        "z_patch_count": z_patch_count,
        "compatible_pair_count": compatible_pair_count,
        "output_projector_count": output_projector_count,
        "cartesian_pair_count": cartesian_pair_count,
        "cartesian_output_patch_count": cartesian_output_patch_count,
        "patch_layout": "balanced_xyz",
        "output_sum_method": "cap_initial_then_fit_sum",
        "output_max_bond_dim": serde_json::Value::Null,
        "left_input_max_patch_chi": left_patch_chi, "right_input_max_patch_chi": right_patch_chi,
        "left_input_patch_params": left_patch_params, "right_input_patch_params": right_patch_params,
        "cache_key": input.cache_key, "cache_hit": input.cache_hit,
        "cache_load_secs": input.cache_load.as_secs_f64(),
        "input_build_secs": input.build.as_secs_f64(),
        "input_compression_secs": input.compression.as_secs_f64(),
        "patch_build_secs": patch_secs,
        "left_input_sampled_relative_l2": left_input_error,
        "right_input_sampled_relative_l2": right_input_error,
        "left_input_principal_axis_relative_l2": left_axis_error,
        "right_input_principal_axis_relative_l2": right_axis_error,
        "error_samples": ERROR_SAMPLES,
        "internal_tolerance": INPUT_L2_RTOL,
        "internal_tolerance_metric": "relative_l2_svd",
        "external_error_metric": "sampled_relative_l2"
    })
}
