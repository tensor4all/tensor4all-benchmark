//! Global and patched chain-TreeTN fit contraction of anisotropic Gaussian MPOs.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use tensor4all_hdf5::{append_mps, load_mps, save_mps};

use t4a_bench::gaussian::{to_quantics_mpo_field, AnisoMixture2D};
use t4a_bench::harness::time_median;
use t4a_bench::mpo_contract::{max_rel_error_vs_aniso_grid, mpo_n_params};
use t4a_bench::patched_mpo::{PatchedMpoOutput, PatchedMpoPair};
use t4a_bench::record::{write_record, RunRecord, SCHEMA_VERSION};

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn main() -> anyhow::Result<()> {
    let n_gauss: usize = env_or("BENCH_NGAUSS", 512);
    let sigma: f64 = env_or("BENCH_ANISO_SIGMA", 0.05);
    let rho_max: f64 = env_or("BENCH_ANISO_RHO_MAX", 8.0);
    let spacing: f64 = env_or("BENCH_ANISO_SPACING", 3.0);
    let box_padding: f64 = env_or("BENCH_BOX_PADDING", 1.0);
    let extra_bits: usize = env_or("BENCH_R_EXTRA", 0);
    let rtol: f64 = env_or("BENCH_RTOL", 1e-8);
    let input_tci_cap: usize = env_or("BENCH_MAX_BOND", 384);
    let patch_cap: usize = env_or("BENCH_PATCH_MAX_BOND", 128);
    let max_input_chi: usize = env_or("BENCH_MAX_INPUT_CHI", 256);
    let runs: usize = env_or("BENCH_RUNS", 1);
    let warmups: usize = env_or("BENCH_WARMUPS", 0);
    let seed: u64 = env_or("BENCH_SEED", 0);
    let sanity: f64 = env_or("BENCH_SANITY", 1e-4);
    let n_error_samples: usize = env_or("BENCH_ERROR_SAMPLES", 128);
    let out_dir =
        PathBuf::from(std::env::var("OUT_DIR").unwrap_or_else(|_| "result/dev/raw".into()));
    let cache_dir = PathBuf::from(
        std::env::var("BENCH_INPUT_CACHE_DIR").unwrap_or_else(|_| ".cache/inputs".into()),
    );
    let refresh_cache = std::env::var_os("BENCH_INPUT_CACHE_REFRESH").is_some();

    anyhow::ensure!(n_gauss > 0 && sigma > 0.0 && spacing > 0.0);
    anyhow::ensure!(box_padding >= 1.0 && rho_max >= 1.0);
    anyhow::ensure!(rtol.is_finite() && rtol >= 0.0);
    anyhow::ensure!(input_tci_cap > 0 && patch_cap > 0 && runs > 0);

    // Same constant-density anisotropic family and resolution as case 5.
    let inner_l = spacing * sigma * (n_gauss as f64).sqrt() / 2.0;
    let box_l = box_padding * inner_l;
    let r = (8.0 * box_l / sigma).log2().ceil().max(1.0) as usize + extra_bits;
    let f = AnisoMixture2D::random(n_gauss, inner_l, sigma, rho_max, seed.wrapping_add(1));
    let g = AnisoMixture2D::random(n_gauss, inner_l, sigma, rho_max, seed.wrapping_add(2));

    fs::create_dir_all(&cache_dir)?;
    let cache_key = format!(
        "aniso-mpo-v1-n{n_gauss}-r{r}-extra{extra_bits}-spacing{:016x}-padding{:016x}-inner{:016x}-box{:016x}-sigma{:016x}-rho{:016x}-rtol{:016x}-cap{input_tci_cap}-seed{seed}",
        spacing.to_bits(),
        box_padding.to_bits(),
        inner_l.to_bits(),
        box_l.to_bits(),
        sigma.to_bits(),
        rho_max.to_bits(),
        rtol.to_bits(),
    );
    let cache_path = cache_dir.join(format!("{cache_key}.h5"));
    let dy = 2.0 * box_l / 2.0_f64.powi(r as i32);
    let build_started = Instant::now();
    let cache_hit = cache_path.is_file() && !refresh_cache;
    let (patched, input_cache_load_secs) = if cache_hit {
        eprintln!("loading cached input {}", cache_path.display());
        let load_started = Instant::now();
        let path = cache_path.to_string_lossy();
        let left = load_mps(&path, "left")?;
        let right = load_mps(&path, "right")?;
        let load_secs = load_started.elapsed().as_secs_f64();
        (
            PatchedMpoPair::from_tensortrains(left, right, rtol, patch_cap)?,
            load_secs,
        )
    } else {
        eprintln!("building N={n_gauss} R={r} box_l={box_l:.6} input_tci_cap={input_tci_cap}");
        let (left, generated_dy) = to_quantics_mpo_field(&f, r, box_l, rtol, input_tci_cap)?;
        let (right, _) = to_quantics_mpo_field(&g, r, box_l, rtol, input_tci_cap)?;
        anyhow::ensure!((generated_dy - dy).abs() <= f64::EPSILON * dy.abs().max(1.0));
        let patched = PatchedMpoPair::new(&left, &right, rtol, patch_cap)?;
        let temporary = cache_dir.join(format!(".{cache_key}.{}.tmp", std::process::id()));
        let temporary_path = temporary.to_string_lossy();
        let (left_train, right_train) = patched.global_inputs();
        let write_result = (|| -> anyhow::Result<()> {
            save_mps(&temporary_path, "left", left_train)?;
            append_mps(&temporary_path, "right", right_train)?;
            fs::rename(&temporary, &cache_path)?;
            Ok(())
        })();
        if write_result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        write_result?;
        (patched, 0.0)
    };
    let input_build_secs = build_started.elapsed().as_secs_f64();
    let raw_input_chi = patched.input_max_bond();
    anyhow::ensure!(
        raw_input_chi <= max_input_chi,
        "input chi {raw_input_chi} exceeds BENCH_MAX_INPUT_CHI={max_input_chi}"
    );
    let input_chi = patched.input_max_bond();
    let input_patches = patched.input_patch_counts();
    let input_patch_bonds = patched.input_patch_max_bonds();
    anyhow::ensure!(
        input_chi == raw_input_chi,
        "MPO bridge changed the input rank"
    );
    eprintln!(
        "prepared input_chi={input_chi} patches={input_patches:?} patch_chi={input_patch_bonds:?} build={input_build_secs:.3}s"
    );

    let (global_train, global_timing) = time_median(warmups, runs, || {
        patched
            .contract_fit_global(rtol, input_chi)
            .expect("global fit contraction failed")
    });
    let (tree_partitioned, tree_timing) = time_median(warmups, runs, || {
        patched
            .contract_fit_treetn_partitioned(rtol, patch_cap)
            .expect("PartitionedTreeTN fit contraction failed")
    });
    let global_output = patched.finish_global_output(&global_train)?;
    let tree_output = patched.finish_treetn_output(tree_partitioned)?;

    let error_seed = seed.wrapping_add(99);
    let global_error = max_rel_error_vs_aniso_grid(
        &global_output,
        dy,
        &f,
        &g,
        r,
        box_l,
        n_error_samples,
        error_seed,
    );
    let tree_error = max_rel_error_vs_aniso_grid(
        &tree_output.mpo,
        dy,
        &f,
        &g,
        r,
        box_l,
        n_error_samples,
        error_seed,
    );
    let common_params = serde_json::json!({
        "n_gauss": n_gauss,
        "r": r,
        "inner_l": inner_l,
        "box_l": box_l,
        "box_padding": box_padding,
        "sigma": sigma,
        "rho_max": rho_max,
        "spacing": spacing,
        "r_extra": extra_bits,
        "input_tci_cap": input_tci_cap,
        "input_patch_cap": patch_cap,
        "input_patch_counts": [input_patches.0, input_patches.1],
        "input_patch_max_bonds": [input_patch_bonds.0, input_patch_bonds.1],
        "global_contract_max_bond": input_chi,
        "patched_contract_max_bond": patch_cap,
        "fit_nfullsweeps": 1,
        "contraction_svd_policy": "relative_squared_discarded_tail_sum",
        "split_strategy": "sequential_y_first",
        "runs": runs,
        "warmups": warmups,
        "n_error_samples": n_error_samples,
        "error_seed": error_seed,
        "input_cache_key": cache_key,
        "input_cache_hit": cache_hit,
        "input_cache_load_secs": input_cache_load_secs,
        "error_metric": "max_rel_vs_quantics_grid_reference",
    });
    let global_record = RunRecord {
        schema_version: SCHEMA_VERSION,
        case: "mpo_mpo_aniso_patched".into(),
        algorithm: "global_fit_treetn".into(),
        params: common_params.clone(),
        seed,
        tolerance: rtol,
        wall_time_median_secs: global_timing.median_secs,
        wall_times_secs: global_timing.runs_secs.clone(),
        max_error: global_error,
        input_max_bond_dim: input_chi,
        output_max_bond_dim: global_output.rank(),
        output_bond_dims: global_output.link_dims(),
        n_params: Some(mpo_n_params(&global_output)),
        n_patches: None,
        max_patch_bond: None,
        rtol: Some(rtol),
        input_build_secs: Some(input_build_secs),
    };
    write_record(
        &out_dir,
        &format!("mpo_mpo_aniso_patched-global_fit_treetn-n{n_gauss}-chi{input_chi}"),
        &global_record,
    )?;
    write_arm(
        &out_dir,
        "patched_fit_treetn",
        &tree_output,
        &tree_timing,
        tree_error,
        input_chi,
        input_build_secs,
        seed,
        rtol,
        &common_params,
    )?;

    eprintln!(
        "global_fit_treetn: t={:.6}s error={global_error:.3e} output_chi={}",
        global_timing.median_secs,
        global_output.rank()
    );
    eprintln!(
        "patched_fit_treetn: t={:.6}s error={tree_error:.3e} output_patches={} output_chi={}",
        tree_timing.median_secs, tree_output.n_patches, tree_output.max_patch_bond
    );
    anyhow::ensure!(
        global_error <= sanity,
        "global fit relative error {global_error:.3e} > {sanity:.3e}"
    );
    anyhow::ensure!(
        tree_error <= sanity,
        "TreeTN relative error {tree_error:.3e} > {sanity:.3e}"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_arm(
    out_dir: &std::path::Path,
    algorithm: &str,
    output: &PatchedMpoOutput,
    timing: &t4a_bench::harness::Timing,
    max_error: f64,
    input_chi: usize,
    input_build_secs: f64,
    seed: u64,
    rtol: f64,
    params: &serde_json::Value,
) -> anyhow::Result<()> {
    let record = RunRecord {
        schema_version: SCHEMA_VERSION,
        case: "mpo_mpo_aniso_patched".into(),
        algorithm: algorithm.into(),
        params: params.clone(),
        seed,
        tolerance: rtol,
        wall_time_median_secs: timing.median_secs,
        wall_times_secs: timing.runs_secs.clone(),
        max_error,
        input_max_bond_dim: input_chi,
        output_max_bond_dim: output.mpo.rank(),
        output_bond_dims: output.mpo.link_dims(),
        n_params: None,
        n_patches: Some(output.n_patches),
        max_patch_bond: Some(output.max_patch_bond),
        rtol: Some(rtol),
        input_build_secs: Some(input_build_secs),
    };
    write_record(
        out_dir,
        &format!(
            "mpo_mpo_aniso_patched-{algorithm}-n{}-chi{input_chi}",
            params["n_gauss"]
        ),
        &record,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn cached_inputs_round_trip_with_shared_index_identity() {
        let field = AnisoMixture2D::random(2, 1.0, 0.2, 3.0, 7);
        let other = AnisoMixture2D::random(2, 1.0, 0.2, 3.0, 8);
        let (left, _) = to_quantics_mpo_field(&field, 4, 1.0, 1.0e-8, 32).unwrap();
        let (right, _) = to_quantics_mpo_field(&other, 4, 1.0, 1.0e-8, 32).unwrap();
        let original = PatchedMpoPair::new(&left, &right, 1.0e-8, 16).unwrap();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("t4a-mpo-cache-{}-{nonce}.h5", std::process::id()));
        let path_str = path.to_string_lossy();
        let (left_train, right_train) = original.global_inputs();
        save_mps(&path_str, "left", left_train).unwrap();
        append_mps(&path_str, "right", right_train).unwrap();

        let cached = PatchedMpoPair::from_tensortrains(
            load_mps(&path_str, "left").unwrap(),
            load_mps(&path_str, "right").unwrap(),
            1.0e-8,
            16,
        )
        .unwrap();
        fs::remove_file(path).unwrap();

        assert_eq!(cached.input_max_bond(), original.input_max_bond());
        assert_eq!(cached.input_patch_counts(), original.input_patch_counts());
        assert_eq!(
            cached.input_patch_max_bonds(),
            original.input_patch_max_bonds()
        );
    }
}
