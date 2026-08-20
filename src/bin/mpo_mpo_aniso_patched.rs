//! Global and patched chain-TreeTN fit contraction of anisotropic Gaussian MPOs.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use tensor4all_hdf5::{append_mps, load_mps, save_mps};
use tensor4all_itensorlike::TensorTrain;
use tensor4all_simplett::mpo::MPO;

use t4a_bench::gaussian::{
    grid_coord, to_quantics_mpo_field_with_pivots, AnisoMixture2D, Field2D, LocalizedAnisoField,
};
use t4a_bench::harness::{index_to_bits, sample_grid_indices, time_median};
use t4a_bench::mpo_contract::{max_rel_error_vs_aniso_grid, mpo_n_params, tensortrain_to_mpo};
use t4a_bench::patched_mpo::{
    mpo_pair_to_tensortrains, tensortrain_n_params, truncate_tensortrain_l2, PatchedMpoOutput,
    PatchedMpoPair,
};
use t4a_bench::record::{write_record, RunRecord, SCHEMA_VERSION};

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn tensortrain_pair_to_mpos(
    left: &TensorTrain,
    right: &TensorTrain,
) -> anyhow::Result<(MPO<f64>, MPO<f64>)> {
    let left_sites = left.site_indices();
    let right_sites = right.site_indices();
    let mut x = Vec::with_capacity(left.len());
    let mut y = Vec::with_capacity(left.len());
    let mut z = Vec::with_capacity(left.len());
    for site in 0..left.len() {
        let common = left_sites[site]
            .iter()
            .find(|index| right_sites[site].contains(index))
            .ok_or_else(|| anyhow::anyhow!("input site {site} has no shared index"))?;
        let left_unique = left_sites[site]
            .iter()
            .find(|index| *index != common)
            .ok_or_else(|| anyhow::anyhow!("left input site {site} has no unique index"))?;
        let right_unique = right_sites[site]
            .iter()
            .find(|index| *index != common)
            .ok_or_else(|| anyhow::anyhow!("right input site {site} has no unique index"))?;
        x.push(left_unique.clone());
        y.push(common.clone());
        z.push(right_unique.clone());
    }
    Ok((
        tensortrain_to_mpo(left, &x, &y)?,
        tensortrain_to_mpo(right, &y, &z)?,
    ))
}

fn max_rel_input_error(
    mpo: &MPO<f64>,
    field: &impl Field2D,
    r: usize,
    box_l: f64,
    n_samples: usize,
    seed: u64,
) -> anyhow::Result<f64> {
    let xs = sample_grid_indices(r, n_samples, seed);
    let ys = sample_grid_indices(r, n_samples, seed.wrapping_add(1));
    let mut max_error = 0.0_f64;
    let mut max_reference = 0.0_f64;
    for (&ix, &iy) in xs.iter().zip(&ys) {
        let x = grid_coord(ix, r, box_l);
        let y = grid_coord(iy, r, box_l);
        let indices = index_to_bits(ix, r)
            .into_iter()
            .zip(index_to_bits(iy, r))
            .flat_map(|(x_bit, y_bit)| [x_bit, y_bit])
            .collect::<Vec<_>>();
        let reference = field.eval(x, y);
        let value = mpo.evaluate(&indices)?;
        max_error = max_error.max((value - reference).abs());
        max_reference = max_reference.max(reference.abs());
    }
    Ok(max_error / max_reference.max(f64::MIN_POSITIVE))
}

fn center_grid_pivots(
    mixture: &AnisoMixture2D,
    count: usize,
    r: usize,
    box_l: f64,
) -> Vec<Vec<usize>> {
    let grid_size = 1usize << r;
    (0..count.min(mixture.centers.len()))
        .map(|pivot| {
            let component = pivot * mixture.centers.len() / count.min(mixture.centers.len());
            let (x, y) = mixture.centers[component];
            [x, y]
                .into_iter()
                .map(|coordinate| {
                    ((coordinate + box_l) * grid_size as f64 / (2.0 * box_l)).floor() as usize
                })
                .map(|index| index.min(grid_size - 1))
                .collect()
        })
        .collect()
}

fn main() -> anyhow::Result<()> {
    let n_gauss: usize = env_or("BENCH_NGAUSS", 512);
    let sigma: f64 = env_or("BENCH_ANISO_SIGMA", 0.05);
    let rho_max: f64 = env_or("BENCH_ANISO_RHO_MAX", 8.0);
    let spacing: f64 = env_or("BENCH_ANISO_SPACING", 3.0);
    let box_padding: f64 = env_or("BENCH_BOX_PADDING", 1.0);
    let extra_bits: usize = env_or("BENCH_R_EXTRA", 0);
    let rtol: f64 = env_or("BENCH_RTOL", 1e-8);
    let input_generator = std::env::var("BENCH_INPUT_GENERATOR").unwrap_or_else(|_| "tci".into());
    let input_tci_rtol: f64 = env_or("BENCH_INPUT_TCI_RTOL", 1e-8);
    let input_poly_degree: usize = env_or("BENCH_INPUT_POLY_DEGREE", 28);
    let input_add_rtol: f64 = env_or("BENCH_INPUT_ADD_RTOL", 1e-10);
    let input_tci_local_abs_tol: f64 = env_or("BENCH_TCI_LOCAL_ABS_TOL", 1e-12);
    let input_tci_initial_pivots: usize = env_or("BENCH_TCI_INITIAL_PIVOTS", 8);
    let input_svd_l2_rtol: f64 = env_or("BENCH_INPUT_SVD_RTOL", 1e-6);
    let input_only = std::env::var_os("BENCH_INPUT_ONLY").is_some();
    let input_tci_cap: usize = env_or("BENCH_MAX_BOND", 384);
    let patch_cap: usize = env_or("BENCH_PATCH_MAX_BOND", 128);
    let max_input_chi: usize = env_or("BENCH_MAX_INPUT_CHI", 256);
    let runs: usize = env_or("BENCH_RUNS", 1);
    let warmups: usize = env_or("BENCH_WARMUPS", 0);
    let seed: u64 = env_or("BENCH_SEED", 0);
    let sanity: f64 = env_or("BENCH_SANITY", 1e-4);
    let input_sanity: f64 = env_or("BENCH_INPUT_SANITY", 1e-4);
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
    anyhow::ensure!(input_generator == "tci" || input_generator == "multiscale");
    anyhow::ensure!(input_tci_rtol.is_finite() && input_tci_rtol >= 0.0);
    anyhow::ensure!(input_poly_degree >= 2);
    anyhow::ensure!(input_add_rtol.is_finite() && input_add_rtol >= 0.0);
    anyhow::ensure!(input_tci_local_abs_tol.is_finite() && input_tci_local_abs_tol > 0.0);
    anyhow::ensure!(input_tci_initial_pivots > 0);
    anyhow::ensure!(input_svd_l2_rtol.is_finite() && input_svd_l2_rtol >= 0.0);
    anyhow::ensure!(input_tci_cap > 0 && patch_cap > 0 && runs > 0);

    // Same constant-density anisotropic family and resolution as case 5.
    let inner_l = spacing * sigma * (n_gauss as f64).sqrt() / 2.0;
    let box_l = box_padding * inner_l;
    let r = (8.0 * box_l / sigma).log2().ceil().max(1.0) as usize + extra_bits;
    let f = AnisoMixture2D::random(n_gauss, inner_l, sigma, rho_max, seed.wrapping_add(1));
    let g = AnisoMixture2D::random(n_gauss, inner_l, sigma, rho_max, seed.wrapping_add(2));

    fs::create_dir_all(&cache_dir)?;
    let cache_key = if input_generator == "tci" {
        format!(
            "aniso-v5-tci-n{n_gauss}-r{r}-e{extra_bits}-s{:x}-p{:x}-sig{:x}-rho{:x}-tol{:x}-loc{:x}-piv{input_tci_initial_pivots}-cap{input_tci_cap}-seed{seed}",
            spacing.to_bits(),
            box_padding.to_bits(),
            sigma.to_bits(),
            rho_max.to_bits(),
            input_tci_rtol.to_bits(),
            input_tci_local_abs_tol.to_bits(),
        )
    } else {
        format!(
            "aniso-v4-ms-n{n_gauss}-r{r}-e{extra_bits}-s{:x}-p{:x}-sig{:x}-rho{:x}-tol{:x}-d{input_poly_degree}-add{:x}-seed{seed}",
            spacing.to_bits(),
            box_padding.to_bits(),
            sigma.to_bits(),
            rho_max.to_bits(),
            input_tci_rtol.to_bits(),
            input_add_rtol.to_bits(),
        )
    };
    let cache_path = cache_dir.join(format!("{cache_key}.h5"));
    let dy = 2.0 * box_l / 2.0_f64.powi(r as i32);
    let build_started = Instant::now();
    let cache_hit = cache_path.is_file() && !refresh_cache;
    let (mut left_train, mut right_train, input_cache_load_secs) = if cache_hit {
        eprintln!("loading cached input {}", cache_path.display());
        let load_started = Instant::now();
        let path = cache_path.to_string_lossy();
        let left = load_mps(&path, "left")?;
        let right = load_mps(&path, "right")?;
        (left, right, load_started.elapsed().as_secs_f64())
    } else {
        eprintln!(
            "building generator={input_generator} N={n_gauss} R={r} box_l={box_l:.6} input_rtol={input_tci_rtol:.1e} input_tci_cap={input_tci_cap}"
        );
        let (left, right) = if input_generator == "tci" {
            let localized_f = LocalizedAnisoField::new(f.clone(), input_tci_local_abs_tol)?;
            let localized_g = LocalizedAnisoField::new(g.clone(), input_tci_local_abs_tol)?;
            let (left, generated_dy) = to_quantics_mpo_field_with_pivots(
                &localized_f,
                r,
                box_l,
                input_tci_rtol,
                input_tci_cap,
                center_grid_pivots(&f, input_tci_initial_pivots, r, box_l),
            )?;
            let (right, _) = to_quantics_mpo_field_with_pivots(
                &localized_g,
                r,
                box_l,
                input_tci_rtol,
                input_tci_cap,
                center_grid_pivots(&g, input_tci_initial_pivots, r, box_l),
            )?;
            anyhow::ensure!((generated_dy - dy).abs() <= f64::EPSILON * dy.abs().max(1.0));
            (left, right)
        } else {
            (
                f.to_multiscale_mpo(r, box_l, input_poly_degree, input_tci_rtol, input_add_rtol)?,
                g.to_multiscale_mpo(r, box_l, input_poly_degree, input_tci_rtol, input_add_rtol)?,
            )
        };
        let (left_train, right_train) = mpo_pair_to_tensortrains(&left, &right)?;
        let temporary = cache_dir.join(format!(".{cache_key}.{}.tmp", std::process::id()));
        let temporary_path = temporary.to_string_lossy();
        let write_result = (|| -> anyhow::Result<()> {
            save_mps(&temporary_path, "left", &left_train)?;
            append_mps(&temporary_path, "right", &right_train)?;
            fs::rename(&temporary, &cache_path)?;
            Ok(())
        })();
        if write_result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        write_result?;
        (left_train, right_train, 0.0)
    };
    anyhow::ensure!(
        left_train.len() == r && right_train.len() == r,
        "cached input length does not match R={r}"
    );
    let raw_input_chi = left_train.max_bond_dim().max(right_train.max_bond_dim());
    let raw_input_params = (
        tensortrain_n_params(&left_train),
        tensortrain_n_params(&right_train),
    );
    let compression_started = Instant::now();
    truncate_tensortrain_l2(&mut left_train, input_svd_l2_rtol)?;
    truncate_tensortrain_l2(&mut right_train, input_svd_l2_rtol)?;
    let input_compression_secs = compression_started.elapsed().as_secs_f64();
    let input_chi = left_train.max_bond_dim().max(right_train.max_bond_dim());
    anyhow::ensure!(
        input_chi <= max_input_chi,
        "compressed input chi {input_chi} exceeds BENCH_MAX_INPUT_CHI={max_input_chi}"
    );
    let input_params = (
        tensortrain_n_params(&left_train),
        tensortrain_n_params(&right_train),
    );
    let (left_mpo, right_mpo) = tensortrain_pair_to_mpos(&left_train, &right_train)?;
    let input_errors = (
        max_rel_input_error(
            &left_mpo,
            &f,
            r,
            box_l,
            n_error_samples,
            seed.wrapping_add(11),
        )?,
        max_rel_input_error(
            &right_mpo,
            &g,
            r,
            box_l,
            n_error_samples,
            seed.wrapping_add(12),
        )?,
    );
    anyhow::ensure!(
        input_errors.0 <= input_sanity && input_errors.1 <= input_sanity,
        "input sampled relative errors {input_errors:?} exceed {input_sanity:.3e}"
    );
    let patched = PatchedMpoPair::from_tensortrains_with_input_rtol(
        left_train,
        right_train,
        input_svd_l2_rtol,
        rtol,
        patch_cap,
    )?;
    let input_build_secs = build_started.elapsed().as_secs_f64();
    let input_patches = patched.input_patch_counts();
    let input_patch_bonds = patched.input_patch_max_bonds();
    let input_patch_params = patched.input_patch_n_params();
    eprintln!(
        "prepared raw_chi={raw_input_chi} input_chi={input_chi} raw_params={raw_input_params:?} input_params={input_params:?} input_errors={input_errors:?} np_over_chi2=({:.3},{:.3}) patches={input_patches:?} patch_chi={input_patch_bonds:?} patch_params={input_patch_params:?} compress={input_compression_secs:.3}s build={input_build_secs:.3}s",
        input_params.0 as f64 / (input_chi * input_chi) as f64,
        input_params.1 as f64 / (input_chi * input_chi) as f64,
    );
    let input_record = serde_json::json!({
        "case": "mpo_mpo_aniso_input",
        "n_gauss": n_gauss,
        "r": r,
        "box_l": box_l,
        "sigma": sigma,
        "rho_max": rho_max,
        "spacing": spacing,
        "box_padding": box_padding,
        "r_extra": extra_bits,
        "seed": seed,
        "input_generator": input_generator,
        "input_cache_key": cache_key,
        "input_tci_rtol": input_tci_rtol,
        "input_tci_cap": input_tci_cap,
        "input_poly_degree": input_poly_degree,
        "input_add_rtol": input_add_rtol,
        "input_tci_local_abs_tol": input_tci_local_abs_tol,
        "input_tci_initial_pivots": input_tci_initial_pivots,
        "input_svd_l2_rtol": input_svd_l2_rtol,
        "raw_input_chi": raw_input_chi,
        "input_chi": input_chi,
        "raw_input_params": raw_input_params,
        "input_params": input_params,
        "input_sampled_relative_errors": input_errors,
        "input_sanity": input_sanity,
        "input_error_samples": n_error_samples,
        "max_input_chi": max_input_chi,
        "np_over_chi_squared": [
            input_params.0 as f64 / (input_chi * input_chi) as f64,
            input_params.1 as f64 / (input_chi * input_chi) as f64,
        ],
        "patch_cap": patch_cap,
        "input_patch_counts": input_patches,
        "input_patch_max_bonds": input_patch_bonds,
        "input_patch_params": input_patch_params,
        "input_cache_hit": cache_hit,
        "input_cache_load_secs": input_cache_load_secs,
        "input_compression_secs": input_compression_secs,
        "input_build_secs": input_build_secs,
    });
    fs::create_dir_all(&out_dir)?;
    fs::write(
        out_dir.join(format!(
            "mpo_mpo_aniso_input-{input_generator}-n{n_gauss}-chi{input_chi}.json"
        )),
        serde_json::to_string_pretty(&input_record)?,
    )?;
    if input_only {
        return Ok(());
    }

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
        "input_tci_rtol": input_tci_rtol,
        "input_tci_cap": input_tci_cap,
        "input_svd_l2_rtol": input_svd_l2_rtol,
        "raw_input_chi": raw_input_chi,
        "raw_input_params": [raw_input_params.0, raw_input_params.1],
        "input_params": [input_params.0, input_params.1],
        "input_compression_secs": input_compression_secs,
        "input_patch_cap": patch_cap,
        "input_patch_counts": [input_patches.0, input_patches.1],
        "input_patch_max_bonds": [input_patch_bonds.0, input_patch_bonds.1],
        "input_patch_params": [input_patch_params.0, input_patch_params.1],
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
    use t4a_bench::gaussian::to_quantics_mpo_field;

    #[test]
    fn l2_compression_reduces_or_preserves_input_size() {
        let field = AnisoMixture2D::random(4, 1.0, 0.2, 3.0, 9);
        let other = AnisoMixture2D::random(4, 1.0, 0.2, 3.0, 10);
        let (left, _) = to_quantics_mpo_field(&field, 5, 1.0, 1.0e-8, 64).unwrap();
        let (right, _) = to_quantics_mpo_field(&other, 5, 1.0, 1.0e-8, 64).unwrap();
        let (mut left, mut right) = mpo_pair_to_tensortrains(&left, &right).unwrap();
        let before = (tensortrain_n_params(&left), tensortrain_n_params(&right));

        truncate_tensortrain_l2(&mut left, 1.0e-6).unwrap();
        truncate_tensortrain_l2(&mut right, 1.0e-6).unwrap();

        assert!(tensortrain_n_params(&left) <= before.0);
        assert!(tensortrain_n_params(&right) <= before.1);
        PatchedMpoPair::from_tensortrains_with_input_rtol(left, right, 1.0e-6, 1.0e-8, 16).unwrap();
    }

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
