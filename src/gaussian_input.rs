//! Shared deterministic Gaussian inputs for the two Gaussian benchmark cases.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use tensor4all_simplett::{
    AbstractTensorTrain, CompressionMethod, CompressionOptions, SimpleTensorTrain, Tensor3Ops,
};

use crate::gaussian::{grid_coord, AnisoMixture2D};
use crate::harness::{index_to_bits, sample_grid_indices};
use crate::hdf5_export::{load_tt_from_mps, save_tt_as_mps};

/// Fixed patch rank cap used by every patched benchmark arm.
pub const PATCH_CAP: usize = 128;
/// Final global relative-L2 input tolerance.
pub const INPUT_L2_RTOL: f64 = 1e-6;
const CACHE_SCHEMA: &str = "interpolative-gaussian-pair-v2";
const TENSOR4ALL_REV: &str = "9e9aedaebe0d3918b34dd399ff0981e337f3835b";

/// Parameters that uniquely define one cached input pair.
#[derive(Clone, Debug)]
pub struct GaussianInputConfig {
    pub n: usize,
    pub sigma_minor: f64,
    pub rho_max: f64,
    pub spacing: f64,
    pub polynomial_degree: usize,
    pub interpolation_tolerance: f64,
    pub addition_tolerance: f64,
    pub seed: u64,
    pub cache_dir: PathBuf,
    pub refresh: bool,
}

/// Prepared pair and complete untimed preparation metadata.
pub struct GaussianInputPair {
    pub left: SimpleTensorTrain<f64>,
    pub right: SimpleTensorTrain<f64>,
    pub left_mixture: AnisoMixture2D,
    pub right_mixture: AnisoMixture2D,
    pub r: usize,
    pub box_l: f64,
    pub cache_key: String,
    pub cache_hit: bool,
    pub cache_load: Duration,
    pub build: Duration,
    pub compression: Duration,
    pub raw_left_chi: usize,
    pub raw_right_chi: usize,
    pub raw_left_params: usize,
    pub raw_right_params: usize,
}

/// Choose the box and the smallest bit count resolving the minor width by eight cells.
pub fn box_and_bits(n: usize, sigma_minor: f64, spacing: f64) -> (f64, usize) {
    let box_l = 0.5 * spacing * sigma_minor * (n as f64).sqrt();
    let mut r = 2;
    while 2.0 * box_l / (1usize << r) as f64 > sigma_minor / 8.0 {
        r += 1;
    }
    (box_l, r)
}

/// Number of stored scalar entries in a tensor train.
pub fn tensortrain_n_params(tt: &SimpleTensorTrain<f64>) -> usize {
    tt.site_tensors()
        .iter()
        .map(|core| core.left_dim() * core.site_dim() * core.right_dim())
        .sum()
}

/// Load or build, then globally L2-compress, one deterministic Gaussian input pair.
pub fn prepare_gaussian_pair(config: &GaussianInputConfig) -> anyhow::Result<GaussianInputPair> {
    anyhow::ensure!(config.n > 0, "Gaussian count must be positive");
    let (box_l, r) = box_and_bits(config.n, config.sigma_minor, config.spacing);
    let left_mixture = AnisoMixture2D::random(
        config.n,
        box_l,
        config.sigma_minor,
        config.rho_max,
        config.seed.wrapping_add(1),
    );
    let right_mixture = AnisoMixture2D::random(
        config.n,
        box_l,
        config.sigma_minor,
        config.rho_max,
        config.seed.wrapping_add(2),
    );
    let cache_key = format!(
        "{CACHE_SCHEMA}-rev{TENSOR4ALL_REV}-n{}-r{r}-sigma{:016x}-rho{:016x}-spacing{:016x}-degree{}-interp{:016x}-add{:016x}-seed{}",
        config.n,
        config.sigma_minor.to_bits(),
        config.rho_max.to_bits(),
        config.spacing.to_bits(),
        config.polynomial_degree,
        config.interpolation_tolerance.to_bits(),
        config.addition_tolerance.to_bits(),
        config.seed,
    );
    std::fs::create_dir_all(&config.cache_dir)?;
    let path = config.cache_dir.join(format!("{cache_key}.h5"));
    let load_start = Instant::now();
    let (mut left, mut right, cache_hit, cache_load, build) = if path.exists() && !config.refresh {
        let left = load_tt_from_mps(path_str(&path)?, "left")?;
        let right = load_tt_from_mps(path_str(&path)?, "right")?;
        validate_cached_pair(&left, &right, r)?;
        (left, right, true, load_start.elapsed(), Duration::ZERO)
    } else {
        let build_start = Instant::now();
        let left = left_mixture.to_interpolative_qtt(
            r,
            box_l,
            config.polynomial_degree,
            config.interpolation_tolerance,
            config.addition_tolerance,
        )?;
        let right = right_mixture.to_interpolative_qtt(
            r,
            box_l,
            config.polynomial_degree,
            config.interpolation_tolerance,
            config.addition_tolerance,
        )?;
        validate_cached_pair(&left, &right, r)?;
        write_pair_atomically(&path, &left, &right)?;
        (left, right, false, Duration::ZERO, build_start.elapsed())
    };
    let raw_left_chi = left.rank();
    let raw_right_chi = right.rank();
    let raw_left_params = tensortrain_n_params(&left);
    let raw_right_params = tensortrain_n_params(&right);
    let compression_start = Instant::now();
    let options = CompressionOptions {
        method: CompressionMethod::SVD,
        tolerance: INPUT_L2_RTOL,
        max_bond_dim: None,
        normalize_error: true,
    };
    left.compress(&options)?;
    right.compress(&options)?;
    let compression = compression_start.elapsed();
    Ok(GaussianInputPair {
        left,
        right,
        left_mixture,
        right_mixture,
        r,
        box_l,
        cache_key,
        cache_hit,
        cache_load,
        build,
        compression,
        raw_left_chi,
        raw_right_chi,
        raw_left_params,
        raw_right_params,
    })
}

/// Deterministic holdout estimate of the pair's common relative-L2 input error.
pub fn sampled_input_relative_l2(
    pair: &GaussianInputPair,
    samples: usize,
    seed: u64,
) -> anyhow::Result<(f64, f64)> {
    let mut left_error = 0.0;
    let mut left_reference = 0.0;
    let mut right_error = 0.0;
    let mut right_reference = 0.0;
    for index in sample_grid_indices(2 * pair.r, samples, seed) {
        let mask = (1u64 << pair.r) - 1;
        let ix = index >> pair.r;
        let iy = index & mask;
        let point: Vec<_> = index_to_bits(ix, pair.r)
            .into_iter()
            .zip(index_to_bits(iy, pair.r))
            .map(|(x, y)| x + 2 * y)
            .collect();
        let x = grid_coord(ix, pair.r, pair.box_l);
        let y = grid_coord(iy, pair.r, pair.box_l);
        let expected_left = pair.left_mixture.eval(x, y);
        let expected_right = pair.right_mixture.eval(x, y);
        let delta_left = pair.left.evaluate(&point)? - expected_left;
        let delta_right = pair.right.evaluate(&point)? - expected_right;
        left_error += delta_left * delta_left;
        left_reference += expected_left * expected_left;
        right_error += delta_right * delta_right;
        right_reference += expected_right * expected_right;
    }
    anyhow::ensure!(
        left_reference > 0.0 && right_reference > 0.0,
        "zero holdout reference norm"
    );
    Ok((
        (left_error / left_reference).sqrt(),
        (right_error / right_reference).sqrt(),
    ))
}

fn validate_cached_pair(
    left: &SimpleTensorTrain<f64>,
    right: &SimpleTensorTrain<f64>,
    r: usize,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        left.len() == r && right.len() == r,
        "cached site count mismatch"
    );
    anyhow::ensure!(
        left.site_tensors().iter().all(|core| core.site_dim() == 4)
            && right.site_tensors().iter().all(|core| core.site_dim() == 4),
        "cached site dimension mismatch"
    );
    Ok(())
}

fn write_pair_atomically(
    path: &Path,
    left: &SimpleTensorTrain<f64>,
    right: &SimpleTensorTrain<f64>,
) -> anyhow::Result<()> {
    let temporary = path.with_extension(format!("{}.tmp", std::process::id()));
    let _ = std::fs::remove_file(&temporary);
    save_tt_as_mps(path_str(&temporary)?, "left", left, false)?;
    save_tt_as_mps(path_str(&temporary)?, "right", right, true)?;
    std::fs::rename(&temporary, path)?;
    Ok(())
}

fn path_str(path: &Path) -> anyhow::Result<&str> {
    path.to_str()
        .ok_or_else(|| anyhow::anyhow!("non-UTF-8 cache path: {path:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_pair_round_trip_preserves_values_and_rank() {
        let cache_dir =
            std::env::temp_dir().join(format!("t4a-gaussian-cache-{}", std::process::id()));
        let config = GaussianInputConfig {
            n: 2,
            sigma_minor: 0.12,
            rho_max: 2.0,
            spacing: 3.0,
            polynomial_degree: 12,
            interpolation_tolerance: 1e-9,
            addition_tolerance: 1e-10,
            seed: 31,
            cache_dir: cache_dir.clone(),
            refresh: true,
        };
        let first = prepare_gaussian_pair(&config).unwrap();
        let second = prepare_gaussian_pair(&GaussianInputConfig {
            refresh: false,
            ..config
        })
        .unwrap();
        assert!(!first.cache_hit);
        assert!(second.cache_hit);
        assert_eq!(first.raw_left_chi, second.raw_left_chi);
        assert_eq!(first.raw_right_chi, second.raw_right_chi);
        for point in [vec![0; first.r], (0..first.r).map(|i| i % 4).collect()] {
            assert!(
                (first.left.evaluate(&point).unwrap() - second.left.evaluate(&point).unwrap())
                    .abs()
                    < 1e-12
            );
        }
        let left_mpo = crate::gaussian::fused_qtt_to_mpo(&second.left).unwrap();
        let right_mpo = crate::gaussian::fused_qtt_to_mpo(&second.right).unwrap();
        let (left_train, right_train) =
            crate::patched_mpo::mpo_pair_to_tensortrains(&left_mpo, &right_mpo).unwrap();
        for (left_sites, right_sites) in left_train
            .site_indices()
            .iter()
            .zip(right_train.site_indices())
        {
            assert_eq!(
                left_sites
                    .iter()
                    .filter(|index| right_sites.contains(index))
                    .count(),
                1
            );
        }
        std::fs::remove_dir_all(cache_dir).unwrap();
    }
}
