//! Shared deterministic Gaussian inputs for the two Gaussian benchmark cases.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use tensor4all_simplett::{
    AbstractTensorTrain, CompressionMethod, CompressionOptions, SimpleTensorTrain, Tensor3Ops,
};

use crate::gaussian::{global_tci_qtt, grid_coord, AnisoMixture2D, LocalizedAnisoField};
use crate::harness::{index_to_bits, sample_grid_indices};
use crate::hdf5_export::{load_tt_from_mps, save_tt_as_mps};

/// Fixed quantics bit count per physical axis.
pub const GAUSSIAN_R: usize = 16;
/// Ratio of computational half-width to the active Gaussian-center half-width.
pub const GAUSSIAN_PADDING_FACTOR: f64 = 4.0;
/// Fixed patch rank cap used by every patched benchmark arm.
pub const PATCH_CAP: usize = 128;
/// Final global relative-L2 input tolerance.
pub const INPUT_L2_RTOL: f64 = 1e-6;
/// Global TCI residual tolerance used to construct each raw input.
pub const INPUT_TCI_TOLERANCE: f64 = 1e-8;
/// Maximum raw TCI bond dimension before final L2 compression.
pub const INPUT_TCI_MAX_BOND: usize = 1024;
/// Rigorous pointwise tail budget of the localized Gaussian evaluator.
pub const INPUT_LOCAL_ABS_TOLERANCE: f64 = 1e-12;
/// Gaussian components supplying deterministic center and principal-axis pivots.
pub const INPUT_TCI_PIVOT_COMPONENTS: usize = 16;
const CACHE_SCHEMA: &str = "global-tci-gaussian-pair-v4-padded";
const TENSOR4ALL_REV: &str = "9e9aedaebe0d3918b34dd399ff0981e337f3835b";

/// Parameters that uniquely define one cached input pair.
#[derive(Clone, Debug)]
pub struct GaussianInputConfig {
    pub n: usize,
    pub sigma_minor: f64,
    pub rho_max: f64,
    pub spacing: f64,
    pub tci_tolerance: f64,
    pub tci_max_bond_dim: usize,
    pub localized_absolute_tolerance: f64,
    pub tci_pivot_components: usize,
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
    /// Half-width containing the generated Gaussian centers.
    pub active_box_l: f64,
    /// Half-width of the padded quantics computational domain.
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

/// Choose the active and padded boxes while keeping the bit count fixed.
pub fn boxes_and_bits(n: usize, sigma_minor: f64, spacing: f64) -> (f64, f64, usize) {
    let active_box_l = 0.5 * spacing * sigma_minor * (n as f64).sqrt();
    (
        active_box_l,
        GAUSSIAN_PADDING_FACTOR * active_box_l,
        GAUSSIAN_R,
    )
}

/// Number of stored scalar entries in a tensor train.
pub fn tensortrain_n_params(tt: &SimpleTensorTrain<f64>) -> usize {
    tt.site_tensors()
        .iter()
        .map(|core| core.left_dim() * core.site_dim() * core.right_dim())
        .sum()
}

fn grid_pivots(
    mixture: &AnisoMixture2D,
    count: usize,
    r: usize,
    box_l: f64,
    shifted: bool,
) -> Vec<Vec<usize>> {
    let count = count.min((mixture.centers.len() / 2).max(1));
    let grid_size = 1usize << r;
    let mut pivots = Vec::with_capacity(5 * count);
    for pivot in 0..count {
        let component = ((2 * pivot + usize::from(shifted)) * mixture.centers.len() / (2 * count))
            .min(mixture.centers.len() - 1);
        let center = mixture.centers[component];
        let (a, b, c) = mixture.quad[component];
        let discriminant = ((a - c).powi(2) + 4.0 * b * b).sqrt();
        let lambda_major = 0.5 * (a + c - discriminant);
        let lambda_minor = 0.5 * (a + c + discriminant);
        let mut major = if b.abs() > f64::EPSILON * a.abs().max(c.abs()) {
            (b, lambda_major - a)
        } else if a <= c {
            (1.0, 0.0)
        } else {
            (0.0, 1.0)
        };
        let norm = major.0.hypot(major.1);
        major.0 /= norm;
        major.1 /= norm;
        let minor = (-major.1, major.0);
        let sigma_major = (2.0 * lambda_major).sqrt().recip();
        let sigma_minor = (2.0 * lambda_minor).sqrt().recip();
        for point in [
            center,
            (
                center.0 + sigma_major * major.0,
                center.1 + sigma_major * major.1,
            ),
            (
                center.0 - sigma_major * major.0,
                center.1 - sigma_major * major.1,
            ),
            (
                center.0 + sigma_minor * minor.0,
                center.1 + sigma_minor * minor.1,
            ),
            (
                center.0 - sigma_minor * minor.0,
                center.1 - sigma_minor * minor.1,
            ),
        ] {
            let grid_point = [point.0, point.1]
                .into_iter()
                .map(|coordinate| {
                    let scaled = ((coordinate + box_l) * grid_size as f64 / (2.0 * box_l)).floor();
                    scaled.clamp(0.0, (grid_size - 1) as f64) as usize
                })
                .collect::<Vec<_>>();
            if !pivots.contains(&grid_point) {
                pivots.push(grid_point);
            }
        }
    }
    pivots
}

/// Load or build a global-TCI input pair, then apply final L2 compression.
pub fn prepare_gaussian_pair(config: &GaussianInputConfig) -> anyhow::Result<GaussianInputPair> {
    anyhow::ensure!(config.n > 0, "Gaussian count must be positive");
    anyhow::ensure!(
        config.tci_tolerance.is_finite() && config.tci_tolerance > 0.0,
        "TCI tolerance must be positive and finite"
    );
    anyhow::ensure!(config.tci_max_bond_dim > 0, "TCI rank cap must be positive");
    anyhow::ensure!(
        config.tci_pivot_components > 0,
        "TCI pivot count must be positive"
    );
    let (active_box_l, box_l, r) = boxes_and_bits(config.n, config.sigma_minor, config.spacing);
    let left_mixture = AnisoMixture2D::random(
        config.n,
        active_box_l,
        config.sigma_minor,
        config.rho_max,
        config.seed.wrapping_add(1),
    );
    let right_mixture = AnisoMixture2D::random(
        config.n,
        active_box_l,
        config.sigma_minor,
        config.rho_max,
        config.seed.wrapping_add(2),
    );
    let cache_key = format!(
        "{CACHE_SCHEMA}-rev{TENSOR4ALL_REV}-n{}-r{r}-padding{:016x}-sigma{:016x}-rho{:016x}-spacing{:016x}-tci{:016x}-local{:016x}-piv{}-cap{}-seed{}",
        config.n,
        GAUSSIAN_PADDING_FACTOR.to_bits(),
        config.sigma_minor.to_bits(),
        config.rho_max.to_bits(),
        config.spacing.to_bits(),
        config.tci_tolerance.to_bits(),
        config.localized_absolute_tolerance.to_bits(),
        config.tci_pivot_components,
        config.tci_max_bond_dim,
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
        let localized_left =
            LocalizedAnisoField::new(left_mixture.clone(), config.localized_absolute_tolerance)?;
        let localized_right =
            LocalizedAnisoField::new(right_mixture.clone(), config.localized_absolute_tolerance)?;
        let left = global_tci_qtt(
            &localized_left,
            r,
            box_l,
            config.tci_tolerance,
            config.tci_max_bond_dim,
            grid_pivots(&left_mixture, config.tci_pivot_components, r, box_l, false),
        )?;
        let right = global_tci_qtt(
            &localized_right,
            r,
            box_l,
            config.tci_tolerance,
            config.tci_max_bond_dim,
            grid_pivots(&right_mixture, config.tci_pivot_components, r, box_l, false),
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
        active_box_l,
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

fn relative_l2_at_grid_points(
    qtt: &SimpleTensorTrain<f64>,
    mixture: &AnisoMixture2D,
    points: &[Vec<usize>],
    r: usize,
    box_l: f64,
) -> anyhow::Result<f64> {
    let mut squared_error = 0.0;
    let mut squared_reference = 0.0;
    for point in points {
        anyhow::ensure!(point.len() == 2, "principal-axis grid point is not 2D");
        let (ix, iy) = (point[0] as u64, point[1] as u64);
        let fused = index_to_bits(ix, r)
            .into_iter()
            .zip(index_to_bits(iy, r))
            .map(|(x, y)| x + 2 * y)
            .collect::<Vec<_>>();
        let expected = mixture.eval(grid_coord(ix, r, box_l), grid_coord(iy, r, box_l));
        let error = qtt.evaluate(&fused)? - expected;
        squared_error += error * error;
        squared_reference += expected * expected;
    }
    anyhow::ensure!(
        squared_reference > 0.0,
        "zero principal-axis reference norm"
    );
    Ok((squared_error / squared_reference).sqrt())
}

/// Relative-L2 errors on centers and principal axes not used as TCI pivots.
pub fn principal_axis_input_relative_l2(
    pair: &GaussianInputPair,
    components: usize,
) -> anyhow::Result<(f64, f64)> {
    let left_points = grid_pivots(&pair.left_mixture, components, pair.r, pair.box_l, true);
    let right_points = grid_pivots(&pair.right_mixture, components, pair.r, pair.box_l, true);
    Ok((
        relative_l2_at_grid_points(
            &pair.left,
            &pair.left_mixture,
            &left_points,
            pair.r,
            pair.box_l,
        )?,
        relative_l2_at_grid_points(
            &pair.right,
            &pair.right_mixture,
            &right_points,
            pair.r,
            pair.box_l,
        )?,
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
    fn padded_production_box_makes_boundary_tail_negligible() {
        let n = 512;
        let sigma_minor = 0.05;
        let rho_max = 8.0;
        let (active, padded, r) = boxes_and_bits(n, sigma_minor, 3.0);
        assert_eq!(r, GAUSSIAN_R);
        assert_eq!(padded, GAUSSIAN_PADDING_FACTOR * active);
        let boundary_gap = padded - 0.9 * active;
        let sigma_major = rho_max * sigma_minor;
        let worst_component_tail = (-(boundary_gap / sigma_major).powi(2) / 2.0).exp();
        assert!(1.5 * n as f64 * worst_component_tail < 1e-30);
    }

    #[test]
    fn cached_pair_round_trip_preserves_values_and_rank() {
        let cache_dir =
            std::env::temp_dir().join(format!("t4a-gaussian-cache-{}", std::process::id()));
        let config = GaussianInputConfig {
            n: 2,
            sigma_minor: 0.12,
            rho_max: 2.0,
            spacing: 3.0,
            tci_tolerance: 1e-8,
            tci_max_bond_dim: 64,
            localized_absolute_tolerance: 1e-12,
            tci_pivot_components: 2,
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
        assert_eq!(first.r, GAUSSIAN_R);
        assert_eq!(first.box_l, GAUSSIAN_PADDING_FACTOR * first.active_box_l);
        let sampled = sampled_input_relative_l2(&first, 64, 41).unwrap();
        let principal = principal_axis_input_relative_l2(&first, 2).unwrap();
        assert!(sampled.0 < 1e-4 && sampled.1 < 1e-4, "sampled={sampled:?}");
        assert!(
            principal.0 < 1e-4 && principal.1 < 1e-4,
            "principal={principal:?}"
        );
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
