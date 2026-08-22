//! Input-only fully correlated 3D Gaussian rank probe.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tensor4all_quanticstci::{
    quanticscrossinterpolate, DiscretizedGrid, QtciOptions, UnfoldingScheme,
};
use tensor4all_simplett::mpo::{tensor4_from_data, MPO};
use tensor4all_simplett::{
    AbstractTensorTrain, CompressionMethod, CompressionOptions, SimpleTensorTrain, Tensor3Ops,
};

use crate::gaussian_input::{tensortrain_n_params, GAUSSIAN_PADDING_FACTOR, GAUSSIAN_R};
use crate::harness::index_to_bits;
use crate::hdf5_export::{load_tt_from_mps, save_tt_as_mps};

const CACHE_SCHEMA: &str = "global-tci-gaussian-3d-batch-diagonal-v3";
const TENSOR4ALL_REV: &str = "9e9aedaebe0d3918b34dd399ff0981e337f3835b";

/// Parameters for one input-only 3D Gaussian rank probe.
#[derive(Clone, Debug)]
pub struct Gaussian3dInputConfig {
    pub n: usize,
    pub sigma_minor: f64,
    pub rho_max: f64,
    pub spacing: f64,
    pub tci_tolerance: f64,
    pub tci_max_bond_dim: usize,
    pub localized_absolute_tolerance: f64,
    pub tci_pivot_components: usize,
    pub input_l2_rtol: f64,
    /// Optional Gaussian count whose constant-density box size is reused.
    pub fixed_box_n: Option<usize>,
    pub seed: u64,
    pub cache_dir: PathBuf,
    pub refresh: bool,
}

/// Prepared 3D QTT and its batch-diagonal MPO embedding.
pub struct PreparedGaussian3dInput {
    pub qtt: SimpleTensorTrain<f64>,
    pub batch_diagonal_mpo: MPO<f64>,
    pub r: usize,
    pub active_box_l: f64,
    pub box_l: f64,
    pub cache_key: String,
    pub cache_hit: bool,
    pub cache_load: Duration,
    pub build: Duration,
    pub compression: Duration,
    pub embedding: Duration,
    pub raw_chi: usize,
    pub raw_params: usize,
    pub local_compression_rtol: f64,
    pub principal_axis_relative_l2: f64,
}

#[derive(Clone, Debug)]
struct GaussianMixture3d {
    weights: Vec<f64>,
    centers: Vec<[f64; 3]>,
    axes: Vec<[[f64; 3]; 3]>,
    lambdas: Vec<[f64; 3]>,
}

impl GaussianMixture3d {
    fn random(n: usize, box_l: f64, sigma_minor: f64, rho_max: f64, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let half = 0.9 * box_l;
        let lambda_minor = 1.0 / (2.0 * sigma_minor * sigma_minor);
        let mut weights = Vec::with_capacity(n);
        let mut centers = Vec::with_capacity(n);
        let mut axes = Vec::with_capacity(n);
        let mut lambdas = Vec::with_capacity(n);
        for _ in 0..n {
            weights.push(rng.random_range(0.5..1.5));
            centers.push([
                rng.random_range(-half..half),
                rng.random_range(-half..half),
                rng.random_range(-half..half),
            ]);
            axes.push(random_rotation(&mut rng));
            let rho = rho_max.powf(rng.random_range(0.0..1.0));
            lambdas.push([lambda_minor / (rho * rho), lambda_minor / rho, lambda_minor]);
        }
        Self {
            weights,
            centers,
            axes,
            lambdas,
        }
    }

    fn eval(&self, point: [f64; 3]) -> f64 {
        (0..self.weights.len())
            .map(|i| self.component(i, point))
            .sum()
    }

    fn component(&self, i: usize, point: [f64; 3]) -> f64 {
        let delta = sub(point, self.centers[i]);
        let exponent = (0..3)
            .map(|axis| self.lambdas[i][axis] * dot(delta, self.axes[i][axis]).powi(2))
            .sum::<f64>();
        self.weights[i] * (-exponent).exp()
    }
}

#[derive(Clone, Debug)]
struct LocalizedGaussianMixture3d {
    mixture: GaussianMixture3d,
    bin_width: f64,
    cutoff_squared: Vec<f64>,
    bins: HashMap<(i64, i64, i64), Vec<usize>>,
}

impl LocalizedGaussianMixture3d {
    fn new(mixture: GaussianMixture3d, absolute_tolerance: f64) -> anyhow::Result<Self> {
        anyhow::ensure!(
            absolute_tolerance.is_finite() && absolute_tolerance > 0.0,
            "localized tolerance must be positive and finite"
        );
        let total_weight = mixture.weights.iter().sum::<f64>();
        anyhow::ensure!(
            total_weight > absolute_tolerance,
            "localized tolerance is too large"
        );
        let exponent_cutoff = (total_weight / absolute_tolerance).ln();
        let cutoff_squared = mixture
            .lambdas
            .iter()
            .map(|lambda| exponent_cutoff / lambda[0])
            .collect::<Vec<_>>();
        let bin_width = cutoff_squared
            .iter()
            .copied()
            .fold(0.0_f64, f64::max)
            .sqrt();
        let mut bins = HashMap::<(i64, i64, i64), Vec<usize>>::new();
        for (i, center) in mixture.centers.iter().enumerate() {
            bins.entry(bin_key(*center, bin_width)).or_default().push(i);
        }
        Ok(Self {
            mixture,
            bin_width,
            cutoff_squared,
            bins,
        })
    }

    fn eval(&self, point: [f64; 3]) -> f64 {
        let (bx, by, bz) = bin_key(point, self.bin_width);
        let mut value = 0.0;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(indices) = self.bins.get(&(bx + dx, by + dy, bz + dz)) {
                        for &i in indices {
                            let delta = sub(point, self.mixture.centers[i]);
                            if dot(delta, delta) <= self.cutoff_squared[i] {
                                value += self.mixture.component(i, point);
                            }
                        }
                    }
                }
            }
        }
        value
    }
}

fn random_rotation(rng: &mut ChaCha8Rng) -> [[f64; 3]; 3] {
    let (u1, u2, u3) = (
        rng.random_range(0.0_f64..1.0),
        rng.random_range(0.0_f64..1.0),
        rng.random_range(0.0_f64..1.0),
    );
    let (s2, c2) = (2.0 * std::f64::consts::PI * u2).sin_cos();
    let (s3, c3) = (2.0 * std::f64::consts::PI * u3).sin_cos();
    let (x, y, z, w) = (
        (1.0 - u1).sqrt() * s2,
        (1.0 - u1).sqrt() * c2,
        u1.sqrt() * s3,
        u1.sqrt() * c3,
    );
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
        ],
        [
            2.0 * (x * y - z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + x * w),
        ],
        [
            2.0 * (x * z + y * w),
            2.0 * (y * z - x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.into_iter().zip(b).map(|(x, y)| x * y).sum()
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn stable_key_hash(value: &str) -> u64 {
    value
        .as_bytes()
        .iter()
        .fold(0xcbf29ce484222325, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        })
}

fn bin_key(point: [f64; 3], width: f64) -> (i64, i64, i64) {
    (
        (point[0] / width).floor() as i64,
        (point[1] / width).floor() as i64,
        (point[2] / width).floor() as i64,
    )
}

fn grid_pivots(mixture: &GaussianMixture3d, count: usize, r: usize, box_l: f64) -> Vec<Vec<usize>> {
    let count = count.min(mixture.centers.len());
    let grid_size = 1usize << r;
    let mut pivots = Vec::with_capacity(7 * count);
    for pivot in 0..count {
        let component = (pivot * mixture.centers.len() / count).min(mixture.centers.len() - 1);
        let center = mixture.centers[component];
        let points = std::iter::once(center).chain((0..3).flat_map(|axis| {
            let sigma = (2.0 * mixture.lambdas[component][axis]).sqrt().recip();
            let direction = mixture.axes[component][axis];
            [-1.0, 1.0].map(move |sign| {
                [
                    center[0] + sign * sigma * direction[0],
                    center[1] + sign * sigma * direction[1],
                    center[2] + sign * sigma * direction[2],
                ]
            })
        }));
        for point in points {
            let grid_point = point
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

fn principal_axis_relative_l2(
    qtt: &SimpleTensorTrain<f64>,
    mixture: &GaussianMixture3d,
    r: usize,
    box_l: f64,
) -> anyhow::Result<f64> {
    let grid_size = 1usize << r;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for component in 0..mixture.centers.len() {
        let center = mixture.centers[component];
        let points = std::iter::once(center).chain((0..3).flat_map(|axis| {
            let sigma = (2.0 * mixture.lambdas[component][axis]).sqrt().recip();
            let direction = mixture.axes[component][axis];
            [-0.5, 0.5].map(move |sign| {
                [
                    center[0] + sign * sigma * direction[0],
                    center[1] + sign * sigma * direction[1],
                    center[2] + sign * sigma * direction[2],
                ]
            })
        }));
        for point in points {
            let indices = point.map(|coordinate| {
                let scaled = ((coordinate + box_l) * grid_size as f64 / (2.0 * box_l)).floor();
                scaled.clamp(0.0, (grid_size - 1) as f64) as usize
            });
            let bits = indices.map(|index| index_to_bits(index as u64, r));
            let local = (0..r)
                .map(|site| bits[0][site] + 2 * bits[1][site] + 4 * bits[2][site])
                .collect::<Vec<_>>();
            let grid_point =
                indices.map(|index| -box_l + index as f64 * (2.0 * box_l / grid_size as f64));
            let expected = mixture.eval(grid_point);
            let error = qtt.evaluate(&local)? - expected;
            numerator += error * error;
            denominator += expected * expected;
        }
    }
    anyhow::ensure!(
        denominator > 0.0 && denominator.is_finite(),
        "invalid validation norm"
    );
    Ok((numerator / denominator).sqrt())
}

fn global_tci_qtt(
    field: &LocalizedGaussianMixture3d,
    r: usize,
    box_l: f64,
    tolerance: f64,
    max_bond_dim: usize,
    initial_pivots: Vec<Vec<usize>>,
) -> anyhow::Result<SimpleTensorTrain<f64>> {
    let grid = DiscretizedGrid::builder(&[r, r, r])
        .with_lower_bound(&[-box_l; 3])
        .with_upper_bound(&[box_l; 3])
        .with_unfolding_scheme(UnfoldingScheme::Fused)
        .build()?;
    let sampled = field.clone();
    let options = QtciOptions::default()
        .with_tolerance(tolerance)
        .with_max_bond_dim(max_bond_dim)
        .with_unfoldingscheme(UnfoldingScheme::Fused)
        .with_nrandominitpivot(0);
    let (qtt, _, _) = quanticscrossinterpolate(
        &grid,
        move |bxy: &[f64]| sampled.eval([bxy[0], bxy[1], bxy[2]]),
        Some(initial_pivots),
        options,
    )?;
    Ok(qtt.tensor_train())
}

/// Embed `A(b,x,y)` as `delta(b,b') A(b,x,y)` with fused MPO legs `(b,x)` and `(b',y)`.
pub fn batch_diagonal_mpo(qtt: &SimpleTensorTrain<f64>) -> anyhow::Result<MPO<f64>> {
    let tensors = qtt
        .site_tensors()
        .iter()
        .map(|core| {
            anyhow::ensure!(core.site_dim() == 8, "expected fused site dimension 8");
            let mut data = Vec::with_capacity(core.left_dim() * 16 * core.right_dim());
            for right in 0..core.right_dim() {
                for input in 0..4 {
                    let batch_input = input % 2;
                    let y = input / 2;
                    for output in 0..4 {
                        let batch_output = output % 2;
                        let x = output / 2;
                        for left in 0..core.left_dim() {
                            let value = if batch_output == batch_input {
                                *core.get3(left, batch_output + 2 * x + 4 * y, right)
                            } else {
                                0.0
                            };
                            data.push(value);
                        }
                    }
                }
            }
            tensor4_from_data(data, core.left_dim(), 4, 4, core.right_dim())
                .map_err(anyhow::Error::from)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    MPO::new(tensors).map_err(anyhow::Error::from)
}

/// Load or construct one fixed-R 3D Gaussian mixture and its diagonal MPO embedding.
pub fn prepare_gaussian3d_input(
    config: &Gaussian3dInputConfig,
) -> anyhow::Result<PreparedGaussian3dInput> {
    prepare_gaussian3d_input_with_r(config, GAUSSIAN_R)
}

fn prepare_gaussian3d_input_with_r(
    config: &Gaussian3dInputConfig,
    r: usize,
) -> anyhow::Result<PreparedGaussian3dInput> {
    anyhow::ensure!(config.n > 0, "Gaussian count must be positive");
    anyhow::ensure!(
        config.fixed_box_n.is_none_or(|n| n > 0),
        "fixed-box reference count must be positive"
    );
    anyhow::ensure!(
        config.sigma_minor > 0.0 && config.sigma_minor.is_finite(),
        "invalid width"
    );
    anyhow::ensure!(
        config.rho_max > 1.0 && config.rho_max.is_finite(),
        "invalid aspect ratio"
    );
    anyhow::ensure!(
        config.input_l2_rtol > 0.0 && config.input_l2_rtol.is_finite(),
        "invalid input tolerance"
    );
    anyhow::ensure!(
        config.tci_tolerance > 0.0 && config.tci_tolerance.is_finite(),
        "invalid TCI tolerance"
    );
    anyhow::ensure!(config.tci_max_bond_dim > 0, "invalid TCI rank cap");
    anyhow::ensure!(config.tci_pivot_components > 0, "invalid pivot count");

    let box_n = config.fixed_box_n.unwrap_or(config.n);
    let active_box_l = 0.5 * config.spacing * config.sigma_minor * (box_n as f64).cbrt();
    let box_l = GAUSSIAN_PADDING_FACTOR * active_box_l;
    let mixture = GaussianMixture3d::random(
        config.n,
        active_box_l,
        config.sigma_minor,
        config.rho_max,
        config.seed,
    );
    let cache_identity = format!(
        "{CACHE_SCHEMA}-rev{TENSOR4ALL_REV}-n{}-boxn{}-r{r}-padding{:016x}-sigma{:016x}-rho{:016x}-spacing{:016x}-tci{:016x}-local{:016x}-piv{}-cap{}-seed{}",
        config.n,
        box_n,
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
    let cache_key = format!(
        "{CACHE_SCHEMA}-n{}-r{r}-{:016x}",
        config.n,
        stable_key_hash(&cache_identity)
    );
    std::fs::create_dir_all(&config.cache_dir)?;
    let path = config.cache_dir.join(format!("{cache_key}.h5"));
    let load_start = Instant::now();
    let (mut qtt, cache_hit, cache_load, build) = if path.exists() && !config.refresh {
        let qtt = load_tt_from_mps(path_str(&path)?, "input")?;
        validate_cached(&qtt, r)?;
        (qtt, true, load_start.elapsed(), Duration::ZERO)
    } else {
        let build_start = Instant::now();
        let localized =
            LocalizedGaussianMixture3d::new(mixture.clone(), config.localized_absolute_tolerance)?;
        let qtt = global_tci_qtt(
            &localized,
            r,
            box_l,
            config.tci_tolerance,
            config.tci_max_bond_dim,
            grid_pivots(&mixture, config.tci_pivot_components, r, box_l),
        )?;
        validate_cached(&qtt, r)?;
        write_atomically(&path, &qtt)?;
        (qtt, false, Duration::ZERO, build_start.elapsed())
    };
    let raw_chi = qtt.rank();
    let raw_params = tensortrain_n_params(&qtt);
    let local_compression_rtol = config.input_l2_rtol / (r.saturating_sub(1).max(1) as f64).sqrt();
    let compression_start = Instant::now();
    qtt.compress(&CompressionOptions {
        method: CompressionMethod::SVD,
        tolerance: local_compression_rtol,
        max_bond_dim: None,
        normalize_error: true,
    })?;
    let compression = compression_start.elapsed();
    let principal_axis_relative_l2 = principal_axis_relative_l2(&qtt, &mixture, r, box_l)?;
    let embedding_start = Instant::now();
    let batch_diagonal_mpo = batch_diagonal_mpo(&qtt)?;
    let embedding = embedding_start.elapsed();
    Ok(PreparedGaussian3dInput {
        qtt,
        batch_diagonal_mpo,
        r,
        active_box_l,
        box_l,
        cache_key,
        cache_hit,
        cache_load,
        build,
        compression,
        embedding,
        raw_chi,
        raw_params,
        local_compression_rtol,
        principal_axis_relative_l2,
    })
}

fn validate_cached(qtt: &SimpleTensorTrain<f64>, r: usize) -> anyhow::Result<()> {
    anyhow::ensure!(qtt.len() == r, "cached site count mismatch");
    anyhow::ensure!(
        qtt.site_tensors().iter().all(|core| core.site_dim() == 8),
        "cached site dimension mismatch"
    );
    Ok(())
}

fn write_atomically(path: &Path, qtt: &SimpleTensorTrain<f64>) -> anyhow::Result<()> {
    let temporary = path.with_extension(format!("{}.tmp", std::process::id()));
    let _ = std::fs::remove_file(&temporary);
    save_tt_as_mps(path_str(&temporary)?, "input", qtt, false)?;
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
    use crate::gaussian_input::INPUT_L2_RTOL;
    use tensor4all_simplett::mpo::Tensor4Ops;

    #[test]
    fn random_rotation_is_orthonormal() {
        let rotation = random_rotation(&mut ChaCha8Rng::seed_from_u64(7));
        for i in 0..3 {
            for j in 0..3 {
                let expected = f64::from(i == j);
                assert!((dot(rotation[i], rotation[j]) - expected).abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn localized_evaluator_respects_the_exact_positive_sum() {
        let mixture = GaussianMixture3d::random(8, 1.0, 0.1, 4.0, 11);
        let localized = LocalizedGaussianMixture3d::new(mixture.clone(), 1.0e-12).unwrap();
        for point in [[0.0; 3], [0.3, -0.2, 0.1], [-0.7, 0.4, 0.5]] {
            let omitted = mixture.eval(point) - localized.eval(point);
            assert!((-1.0e-14..=1.01e-12).contains(&omitted));
        }
    }

    #[test]
    fn cache_round_trip_and_diagonal_embedding() {
        let cache_dir =
            std::env::temp_dir().join(format!("t4a-gaussian3d-cache-{}", std::process::id()));
        let config = Gaussian3dInputConfig {
            n: 2,
            sigma_minor: 0.12,
            rho_max: 2.0,
            spacing: 3.0,
            tci_tolerance: 1.0e-8,
            tci_max_bond_dim: 32,
            localized_absolute_tolerance: 1.0e-12,
            tci_pivot_components: 2,
            input_l2_rtol: INPUT_L2_RTOL,
            fixed_box_n: Some(8),
            seed: 13,
            cache_dir: cache_dir.clone(),
            refresh: true,
        };
        let first = prepare_gaussian3d_input_with_r(&config, 4).unwrap();
        let second = prepare_gaussian3d_input_with_r(
            &Gaussian3dInputConfig {
                refresh: false,
                ..config
            },
            4,
        )
        .unwrap();
        assert!(!first.cache_hit && second.cache_hit);
        assert_eq!(first.active_box_l, 0.5 * 3.0 * 0.12 * (8.0_f64).cbrt());
        assert_eq!(first.raw_chi, second.raw_chi);
        assert_eq!(second.qtt.rank(), second.batch_diagonal_mpo.rank());
        for site in 0..second.batch_diagonal_mpo.len() {
            let core = second.batch_diagonal_mpo.site_tensor(site);
            let qtt_core = &second.qtt.site_tensors()[site];
            for input in 0..4 {
                for output in 0..4 {
                    for left in 0..core.left_dim() {
                        for right in 0..core.right_dim() {
                            let expected = if input % 2 == output % 2 {
                                *qtt_core.get3(
                                    left,
                                    output % 2 + 2 * (output / 2) + 4 * (input / 2),
                                    right,
                                )
                            } else {
                                0.0
                            };
                            assert_eq!(*core.get4(left, output, input, right), expected);
                        }
                    }
                }
            }
        }
        std::fs::remove_dir_all(cache_dir).unwrap();
    }
}
