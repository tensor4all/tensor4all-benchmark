//! Localized analytic contraction of positive anisotropic Gaussian mixtures.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::gaussian::{AnisoMixture2D, LocalizedAnisoField};

const CACHE_MAGIC: &[u8; 8] = b"T4AIG02\0";
/// Global pointwise absolute omission budget for output references.
pub const OUTPUT_REFERENCE_ABS_TOLERANCE: f64 = 1e-12;

/// Work and omission metadata for one integrated Gaussian mixture.
#[derive(Clone, Debug)]
pub struct IntegratedGaussianStats {
    /// Cartesian pair count avoided by the neighbor list.
    pub total_pair_count: usize,
    /// Pairs examined after the one-dimensional y cell lookup.
    pub candidate_pair_count: usize,
    /// Output Gaussian components retained by the rigorous peak bound.
    pub retained_pair_count: usize,
    /// Proven global pointwise bound on all omitted pair peaks.
    pub omitted_absolute_bound: f64,
    /// Width of the y cells used for candidate enumeration.
    pub y_cell_width: f64,
}

/// Cached and spatially indexed integrated-mixture reference.
pub struct IntegratedGaussianReference {
    /// Localized x/z evaluator containing the retained Gaussian metadata.
    pub field: LocalizedAnisoField,
    /// Pair-enumeration statistics from cache construction.
    pub stats: IntegratedGaussianStats,
    /// Whether the retained mixture was loaded from disk.
    pub cache_hit: bool,
    /// Time spent loading the retained mixture.
    pub cache_load: Duration,
    /// Time spent enumerating and writing the retained mixture.
    pub build: Duration,
    /// Cache artifact path.
    pub cache_path: PathBuf,
}

/// Load or build and spatially index an integrated-mixture reference.
pub fn prepare_integrated_reference(
    left: &AnisoMixture2D,
    right: &AnisoMixture2D,
    absolute_tolerance: f64,
    cache_path: &Path,
    refresh: bool,
) -> anyhow::Result<IntegratedGaussianReference> {
    let load_start = Instant::now();
    let (mixture, stats, cache_hit, cache_load, build) = if cache_path.exists() && !refresh {
        match load_cache(cache_path) {
            Ok((mixture, stats)) => (mixture, stats, true, load_start.elapsed(), Duration::ZERO),
            Err(_) => {
                let cache_load = load_start.elapsed();
                let (mixture, stats, build) =
                    build_cache(left, right, absolute_tolerance, cache_path)?;
                (mixture, stats, false, cache_load, build)
            }
        }
    } else {
        let (mixture, stats, build) = build_cache(left, right, absolute_tolerance, cache_path)?;
        (mixture, stats, false, Duration::ZERO, build)
    };
    let field = LocalizedAnisoField::new(mixture, absolute_tolerance)?;
    Ok(IntegratedGaussianReference {
        field,
        stats,
        cache_hit,
        cache_load,
        build,
        cache_path: cache_path.to_path_buf(),
    })
}

fn build_cache(
    left: &AnisoMixture2D,
    right: &AnisoMixture2D,
    absolute_tolerance: f64,
    cache_path: &Path,
) -> anyhow::Result<(AnisoMixture2D, IntegratedGaussianStats, Duration)> {
    let build_start = Instant::now();
    let (mixture, stats) = integrate_mixtures(left, right, absolute_tolerance)?;
    if let Some(parent) = cache_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    write_cache(cache_path, &mixture, &stats)?;
    Ok((mixture, stats, build_start.elapsed()))
}

#[derive(Clone, Copy)]
struct YProfile {
    center: f64,
    precision: f64,
    radius: f64,
}

struct IntegratedComponent {
    weight: f64,
    quad: (f64, f64, f64),
    center: (f64, f64),
}

fn y_profiles(
    mixture: &AnisoMixture2D,
    first_coordinate_is_y: bool,
    exponent_cutoff: f64,
) -> anyhow::Result<Vec<YProfile>> {
    mixture
        .quad
        .iter()
        .zip(&mixture.centers)
        .map(|(&(a, b, c), &(first, second))| {
            let (center, precision) = if first_coordinate_is_y {
                (first, a - b * b / c)
            } else {
                (second, c - b * b / a)
            };
            anyhow::ensure!(precision.is_finite() && precision > 0.0);
            Ok(YProfile {
                center,
                precision,
                radius: (exponent_cutoff / precision).sqrt(),
            })
        })
        .collect()
}

/// Integrate `left(x,y) * right(y,z)` over the infinite y axis.
///
/// A linked-cell neighbor list avoids the Cartesian pair scan. Components whose
/// maximum possible contribution is omitted have a combined pointwise value no
/// larger than `absolute_tolerance`.
pub fn integrate_mixtures(
    left: &AnisoMixture2D,
    right: &AnisoMixture2D,
    absolute_tolerance: f64,
) -> anyhow::Result<(AnisoMixture2D, IntegratedGaussianStats)> {
    let mut weights = Vec::new();
    let mut quad = Vec::new();
    let mut centers = Vec::new();
    let stats = enumerate_significant_pairs(
        left,
        right,
        absolute_tolerance,
        |i, j| -> anyhow::Result<()> {
            let component = integrate_pair(left, i, right, j)?;
            weights.push(component.weight);
            quad.push(component.quad);
            centers.push(component.center);
            Ok(())
        },
    )?;
    anyhow::ensure!(
        !weights.is_empty(),
        "all integrated Gaussian pairs were omitted"
    );
    Ok((
        AnisoMixture2D {
            weights,
            quad,
            centers,
        },
        stats,
    ))
}

/// Count significant integrated components without constructing or storing them.
pub fn count_integrated_components(
    left: &AnisoMixture2D,
    right: &AnisoMixture2D,
    absolute_tolerance: f64,
) -> anyhow::Result<IntegratedGaussianStats> {
    enumerate_significant_pairs(left, right, absolute_tolerance, |_, _| Ok(()))
}

fn enumerate_significant_pairs(
    left: &AnisoMixture2D,
    right: &AnisoMixture2D,
    absolute_tolerance: f64,
    mut visit: impl FnMut(usize, usize) -> anyhow::Result<()>,
) -> anyhow::Result<IntegratedGaussianStats> {
    anyhow::ensure!(
        absolute_tolerance.is_finite() && absolute_tolerance > 0.0,
        "integrated-mixture tolerance must be positive and finite"
    );
    anyhow::ensure!(
        !left.weights.is_empty() && !right.weights.is_empty(),
        "integrated mixtures must be nonempty"
    );
    let left_weight = left.weights.iter().sum::<f64>();
    let right_weight = right.weights.iter().sum::<f64>();
    anyhow::ensure!(left_weight.is_finite() && right_weight.is_finite());

    let left_base = y_profiles(left, false, 1.0)?;
    let right_base = y_profiles(right, true, 1.0)?;
    let min_precision = left_base
        .iter()
        .map(|profile| profile.precision)
        .fold(f64::INFINITY, f64::min)
        + right_base
            .iter()
            .map(|profile| profile.precision)
            .fold(f64::INFINITY, f64::min);
    let total_peak_prefactor_bound =
        left_weight * right_weight * (std::f64::consts::PI / min_precision).sqrt();
    anyhow::ensure!(
        total_peak_prefactor_bound.is_finite() && total_peak_prefactor_bound > absolute_tolerance,
        "invalid integrated-mixture peak bound"
    );
    let exponent_cutoff = (total_peak_prefactor_bound / absolute_tolerance).ln();
    let left_profiles = y_profiles(left, false, exponent_cutoff)?;
    let right_profiles = y_profiles(right, true, exponent_cutoff)?;
    let max_radius = left_profiles
        .iter()
        .chain(&right_profiles)
        .map(|profile| profile.radius)
        .fold(0.0, f64::max);
    anyhow::ensure!(max_radius.is_finite() && max_radius > 0.0);
    let cell_width = max_radius / 4.0;
    let max_right_radius = right_profiles
        .iter()
        .map(|profile| profile.radius)
        .fold(0.0, f64::max);
    let cell = |coordinate: f64| (coordinate / cell_width).floor() as i64;
    let mut right_cells: HashMap<i64, Vec<usize>> = HashMap::new();
    for (index, profile) in right_profiles.iter().enumerate() {
        right_cells
            .entry(cell(profile.center))
            .or_default()
            .push(index);
    }

    let total_pair_count = left
        .weights
        .len()
        .checked_mul(right.weights.len())
        .ok_or_else(|| anyhow::anyhow!("Gaussian pair count overflow"))?;
    let mut candidate_pair_count = 0usize;
    let mut retained_pair_count = 0usize;
    for (i, left_profile) in left_profiles.iter().enumerate() {
        let center_cell = cell(left_profile.center);
        let cell_reach = ((left_profile.radius + max_right_radius) / cell_width).ceil() as i64 + 1;
        for neighbor_cell in center_cell - cell_reach..=center_cell + cell_reach {
            let Some(candidates) = right_cells.get(&neighbor_cell) else {
                continue;
            };
            for &j in candidates {
                candidate_pair_count += 1;
                let right_profile = right_profiles[j];
                let dy = left_profile.center - right_profile.center;
                if dy.abs() > left_profile.radius + right_profile.radius {
                    continue;
                }
                let q = left_profile.precision + right_profile.precision;
                let mismatch = left_profile.precision * right_profile.precision / q * dy * dy;
                if mismatch > exponent_cutoff {
                    continue;
                }
                visit(i, j)?;
                retained_pair_count += 1;
            }
        }
    }
    Ok(IntegratedGaussianStats {
        total_pair_count,
        candidate_pair_count,
        retained_pair_count,
        omitted_absolute_bound: absolute_tolerance,
        y_cell_width: cell_width,
    })
}

fn integrate_pair(
    left: &AnisoMixture2D,
    i: usize,
    right: &AnisoMixture2D,
    j: usize,
) -> anyhow::Result<IntegratedComponent> {
    let (fa, fb, fc) = left.quad[i];
    let (ga, gb, gc) = right.quad[j];
    let (fcx, fcy) = left.centers[i];
    let (gcy, gcz) = right.centers[j];
    let qy = fc + ga;
    let a = fa - fb * fb / qy;
    let b = -fb * gb / qy;
    let c = gc - gb * gb / qy;
    let hx = fa * fcx + fb * fcy;
    let hy = fb * fcx + fc * fcy + ga * gcy + gb * gcz;
    let hz = gb * gcy + gc * gcz;
    let ux = hx - fb * hy / qy;
    let uz = hz - gb * hy / qy;
    let determinant = a * c - b * b;
    anyhow::ensure!(qy > 0.0 && determinant > 0.0);
    let center = (
        (c * ux - b * uz) / determinant,
        (a * uz - b * ux) / determinant,
    );
    let constant = fa * fcx * fcx
        + 2.0 * fb * fcx * fcy
        + fc * fcy * fcy
        + ga * gcy * gcy
        + 2.0 * gb * gcy * gcz
        + gc * gcz * gcz
        - hy * hy / qy;
    let completed = constant
        - (a * center.0 * center.0 + 2.0 * b * center.0 * center.1 + c * center.1 * center.1);
    let weight = left.weights[i]
        * right.weights[j]
        * (std::f64::consts::PI / qy).sqrt()
        * (-completed).exp();
    anyhow::ensure!(weight.is_finite() && weight >= 0.0);
    Ok(IntegratedComponent {
        weight,
        quad: (a, b, c),
        center,
    })
}

fn write_cache(
    path: &Path,
    mixture: &AnisoMixture2D,
    stats: &IntegratedGaussianStats,
) -> anyhow::Result<()> {
    let temporary = path.with_extension(format!("{}.tmp", std::process::id()));
    let mut file = std::io::BufWriter::new(std::fs::File::create(&temporary)?);
    file.write_all(CACHE_MAGIC)?;
    for value in [
        mixture.weights.len() as u64,
        stats.total_pair_count as u64,
        stats.candidate_pair_count as u64,
        stats.retained_pair_count as u64,
    ] {
        file.write_all(&value.to_le_bytes())?;
    }
    file.write_all(&stats.omitted_absolute_bound.to_le_bytes())?;
    file.write_all(&stats.y_cell_width.to_le_bytes())?;
    for ((&weight, &(a, b, c)), &(x, z)) in mixture
        .weights
        .iter()
        .zip(&mixture.quad)
        .zip(&mixture.centers)
    {
        for value in [weight, a, b, c, x, z] {
            file.write_all(&value.to_le_bytes())?;
        }
    }
    file.flush()?;
    drop(file);
    std::fs::rename(temporary, path)?;
    Ok(())
}

fn load_cache(path: &Path) -> anyhow::Result<(AnisoMixture2D, IntegratedGaussianStats)> {
    let mut file = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic)?;
    anyhow::ensure!(
        &magic == CACHE_MAGIC,
        "integrated Gaussian cache schema mismatch"
    );
    let read_u64 = |reader: &mut std::io::BufReader<std::fs::File>| -> anyhow::Result<u64> {
        let mut bytes = [0u8; 8];
        reader.read_exact(&mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    };
    let count = usize::try_from(read_u64(&mut file)?)?;
    let total_pair_count = usize::try_from(read_u64(&mut file)?)?;
    let candidate_pair_count = usize::try_from(read_u64(&mut file)?)?;
    let retained_pair_count = usize::try_from(read_u64(&mut file)?)?;
    anyhow::ensure!(
        count == retained_pair_count,
        "integrated cache count mismatch"
    );
    let read_f64 = |reader: &mut std::io::BufReader<std::fs::File>| -> anyhow::Result<f64> {
        let mut bytes = [0u8; 8];
        reader.read_exact(&mut bytes)?;
        Ok(f64::from_le_bytes(bytes))
    };
    let omitted_absolute_bound = read_f64(&mut file)?;
    let y_cell_width = read_f64(&mut file)?;
    anyhow::ensure!(
        omitted_absolute_bound.is_finite() && omitted_absolute_bound > 0.0,
        "invalid cached omission bound"
    );
    anyhow::ensure!(
        y_cell_width.is_finite() && y_cell_width > 0.0,
        "invalid cached y-cell width"
    );
    let expected_bytes = 56u64
        .checked_add(
            u64::try_from(count)?
                .checked_mul(48)
                .ok_or_else(|| anyhow::anyhow!("integrated cache byte count overflow"))?,
        )
        .ok_or_else(|| anyhow::anyhow!("integrated cache byte count overflow"))?;
    anyhow::ensure!(
        std::fs::metadata(path)?.len() == expected_bytes,
        "integrated cache length mismatch"
    );
    let mut weights = Vec::with_capacity(count);
    let mut quad = Vec::with_capacity(count);
    let mut centers = Vec::with_capacity(count);
    for _ in 0..count {
        let weight = read_f64(&mut file)?;
        let a = read_f64(&mut file)?;
        let b = read_f64(&mut file)?;
        let c = read_f64(&mut file)?;
        let x = read_f64(&mut file)?;
        let z = read_f64(&mut file)?;
        anyhow::ensure!(
            weight.is_finite()
                && weight >= 0.0
                && a.is_finite()
                && b.is_finite()
                && c.is_finite()
                && a > 0.0
                && a * c - b * b > 0.0
                && x.is_finite()
                && z.is_finite(),
            "invalid integrated Gaussian cache component"
        );
        weights.push(weight);
        quad.push((a, b, c));
        centers.push((x, z));
    }
    Ok((
        AnisoMixture2D {
            weights,
            quad,
            centers,
        },
        IntegratedGaussianStats {
            total_pair_count,
            candidate_pair_count,
            retained_pair_count,
            omitted_absolute_bound,
            y_cell_width,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaussian::analytic_contraction_aniso;

    #[test]
    fn localized_integrated_mixture_matches_all_pairs_reference() {
        let left = AnisoMixture2D::random(5, 2.0, 0.2, 3.0, 1);
        let right = AnisoMixture2D::random(6, 2.0, 0.2, 3.0, 2);
        let (output, stats) = integrate_mixtures(&left, &right, 1e-14).unwrap();
        assert_eq!(stats.retained_pair_count, output.weights.len());
        assert!(stats.candidate_pair_count <= stats.total_pair_count);
        for (x, z) in [(-0.8, 0.3), (0.1, -0.2), (1.2, 0.9)] {
            let expected = analytic_contraction_aniso(&left, &right, x, z);
            assert!((output.eval(x, z) - expected).abs() < 1e-11);
        }
    }

    #[test]
    fn cache_round_trip_preserves_integrated_components() {
        let left = AnisoMixture2D::random(3, 2.0, 0.2, 3.0, 11);
        let right = AnisoMixture2D::random(4, 2.0, 0.2, 3.0, 12);
        let path = std::env::temp_dir().join(format!(
            "t4a-integrated-gaussian-cache-{}.bin",
            std::process::id()
        ));
        let first = prepare_integrated_reference(&left, &right, 1e-14, &path, true).unwrap();
        let second = prepare_integrated_reference(&left, &right, 1e-14, &path, false).unwrap();
        assert!(!first.cache_hit);
        assert!(second.cache_hit);
        assert_eq!(
            first.stats.retained_pair_count,
            second.stats.retained_pair_count
        );
        assert_eq!(
            first.field.mixture().weights,
            second.field.mixture().weights
        );
        assert_eq!(first.field.mixture().quad, second.field.mixture().quad);
        assert_eq!(
            first.field.mixture().centers,
            second.field.mixture().centers
        );
        std::fs::write(&path, b"corrupt").unwrap();
        let rebuilt = prepare_integrated_reference(&left, &right, 1e-14, &path, false).unwrap();
        assert!(!rebuilt.cache_hit);
        assert_eq!(
            first.field.mixture().weights,
            rebuilt.field.mixture().weights
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn distant_y_centers_are_removed_without_all_pairs_scan() {
        let mut left = AnisoMixture2D::random(2, 1.0, 0.1, 2.0, 3);
        let mut right = AnisoMixture2D::random(2, 1.0, 0.1, 2.0, 4);
        left.centers.fill((0.0, -10.0));
        right.centers.fill((10.0, 0.0));
        assert!(integrate_mixtures(&left, &right, 1e-12).is_err());
    }
}
