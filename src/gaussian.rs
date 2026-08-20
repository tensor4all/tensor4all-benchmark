//! Randomly rotated anisotropic Gaussian mixtures and their quantics representation.

use std::collections::HashMap;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tensor4all_interpolativeqtt::{interpolate_multi_scale_nd, InterpolativeQttOptions};
use tensor4all_quanticstci::{
    quanticscrossinterpolate, DiscretizedGrid, QtciOptions, UnfoldingScheme,
};
use tensor4all_simplett::mpo::{tensor4_from_data, MPO};
use tensor4all_simplett::{
    AbstractTensorTrain, CompressionMethod, CompressionOptions, SimpleTensorTrain, Tensor3Ops,
};

/// A scalar function of two variables.
pub trait Field2D {
    /// Value at `(x, y)`.
    fn eval(&self, x: f64, y: f64) -> f64;
}

/// A sum of independently rotated anisotropic 2D Gaussians.
///
/// Term `i` is `w_i exp(-(a_i dx^2 + 2 b_i dx dy + c_i dy^2))`.
#[derive(Clone, Debug)]
pub struct AnisoMixture2D {
    /// Positive prefactors `w_i`.
    pub weights: Vec<f64>,
    /// Positive-definite quadratic forms `(a_i, b_i, c_i)`.
    pub quad: Vec<(f64, f64, f64)>,
    /// Centers `(cx_i, cy_i)`.
    pub centers: Vec<(f64, f64)>,
}

impl AnisoMixture2D {
    /// Draw the benchmark family on `[-box_l, box_l)^2`.
    ///
    /// Weights are uniform in `[0.5, 1.5)`, aspect ratios are log-uniform in
    /// `[1, rho_max]`, orientations are uniform in `[0, pi)`, and centers are
    /// uniform in the inner 90 percent of the box. Draw order is part of the
    /// deterministic instance definition.
    pub fn random(n: usize, box_l: f64, sigma_minor: f64, rho_max: f64, seed: u64) -> Self {
        assert!(n > 0);
        assert!(box_l.is_finite() && box_l > 0.0);
        assert!(sigma_minor.is_finite() && sigma_minor > 0.0);
        assert!(rho_max.is_finite() && rho_max > 1.0);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let half = 0.9 * box_l;
        let a_minor = 1.0 / (2.0 * sigma_minor * sigma_minor);
        let mut weights = Vec::with_capacity(n);
        let mut quad = Vec::with_capacity(n);
        let mut centers = Vec::with_capacity(n);
        for _ in 0..n {
            weights.push(rng.random_range(0.5..1.5));
            let rho = rho_max.powf(rng.random_range(0.0..1.0));
            let theta = rng.random_range(0.0..std::f64::consts::PI);
            let a_major = a_minor / (rho * rho);
            let (sin, cos) = theta.sin_cos();
            quad.push((
                a_major * cos * cos + a_minor * sin * sin,
                (a_major - a_minor) * sin * cos,
                a_major * sin * sin + a_minor * cos * cos,
            ));
            centers.push((rng.random_range(-half..half), rng.random_range(-half..half)));
        }
        Self {
            weights,
            quad,
            centers,
        }
    }

    /// Evaluate the exact mixture.
    pub fn eval(&self, x: f64, y: f64) -> f64 {
        self.weights
            .iter()
            .zip(&self.quad)
            .zip(&self.centers)
            .map(|((&weight, &(a, b, c)), &(cx, cy))| {
                let (dx, dy) = (x - cx, y - cy);
                weight * (-(a * dx * dx + 2.0 * b * dx * dy + c * dy * dy)).exp()
            })
            .sum()
    }

    /// First derivatives with respect to the first and second coordinates.
    pub fn gradient(&self, x: f64, y: f64) -> (f64, f64) {
        self.weights.iter().zip(&self.quad).zip(&self.centers).fold(
            (0.0, 0.0),
            |(gx, gy), ((&weight, &(a, b, c)), &(cx, cy))| {
                let (dx, dy) = (x - cx, y - cy);
                let value = weight * (-(a * dx * dx + 2.0 * b * dx * dy + c * dy * dy)).exp();
                (
                    gx - 2.0 * (a * dx + b * dy) * value,
                    gy - 2.0 * (b * dx + c * dy) * value,
                )
            },
        )
    }

    /// Interpolate each Gaussian independently and add the QTTs by balanced reduction.
    ///
    /// Every pair is compressed immediately, so reduction depth is logarithmic and
    /// no exact sum with bond dimension proportional to the term count is formed.
    /// `r` must be large enough to resolve the minor Gaussian width.
    pub fn to_interpolative_qtt(
        &self,
        r: usize,
        box_l: f64,
        polynomial_degree: usize,
        interpolation_tolerance: f64,
        addition_tolerance: f64,
    ) -> anyhow::Result<SimpleTensorTrain<f64>> {
        anyhow::ensure!(
            self.weights.len() == self.quad.len() && self.weights.len() == self.centers.len(),
            "anisotropic mixture arrays have different lengths"
        );
        anyhow::ensure!(!self.weights.is_empty(), "anisotropic mixture is empty");
        anyhow::ensure!(
            addition_tolerance.is_finite() && addition_tolerance >= 0.0,
            "invalid addition tolerance"
        );
        let compression = CompressionOptions {
            method: CompressionMethod::SVD,
            tolerance: addition_tolerance,
            max_bond_dim: None,
            normalize_error: true,
        };
        let mut level = self
            .weights
            .iter()
            .zip(&self.quad)
            .zip(&self.centers)
            .map(|((&weight, &quad), &center)| {
                rotated_gaussian_qtt(
                    r,
                    box_l,
                    center,
                    quad,
                    weight,
                    polynomial_degree,
                    interpolation_tolerance,
                )
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        while level.len() > 1 {
            let mut next = Vec::with_capacity(level.len().div_ceil(2));
            let mut terms = level.into_iter();
            while let Some(left) = terms.next() {
                if let Some(right) = terms.next() {
                    let mut sum = left.add(&right)?;
                    sum.compress(&compression)?;
                    next.push(sum);
                } else {
                    next.push(left);
                }
            }
            level = next;
        }
        level
            .pop()
            .ok_or_else(|| anyhow::anyhow!("balanced Gaussian sum is empty"))
    }

    /// Construct the same interpolative mixture as a quantics MPO.
    pub fn to_interpolative_mpo(
        &self,
        r: usize,
        box_l: f64,
        polynomial_degree: usize,
        interpolation_tolerance: f64,
        addition_tolerance: f64,
    ) -> anyhow::Result<MPO<f64>> {
        fused_qtt_to_mpo(&self.to_interpolative_qtt(
            r,
            box_l,
            polynomial_degree,
            interpolation_tolerance,
            addition_tolerance,
        )?)
    }
}

impl Field2D for AnisoMixture2D {
    fn eval(&self, x: f64, y: f64) -> f64 {
        AnisoMixture2D::eval(self, x, y)
    }
}

/// Spatially indexed evaluator with a rigorous global absolute tail bound.
///
/// Positive Gaussian components are omitted only when their combined value is
/// bounded by `absolute_tolerance` at every point.
#[derive(Clone, Debug)]
pub struct LocalizedAnisoField {
    mixture: AnisoMixture2D,
    absolute_tolerance: f64,
    bin_width: f64,
    cutoff_squared: Vec<f64>,
    bins: HashMap<(i64, i64), Vec<usize>>,
}

impl LocalizedAnisoField {
    /// Build a local evaluator for one positive anisotropic mixture.
    pub fn new(mixture: AnisoMixture2D, absolute_tolerance: f64) -> anyhow::Result<Self> {
        anyhow::ensure!(
            absolute_tolerance.is_finite() && absolute_tolerance > 0.0,
            "localized evaluator tolerance must be positive and finite"
        );
        anyhow::ensure!(
            mixture.weights.len() == mixture.quad.len()
                && mixture.weights.len() == mixture.centers.len()
                && !mixture.weights.is_empty(),
            "invalid anisotropic mixture"
        );
        anyhow::ensure!(
            mixture
                .weights
                .iter()
                .all(|weight| weight.is_finite() && *weight >= 0.0),
            "localized evaluator requires finite non-negative weights"
        );
        let total_weight = mixture.weights.iter().sum::<f64>();
        anyhow::ensure!(
            total_weight.is_finite() && total_weight > absolute_tolerance,
            "localized evaluator tolerance must be smaller than total weight"
        );
        let exponent_cutoff = (total_weight / absolute_tolerance).ln();
        let cutoff_squared = mixture
            .quad
            .iter()
            .map(|&(a, b, c)| {
                let discriminant = ((a - c).powi(2) + 4.0 * b * b).sqrt();
                let lambda_min = 0.5 * (a + c - discriminant);
                anyhow::ensure!(lambda_min.is_finite() && lambda_min > 0.0);
                Ok(exponent_cutoff / lambda_min)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let bin_width = cutoff_squared
            .iter()
            .copied()
            .fold(0.0_f64, f64::max)
            .sqrt();
        let mut bins = HashMap::<(i64, i64), Vec<usize>>::new();
        for (i, &(x, y)) in mixture.centers.iter().enumerate() {
            bins.entry(Self::bin_key(x, y, bin_width))
                .or_default()
                .push(i);
        }
        Ok(Self {
            mixture,
            absolute_tolerance,
            bin_width,
            cutoff_squared,
            bins,
        })
    }

    fn bin_key(x: f64, y: f64, bin_width: f64) -> (i64, i64) {
        (
            (x / bin_width).floor() as i64,
            (y / bin_width).floor() as i64,
        )
    }

    /// Guaranteed pointwise absolute error bound relative to exact evaluation.
    pub fn absolute_tolerance(&self) -> f64 {
        self.absolute_tolerance
    }

    /// Evaluate the rigorously localized Gaussian sum.
    pub fn eval(&self, x: f64, y: f64) -> f64 {
        let (bx, by) = Self::bin_key(x, y, self.bin_width);
        let mut value = 0.0;
        for dx_bin in -1..=1 {
            for dy_bin in -1..=1 {
                if let Some(indices) = self.bins.get(&(bx + dx_bin, by + dy_bin)) {
                    for &i in indices {
                        let (cx, cy) = self.mixture.centers[i];
                        let (dx, dy) = (x - cx, y - cy);
                        if dx * dx + dy * dy <= self.cutoff_squared[i] {
                            let (a, b, c) = self.mixture.quad[i];
                            value += self.mixture.weights[i]
                                * (-(a * dx * dx + 2.0 * b * dx * dy + c * dy * dy)).exp();
                        }
                    }
                }
            }
        }
        value
    }
}

impl Field2D for LocalizedAnisoField {
    fn eval(&self, x: f64, y: f64) -> f64 {
        LocalizedAnisoField::eval(self, x, y)
    }
}

/// Cross-interpolate one 2D field into a fused dimension-4 QTT.
pub fn global_tci_qtt<M: Field2D + Clone + 'static>(
    field: &M,
    r: usize,
    box_l: f64,
    tolerance: f64,
    max_bond_dim: usize,
    initial_pivots: Vec<Vec<usize>>,
) -> anyhow::Result<SimpleTensorTrain<f64>> {
    anyhow::ensure!(!initial_pivots.is_empty(), "initial pivot list is empty");
    let grid = DiscretizedGrid::builder(&[r, r])
        .with_lower_bound(&[-box_l, -box_l])
        .with_upper_bound(&[box_l, box_l])
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
        move |xy: &[f64]| sampled.eval(xy[0], xy[1]),
        Some(initial_pivots),
        options,
    )?;
    Ok(qtt.tensor_train())
}

fn rotated_gaussian_qtt(
    r: usize,
    box_l: f64,
    center: (f64, f64),
    quad: (f64, f64, f64),
    weight: f64,
    polynomial_degree: usize,
    tolerance: f64,
) -> anyhow::Result<SimpleTensorTrain<f64>> {
    anyhow::ensure!(r >= 2, "an interpolative QTT needs at least two bits");
    anyhow::ensure!(box_l.is_finite() && box_l > 0.0, "invalid box size");
    anyhow::ensure!(
        center.0.abs() < box_l && center.1.abs() < box_l,
        "Gaussian center is outside the box"
    );
    let (a, b, c) = quad;
    let discriminant = ((a - c).powi(2) + 4.0 * b * b).sqrt();
    let lambda_major = 0.5 * (a + c - discriminant);
    let lambda_minor = 0.5 * (a + c + discriminant);
    anyhow::ensure!(
        lambda_major.is_finite()
            && lambda_major > 0.0
            && lambda_minor.is_finite()
            && lambda_minor >= lambda_major,
        "Gaussian quadratic form is not positive definite"
    );
    anyhow::ensure!(
        weight.is_finite() && weight >= 0.0,
        "invalid Gaussian weight"
    );
    anyhow::ensure!(
        tolerance.is_finite() && tolerance >= 0.0,
        "invalid interpolation tolerance"
    );

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
    let sigma_major = (2.0 * lambda_major).sqrt().recip();
    let sigma_minor = (2.0 * lambda_minor).sqrt().recip();
    let spacing = 0.5 * sigma_minor;
    let extent = (4.0 * sigma_major / spacing).ceil() as isize;
    let unsafe_points = (-extent..=extent)
        .filter_map(|i| {
            let distance = i as f64 * spacing;
            let point = [center.0 + distance * major.0, center.1 + distance * major.1];
            (point[0].abs() < box_l && point[1].abs() < box_l).then(|| point.to_vec())
        })
        .collect::<Vec<_>>();
    interpolate_multi_scale_nd(
        |xy| {
            let (dx, dy) = (xy[0] - center.0, xy[1] - center.1);
            weight * (-(a * dx * dx + 2.0 * b * dx * dy + c * dy * dy)).exp()
        },
        &[-box_l, -box_l],
        &[box_l, box_l],
        r,
        polynomial_degree,
        &unsafe_points,
        &InterpolativeQttOptions::default().with_tolerance(tolerance),
    )
    .map_err(anyhow::Error::from)
}

/// Convert a fused site-dimension-4 QTT to a two-index-per-site MPO.
pub fn fused_qtt_to_mpo(qtt: &SimpleTensorTrain<f64>) -> anyhow::Result<MPO<f64>> {
    let tensors = qtt
        .site_tensors()
        .iter()
        .map(|core| {
            anyhow::ensure!(core.site_dim() == 4, "expected fused site dimension 4");
            let mut data = Vec::with_capacity(core.left_dim() * 4 * core.right_dim());
            for right in 0..core.right_dim() {
                for y_bit in 0..2 {
                    for x_bit in 0..2 {
                        for left in 0..core.left_dim() {
                            data.push(*core.get3(left, x_bit + 2 * y_bit, right));
                        }
                    }
                }
            }
            tensor4_from_data(data, core.left_dim(), 2, 2, core.right_dim())
                .map_err(anyhow::Error::from)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    MPO::new(tensors).map_err(anyhow::Error::from)
}

/// Coordinate of grid index `i` on the half-open interval `[-box_l, box_l)`.
pub fn grid_coord(i: u64, r: usize, box_l: f64) -> f64 {
    -box_l + i as f64 * (2.0 * box_l / (1u64 << r) as f64)
}

/// Closed form of the infinite-domain contraction `int f(x,y) g(y,z) dy`.
pub fn analytic_contraction_aniso(f: &AnisoMixture2D, g: &AnisoMixture2D, x: f64, z: f64) -> f64 {
    analytic_contraction_aniso_with(f, g, x, z, |quadratic, linear, constant| {
        (std::f64::consts::PI / quadratic).sqrt()
            * (-(constant - linear * linear / quadratic)).exp()
    })
}

/// Closed form of `int f(x,y) g(y,z) dy` over `[-box_l, box_l]`.
pub fn analytic_contraction_aniso_box(
    f: &AnisoMixture2D,
    g: &AnisoMixture2D,
    x: f64,
    z: f64,
    box_l: f64,
) -> f64 {
    analytic_contraction_aniso_with(f, g, x, z, |quadratic, linear, constant| {
        let root = quadratic.sqrt();
        let center = -linear / quadratic;
        let bounded = libm::erf(root * (box_l - center)) - libm::erf(root * (-box_l - center));
        0.5 * (std::f64::consts::PI / quadratic).sqrt()
            * (-(constant - linear * linear / quadratic)).exp()
            * bounded
    })
}

/// Euler-Maclaurin reference for the left-endpoint quantics grid sum times its step.
pub fn discrete_contraction_aniso_reference(
    f: &AnisoMixture2D,
    g: &AnisoMixture2D,
    x: f64,
    z: f64,
    r: usize,
    box_l: f64,
) -> f64 {
    let step = 2.0 * box_l / (1u64 << r) as f64;
    let boundary = |y: f64| {
        let fv = f.eval(x, y);
        let gv = g.eval(y, z);
        let fy = f.gradient(x, y).1;
        let gy = g.gradient(y, z).0;
        (fv * gv, fy * gv + fv * gy)
    };
    let (left, left_derivative) = boundary(-box_l);
    let (right, right_derivative) = boundary(box_l);
    analytic_contraction_aniso_box(f, g, x, z, box_l)
        + 0.5 * step * (left - right)
        + step * step * (right_derivative - left_derivative) / 12.0
}

fn analytic_contraction_aniso_with(
    f: &AnisoMixture2D,
    g: &AnisoMixture2D,
    x: f64,
    z: f64,
    integral: impl Fn(f64, f64, f64) -> f64,
) -> f64 {
    let mut sum = 0.0;
    for i in 0..f.weights.len() {
        let (fcx, fcy) = f.centers[i];
        let (fa, fb, fc) = f.quad[i];
        let dx = x - fcx;
        for j in 0..g.weights.len() {
            let (gcy, gcz) = g.centers[j];
            let (ga, gb, gc) = g.quad[j];
            let dz = z - gcz;
            let quadratic = fc + ga;
            let linear = fb * dx - fc * fcy + gb * dz - ga * gcy;
            let constant = fa * dx * dx - 2.0 * fb * dx * fcy + fc * fcy * fcy + ga * gcy * gcy
                - 2.0 * gb * gcy * dz
                + gc * dz * dz;
            sum += f.weights[i] * g.weights[j] * integral(quadratic, linear, constant);
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::index_to_bits;

    fn fused_point(ix: u64, iy: u64, r: usize) -> Vec<usize> {
        index_to_bits(ix, r)
            .into_iter()
            .zip(index_to_bits(iy, r))
            .map(|(x, y)| x + 2 * y)
            .collect()
    }

    #[test]
    fn rotated_gaussian_interpolation_resolves_a_narrow_ridge() {
        let (r, box_l) = (10, 1.0);
        let center = (0.07, -0.09);
        let (sigma, rho, angle) = (0.05_f64, 8.0_f64, std::f64::consts::FRAC_PI_4);
        let minor = 1.0 / (2.0 * sigma.powi(2));
        let major = minor / rho.powi(2);
        let (sin, cos) = angle.sin_cos();
        let quad = (
            major * cos * cos + minor * sin * sin,
            (major - minor) * sin * cos,
            major * sin * sin + minor * cos * cos,
        );
        let qtt = rotated_gaussian_qtt(r, box_l, center, quad, 1.3, 28, 1e-12).unwrap();
        let step = 2.0 * box_l / (1usize << r) as f64;
        let mut max_error = 0.0_f64;
        for sample in 0..256 {
            let ix = ((73 * sample + 11) % (1usize << r)) as u64;
            let iy = ((151 * sample + 29) % (1usize << r)) as u64;
            let x = grid_coord(ix, r, box_l);
            let y = grid_coord(iy, r, box_l);
            let (dx, dy) = (x - center.0, y - center.1);
            let expected =
                1.3 * (-(quad.0 * dx * dx + 2.0 * quad.1 * dx * dy + quad.2 * dy * dy)).exp();
            max_error =
                max_error.max((qtt.evaluate(&fused_point(ix, iy, r)).unwrap() - expected).abs());
        }
        assert!(
            max_error < 5e-8,
            "rank={}, error={max_error:.3e}",
            qtt.rank()
        );
        assert!(step <= sigma / 4.0);
    }

    #[test]
    fn balanced_interpolative_sum_matches_the_mixture() {
        let (r, box_l) = (8, 1.0);
        let mixture = AnisoMixture2D::random(5, 0.7, 0.08, 3.0, 17);
        let qtt = mixture
            .to_interpolative_qtt(r, box_l, 24, 1e-12, 1e-10)
            .unwrap();
        let mut squared_error = 0.0;
        let mut squared_reference = 0.0;
        for sample in 0..128 {
            let ix = ((73 * sample + 11) % (1usize << r)) as u64;
            let iy = ((151 * sample + 29) % (1usize << r)) as u64;
            let expected = mixture.eval(grid_coord(ix, r, box_l), grid_coord(iy, r, box_l));
            let error = qtt.evaluate(&fused_point(ix, iy, r)).unwrap() - expected;
            squared_error += error * error;
            squared_reference += expected * expected;
        }
        assert!((squared_error / squared_reference).sqrt() < 1e-7);
    }

    #[test]
    fn localized_evaluator_respects_its_global_tail_bound() {
        let mixture = AnisoMixture2D::random(64, 2.0, 0.05, 8.0, 29);
        let tolerance = 1e-10;
        let localized = LocalizedAnisoField::new(mixture.clone(), tolerance).unwrap();
        assert_eq!(localized.absolute_tolerance(), tolerance);
        for sample in 0..256 {
            let x = -2.0 + 4.0 * ((73 * sample + 11) % 1024) as f64 / 1024.0;
            let y = -2.0 + 4.0 * ((151 * sample + 29) % 1024) as f64 / 1024.0;
            assert!((mixture.eval(x, y) - localized.eval(x, y)).abs() <= 1.01 * tolerance);
        }
    }

    #[test]
    fn anisotropic_grid_reference_matches_the_explicit_sum() {
        let f = AnisoMixture2D::random(3, 1.0, 0.3, 4.0, 1);
        let g = AnisoMixture2D::random(2, 1.0, 0.3, 4.0, 2);
        let (x, z, r, box_l) = (0.3, -0.7, 10, 1.0);
        let step = 2.0 * box_l / (1u64 << r) as f64;
        let direct = (0..1u64 << r)
            .map(|i| f.eval(x, grid_coord(i, r, box_l)) * g.eval(grid_coord(i, r, box_l), z) * step)
            .sum::<f64>();
        let reference = discrete_contraction_aniso_reference(&f, &g, x, z, r, box_l);
        assert!((direct - reference).abs() < 1e-10 * direct.abs().max(1.0));
    }

    #[test]
    fn random_draws_the_declared_anisotropic_family() {
        let (sigma, rho_max) = (0.05, 8.0);
        let mix = AnisoMixture2D::random(64, 1.0, sigma, rho_max, 7);
        assert_eq!(
            AnisoMixture2D::random(64, 1.0, sigma, rho_max, 7).quad,
            mix.quad
        );
        let expected_minor = 1.0 / (2.0 * sigma * sigma);
        for (i, &(a, b, c)) in mix.quad.iter().enumerate() {
            let mean = 0.5 * (a + c);
            let gap = (0.25 * (a - c).powi(2) + b * b).sqrt();
            let rho = (expected_minor / (mean - gap)).sqrt();
            assert!(
                (mean + gap - expected_minor).abs() < 1e-9 * expected_minor,
                "term {i}"
            );
            assert!((1.0..=rho_max).contains(&rho), "term {i}: rho={rho}");
            assert!((0.5..1.5).contains(&mix.weights[i]));
        }
    }
}
