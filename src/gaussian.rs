//! 2D Gaussian mixtures: analytic y-integral and quantics MPO construction.

use std::collections::HashMap;

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor4all_interpolativeqtt::{
    direct_product_core_tensors, interpolate_multi_scale, interpolate_multi_scale_nd,
    InterpolativeQttOptions,
};
use tensor4all_quanticstci::{
    quanticscrossinterpolate, DiscretizedGrid, QtciOptions, UnfoldingScheme,
};
use tensor4all_simplett::mpo::{tensor4_from_data, MPO};
use tensor4all_simplett::{
    AbstractTensorTrain, CompressionMethod, CompressionOptions, SimpleTensorTrain, Tensor3Ops,
};

/// A scalar function of two variables that a benchmark case can compress.
///
/// The two instance families of case 5 differ only in what they evaluate, so
/// every piece of machinery that samples a function (the quantics construction,
/// the degeneracy guard, the error metric, the patched input construction) is
/// written against this trait rather than against one concrete mixture. `Clone`
/// and `'static` are required because the quantics construction moves an owned
/// copy of the function into the closure it hands to the cross interpolation.
pub trait Field2D: Clone + 'static {
    /// Value at `(x, y)`.
    fn eval(&self, x: f64, y: f64) -> f64;
}

/// Construct one axis-aligned Gaussian as a quantics MPO.
///
/// Each one-variable factor is built by multiscale interpolative QTT with its
/// center marked as the interval that must remain refined. The factors are then
/// combined core-wise, without another interpolation or approximation.
///
/// This construction assumes `r` is large enough to resolve both Gaussian
/// widths on `[-box_l, box_l)`. It does not attempt to detect an under-resolved
/// grid.
pub fn axis_aligned_gaussian_mpo(
    r: usize,
    box_l: f64,
    center: (f64, f64),
    alpha: (f64, f64),
    weight: f64,
    polynomial_degree: usize,
    tolerance: f64,
) -> anyhow::Result<MPO<f64>> {
    anyhow::ensure!(r >= 2, "an interpolative QTT needs at least two bits");
    anyhow::ensure!(box_l.is_finite() && box_l > 0.0, "invalid box size");
    anyhow::ensure!(
        center.0.abs() < box_l && center.1.abs() < box_l,
        "Gaussian center is outside the box"
    );
    anyhow::ensure!(
        alpha.0.is_finite() && alpha.0 > 0.0 && alpha.1.is_finite() && alpha.1 > 0.0,
        "invalid Gaussian width"
    );
    anyhow::ensure!(weight.is_finite(), "invalid Gaussian weight");
    anyhow::ensure!(
        tolerance.is_finite() && tolerance >= 0.0,
        "invalid interpolation tolerance"
    );

    let options = InterpolativeQttOptions::default().with_tolerance(tolerance);
    let x = interpolate_multi_scale(
        |value| weight * (-alpha.0 * (value - center.0).powi(2)).exp(),
        -box_l,
        box_l,
        r,
        polynomial_degree,
        &[center.0],
        &options,
    )?;
    let y = interpolate_multi_scale(
        |value| (-alpha.1 * (value - center.1).powi(2)).exp(),
        -box_l,
        box_l,
        r,
        polynomial_degree,
        &[center.1],
        &options,
    )?;

    let tensors = x
        .site_tensors()
        .iter()
        .zip(y.site_tensors())
        .map(|(x_core, y_core)| {
            let fused = direct_product_core_tensors(&[x_core.clone(), y_core.clone()])?;
            let mut data =
                Vec::with_capacity(fused.left_dim() * fused.site_dim() * fused.right_dim());
            for right in 0..fused.right_dim() {
                for y_bit in 0..2 {
                    for x_bit in 0..2 {
                        for left in 0..fused.left_dim() {
                            data.push(*fused.get3(left, x_bit + 2 * y_bit, right));
                        }
                    }
                }
            }
            tensor4_from_data(data, fused.left_dim(), 2, 2, fused.right_dim())
                .map_err(anyhow::Error::from)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    MPO::new(tensors).map_err(anyhow::Error::from)
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
        lambda_major.is_finite() && lambda_major > 0.0 && lambda_minor.is_finite(),
        "Gaussian quadratic form is not positive definite"
    );
    anyhow::ensure!(weight.is_finite(), "invalid Gaussian weight");
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
    let unsafe_spacing = 0.5 * sigma_minor;
    let n = (4.0 * sigma_major / unsafe_spacing).ceil() as isize;
    let cusp_locations = (-n..=n)
        .filter_map(|i| {
            let distance = i as f64 * unsafe_spacing;
            let point = [center.0 + distance * major.0, center.1 + distance * major.1];
            (point[0].abs() < box_l && point[1].abs() < box_l).then(|| point.to_vec())
        })
        .collect::<Vec<_>>();
    let options = InterpolativeQttOptions::default().with_tolerance(tolerance);
    interpolate_multi_scale_nd(
        |xy| {
            let (dx, dy) = (xy[0] - center.0, xy[1] - center.1);
            weight * (-(a * dx * dx + 2.0 * b * dx * dy + c * dy * dy)).exp()
        },
        &[-box_l, -box_l],
        &[box_l, box_l],
        r,
        polynomial_degree,
        &cusp_locations,
        &options,
    )
    .map_err(anyhow::Error::from)
}

fn fused_qtt_to_mpo(qtt: &SimpleTensorTrain<f64>) -> anyhow::Result<MPO<f64>> {
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

/// Construct one rotated anisotropic Gaussian as a multiscale quantics MPO.
///
/// Unsafe interpolation points are spaced by half the minor-axis standard
/// deviation along the major axis through the center. This keeps every box
/// intersecting the narrow ridge refined without making the entire 2D box unsafe.
/// `r` must be large enough to resolve the minor width.
pub fn rotated_gaussian_mpo(
    r: usize,
    box_l: f64,
    center: (f64, f64),
    quad: (f64, f64, f64),
    weight: f64,
    polynomial_degree: usize,
    tolerance: f64,
) -> anyhow::Result<MPO<f64>> {
    fused_qtt_to_mpo(&rotated_gaussian_qtt(
        r,
        box_l,
        center,
        quad,
        weight,
        polynomial_degree,
        tolerance,
    )?)
}

/// A sum of isotropic 2D Gaussians `w_i exp(-a_i ((x-cx_i)^2 + (y-cy_i)^2))`.
#[derive(Clone, Debug)]
pub struct GaussianMixture2D {
    /// Prefactors `w_i`.
    pub weights: Vec<f64>,
    /// Inverse widths `a_i`.
    pub alphas: Vec<f64>,
    /// Centers `(cx_i, cy_i)`.
    pub centers: Vec<(f64, f64)>,
}

impl GaussianMixture2D {
    /// Draw `n` Gaussians: centers uniform in `[-L/2, L/2]^2` (kept away from the
    /// box edge so tail truncation stays small), weights `U[0.5, 1.5]`, alphas
    /// log-uniform in `alpha_range`.
    pub fn random(n: usize, box_l: f64, alpha_range: (f64, f64), seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let half = box_l / 2.0;
        let (a_lo, a_hi) = alpha_range;
        let mut weights = Vec::with_capacity(n);
        let mut alphas = Vec::with_capacity(n);
        let mut centers = Vec::with_capacity(n);
        for _ in 0..n {
            weights.push(rng.random_range(0.5..1.5));
            alphas.push(rng.random_range(a_lo.ln()..a_hi.ln()).exp());
            centers.push((rng.random_range(-half..half), rng.random_range(-half..half)));
        }
        Self {
            weights,
            alphas,
            centers,
        }
    }

    /// Evaluate the mixture at `(x, y)`.
    pub fn eval(&self, x: f64, y: f64) -> f64 {
        (0..self.weights.len())
            .map(|i| {
                let (cx, cy) = self.centers[i];
                self.weights[i] * (-self.alphas[i] * ((x - cx).powi(2) + (y - cy).powi(2))).exp()
            })
            .sum()
    }
}

impl Field2D for GaussianMixture2D {
    fn eval(&self, x: f64, y: f64) -> f64 {
        GaussianMixture2D::eval(self, x, y)
    }
}

/// A sum of anisotropic 2D Gaussians: narrow spikes of a fixed minor width, each
/// one stretched by its own aspect ratio along its own random direction.
///
/// Spike `i` is `w_i exp(-(a_i dx^2 + 2 b_i dx dy + c_i dy^2))` with
/// `dx = x - cx_i`, `dy = y - cy_i`, and the quadratic form obtained by rotating
/// `diag(a_major, a_minor)` by the orientation angle, where
/// `a_minor = 1 / (2 sigma^2)` fixes the minor width and
/// `a_major = a_minor / rho^2` stretches the spike by `rho` along the major axis.
///
/// This is the family case 5 defaults to. Measured at the case-5 settings its
/// global quantics rank grows like `N^0.5` and reaches the geometric bound of the
/// bit count at `N` = 1024, which is what makes it the family where a global
/// representation runs out of room and a patched one, held at its per-patch cap by
/// construction, has something to win.
///
/// The per-spike orientation and aspect ratio are drawn so that the family is not
/// a pure shift family, which a mixture of one common shape would be: a low
/// dimensional manifold, and the degenerate case a compression study should not
/// rest on. They are not what makes the rank grow, though. Setting `rho_max = 1`
/// gives exactly that common-shape control, and measured at the same settings its
/// rank grows the same way, so the growth comes from the density of narrow spikes.
#[derive(Clone, Debug)]
pub struct AnisoMixture2D {
    /// Prefactors `w_i`.
    pub weights: Vec<f64>,
    /// Quadratic forms `(a_i, b_i, c_i)` of `a dx^2 + 2 b dx dy + c dy^2`.
    pub quad: Vec<(f64, f64, f64)>,
    /// Centers `(cx_i, cy_i)`.
    pub centers: Vec<(f64, f64)>,
}

impl AnisoMixture2D {
    /// Construct this mixture by adding individually interpolated Gaussian QTTs.
    ///
    /// The sum is SVD-compressed after every addition with the supplied relative
    /// per-bond tolerance; the benchmark records this separately from its final
    /// global relative-L2 truncation.
    pub fn to_multiscale_mpo(
        &self,
        r: usize,
        box_l: f64,
        polynomial_degree: usize,
        interpolation_tolerance: f64,
        addition_tolerance: f64,
    ) -> anyhow::Result<MPO<f64>> {
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
        let mut sum = rotated_gaussian_qtt(
            r,
            box_l,
            self.centers[0],
            self.quad[0],
            self.weights[0],
            polynomial_degree,
            interpolation_tolerance,
        )?;
        for i in 1..self.weights.len() {
            sum = sum.add(&rotated_gaussian_qtt(
                r,
                box_l,
                self.centers[i],
                self.quad[i],
                self.weights[i],
                polynomial_degree,
                interpolation_tolerance,
            )?)?;
            sum.compress(&compression)?;
        }
        fused_qtt_to_mpo(&sum)
    }

    /// Draw `n` anisotropic spikes on the box `[-box_l, box_l)^2`.
    ///
    /// Weights are `U[0.5, 1.5]`, the aspect ratio `rho` is log-uniform in
    /// `[1, rho_max]`, the orientation is uniform in `[0, pi)` and the centers are
    /// uniform in `0.9 * box_l`, kept off the box edge so tail truncation stays
    /// small. The draw order is weight, rho, theta, center, per spike, and it is
    /// part of the instance definition: the stream is a single `ChaCha8` sequence,
    /// so reordering the draws would silently change every instance.
    pub fn random(n: usize, box_l: f64, sigma_minor: f64, rho_max: f64, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let half = 0.9 * box_l;
        let a_minor = 1.0 / (2.0 * sigma_minor * sigma_minor);
        let mut weights = Vec::with_capacity(n);
        let mut quad = Vec::with_capacity(n);
        let mut centers = Vec::with_capacity(n);
        for _ in 0..n {
            weights.push(rng.random_range(0.5..1.5));
            // Log-uniform in [1, rho_max], written as a power rather than as
            // `exp(U[0, ln rho_max])` so that `rho_max = 1` is a legal setting: it
            // gives the isotropic control family, spikes of a common shape, which
            // is the comparison that shows the rank growth comes from the density
            // of narrow spikes and not from the anisotropy. An empty range would
            // panic instead.
            let rho = rho_max.powf(rng.random_range(0.0..1.0));
            let theta = rng.random_range(0.0..std::f64::consts::PI);
            let a_major = a_minor / (rho * rho);
            let (s, c) = theta.sin_cos();
            // Rotate diag(a_major, a_minor) into the coordinate axes: the major
            // axis of the spike points along theta and carries the smaller
            // quadratic coefficient, so the spike is rho times longer there.
            quad.push((
                a_major * c * c + a_minor * s * s,
                (a_major - a_minor) * s * c,
                a_major * s * s + a_minor * c * c,
            ));
            centers.push((rng.random_range(-half..half), rng.random_range(-half..half)));
        }
        Self {
            weights,
            quad,
            centers,
        }
    }

    /// Evaluate the mixture at `(x, y)`.
    pub fn eval(&self, x: f64, y: f64) -> f64 {
        (0..self.weights.len())
            .map(|i| {
                let (cx, cy) = self.centers[i];
                let (a, b, c) = self.quad[i];
                let (dx, dy) = (x - cx, y - cy);
                self.weights[i] * (-(a * dx * dx + 2.0 * b * dx * dy + c * dy * dy)).exp()
            })
            .sum()
    }

    /// First derivatives with respect to the first and second coordinates.
    pub fn gradient(&self, x: f64, y: f64) -> (f64, f64) {
        (0..self.weights.len()).fold((0.0, 0.0), |(gx, gy), i| {
            let (cx, cy) = self.centers[i];
            let (a, b, c) = self.quad[i];
            let (dx, dy) = (x - cx, y - cy);
            let value = self.weights[i] * (-(a * dx * dx + 2.0 * b * dx * dy + c * dy * dy)).exp();
            (
                gx - 2.0 * (a * dx + b * dy) * value,
                gy - 2.0 * (b * dx + c * dy) * value,
            )
        })
    }
}

impl Field2D for AnisoMixture2D {
    fn eval(&self, x: f64, y: f64) -> f64 {
        AnisoMixture2D::eval(self, x, y)
    }
}

/// Spatially indexed evaluator with a rigorous global absolute tail bound.
///
/// Components whose center is farther than their major-axis cutoff radius are
/// omitted. The common exponent threshold is chosen so the sum of all omitted
/// positive Gaussian tails is at most `absolute_tolerance` at every point.
#[derive(Clone, Debug)]
pub struct LocalizedAnisoField {
    mixture: AnisoMixture2D,
    absolute_tolerance: f64,
    bin_width: f64,
    cutoff_squared: Vec<f64>,
    bins: HashMap<(i64, i64), Vec<usize>>,
}

impl LocalizedAnisoField {
    /// Build a spatial index without changing the underlying random mixture.
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
        anyhow::ensure!(
            mixture
                .centers
                .iter()
                .all(|(x, y)| x.is_finite() && y.is_finite()),
            "localized evaluator requires finite centers"
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

    /// Evaluate using only components whose tail can exceed the global budget.
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

/// Closed form of `int f(x,y) g(y,z) dy` over the whole real line.
pub fn analytic_contraction(f: &GaussianMixture2D, g: &GaussianMixture2D, x: f64, z: f64) -> f64 {
    let mut s = 0.0;
    for i in 0..f.weights.len() {
        let (fcx, fcy) = f.centers[i];
        let a = f.alphas[i];
        let fx = f.weights[i] * (-a * (x - fcx).powi(2)).exp();
        for j in 0..g.weights.len() {
            let (gcy, gcz) = g.centers[j];
            let b = g.alphas[j];
            let gz = g.weights[j] * (-b * (z - gcz).powi(2)).exp();
            let ab = a + b;
            let yfac =
                (std::f64::consts::PI / ab).sqrt() * (-(a * b / ab) * (fcy - gcy).powi(2)).exp();
            s += fx * gz * yfac;
        }
    }
    s
}

/// Closed form of `int f(x,y) g(y,z) dy` for anisotropic mixtures.
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

/// Coordinate of grid index `i`: `-L + i * 2L/2^R` (the grid is `[-L, L)`).
pub fn grid_coord(i: u64, r: usize, box_l: f64) -> f64 {
    let step = 2.0 * box_l / (1u64 << r) as f64;
    -box_l + i as f64 * step
}

/// Cross-interpolate `mix` into a fused 2D quantics TT on `[-L, L)^2` with `r`
/// bits per variable. Returns the TT (site dim 4 per site) and the grid step.
pub fn to_quantics_fused_tt(
    mix: &GaussianMixture2D,
    r: usize,
    box_l: f64,
    tol: f64,
    max_bond: usize,
) -> anyhow::Result<(SimpleTensorTrain<f64>, f64)> {
    to_quantics_fused_tt_field(mix, r, box_l, tol, max_bond)
}

/// [`to_quantics_fused_tt`] for any [`Field2D`], which is the one case-5 uses so
/// that its two instance families share a single construction. The grid, the
/// unfolding scheme and the `QtciOptions` are identical, so a train built here is
/// the same object the typed entry point returns.
pub fn to_quantics_fused_tt_field<M: Field2D>(
    mix: &M,
    r: usize,
    box_l: f64,
    tol: f64,
    max_bond: usize,
) -> anyhow::Result<(SimpleTensorTrain<f64>, f64)> {
    to_quantics_fused_tt_field_with_pivots(mix, r, box_l, tol, max_bond, None)
}

fn to_quantics_fused_tt_field_with_pivots<M: Field2D>(
    mix: &M,
    r: usize,
    box_l: f64,
    tol: f64,
    max_bond: usize,
    initial_pivots: Option<Vec<Vec<usize>>>,
) -> anyhow::Result<(SimpleTensorTrain<f64>, f64)> {
    let grid = DiscretizedGrid::builder(&[r, r])
        .with_lower_bound(&[-box_l, -box_l])
        .with_upper_bound(&[box_l, box_l])
        .with_unfolding_scheme(UnfoldingScheme::Fused)
        .build()?;
    let m = mix.clone();
    let mut opts = QtciOptions::default()
        .with_tolerance(tol)
        .with_max_bond_dim(max_bond)
        .with_unfoldingscheme(UnfoldingScheme::Fused);
    if initial_pivots.is_some() {
        opts = opts.with_nrandominitpivot(0);
    }
    let (qtci, _ranks, _errs) = quanticscrossinterpolate(
        &grid,
        move |xy: &[f64]| m.eval(xy[0], xy[1]),
        initial_pivots,
        opts,
    )?;
    let step = 2.0 * box_l / (1u64 << r) as f64;
    Ok((qtci.tensor_train(), step))
}

/// `f(v1, v2)` as a quantics MPO: site `n` carries (bit `n` of `v1`, bit `n` of
/// `v2`), most significant bit first, `r` sites. Returns `(mpo, grid_step)`.
pub fn to_quantics_mpo(
    mix: &GaussianMixture2D,
    r: usize,
    box_l: f64,
    tol: f64,
    max_bond: usize,
) -> anyhow::Result<(MPO<f64>, f64)> {
    to_quantics_mpo_field(mix, r, box_l, tol, max_bond)
}

/// [`to_quantics_mpo`] for any [`Field2D`].
pub fn to_quantics_mpo_field<M: Field2D>(
    mix: &M,
    r: usize,
    box_l: f64,
    tol: f64,
    max_bond: usize,
) -> anyhow::Result<(MPO<f64>, f64)> {
    let (tt, step) = to_quantics_fused_tt_field(mix, r, box_l, tol, max_bond)?;
    Ok((fused_qtt_to_mpo(&tt)?, step))
}

/// Deterministic [`to_quantics_mpo_field`] variant using explicit zero-based grid pivots.
///
/// Random initial pivots are disabled when this entry point is used, so identical
/// inputs and pivots produce cacheable input tensors.
pub fn to_quantics_mpo_field_with_pivots<M: Field2D>(
    mix: &M,
    r: usize,
    box_l: f64,
    tol: f64,
    max_bond: usize,
    initial_pivots: Vec<Vec<usize>>,
) -> anyhow::Result<(MPO<f64>, f64)> {
    anyhow::ensure!(!initial_pivots.is_empty(), "initial pivot list is empty");
    let (tt, step) =
        to_quantics_fused_tt_field_with_pivots(mix, r, box_l, tol, max_bond, Some(initial_pivots))?;
    Ok((fused_qtt_to_mpo(&tt)?, step))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::index_to_bits;

    fn quadrature(f: impl Fn(f64) -> f64, lo: f64, hi: f64, intervals: usize) -> f64 {
        let step = (hi - lo) / intervals as f64;
        (0..=intervals)
            .map(|i| {
                let weight = if i == 0 || i == intervals { 0.5 } else { 1.0 };
                weight * f(lo + i as f64 * step) * step
            })
            .sum()
    }

    #[test]
    fn multiscale_axis_aligned_gaussian_mpo_matches_grid_values() {
        let (r, box_l) = (8, 1.0);
        let center = (0.17, -0.23);
        let alpha = (8.0, 20.0);
        let weight = 1.3;
        let mpo = axis_aligned_gaussian_mpo(r, box_l, center, alpha, weight, 12, 1e-12).unwrap();
        let step = 2.0 * box_l / (1usize << r) as f64;
        for (ix, iy) in [(0, 0), (23, 197), (128, 128), (150, 98), (255, 255)] {
            let x = -box_l + ix as f64 * step;
            let y = -box_l + iy as f64 * step;
            let indices = index_to_bits(ix, r)
                .into_iter()
                .zip(index_to_bits(iy, r))
                .flat_map(|(x_bit, y_bit)| [x_bit, y_bit])
                .collect::<Vec<_>>();
            let got = mpo.evaluate(&indices).unwrap();
            let expected = weight
                * (-alpha.0 * (x - center.0).powi(2) - alpha.1 * (y - center.1).powi(2)).exp();
            assert!(
                (got - expected).abs() < 2e-8,
                "({x}, {y}): {got} != {expected}"
            );
        }
    }

    #[test]
    fn multiscale_gaussian_mpo_stays_accurate_as_r_increases() {
        let box_l = 1.0;
        let center = (0.173, -0.227);
        let alpha = (32.0, 200.0);
        let weight = 1.3;
        let sigma_min = 1.0 / (2.0_f64 * alpha.1).sqrt();

        for r in [8, 10, 12] {
            let step = 2.0 * box_l / (1usize << r) as f64;
            assert!(
                step <= sigma_min / 4.0,
                "R={r} does not resolve the narrow width"
            );
            let mpo =
                axis_aligned_gaussian_mpo(r, box_l, center, alpha, weight, 24, 1e-12).unwrap();
            let grid_size = 1usize << r;
            let mut max_error = 0.0_f64;
            for sample in 0..128 {
                let ix = (73 * sample + 11) % grid_size;
                let iy = (151 * sample + 29) % grid_size;
                let x = -box_l + ix as f64 * step;
                let y = -box_l + iy as f64 * step;
                let indices = index_to_bits(ix as u64, r)
                    .into_iter()
                    .zip(index_to_bits(iy as u64, r))
                    .flat_map(|(x_bit, y_bit)| [x_bit, y_bit])
                    .collect::<Vec<_>>();
                let got = mpo.evaluate(&indices).unwrap();
                let expected = weight
                    * (-alpha.0 * (x - center.0).powi(2) - alpha.1 * (y - center.1).powi(2)).exp();
                max_error = max_error.max((got - expected).abs());
            }
            eprintln!("R={r} rank={} max_error={max_error:.3e}", mpo.rank());
            assert!(
                max_error < 5e-8,
                "R={r}, rank={}, max error={max_error:.3e}",
                mpo.rank()
            );
            assert!(
                mpo.rank() <= 400,
                "R={r} unexpectedly increased rank to {}",
                mpo.rank()
            );
        }
    }

    #[test]
    fn multiscale_rotated_gaussian_mpo_resolves_a_narrow_ridge() {
        let (r, box_l) = (10, 1.0);
        let center = (0.07, -0.09);
        let (sigma_minor, rho, angle) = (0.05_f64, 8.0_f64, std::f64::consts::FRAC_PI_4);
        let lambda_minor = 1.0 / (2.0 * sigma_minor.powi(2));
        let lambda_major = lambda_minor / rho.powi(2);
        let (sin, cos) = angle.sin_cos();
        let quad = (
            lambda_major * cos * cos + lambda_minor * sin * sin,
            (lambda_major - lambda_minor) * sin * cos,
            lambda_major * sin * sin + lambda_minor * cos * cos,
        );
        let mpo = rotated_gaussian_mpo(r, box_l, center, quad, 1.3, 28, 1e-12).unwrap();
        let step = 2.0 * box_l / (1usize << r) as f64;
        let mut max_error = 0.0_f64;
        {
            let mut check_point = |ix: usize, iy: usize| {
                let x = -box_l + ix as f64 * step;
                let y = -box_l + iy as f64 * step;
                let indices = index_to_bits(ix as u64, r)
                    .into_iter()
                    .zip(index_to_bits(iy as u64, r))
                    .flat_map(|(x_bit, y_bit)| [x_bit, y_bit])
                    .collect::<Vec<_>>();
                let got = mpo.evaluate(&indices).unwrap();
                let (dx, dy) = (x - center.0, y - center.1);
                let expected =
                    1.3 * (-(quad.0 * dx * dx + 2.0 * quad.1 * dx * dy + quad.2 * dy * dy)).exp();
                max_error = max_error.max((got - expected).abs());
            };
            for sample in 0..256 {
                check_point(
                    (73 * sample + 11) % (1usize << r),
                    (151 * sample + 29) % (1usize << r),
                );
            }
            for major_step in -6..=6 {
                for minor_step in -4..=4 {
                    let u = major_step as f64 * 0.5 * sigma_minor * rho;
                    let v = minor_step as f64 * 0.5 * sigma_minor;
                    let x = center.0 + cos * u - sin * v;
                    let y = center.1 + sin * u + cos * v;
                    let ix = ((x + box_l) / step).round() as usize;
                    let iy = ((y + box_l) / step).round() as usize;
                    check_point(ix.min((1usize << r) - 1), iy.min((1usize << r) - 1));
                }
            }
        }
        eprintln!("rotated rank={} max_error={max_error:.3e}", mpo.rank());
        assert!(
            max_error < 5e-8,
            "rank={}, max error={max_error:.3e}",
            mpo.rank()
        );
    }

    #[test]
    fn multiscale_anisotropic_mixture_matches_grid_values() {
        let (r, box_l) = (8, 1.0);
        let mixture = AnisoMixture2D::random(2, 0.7, 0.08, 3.0, 17);
        let mpo = mixture
            .to_multiscale_mpo(r, box_l, 24, 1e-12, 1e-10)
            .unwrap();
        let step = 2.0 * box_l / (1usize << r) as f64;
        let mut max_error = 0.0_f64;
        for sample in 0..128 {
            let ix = (73 * sample + 11) % (1usize << r);
            let iy = (151 * sample + 29) % (1usize << r);
            let x = -box_l + ix as f64 * step;
            let y = -box_l + iy as f64 * step;
            let indices = index_to_bits(ix as u64, r)
                .into_iter()
                .zip(index_to_bits(iy as u64, r))
                .flat_map(|(x_bit, y_bit)| [x_bit, y_bit])
                .collect::<Vec<_>>();
            max_error = max_error.max((mpo.evaluate(&indices).unwrap() - mixture.eval(x, y)).abs());
        }
        assert!(
            max_error < 1e-7,
            "rank={}, max error={max_error:.3e}",
            mpo.rank()
        );
    }

    #[test]
    fn localized_anisotropic_evaluator_respects_absolute_tail_bound() {
        let mixture = AnisoMixture2D::random(512, 4.0, 0.05, 8.0, 23);
        let localized = LocalizedAnisoField::new(mixture.clone(), 1e-10).unwrap();
        assert_eq!(localized.absolute_tolerance(), 1e-10);
        for sample in 0..256 {
            let x = -5.0 + 10.0 * ((73 * sample + 11) % 257) as f64 / 256.0;
            let y = -5.0 + 10.0 * ((151 * sample + 29) % 257) as f64 / 256.0;
            let error = (localized.eval(x, y) - mixture.eval(x, y)).abs();
            assert!(error <= 1e-10, "({x}, {y}): error={error:.3e}");
        }
    }

    #[test]
    fn explicit_tci_pivots_make_input_construction_deterministic() {
        let mixture = AnisoMixture2D::random(4, 1.0, 0.1, 3.0, 29);
        let pivots = vec![vec![16, 16], vec![8, 24], vec![24, 8]];
        let (first, _) =
            to_quantics_mpo_field_with_pivots(&mixture, 5, 1.0, 1e-8, 64, pivots.clone()).unwrap();
        let (second, _) =
            to_quantics_mpo_field_with_pivots(&mixture, 5, 1.0, 1e-8, 64, pivots).unwrap();
        assert_eq!(first.rank(), second.rank());
        for (left, right) in first.site_tensors().iter().zip(second.site_tensors()) {
            assert_eq!(left.to_col_major_vec(), right.to_col_major_vec());
        }
    }

    #[test]
    fn localized_anisotropic_evaluator_rejects_negative_weights() {
        let mut mixture = AnisoMixture2D::random(1, 1.0, 0.05, 8.0, 23);
        mixture.weights[0] = -1.0;
        assert!(LocalizedAnisoField::new(mixture, 1e-10).is_err());
    }

    #[test]
    fn analytic_contraction_matches_quadrature() {
        let f = GaussianMixture2D::random(3, 4.0, (0.5, 2.0), 1);
        let g = GaussianMixture2D::random(2, 4.0, (0.5, 2.0), 2);
        let (x, z) = (0.3, -0.7);
        let numeric = quadrature(|y| f.eval(x, y) * g.eval(y, z), -8.0, 8.0, 20_000);
        let analytic = analytic_contraction(&f, &g, x, z);
        assert!((numeric - analytic).abs() < 1e-8 * analytic.abs().max(1.0));
    }

    #[test]
    fn anisotropic_analytic_contraction_matches_quadrature() {
        let f = AnisoMixture2D::random(3, 4.0, 0.3, 4.0, 1);
        let g = AnisoMixture2D::random(2, 4.0, 0.3, 4.0, 2);
        let (x, z) = (0.3, -0.7);
        let numeric = quadrature(|y| f.eval(x, y) * g.eval(y, z), -12.0, 12.0, 40_000);
        let analytic = analytic_contraction_aniso(&f, &g, x, z);
        assert!((numeric - analytic).abs() < 1e-8 * analytic.abs().max(1.0));
    }

    #[test]
    fn bounded_anisotropic_contraction_matches_quadrature() {
        let f = AnisoMixture2D::random(3, 1.0, 0.3, 4.0, 1);
        let g = AnisoMixture2D::random(2, 1.0, 0.3, 4.0, 2);
        let (x, z, box_l) = (0.3, -0.7, 1.0);
        let numeric = quadrature(|y| f.eval(x, y) * g.eval(y, z), -box_l, box_l, 40_000);
        let analytic = analytic_contraction_aniso_box(&f, &g, x, z, box_l);
        assert!((numeric - analytic).abs() < 1e-8 * analytic.abs().max(1.0));
    }

    #[test]
    fn anisotropic_grid_reference_matches_the_explicit_sum() {
        let f = AnisoMixture2D::random(3, 1.0, 0.3, 4.0, 1);
        let g = AnisoMixture2D::random(2, 1.0, 0.3, 4.0, 2);
        let (x, z, r, box_l) = (0.3, -0.7, 10, 1.0);
        let step = 2.0 * box_l / (1u64 << r) as f64;
        let direct = (0..1u64 << r)
            .map(|i| {
                let y = -box_l + i as f64 * step;
                f.eval(x, y) * g.eval(y, z) * step
            })
            .sum::<f64>();
        let reference = discrete_contraction_aniso_reference(&f, &g, x, z, r, box_l);
        assert!((direct - reference).abs() < 1e-10 * direct.abs().max(1.0));
    }

    /// The rotated quadratic form, against a value computed by hand.
    ///
    /// One spike at the origin with `a_minor = 2`, `a_major = 0.5` (so `rho = 2`)
    /// rotated by 45 degrees: `a = c = 1.25` and `b = -0.75`. The point `(1, 1)`
    /// lies on the major axis at `u = sqrt(2)`, so the exponent is
    /// `a_major u^2 = 1`, and the point `(1, -1)` lies on the minor axis at
    /// `v = sqrt(2)`, where the exponent is `a_minor v^2 = 4`.
    #[test]
    fn aniso_eval_matches_a_hand_computed_rotated_form() {
        let mix = AnisoMixture2D {
            weights: vec![2.0],
            quad: vec![(1.25, -0.75, 1.25)],
            centers: vec![(0.0, 0.0)],
        };
        let major = mix.eval(1.0, 1.0);
        let minor = mix.eval(1.0, -1.0);
        assert!(
            (major - 2.0 * (-1.0f64).exp()).abs() < 1e-14,
            "along the major axis: {major}"
        );
        assert!(
            (minor - 2.0 * (-4.0f64).exp()).abs() < 1e-14,
            "along the minor axis: {minor}"
        );
        // The center is the maximum and carries the weight.
        assert!((mix.eval(0.0, 0.0) - 2.0).abs() < 1e-14);
    }

    /// `random` has to produce exactly that family: every drawn form is a rotation
    /// of `diag(a_minor / rho^2, a_minor)` with `a_minor = 1 / (2 sigma^2)` and
    /// `rho` in `[1, rho_max]`. Rotation leaves the trace and the determinant
    /// alone, so the two eigenvalues are recoverable from the stored `(a, b, c)`
    /// and can be checked against sigma and the aspect range without knowing the
    /// angle. The draw is also required to be deterministic in the seed.
    #[test]
    fn aniso_random_draws_the_declared_family() {
        let (sigma, rho_max) = (0.05, 8.0);
        let mix = AnisoMixture2D::random(64, 1.0, sigma, rho_max, 7);
        assert_eq!(
            AnisoMixture2D::random(64, 1.0, sigma, rho_max, 7).quad,
            mix.quad
        );
        let a_minor = 1.0 / (2.0 * sigma * sigma);
        for (i, &(a, b, c)) in mix.quad.iter().enumerate() {
            // Eigenvalues of [[a, b], [b, c]].
            let mean = 0.5 * (a + c);
            let gap = (0.25 * (a - c).powi(2) + b * b).sqrt();
            let (lo, hi) = (mean - gap, mean + gap);
            assert!(
                (hi - a_minor).abs() < 1e-9 * a_minor,
                "spike {i}: minor {hi}"
            );
            let rho = (a_minor / lo).sqrt();
            assert!(
                (1.0..=rho_max).contains(&rho),
                "spike {i}: aspect {rho} outside [1, {rho_max}]"
            );
            let (cx, cy) = mix.centers[i];
            assert!(cx.abs() <= 0.9 && cy.abs() <= 0.9);
            assert!((0.5..1.5).contains(&mix.weights[i]));
        }
    }

    #[test]
    fn quantics_mpo_evaluates_to_function_values() {
        let r = 8;
        let l = 4.0;
        let mix = GaussianMixture2D::random(3, l, (0.5, 2.0), 3);
        let (mpo, _dy) = to_quantics_mpo(&mix, r, l, 1e-10, 200).unwrap();
        assert_eq!(mpo.len(), r);
        for &(i, j) in &[(0u64, 0u64), (37, 200), (255, 1), (128, 128)] {
            let x = grid_coord(i, r, l);
            let y = grid_coord(j, r, l);
            let xb = crate::harness::index_to_bits(i, r);
            let yb = crate::harness::index_to_bits(j, r);
            let mut idx = Vec::with_capacity(2 * r);
            for n in 0..r {
                idx.push(xb[n]);
                idx.push(yb[n]);
            }
            let v = mpo.evaluate(&idx).unwrap();
            assert!(
                (v - mix.eval(x, y)).abs() < 1e-6,
                "at ({x},{y}): {v} vs {}",
                mix.eval(x, y)
            );
        }
    }
}
