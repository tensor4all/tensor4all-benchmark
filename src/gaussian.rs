//! 2D Gaussian mixtures: analytic y-integral and quantics MPO construction.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor4all_quanticstci::{
    quanticscrossinterpolate, DiscretizedGrid, QtciOptions, UnfoldingScheme,
};
use tensor4all_simplett::mpo::{tensor4_from_data, MPO};
use tensor4all_simplett::{AbstractTensorTrain, SimpleTensorTrain, Tensor3Ops};

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
}

impl Field2D for AnisoMixture2D {
    fn eval(&self, x: f64, y: f64) -> f64 {
        AnisoMixture2D::eval(self, x, y)
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
    let grid = DiscretizedGrid::builder(&[r, r])
        .with_lower_bound(&[-box_l, -box_l])
        .with_upper_bound(&[box_l, box_l])
        .with_unfolding_scheme(UnfoldingScheme::Fused)
        .build()?;
    let m = mix.clone();
    let opts = QtciOptions::default()
        .with_tolerance(tol)
        .with_max_bond_dim(max_bond)
        .with_unfoldingscheme(UnfoldingScheme::Fused);
    let (qtci, _ranks, _errs) =
        quanticscrossinterpolate(&grid, move |xy: &[f64]| m.eval(xy[0], xy[1]), None, opts)?;
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
    let (tt, step) = to_quantics_fused_tt(mix, r, box_l, tol, max_bond)?;
    let mut cores4 = Vec::with_capacity(tt.len());
    for c in tt.site_tensors() {
        let (l, s, rd) = (c.left_dim(), c.site_dim(), c.right_dim());
        anyhow::ensure!(s == 4, "expected fused site dim 4, got {s}");
        // Fused local index is `s = s1 + 2*s2` with s1 the bit of variable 1 (x)
        // as the least significant digit. Verified both against the quanticsgrids
        // source (add_fused_indices pushes variables in reverse order, so the last
        // variable lands at pos_in_site 0 and gets place value 2^(len-1) = 2) and
        // empirically by `quantics_mpo_evaluates_to_function_values`.
        // Task 8's Julia mirror must use the same order.
        let mut data = vec![0.0f64; l * 4 * rd];
        for rr in 0..rd {
            for s2 in 0..2 {
                for s1 in 0..2 {
                    for ll in 0..l {
                        let fused = s1 + 2 * s2;
                        data[ll + l * (s1 + 2 * (s2 + 2 * rr))] = *c.get3(ll, fused, rr);
                    }
                }
            }
        }
        cores4.push(tensor4_from_data(data, l, 2, 2, rd)?);
    }
    Ok((MPO::new(cores4)?, step))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytic_contraction_matches_quadrature() {
        let f = GaussianMixture2D::random(3, 4.0, (0.5, 2.0), 1);
        let g = GaussianMixture2D::random(2, 4.0, (0.5, 2.0), 2);
        let (x, z) = (0.3, -0.7);
        // trapezoid quadrature over y in [-8, 8]
        let n = 20_000;
        let (lo, hi) = (-8.0, 8.0);
        let h = (hi - lo) / n as f64;
        let mut s = 0.0;
        for i in 0..=n {
            let y = lo + i as f64 * h;
            let w = if i == 0 || i == n { 0.5 } else { 1.0 };
            s += w * f.eval(x, y) * g.eval(y, z) * h;
        }
        let a = analytic_contraction(&f, &g, x, z);
        assert!(
            (s - a).abs() < 1e-8 * a.abs().max(1.0),
            "quad {s} vs analytic {a}"
        );
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
