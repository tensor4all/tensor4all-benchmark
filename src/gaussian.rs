//! 2D Gaussian mixtures: analytic y-integral and quantics MPO construction.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor4all_quanticstci::{
    quanticscrossinterpolate, DiscretizedGrid, QtciOptions, UnfoldingScheme,
};
use tensor4all_simplett::mpo::{tensor4_from_data, MPO};
use tensor4all_simplett::{AbstractTensorTrain, Tensor3Ops, TensorTrain};

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
) -> anyhow::Result<(TensorTrain<f64>, f64)> {
    let grid = DiscretizedGrid::builder(&[r, r])
        .with_lower_bound(&[-box_l, -box_l])
        .with_upper_bound(&[box_l, box_l])
        .with_unfolding_scheme(UnfoldingScheme::Fused)
        .build()?;
    let m = mix.clone();
    let opts = QtciOptions::default()
        .with_tolerance(tol)
        .with_maxbonddim(max_bond)
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
