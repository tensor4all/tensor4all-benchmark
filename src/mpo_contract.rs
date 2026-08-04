//! MPO-MPO contraction wrappers (case 2) and the analytic accuracy check.
//!
//! Contraction sums over the index shared by `a`'s second site dimension and
//! `b`'s first, so with `f(x,y)` and `g(y,z)` as quantics MPOs, `contract(f, g)`
//! sums over the `y` grid and yields `h(x,z)`.

use tensor4all_simplett::mpo::{
    contract_fit, contract_naive, contract_zipup, ContractionOptions, FitOptions, MPO,
};

use crate::gaussian::{analytic_contraction, grid_coord, GaussianMixture2D};
use crate::harness::{index_to_bits, sample_grid_indices};

/// The MPO-MPO contraction algorithms benchmarked in case 2.
#[derive(Clone, Copy, Debug)]
pub enum MpoAlgo {
    /// Full contraction of the bond dimensions, then compression.
    Naive,
    /// Sweep with truncation of the growing bond on the fly.
    Zipup,
    /// Variational (DMRG-like) fit at a fixed maximum bond dimension.
    Fit,
}

/// Contract two MPOs over their shared site index with the given algorithm.
pub fn mpo_contract(
    algo: MpoAlgo,
    a: &MPO<f64>,
    b: &MPO<f64>,
    tol: f64,
    max_bond: usize,
) -> anyhow::Result<MPO<f64>> {
    let opts = ContractionOptions {
        tolerance: tol,
        max_bond_dim: max_bond,
        ..ContractionOptions::default()
    };
    let out = match algo {
        MpoAlgo::Naive => contract_naive(a, b, Some(opts))?,
        MpoAlgo::Zipup => contract_zipup(a, b, &opts)?,
        MpoAlgo::Fit => {
            let fopts = FitOptions {
                tolerance: tol,
                max_bond_dim: max_bond,
                // At the pinned rev the variational update is a stub upstream
                // (`update_two_site_core` is a placeholder), so every sweep is
                // dead work: it builds environments and changes nothing. Pinned
                // to 1 instead of the default 10 to minimize wasted time. This
                // is part of the benchmark definition, not an incidental value.
                max_sweeps: 1,
                ..FitOptions::default()
            };
            contract_fit(a, b, &fopts, None)?
        }
    };
    Ok(out)
}

/// Relative max error of `h` vs the analytic y-integral, normalized by the max
/// sampled `|analytic|` value. The MPO holds the plain sum over the y grid, so
/// multiply by `dy` to approximate the integral.
#[allow(clippy::too_many_arguments)]
pub fn max_rel_error_vs_analytic(
    h: &MPO<f64>,
    dy: f64,
    f: &GaussianMixture2D,
    g: &GaussianMixture2D,
    r: usize,
    box_l: f64,
    n_samples: usize,
    seed: u64,
) -> f64 {
    let xs = sample_grid_indices(r, n_samples, seed);
    let zs = sample_grid_indices(r, n_samples, seed.wrapping_add(1));
    let mut max_abs = 0.0f64;
    let mut max_ref = 0.0f64;
    for (&ix, &iz) in xs.iter().zip(&zs) {
        let x = grid_coord(ix, r, box_l);
        let z = grid_coord(iz, r, box_l);
        let xb = index_to_bits(ix, r);
        let zb = index_to_bits(iz, r);
        let mut idx = Vec::with_capacity(2 * r);
        for n in 0..r {
            idx.push(xb[n]);
            idx.push(zb[n]);
        }
        let got = h.evaluate(&idx).unwrap() * dy;
        let want = analytic_contraction(f, g, x, z);
        max_abs = max_abs.max((got - want).abs());
        max_ref = max_ref.max(want.abs());
    }
    max_abs / max_ref.max(f64::MIN_POSITIVE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaussian::{to_quantics_mpo, GaussianMixture2D};

    #[test]
    fn naive_and_zipup_agree_with_analytic_integral() {
        let (r, l) = (8, 6.0);
        let f = GaussianMixture2D::random(3, l, (0.5, 2.0), 20);
        let g = GaussianMixture2D::random(3, l, (0.5, 2.0), 21);
        let (fa, dy) = to_quantics_mpo(&f, r, l, 1e-6, 128).unwrap();
        let (gb, _) = to_quantics_mpo(&g, r, l, 1e-6, 128).unwrap();
        for algo in [MpoAlgo::Naive, MpoAlgo::Zipup] {
            let h = mpo_contract(algo, &fa, &gb, 1e-6, 128).unwrap();
            let err = max_rel_error_vs_analytic(&h, dy, &f, &g, r, l, 50, 22);
            // R=8 discretization floor dominates; bound is loose on purpose
            // (measured 4.2e-7 for both algorithms).
            assert!(err < 1e-2, "{algo:?}: rel err {err}");
        }
    }

    /// `contract_fit` is checked at a smaller R than the other two algorithms.
    /// Its environment builders (upstream `mpo/contract_fit.rs`,
    /// `build_right_environment`) are scalar loops over all six bond indices,
    /// so a single environment costs O((chi_c chi_a chi_b)^2 d^3): 29s at R=5
    /// here, and already hours at R=6. R=5 still pins the contracted index
    /// pair and the dy normalization, which is what this test is for.
    #[test]
    fn fit_agrees_with_analytic_integral() {
        let (r, l) = (5, 6.0);
        let f = GaussianMixture2D::random(3, l, (0.5, 2.0), 20);
        let g = GaussianMixture2D::random(3, l, (0.5, 2.0), 21);
        let (fa, dy) = to_quantics_mpo(&f, r, l, 1e-6, 64).unwrap();
        let (gb, _) = to_quantics_mpo(&g, r, l, 1e-6, 64).unwrap();
        let h = mpo_contract(MpoAlgo::Fit, &fa, &gb, 1e-6, 64).unwrap();
        let err = max_rel_error_vs_analytic(&h, dy, &f, &g, r, l, 50, 22);
        // R=5 grid error dominates (measured 2.1e-6, same as naive/zipup).
        assert!(err < 1e-2, "fit: rel err {err}");
    }
}
