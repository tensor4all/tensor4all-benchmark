//! MPO-MPO contraction wrappers (case 2) and the analytic accuracy check.
//!
//! Contraction sums over the index shared by `a`'s second site dimension and
//! `b`'s first, so with `f(x,y)` and `g(y,z)` as quantics MPOs, `contract(f, g)`
//! sums over the `y` grid and yields `h(x,z)`.

use tensor4all_core::{DynIndex, IndexLike, SvdTruncationPolicy, TensorDynLen};
use tensor4all_itensorlike::{ContractOptions, TensorTrain};
use tensor4all_simplett::mpo::{
    contract_naive, contract_zipup, tensor4_from_data, ContractionOptions, Tensor4Ops, MPO,
};

use crate::gaussian::{analytic_contraction, grid_coord, GaussianMixture2D};
use crate::harness::{index_to_bits, sample_grid_indices};

/// Number of full variational sweeps used by the `fit` arm.
///
/// This is part of the benchmark definition, not an incidental value: the fit
/// cost is linear in the sweep count, so any timing comparison against naive
/// and zipup is only meaningful at a fixed, stated number of sweeps. One full
/// sweep (two half-sweeps) on top of the zip-up initializer is the smallest
/// setting that actually exercises the variational update. Upstream requires
/// `nhalfsweeps` to be even for the Fit method, which `with_nsweeps` ensures.
pub const FIT_NSWEEPS: usize = 1;

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
        MpoAlgo::Fit => contract_fit_treetn(a, b, tol, max_bond)?,
    };
    Ok(out)
}

/// Variational (DMRG-like) fit, run on the treetn engine via `itensorlike`.
///
/// `tensor4all_simplett::mpo::contract_fit` is not used: at the pinned upstream
/// rev its local update (`update_two_site_core`) is a placeholder that returns
/// without touching the core, so it degenerates to naive plus dead sweeps. The
/// complete implementation lives in `tensor4all_treetn::treetn::fit`, reached
/// here through `tensor4all_itensorlike::TensorTrain::contract` with
/// `ContractOptions::fit()`. That is the same engine case 1 uses for its
/// elementwise fit.
fn contract_fit_treetn(
    a: &MPO<f64>,
    b: &MPO<f64>,
    tol: f64,
    max_bond: usize,
) -> anyhow::Result<MPO<f64>> {
    let n = a.len();
    anyhow::ensure!(n == b.len(), "fit: MPO lengths differ ({n} vs {})", b.len());

    // `a` is f(x, y) and `b` is g(y, z). Sharing the y index objects between the
    // two trains is what makes `contract` sum over y, exactly as the simplett
    // MPO-MPO convention does.
    let x_inds: Vec<DynIndex> = (0..n).map(|i| DynIndex::new_dyn(a.site_dim(i).0)).collect();
    let y_inds: Vec<DynIndex> = (0..n).map(|i| DynIndex::new_dyn(a.site_dim(i).1)).collect();
    let z_inds: Vec<DynIndex> = (0..n).map(|i| DynIndex::new_dyn(b.site_dim(i).1)).collect();
    for i in 0..n {
        anyhow::ensure!(
            a.site_dim(i).1 == b.site_dim(i).1,
            "fit: contracted site dims differ at site {i}"
        );
    }

    let ta = mpo_to_tensortrain(a, &x_inds, &y_inds)?;
    let tb = mpo_to_tensortrain(b, &y_inds, &z_inds)?;

    let opts = ContractOptions::fit()
        .with_max_rank(max_bond)
        .with_svd_policy(SvdTruncationPolicy::new(tol))
        .with_nsweeps(FIT_NSWEEPS);
    let tc = ta
        .contract(&tb, &opts)
        .map_err(|e| anyhow::anyhow!("fit contraction failed: {e}"))?;

    tensortrain_to_mpo(&tc, &x_inds, &z_inds)
}

/// Bridge an `MPO<f64>` into an `itensorlike` `TensorTrain`.
///
/// MPO cores are `Tensor4` with shape `(left, site1, site2, right)` in
/// column-major order (upstream `mpo/types.rs`), and
/// `TensorDynLen::from_dense` reads column-major data with the *first* index
/// varying fastest, so the index list is given in that same order. Boundary
/// links are dimension 1 in every `MPO` (checked by `MPO::new`), so they are
/// dropped rather than carried as dummy indices: `TensorTrain::new` requires
/// adjacent sites to share exactly one index and identifies links purely by
/// shared index id, and a leading or trailing extent-1 axis does not change
/// the column-major layout.
fn mpo_to_tensortrain(
    m: &MPO<f64>,
    site1: &[DynIndex],
    site2: &[DynIndex],
) -> anyhow::Result<TensorTrain> {
    let n = m.len();
    let links: Vec<DynIndex> = (0..n.saturating_sub(1))
        .map(|i| DynIndex::new_dyn(m.link_dim(i)))
        .collect();

    let mut tensors = Vec::with_capacity(n);
    for i in 0..n {
        let core = m.site_tensor(i);
        anyhow::ensure!(
            i > 0 || core.left_dim() == 1,
            "fit: leading MPO bond is not 1"
        );
        anyhow::ensure!(
            i + 1 < n || core.right_dim() == 1,
            "fit: trailing MPO bond is not 1"
        );
        let mut indices = Vec::with_capacity(4);
        if i > 0 {
            indices.push(links[i - 1].clone());
        }
        indices.push(site1[i].clone());
        indices.push(site2[i].clone());
        if i + 1 < n {
            indices.push(links[i].clone());
        }
        tensors.push(TensorDynLen::from_dense(indices, core.to_col_major_vec())?);
    }

    TensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("fit: TensorTrain::new failed: {e}"))
}

/// Bridge an `itensorlike` `TensorTrain` back to an `MPO<f64>`.
///
/// The contraction preserves the external site index objects, so each result
/// core is permuted into `(left, site1, site2, right)` and read out as
/// column-major data, which is what `tensor4_from_data` expects.
fn tensortrain_to_mpo(
    tt: &TensorTrain,
    site1: &[DynIndex],
    site2: &[DynIndex],
) -> anyhow::Result<MPO<f64>> {
    let n = tt.len();
    let mut cores = Vec::with_capacity(n);
    for i in 0..n {
        let tensor = tt
            .tensor(i)
            .map_err(|e| anyhow::anyhow!("fit: missing result tensor at site {i}: {e}"))?;
        let left = if i > 0 { tt.linkind(i - 1) } else { None };
        let right = if i + 1 < n { tt.linkind(i) } else { None };
        let mut order = Vec::with_capacity(4);
        order.extend(left.clone());
        order.push(site1[i].clone());
        order.push(site2[i].clone());
        order.extend(right.clone());
        anyhow::ensure!(
            order.len() == tensor.indices().len(),
            "fit: result core at site {i} has {} indices, expected {}",
            tensor.indices().len(),
            order.len()
        );
        let data = tensor.permute_indices(&order)?.to_vec::<f64>()?;
        let ldim = left.map_or(1, |ix| ix.dim());
        let rdim = right.map_or(1, |ix| ix.dim());
        cores.push(tensor4_from_data(
            data,
            ldim,
            site1[i].dim(),
            site2[i].dim(),
            rdim,
        )?);
    }
    Ok(MPO::new(cores)?)
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

    /// Evaluate `h` at `n` pseudo-random `(x, z)` grid points.
    fn sample_values(h: &MPO<f64>, r: usize, n: usize, seed: u64) -> Vec<f64> {
        let xs = sample_grid_indices(r, n, seed);
        let zs = sample_grid_indices(r, n, seed.wrapping_add(1));
        xs.iter()
            .zip(&zs)
            .map(|(&ix, &iz)| {
                let xb = index_to_bits(ix, r);
                let zb = index_to_bits(iz, r);
                let mut idx = Vec::with_capacity(2 * r);
                for k in 0..r {
                    idx.push(xb[k]);
                    idx.push(zb[k]);
                }
                h.evaluate(&idx).unwrap()
            })
            .collect()
    }

    #[test]
    fn all_algorithms_agree_with_analytic_integral() {
        let (r, l) = (8, 6.0);
        let f = GaussianMixture2D::random(3, l, (0.5, 2.0), 20);
        let g = GaussianMixture2D::random(3, l, (0.5, 2.0), 21);
        let (fa, dy) = to_quantics_mpo(&f, r, l, 1e-6, 128).unwrap();
        let (gb, _) = to_quantics_mpo(&g, r, l, 1e-6, 128).unwrap();
        for algo in [MpoAlgo::Naive, MpoAlgo::Zipup, MpoAlgo::Fit] {
            let h = mpo_contract(algo, &fa, &gb, 1e-6, 128).unwrap();
            let err = max_rel_error_vs_analytic(&h, dy, &f, &g, r, l, 50, 22);
            // R=8 discretization floor dominates; bound is loose on purpose
            // (measured 4.2e-7 for all three algorithms).
            assert!(err < 1e-2, "{algo:?}: rel err {err}");
        }
    }

    /// Guard that `MpoAlgo::Fit` reaches the variational engine rather than
    /// silently returning the zip-up result. Untruncated, fit and zipup both
    /// converge to the same tensor, so the two are separated under a bond
    /// dimension small enough to force real truncation error: there the
    /// variational sweep moves the solution away from the one-pass zip-up.
    #[test]
    fn fit_differs_from_zipup_under_forced_truncation() {
        let (r, l, max_bond) = (8, 6.0, 8);
        let f = GaussianMixture2D::random(3, l, (0.5, 2.0), 20);
        let g = GaussianMixture2D::random(3, l, (0.5, 2.0), 21);
        let (fa, _dy) = to_quantics_mpo(&f, r, l, 1e-6, 128).unwrap();
        let (gb, _) = to_quantics_mpo(&g, r, l, 1e-6, 128).unwrap();
        let h_zip = mpo_contract(MpoAlgo::Zipup, &fa, &gb, 1e-6, max_bond).unwrap();
        let h_fit = mpo_contract(MpoAlgo::Fit, &fa, &gb, 1e-6, max_bond).unwrap();
        let vz = sample_values(&h_zip, r, 20, 33);
        let vf = sample_values(&h_fit, r, 20, 33);
        let max_diff = vz
            .iter()
            .zip(&vf)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_diff > 1e-14,
            "fit output is bit-identical to zipup at max_bond={max_bond}: \
             the Fit arm is not reaching the variational engine (max diff {max_diff:e})"
        );
    }
}
