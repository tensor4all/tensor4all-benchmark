//! Four ways to form the elementwise (Hadamard) product of two quantics tensor
//! trains, behind one entry point, plus the sampled max-error metrics of the
//! two cases that use them.
//!
//! The product itself is generic over [`BenchScalar`], because case 1 works on
//! a complex Fourier series and case 3 on a real 2D Gaussian mixture, and both
//! run the same four arms. The error metrics are not generic: each one compares
//! against the analytic reference of its own case, which fixes the scalar type
//! (`Complex64` for case 1, `f64` for case 3).

use num_complex::Complex64;
use tensor4all_simplett::{tensor3_from_data, AbstractTensorTrain, Tensor3Ops, TensorTrain};

use crate::fourier::{compress_svd, FourierSeries};
use crate::gaussian::{grid_coord, GaussianMixture2D};
use crate::harness::{index_to_bits, sample_grid_indices};
use crate::scalar::BenchScalar;

/// The fit arm uses a fixed two full sweeps: the sweep count is part of the
/// benchmark definition, not something we let adapt or inherit from upstream
/// defaults (which is 1 at the pinned rev).
pub const FIT_NFULLSWEEPS: usize = 2;

#[derive(Clone, Copy, Debug)]
pub enum ElementwiseAlgo {
    Naive,
    Zipup,
    Fit,
    Aci,
}

impl ElementwiseAlgo {
    /// Which engine actually runs this arm, recorded in every case-3 record.
    ///
    /// `Naive` is the local bond-Kronecker product plus an SVD sweep written in
    /// this crate on top of `simplett` primitives, so it is labelled `local`
    /// rather than attributed to an upstream contraction engine.
    pub fn engine(self) -> &'static str {
        match self {
            ElementwiseAlgo::Naive => "local",
            ElementwiseAlgo::Zipup | ElementwiseAlgo::Fit => "treetn",
            ElementwiseAlgo::Aci => "aci",
        }
    }
}

pub fn elementwise_product<T: BenchScalar>(
    algo: ElementwiseAlgo,
    a: &TensorTrain<T>,
    b: &TensorTrain<T>,
    tol: f64,
    max_bond: usize,
) -> anyhow::Result<TensorTrain<T>> {
    match algo {
        ElementwiseAlgo::Naive => hadamard_naive(a, b, tol, max_bond),
        ElementwiseAlgo::Zipup => hadamard_treetn(a, b, tol, max_bond, false),
        ElementwiseAlgo::Fit => hadamard_treetn(a, b, tol, max_bond, true),
        ElementwiseAlgo::Aci => hadamard_aci(a, b, tol, max_bond),
    }
}

/// Core-wise Hadamard (bond Kronecker product) followed by SVD compression.
/// This is the O(chi^4) baseline.
fn hadamard_naive<T: BenchScalar>(
    a: &TensorTrain<T>,
    b: &TensorTrain<T>,
    tol: f64,
    max_bond: usize,
) -> anyhow::Result<TensorTrain<T>> {
    anyhow::ensure!(a.len() == b.len(), "site count mismatch");
    let mut cores = Vec::with_capacity(a.len());
    for (ca, cb) in a.site_tensors().iter().zip(b.site_tensors()) {
        let (la, s, ra) = (ca.left_dim(), ca.site_dim(), ca.right_dim());
        let (lb, rb) = (cb.left_dim(), cb.right_dim());
        anyhow::ensure!(s == cb.site_dim(), "site dimension mismatch");
        let mut data = vec![T::default(); la * lb * s * ra * rb];
        for r2 in 0..rb {
            for r1 in 0..ra {
                for si in 0..s {
                    for l2 in 0..lb {
                        for l1 in 0..la {
                            let idx = (l1 + la * l2) + la * lb * (si + s * (r1 + ra * r2));
                            data[idx] = *ca.get3(l1, si, r1) * *cb.get3(l2, si, r2);
                        }
                    }
                }
            }
        }
        cores.push(tensor3_from_data(data, la * lb, s, ra * rb)?);
    }
    let mut tt = TensorTrain::new(cores)?;
    compress_svd(&mut tt, tol, max_bond)?;
    Ok(tt)
}

/// `tensor4all_treetn::hadamard` on the bridged TreeTNs, with either the
/// one-pass zipup or the variational fit contraction.
fn hadamard_treetn<T: BenchScalar>(
    a: &TensorTrain<T>,
    b: &TensorTrain<T>,
    tol: f64,
    max_bond: usize,
    fit: bool,
) -> anyhow::Result<TensorTrain<T>> {
    use tensor4all_core::SvdTruncationPolicy;
    use tensor4all_treetn::contraction::{ContractionMethod, ContractionOptions};
    use tensor4all_treetn::{hadamard, tensor_train_to_treetn, treetn_to_tensor_train};

    let (ta, ia) = tensor_train_to_treetn(a)?;
    let (tb, ib) = tensor_train_to_treetn(b)?;
    let pairs: Vec<_> = ia.into_iter().zip(ib).collect();
    let method = if fit {
        ContractionMethod::Fit
    } else {
        ContractionMethod::Zipup
    };
    let mut opts = ContractionOptions::new(method)
        .with_max_rank(max_bond)
        .with_svd_policy(SvdTruncationPolicy::new(tol));
    if fit {
        opts = opts.with_nfullsweeps(FIT_NFULLSWEEPS);
    }
    let out = hadamard(&ta, &tb, &pairs, &0, opts)
        .map_err(|e| anyhow::anyhow!("hadamard failed: {e:?}"))?;
    treetn_to_tensor_train::<T>(out)
}

/// Adaptive cross interpolation of the pointwise product function.
fn hadamard_aci<T: BenchScalar>(
    a: &TensorTrain<T>,
    b: &TensorTrain<T>,
    tol: f64,
    max_bond: usize,
) -> anyhow::Result<TensorTrain<T>> {
    use tensor4all_aci::{elementwise, AciOptions};
    let opts = AciOptions::<T> {
        tolerance: tol,
        max_bond_dim: max_bond,
        ..AciOptions::default()
    };
    let res = elementwise(|xs: &[T]| xs[0] * xs[1], &[a.clone(), b.clone()], &opts)?;
    Ok(res.tensor_train)
}

/// Max abs error against the exact product series at sampled grid points.
pub fn max_error_vs_series(
    tt: &TensorTrain<Complex64>,
    exact: &FourierSeries,
    r: usize,
    n_samples: usize,
    seed: u64,
) -> f64 {
    sample_grid_indices(r, n_samples, seed)
        .iter()
        .map(|&i| {
            let x = i as f64 / (1u64 << r) as f64;
            let v = tt.evaluate(&index_to_bits(i, r)).unwrap();
            (v - exact.eval(x)).norm()
        })
        .fold(0.0, f64::max)
}

/// Case 3: relative max error of the fused 2D product train against the exact
/// pointwise product of the two Gaussian mixtures.
///
/// `h` is a fused quantics train on `[-L, L)^2` with `r` sites of dimension 4,
/// local index `x_bit + 2 * y_bit` and the most significant bit first, which is
/// the layout `gaussian::to_quantics_fused_tt` produces. The normalization
/// matches case 2: the largest sampled `|reference|`, so the two cases report
/// the same kind of number under `error_metric = "max_rel_vs_analytic"`.
#[allow(clippy::too_many_arguments)]
pub fn max_rel_error_vs_mixture_product(
    h: &TensorTrain<f64>,
    f: &GaussianMixture2D,
    g: &GaussianMixture2D,
    r: usize,
    box_l: f64,
    n_samples: usize,
    seed: u64,
) -> f64 {
    let xs = sample_grid_indices(r, n_samples, seed);
    let ys = sample_grid_indices(r, n_samples, seed.wrapping_add(1));
    let mut max_abs = 0.0f64;
    let mut max_ref = 0.0f64;
    for (&ix, &iy) in xs.iter().zip(&ys) {
        let x = grid_coord(ix, r, box_l);
        let y = grid_coord(iy, r, box_l);
        let xb = index_to_bits(ix, r);
        let yb = index_to_bits(iy, r);
        let fused: Vec<usize> = (0..r).map(|n| xb[n] + 2 * yb[n]).collect();
        let got = h.evaluate(&fused).unwrap();
        let want = f.eval(x, y) * g.eval(x, y);
        max_abs = max_abs.max((got - want).abs());
        max_ref = max_ref.max(want.abs());
    }
    max_abs / max_ref.max(f64::MIN_POSITIVE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fourier::{compress_svd, FourierSeries};

    fn setup(
        r: usize,
        k: usize,
    ) -> (
        TensorTrain<Complex64>,
        TensorTrain<Complex64>,
        FourierSeries,
    ) {
        let f = FourierSeries::random(k, 10);
        let g = FourierSeries::random(k, 11);
        let mut a = f.to_qtt(r).unwrap();
        let mut b = g.to_qtt(r).unwrap();
        compress_svd(&mut a, 1e-12, usize::MAX).unwrap();
        compress_svd(&mut b, 1e-12, usize::MAX).unwrap();
        (a, b, f.product(&g))
    }

    #[test]
    fn all_algorithms_agree_with_exact_product() {
        let r = 10;
        let (a, b, exact) = setup(r, 6);
        for (algo, bound) in [
            (ElementwiseAlgo::Naive, 1e-8),
            (ElementwiseAlgo::Zipup, 1e-8),
            (ElementwiseAlgo::Fit, 1e-3),
            (ElementwiseAlgo::Aci, 1e-6),
        ] {
            let out = elementwise_product(algo, &a, &b, 1e-10, 200).unwrap();
            let err = max_error_vs_series(&out, &exact, r, 100, 5);
            println!("{algo:?}: max abs error {err:.3e} (bound {bound:.0e})");
            assert!(err < bound, "{algo:?}: err {err} exceeds {bound}");
        }
    }

    /// Case 3 at its own fixed output budget: every arm capped at `chi_in`, the
    /// larger input rank, and judged only on the error it returns for it.
    ///
    /// The bounds are per arm because the arms are not comparable here. Measured
    /// at the pinned revision on this instance (r = 8, 3 Gaussians, chi_in 61):
    /// naive 1.4e-8 and fit 1.4e-8 at chi_out 49, aci 1.0e-7 at chi_out 43, all
    /// near the 1e-8 working tolerance, while zipup returns 4.1e-1 having spent
    /// the whole budget. Every bound carries about an order of magnitude of
    /// margin, since the quantics TCI construction is not bit-reproducible and
    /// chi_in moves by one between runs. The zipup bound is loose on purpose:
    /// at this budget a single-pass truncation of an elementwise product has no
    /// accuracy left to defend (the runner's default instance reaches 9.2e-1 at
    /// r = 10), so what the bound guards is that the arm still returns a finite
    /// result of roughly the right scale. This test also covers the real-scalar
    /// (`f64`) path through all four arms, which case 1 does not exercise.
    #[test]
    fn gauss2d_arms_meet_their_error_bounds_at_fixed_budget() {
        use crate::gaussian::{to_quantics_fused_tt, GaussianMixture2D};

        let (r, l) = (8, 6.0);
        let f = GaussianMixture2D::random(3, l, (0.5, 8.0), 1);
        let g = GaussianMixture2D::random(3, l, (0.5, 8.0), 2);
        let (fa, _) = to_quantics_fused_tt(&f, r, l, 1e-8, 512).unwrap();
        let (gb, _) = to_quantics_fused_tt(&g, r, l, 1e-8, 512).unwrap();
        let chi_in = fa.rank().max(gb.rank());

        for (algo, bound) in [
            (ElementwiseAlgo::Naive, 1e-6),
            (ElementwiseAlgo::Zipup, 2.0),
            (ElementwiseAlgo::Fit, 1e-6),
            (ElementwiseAlgo::Aci, 1e-6),
        ] {
            let out = elementwise_product(algo, &fa, &gb, 1e-8, chi_in).unwrap();
            assert!(
                out.rank() <= chi_in,
                "{algo:?}: chi_out {} exceeds the budget {chi_in}",
                out.rank()
            );
            let err = max_rel_error_vs_mixture_product(&out, &f, &g, r, l, 128, 99);
            println!(
                "{algo:?}: rel err {err:.3e} (bound {bound:.0e}), chi_out {} of {chi_in}",
                out.rank()
            );
            assert!(
                err.is_finite() && err < bound,
                "{algo:?}: rel err {err} exceeds {bound}"
            );
        }
    }

    /// Guards against a dispatch swap between the arms of `elementwise_product`.
    /// With a forced truncation (`max_bond = 4` on a k=6 instance) the four
    /// algorithms are no longer interchangeable: they land on genuinely different
    /// approximants, so arm identity becomes an observable property.
    #[test]
    fn algorithms_are_distinguishable_under_forced_truncation() {
        let r = 10;
        let max_bond = 4;
        let (a, b, _exact) = setup(r, 6);
        let idx = sample_grid_indices(r, 20, 7);

        let eval = |tt: &TensorTrain<Complex64>| -> Vec<Complex64> {
            idx.iter()
                .map(|&i| tt.evaluate(&index_to_bits(i, r)).unwrap())
                .collect()
        };
        let max_diff = |x: &[Complex64], y: &[Complex64]| -> f64 {
            x.iter()
                .zip(y)
                .map(|(p, q)| (p - q).norm())
                .fold(0.0, f64::max)
        };

        let mut vals = Vec::new();
        let mut dims = Vec::new();
        for algo in [
            ElementwiseAlgo::Naive,
            ElementwiseAlgo::Zipup,
            ElementwiseAlgo::Fit,
            ElementwiseAlgo::Aci,
        ] {
            let out = elementwise_product(algo, &a, &b, 1e-10, max_bond).unwrap();
            let ld = out.link_dims();
            println!("{algo:?}: link dims {ld:?}");
            // Every arm must honour the rank cap it was handed.
            assert!(
                ld.iter().all(|&d| d <= max_bond),
                "{algo:?}: link dims {ld:?} exceed max_bond {max_bond}"
            );
            vals.push(eval(&out));
            dims.push(ld);
        }
        let (naive, zipup, fit, aci) = (&vals[0], &vals[1], &vals[2], &vals[3]);

        // (a) Zipup (single-pass truncation) and Fit (two variational sweeps) must
        // not produce bit-identical outputs, otherwise the two arms are the same code.
        let d_zipup_fit = max_diff(zipup, fit);
        println!("max |Zipup - Fit| = {d_zipup_fit:.3e}");
        assert!(
            d_zipup_fit > 1e-14,
            "Zipup and Fit outputs are numerically identical (max diff {d_zipup_fit:.3e}); \
             the two arms may be dispatching to the same algorithm"
        );

        // (b) Naive (full Kronecker product then SVD) and Zipup (single-pass
        // truncation) must differ in either the bond-dimension profile or the
        // sampled values under the rank cap.
        let d_naive_zipup = max_diff(naive, zipup);
        println!("max |Naive - Zipup| = {d_naive_zipup:.3e}");
        assert!(
            dims[0] != dims[1] || d_naive_zipup > 1e-14,
            "Naive and Zipup agree in both link dims {:?} and sampled values \
             (max diff {d_naive_zipup:.3e}); the two arms may be dispatching to the \
             same algorithm",
            dims[0]
        );

        // (c) ACI is interpolation-based, Naive is SVD-based, so under truncation they
        // must differ in either the bond-dimension profile or the sampled values.
        let d_naive_aci = max_diff(naive, aci);
        println!("max |Naive - Aci| = {d_naive_aci:.3e}");
        assert!(
            dims[0] != dims[3] || d_naive_aci > 1e-14,
            "Naive and Aci agree in both link dims {:?} and sampled values \
             (max diff {d_naive_aci:.3e}); the two arms may be dispatching to the \
             same algorithm",
            dims[0]
        );
    }
}
