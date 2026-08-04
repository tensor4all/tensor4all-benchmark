//! Four ways to form the elementwise (Hadamard) product of two quantics tensor
//! trains, behind one entry point, plus a sampled max-error metric against the
//! analytically known product series.

use num_complex::Complex64;
use tensor4all_simplett::{tensor3_from_data, AbstractTensorTrain, Tensor3Ops, TensorTrain};

use crate::fourier::{compress_svd, FourierSeries};
use crate::harness::{index_to_bits, sample_grid_indices};

#[derive(Clone, Copy, Debug)]
pub enum ElementwiseAlgo {
    Naive,
    Zipup,
    Fit,
    Aci,
}

pub fn elementwise_product(
    algo: ElementwiseAlgo,
    a: &TensorTrain<Complex64>,
    b: &TensorTrain<Complex64>,
    tol: f64,
    max_bond: usize,
) -> anyhow::Result<TensorTrain<Complex64>> {
    match algo {
        ElementwiseAlgo::Naive => hadamard_naive(a, b, tol, max_bond),
        ElementwiseAlgo::Zipup => hadamard_treetn(a, b, tol, max_bond, false),
        ElementwiseAlgo::Fit => hadamard_treetn(a, b, tol, max_bond, true),
        ElementwiseAlgo::Aci => hadamard_aci(a, b, tol, max_bond),
    }
}

/// Core-wise Hadamard (bond Kronecker product) followed by SVD compression.
/// This is the O(chi^4) baseline.
fn hadamard_naive(
    a: &TensorTrain<Complex64>,
    b: &TensorTrain<Complex64>,
    tol: f64,
    max_bond: usize,
) -> anyhow::Result<TensorTrain<Complex64>> {
    anyhow::ensure!(a.len() == b.len(), "site count mismatch");
    let mut cores = Vec::with_capacity(a.len());
    for (ca, cb) in a.site_tensors().iter().zip(b.site_tensors()) {
        let (la, s, ra) = (ca.left_dim(), ca.site_dim(), ca.right_dim());
        let (lb, rb) = (cb.left_dim(), cb.right_dim());
        anyhow::ensure!(s == cb.site_dim(), "site dimension mismatch");
        let mut data = vec![Complex64::new(0.0, 0.0); la * lb * s * ra * rb];
        for r2 in 0..rb {
            for r1 in 0..ra {
                for si in 0..s {
                    for l2 in 0..lb {
                        for l1 in 0..la {
                            let idx = (l1 + la * l2) + la * lb * (si + s * (r1 + ra * r2));
                            data[idx] = ca.get3(l1, si, r1) * cb.get3(l2, si, r2);
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
fn hadamard_treetn(
    a: &TensorTrain<Complex64>,
    b: &TensorTrain<Complex64>,
    tol: f64,
    max_bond: usize,
    fit: bool,
) -> anyhow::Result<TensorTrain<Complex64>> {
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
    let opts = ContractionOptions::new(method)
        .with_max_rank(max_bond)
        .with_svd_policy(SvdTruncationPolicy::new(tol));
    let out = hadamard(&ta, &tb, &pairs, &0, opts)
        .map_err(|e| anyhow::anyhow!("hadamard failed: {e:?}"))?;
    treetn_to_tensor_train::<Complex64>(out)
}

/// Adaptive cross interpolation of the pointwise product function.
fn hadamard_aci(
    a: &TensorTrain<Complex64>,
    b: &TensorTrain<Complex64>,
    tol: f64,
    max_bond: usize,
) -> anyhow::Result<TensorTrain<Complex64>> {
    use tensor4all_aci::{elementwise, AciOptions};
    let opts = AciOptions::<Complex64> {
        tolerance: tol,
        max_bond_dim: max_bond,
        ..AciOptions::default()
    };
    let res = elementwise(
        |xs: &[Complex64]| xs[0] * xs[1],
        &[a.clone(), b.clone()],
        &opts,
    )?;
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
}
