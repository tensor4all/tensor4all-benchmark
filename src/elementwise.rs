//! Four ways to form the elementwise (Hadamard) product of two quantics tensor
//! trains, behind one entry point, plus the sampled max-error metrics of the
//! two cases that use them.
//!
//! The product itself is generic over [`BenchScalar`], because case 1 works on
//! a complex Fourier series and case 2 on a real 2D Gaussian mixture. The error
//! metrics compare against the analytic reference of their own case, fixing the
//! scalar type to `Complex64` for case 1 and `f64` for case 2.

use num_complex::Complex64;
use tensor4all_simplett::{tensor3_from_data, AbstractTensorTrain, SimpleTensorTrain, Tensor3Ops};

use crate::fourier::{compress_svd, FourierSeries};
use crate::gaussian::{grid_coord, Field2D};
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

/// How the `aci` arm interprets the stopping tolerance it is handed.
///
/// The SVD-based arms take the tolerance as a singular value threshold relative
/// to the largest singular value, so an inert tolerance such as `1e-15` simply
/// never fires and the rank cap alone decides where to truncate. ACI instead
/// compares a pivot error against the tolerance, and whether that comparison is
/// absolute or scaled by the sampled output magnitude of the bond is a separate
/// upstream switch, `AciOptions::scale_tolerance`, whose upstream default is
/// scale-relative since tensor4all-rs#619. This enum makes the choice explicit
/// at every call site rather than inheriting that default. Case 1 and the
/// patched arm of case 2 use an absolute residual. The global ACI arm of case 2
/// uses a scale-relative residual. Records name the selected metric explicitly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AciTolerance {
    /// Absolute pivot error threshold.
    Absolute,
    /// Pivot error divided by the largest sampled output magnitude of the bond,
    /// which is the upstream default.
    ScaleRelative,
}

impl ElementwiseAlgo {
    /// Which engine actually runs this arm.
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
    a: &SimpleTensorTrain<T>,
    b: &SimpleTensorTrain<T>,
    tol: f64,
    max_bond: usize,
    aci_tol: AciTolerance,
) -> anyhow::Result<SimpleTensorTrain<T>> {
    match algo {
        ElementwiseAlgo::Naive => hadamard_naive(a, b, tol, max_bond),
        ElementwiseAlgo::Zipup => hadamard_treetn(a, b, tol, max_bond, false),
        ElementwiseAlgo::Fit => hadamard_treetn(a, b, tol, max_bond, true),
        ElementwiseAlgo::Aci => hadamard_aci(a, b, tol, max_bond, aci_tol),
    }
}

/// Core-wise Hadamard (bond Kronecker product) followed by SVD compression.
/// This is the O(chi^4) baseline.
fn hadamard_naive<T: BenchScalar>(
    a: &SimpleTensorTrain<T>,
    b: &SimpleTensorTrain<T>,
    tol: f64,
    max_bond: usize,
) -> anyhow::Result<SimpleTensorTrain<T>> {
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
    let mut tt = SimpleTensorTrain::new(cores)?;
    compress_svd(&mut tt, tol, max_bond)?;
    Ok(tt)
}

/// `tensor4all_treetn::hadamard` on the bridged TreeTNs, with either the
/// one-pass zipup or the variational fit contraction.
fn hadamard_treetn<T: BenchScalar>(
    a: &SimpleTensorTrain<T>,
    b: &SimpleTensorTrain<T>,
    tol: f64,
    max_bond: usize,
    fit: bool,
) -> anyhow::Result<SimpleTensorTrain<T>> {
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
        .with_max_bond_dim(max_bond)
        .with_svd_policy(SvdTruncationPolicy::new(tol));
    if fit {
        opts = opts.with_nfullsweeps(FIT_NFULLSWEEPS);
    }
    let out = hadamard(&ta, &tb, &pairs, &0, opts)
        .map_err(|e| anyhow::anyhow!("hadamard failed: {e:?}"))?;
    Ok(treetn_to_tensor_train::<T>(out)?)
}

/// Adaptive cross interpolation of the pointwise product function.
fn hadamard_aci<T: BenchScalar>(
    a: &SimpleTensorTrain<T>,
    b: &SimpleTensorTrain<T>,
    tol: f64,
    max_bond: usize,
    aci_tol: AciTolerance,
) -> anyhow::Result<SimpleTensorTrain<T>> {
    use tensor4all_aci::{elementwise, AciOptions};
    let opts = AciOptions::<T> {
        tolerance: tol,
        max_bond_dim: Some(max_bond),
        scale_tolerance: aci_tol == AciTolerance::ScaleRelative,
        ..AciOptions::default()
    };
    let res = elementwise(|xs: &[T]| xs[0] * xs[1], &[a.clone(), b.clone()], &opts)?;
    Ok(res.tensor_train)
}

/// Number of stored parameters of a tensor train, the sum of its core sizes.
///
/// This is the honest size metric when two representations of the same function
/// are not both single trains: a rank is only comparable between trains of the
/// same length, while a parameter count is comparable between a global train and
/// a set of patch trains (see [`crate::patched::total_params`]).
pub fn tt_n_params<T: BenchScalar>(tt: &SimpleTensorTrain<T>) -> usize {
    tt.site_tensors()
        .iter()
        .map(|core| core.left_dim() * core.site_dim() * core.right_dim())
        .sum()
}

/// Max abs error against the exact product series at sampled grid points.
pub fn max_error_vs_series(
    tt: &SimpleTensorTrain<Complex64>,
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

/// Deterministic sampled relative-L2 error against an exact Fourier series.
pub fn sampled_relative_l2_vs_series(
    output: &SimpleTensorTrain<Complex64>,
    exact: &FourierSeries,
    r: usize,
    samples: usize,
    seed: u64,
) -> f64 {
    let (error, reference) = sample_grid_indices(r, samples, seed).into_iter().fold(
        (0.0, 0.0),
        |(error, reference), index| {
            let x = index as f64 / (1u64 << r) as f64;
            let expected = exact.eval(x);
            let delta = output.evaluate(&index_to_bits(index, r)).unwrap() - expected;
            (error + delta.norm_sqr(), reference + expected.norm_sqr())
        },
    );
    (error / reference.max(f64::MIN_POSITIVE)).sqrt()
}

/// Sampled relative maximum error against the exact pointwise product.
pub fn max_rel_error_vs_product<A: Field2D, B: Field2D>(
    h: &SimpleTensorTrain<f64>,
    f: &A,
    g: &B,
    r: usize,
    box_l: f64,
    n_samples: usize,
    seed: u64,
) -> f64 {
    let (error, scale) = sampled_product_values(h, f, g, r, box_l, n_samples, seed)
        .into_iter()
        .fold((0.0_f64, 0.0_f64), |(error, scale), (got, expected)| {
            (error.max((got - expected).abs()), scale.max(expected.abs()))
        });
    error / scale.max(f64::MIN_POSITIVE)
}

/// Deterministic sampled relative-L2 error against the exact pointwise product.
pub fn sampled_relative_l2_vs_product<A: Field2D, B: Field2D>(
    output: &SimpleTensorTrain<f64>,
    left: &A,
    right: &B,
    r: usize,
    box_l: f64,
    samples: usize,
    seed: u64,
) -> f64 {
    let (error, reference) = sampled_product_values(output, left, right, r, box_l, samples, seed)
        .into_iter()
        .fold((0.0, 0.0), |(error, reference), (got, expected)| {
            (
                error + (got - expected).powi(2),
                reference + expected.powi(2),
            )
        });
    (error / reference.max(f64::MIN_POSITIVE)).sqrt()
}

fn sampled_product_values<A: Field2D, B: Field2D>(
    output: &SimpleTensorTrain<f64>,
    left: &A,
    right: &B,
    r: usize,
    box_l: f64,
    samples: usize,
    seed: u64,
) -> Vec<(f64, f64)> {
    let xs = sample_grid_indices(r, samples, seed);
    let ys = sample_grid_indices(r, samples, seed.wrapping_add(1));
    xs.iter()
        .zip(&ys)
        .map(|(&ix, &iy)| {
            let x = grid_coord(ix, r, box_l);
            let y = grid_coord(iy, r, box_l);
            let fused: Vec<_> = index_to_bits(ix, r)
                .into_iter()
                .zip(index_to_bits(iy, r))
                .map(|(x, y)| x + 2 * y)
                .collect();
            (
                output.evaluate(&fused).unwrap(),
                left.eval(x, y) * right.eval(x, y),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests;
