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
/// at every call site rather than inheriting that default, because the two
/// families of cases want opposite things. The fixed-budget cases (2, 3 and 4)
/// want the same "tolerance is unreachable, the cap decides" regime for ACI as
/// for the SVD arms, so they ask for [`AciTolerance::ScaleRelative`]. Cases 1
/// and 5 are tolerance-driven and judged by one error that is normalized
/// globally, so they ask for [`AciTolerance::Absolute`]: a per-bond
/// normalization would hold each region to its own scale, which is not the
/// quantity either case reports.
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

/// Ratio at which the sampled elementwise product counts as numerically zero.
///
/// The relative error of cases 3 and 4 is normalized by the largest sampled
/// `|f * g|`. If the two mixtures barely overlap, that number collapses
/// exponentially while `max |f|` and `max |g|` stay of order one, and the
/// reported relative error stops measuring anything. A product scale below
/// `DEGENERACY_THRESHOLD` times the product of the two input scales means the
/// sampled reference has lost at least six orders of magnitude to the lack of
/// overlap, which is far outside anything a healthy instance produces: the
/// healthy benchmark instances sit many orders of magnitude above the threshold.
pub const DEGENERACY_THRESHOLD: f64 = 1e-6;

/// Scales of the product and both inputs on one sampled grid.
#[derive(Clone, Copy, Debug)]
pub struct MixtureProductScales {
    /// `max |f(x, y) * g(x, y)|`, the normalization of the relative error.
    pub ref_scale: f64,
    /// `max |f(x, y)|`.
    pub input_scale_f: f64,
    /// `max |g(x, y)|`.
    pub input_scale_g: f64,
}

impl MixtureProductScales {
    /// True when the sampled product has collapsed relative to the inputs, so
    /// the relative error metric of the case is meaningless.
    pub fn is_degenerate(&self) -> bool {
        self.ref_scale < DEGENERACY_THRESHOLD * self.input_scale_f * self.input_scale_g
    }
}

/// Reference and input scales on one sampled grid.
pub fn product_scales<A: Field2D, B: Field2D>(
    f: &A,
    g: &B,
    r: usize,
    box_l: f64,
    n_samples: usize,
    seed: u64,
) -> MixtureProductScales {
    let xs = sample_grid_indices(r, n_samples, seed);
    let ys = sample_grid_indices(r, n_samples, seed.wrapping_add(1));
    let mut s = MixtureProductScales {
        ref_scale: 0.0,
        input_scale_f: 0.0,
        input_scale_g: 0.0,
    };
    for (&ix, &iy) in xs.iter().zip(&ys) {
        let x = grid_coord(ix, r, box_l);
        let y = grid_coord(iy, r, box_l);
        let (fv, gv) = (f.eval(x, y), g.eval(x, y));
        s.ref_scale = s.ref_scale.max((fv * gv).abs());
        s.input_scale_f = s.input_scale_f.max(fv.abs());
        s.input_scale_g = s.input_scale_g.max(gv.abs());
    }
    s
}

/// Reject an instance whose sampled product is numerically zero.
pub fn check_product_not_degenerate<A: Field2D, B: Field2D>(
    f: &A,
    g: &B,
    r: usize,
    box_l: f64,
    n_samples: usize,
    seed: u64,
) -> anyhow::Result<MixtureProductScales> {
    let s = product_scales(f, g, r, box_l, n_samples, seed);
    anyhow::ensure!(
        !s.is_degenerate(),
        "degenerate instance at r={r}, box_l={box_l}: the two mixtures barely overlap, so the \
         elementwise product is numerically zero at the sampled points (ref_scale {:.3e} against \
         input_scale_f {:.3e} and input_scale_g {:.3e}, ratio {:.3e} below the threshold {:.0e}) \
         and the relative error metric, which divides by ref_scale, is meaningless. Lower \
         Increase the Gaussian density or use a smaller box.",
        s.ref_scale,
        s.input_scale_f,
        s.input_scale_g,
        s.ref_scale / (s.input_scale_f * s.input_scale_g).max(f64::MIN_POSITIVE),
        DEGENERACY_THRESHOLD
    );
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fourier::{compress_svd, FourierSeries};

    fn setup(
        r: usize,
        k: usize,
    ) -> (
        SimpleTensorTrain<Complex64>,
        SimpleTensorTrain<Complex64>,
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
            let out =
                elementwise_product(algo, &a, &b, 1e-10, 200, AciTolerance::Absolute).unwrap();
            let err = max_error_vs_series(&out, &exact, r, 100, 5);
            println!("{algo:?}: max abs error {err:.3e} (bound {bound:.0e})");
            assert!(err < bound, "{algo:?}: err {err} exceeds {bound}");
        }
    }

    /// Two mixtures whose supports sit in opposite corners of the box: each one
    /// is of order one where it lives, but the product is zero to machine
    /// precision everywhere, so the guard must refuse the instance.
    #[test]
    fn degeneracy_guard_fires_on_non_overlapping_mixtures() {
        use crate::gaussian::AnisoMixture2D;

        let (r, l) = (6, 4.0);
        let far = |sign: f64| AnisoMixture2D {
            weights: vec![1.0],
            quad: vec![(50.0, 0.0, 50.0)],
            centers: vec![(sign * l / 2.0, sign * l / 2.0)],
        };
        let (f, g) = (far(-1.0), far(1.0));

        let s = product_scales(&f, &g, r, l, 128, 7);
        println!(
            "ref_scale {:.3e}, input scales {:.3e} and {:.3e}",
            s.ref_scale, s.input_scale_f, s.input_scale_g
        );
        assert!(s.input_scale_f > 1e-3 && s.input_scale_g > 1e-3);
        assert!(s.is_degenerate(), "guard predicate missed a zero product");

        let err = check_product_not_degenerate(&f, &g, r, l, 128, 7)
            .expect_err("the guard must reject this instance");
        let msg = err.to_string();
        assert!(msg.contains("barely overlap"), "unexpected message: {msg}");

        // A healthy instance of the same shape must pass, so the guard is not
        // simply always firing.
        let f2 = AnisoMixture2D::random(8, 1.0, 0.12, 2.0, 1);
        let g2 = AnisoMixture2D::random(8, 1.0, 0.12, 2.0, 2);
        let s2 = check_product_not_degenerate(&f2, &g2, 8, 1.0, 128, 99).unwrap();
        println!("default-like instance: ref_scale {:.3e}", s2.ref_scale);
        assert!(!s2.is_degenerate());
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

        let eval = |tt: &SimpleTensorTrain<Complex64>| -> Vec<Complex64> {
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
            let out =
                elementwise_product(algo, &a, &b, 1e-10, max_bond, AciTolerance::Absolute).unwrap();
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
