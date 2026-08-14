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
use tensor4all_simplett::{tensor3_from_data, AbstractTensorTrain, SimpleTensorTrain, Tensor3Ops};

use crate::fourier::{compress_svd, FourierSeries};
use crate::gaussian::{grid_coord, Field2D, GaussianMixture2D};
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
/// quantity either case reports (see the case-5 tolerance discussion in the
/// README).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AciTolerance {
    /// Absolute pivot error threshold.
    Absolute,
    /// Pivot error divided by the largest sampled output magnitude of the bond,
    /// which is the upstream default.
    ScaleRelative,
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

/// Case 3: relative max error of the fused 2D product train against the exact
/// pointwise product of the two Gaussian mixtures.
///
/// `h` is a fused quantics train on `[-L, L)^2` with `r` sites of dimension 4,
/// local index `x_bit + 2 * y_bit` and the most significant bit first, which is
/// the layout `gaussian::to_quantics_fused_tt` produces. The normalization
/// matches case 2: the largest sampled `|reference|`, so the two cases report
/// the same kind of number under `error_metric = "max_rel_vs_analytic"`.
pub fn max_rel_error_vs_mixture_product(
    h: &SimpleTensorTrain<f64>,
    f: &GaussianMixture2D,
    g: &GaussianMixture2D,
    r: usize,
    box_l: f64,
    n_samples: usize,
    seed: u64,
) -> f64 {
    max_rel_error_vs_product(h, f, g, r, box_l, n_samples, seed)
}

/// [`max_rel_error_vs_mixture_product`] for any pair of [`Field2D`] instances, so
/// that the two case-5 families share one error metric. Same sampling, same
/// normalization, same reported `error_metric`.
pub fn max_rel_error_vs_product<A: Field2D, B: Field2D>(
    h: &SimpleTensorTrain<f64>,
    f: &A,
    g: &B,
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

/// Ratio at which the sampled elementwise product counts as numerically zero.
///
/// The relative error of cases 3 and 4 is normalized by the largest sampled
/// `|f * g|`. If the two mixtures barely overlap, that number collapses
/// exponentially while `max |f|` and `max |g|` stay of order one, and the
/// reported relative error stops measuring anything. A product scale below
/// `DEGENERACY_THRESHOLD` times the product of the two input scales means the
/// sampled reference has lost at least six orders of magnitude to the lack of
/// overlap, which is far outside anything a healthy instance produces: the
/// default case-3 and case-4 instances sit at a ratio near 0.5, five to six
/// orders of magnitude above the threshold.
pub const DEGENERACY_THRESHOLD: f64 = 1e-6;

/// Scales of the case-3 and case-4 reference, all measured over one and the
/// same set of sampled grid points.
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

/// Reference and input scales at the same sampled grid points that
/// [`max_rel_error_vs_mixture_product`] uses, given the same `r`, `box_l`,
/// `n_samples` and `seed`.
pub fn mixture_product_scales(
    f: &GaussianMixture2D,
    g: &GaussianMixture2D,
    r: usize,
    box_l: f64,
    n_samples: usize,
    seed: u64,
) -> MixtureProductScales {
    product_scales(f, g, r, box_l, n_samples, seed)
}

/// [`mixture_product_scales`] for any pair of [`Field2D`] instances.
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

/// Fail-fast guard for cases 3 and 4: refuse to benchmark an instance whose
/// elementwise product is numerically zero at the sampled points.
///
/// Returns the scales on success so the caller can record them.
pub fn check_mixture_product_not_degenerate(
    f: &GaussianMixture2D,
    g: &GaussianMixture2D,
    r: usize,
    box_l: f64,
    n_samples: usize,
    seed: u64,
) -> anyhow::Result<MixtureProductScales> {
    check_product_not_degenerate(f, g, r, box_l, n_samples, seed)
}

/// [`check_mixture_product_not_degenerate`] for any pair of [`Field2D`]
/// instances, at the same threshold. The anisotropic spike family of case 5 holds
/// its spacing-to-width ratio fixed as `N` grows, which is exactly what keeps this
/// guard passing there: the sampled product stays at about half of
/// `max|f| max|g|` at every `N`.
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
         BENCH_ALPHA_HI so the Gaussians are wider, or raise the density (more Gaussians, or a \
         smaller box).",
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

    /// Case 3 at its own fixed output budget: every arm capped at `chi_in`, the
    /// larger input rank, and judged only on the error it returns for it.
    ///
    /// The budget is the cap alone: the tolerance handed to the arms is the
    /// runner's inert 1e-15 and the aci arm runs scale-relative, so every arm is
    /// expected to spend the whole budget rather than stop at a tolerance.
    ///
    /// The bounds are per arm because the arms are not comparable here. Measured
    /// at the pinned revision on this instance (r = 8, 3 Gaussians, chi_in 62):
    /// naive, fit and aci all land on 1.4e-8 at the full chi_out of 62, and zipup
    /// returns 8.3e-1 for the same budget. Every bound carries about an order of
    /// magnitude of margin, since the quantics TCI construction is not
    /// bit-reproducible and chi_in moves by one between runs. The zipup bound is
    /// loose on purpose: at this budget a single-pass truncation of an
    /// elementwise product has no accuracy left to defend (the runner's default
    /// instance reaches 9.2e-1 at r = 10), so what the bound guards is that the
    /// arm still returns a finite result of roughly the right scale. This test
    /// also covers the real-scalar (`f64`) path through all four arms, which case
    /// 1 does not exercise.
    #[test]
    fn gauss2d_arms_meet_their_error_bounds_at_fixed_budget() {
        use crate::gaussian::{to_quantics_fused_tt, GaussianMixture2D};

        let (r, l) = (8, 6.0);
        let f = GaussianMixture2D::random(3, l, (0.5, 8.0), 1);
        let g = GaussianMixture2D::random(3, l, (0.5, 8.0), 2);
        let (fa, _) = to_quantics_fused_tt(&f, r, l, 1e-8, 512).unwrap();
        let (gb, _) = to_quantics_fused_tt(&g, r, l, 1e-8, 512).unwrap();
        let chi_in = fa.rank().max(gb.rank());

        // The third element says whether the arm is expected to spend the whole
        // budget. The three SVD-based arms keep everything the cap allows, since
        // the tolerance can no longer stop them; aci is interpolation-based and
        // may settle below the cap if its pivot search saturates first, so it is
        // only held to the cap as an upper bound.
        for (algo, bound, exhausts_budget) in [
            (ElementwiseAlgo::Naive, 1e-6, true),
            (ElementwiseAlgo::Zipup, 2.0, true),
            (ElementwiseAlgo::Fit, 1e-6, true),
            (ElementwiseAlgo::Aci, 1e-6, false),
        ] {
            // The budget is the cap, so the tolerance is pinned inert, exactly as
            // the runner does it.
            let out =
                elementwise_product(algo, &fa, &gb, 1e-15, chi_in, AciTolerance::ScaleRelative)
                    .unwrap();
            assert!(
                out.rank() <= chi_in,
                "{algo:?}: chi_out {} exceeds the budget {chi_in}",
                out.rank()
            );
            if exhausts_budget {
                assert_eq!(
                    out.rank(),
                    chi_in,
                    "{algo:?}: chi_out {} fell short of the budget {chi_in}, so something \
                     other than the cap truncated it",
                    out.rank()
                );
            }
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

    /// Two mixtures whose supports sit in opposite corners of the box: each one
    /// is of order one where it lives, but the product is zero to machine
    /// precision everywhere, so the guard must refuse the instance.
    #[test]
    fn degeneracy_guard_fires_on_non_overlapping_mixtures() {
        use crate::gaussian::GaussianMixture2D;

        let (r, l) = (6, 4.0);
        let far = |sign: f64| GaussianMixture2D {
            weights: vec![1.0],
            alphas: vec![50.0],
            centers: vec![(sign * l / 2.0, sign * l / 2.0)],
        };
        let (f, g) = (far(-1.0), far(1.0));

        let s = mixture_product_scales(&f, &g, r, l, 128, 7);
        println!(
            "ref_scale {:.3e}, input scales {:.3e} and {:.3e}",
            s.ref_scale, s.input_scale_f, s.input_scale_g
        );
        assert!(s.input_scale_f > 1e-3 && s.input_scale_g > 1e-3);
        assert!(s.is_degenerate(), "guard predicate missed a zero product");

        let err = check_mixture_product_not_degenerate(&f, &g, r, l, 128, 7)
            .expect_err("the guard must reject this instance");
        let msg = err.to_string();
        assert!(msg.contains("barely overlap"), "unexpected message: {msg}");
        assert!(msg.contains("BENCH_ALPHA_HI"), "unexpected message: {msg}");

        // A healthy instance of the same shape must pass, so the guard is not
        // simply always firing.
        let f2 = GaussianMixture2D::random(8, 6.0, (0.5, 8.0), 1);
        let g2 = GaussianMixture2D::random(8, 6.0, (0.5, 8.0), 2);
        let s2 = check_mixture_product_not_degenerate(&f2, &g2, 8, 6.0, 128, 99).unwrap();
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
