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
        let out = elementwise_product(algo, &a, &b, 1e-10, 200, AciTolerance::Absolute).unwrap();
        let err = max_error_vs_series(&out, &exact, r, 100, 5);
        println!("{algo:?}: max abs error {err:.3e} (bound {bound:.0e})");
        assert!(err < bound, "{algo:?}: err {err} exceeds {bound}");
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
