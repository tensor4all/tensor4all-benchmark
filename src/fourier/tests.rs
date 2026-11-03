use super::*;
use crate::harness::{index_to_bits, sample_grid_indices};
use tensor4all_simplett::AbstractTensorTrain;

#[test]
fn random_series_is_normalized_and_deterministic() {
    let f = FourierSeries::random(8, 42);
    let g = FourierSeries::random(8, 42);
    assert_eq!(f.coeffs, g.coeffs);
    let n2: f64 = f.coeffs.iter().map(|c| c.norm_sqr()).sum();
    assert!((n2 - 1.0).abs() < 1e-12);
    assert_eq!(f.coeffs.len(), 9);
}

#[test]
fn qtt_matches_direct_evaluation() {
    let r = 10;
    let f = FourierSeries::random(6, 1);
    let tt = f.to_qtt(r).unwrap();
    for &i in &sample_grid_indices(r, 50, 2) {
        let x = i as f64 / (1u64 << r) as f64;
        let bits = index_to_bits(i, r);
        let v = tt.evaluate(&bits).unwrap();
        assert!((v - f.eval(x)).norm() < 1e-11, "mismatch at x={x}");
    }
}

#[test]
fn product_is_coefficient_convolution() {
    let f = FourierSeries::random(3, 1);
    let g = FourierSeries::random(4, 2);
    let p = f.product(&g);
    assert_eq!(p.coeffs.len(), 3 + 4 + 1);
    let x = 0.371;
    assert!((p.eval(x) - f.eval(x) * g.eval(x)).norm() < 1e-12);
}

#[test]
fn compression_reduces_rank_below_k_plus_one() {
    let r = 14;
    let k = 32;
    let f = FourierSeries::random(k, 7);
    let mut tt = f.to_qtt(r).unwrap();
    assert_eq!(tt.rank(), k + 1);
    compress_svd(&mut tt, 1e-10, usize::MAX).unwrap();
    assert!(tt.rank() < k + 1, "rank {} not reduced", tt.rank());
    for &i in &sample_grid_indices(r, 20, 3) {
        let x = i as f64 / (1u64 << r) as f64;
        let v = tt.evaluate(&index_to_bits(i, r)).unwrap();
        assert!((v - f.eval(x)).norm() < 1e-8);
    }
}
