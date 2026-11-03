use super::*;
use tensor4all_simplett::{tensor3_from_data, AbstractTensorTrain};

/// A rank-2 train whose second bond direction carries no weight compresses
/// to rank 1, on both scalar types, through the same generic call.
fn compresses_to_rank_one<T: BenchScalar>(one: T, zero: T) {
    let a = tensor3_from_data(vec![one, zero, one, zero], 1, 2, 2).unwrap();
    let b = tensor3_from_data(vec![one, zero, one, zero], 2, 2, 1).unwrap();
    let mut tt = SimpleTensorTrain::new(vec![a, b]).unwrap();
    assert_eq!(tt.rank(), 2);
    T::compress_svd_in_place(&mut tt, 1e-12, usize::MAX).unwrap();
    assert_eq!(tt.rank(), 1);
}

#[test]
fn both_scalars_compress() {
    compresses_to_rank_one(1.0f64, 0.0f64);
    compresses_to_rank_one(Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0));
}
