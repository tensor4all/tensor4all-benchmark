//! The scalar types the benchmark's tensor trains are built on.
//!
//! Case 1 runs on `Complex64` (a Fourier series) and case 3 on `f64` (a real
//! Gaussian mixture), and both go through the same Hadamard code, so that code
//! is generic over this trait.
//!
//! The trait carries the SVD compression as a method rather than leaving it to
//! a plain generic function. `SimpleTensorTrain::compress` is bounded by
//! `f64: From<<T as TensorScalar>::Real>`, and `TensorScalar` lives in
//! `tenferro-tensor`, which no crate this benchmark depends on re-exports, so
//! the bound cannot be written in a `where` clause here. Dispatching through a
//! trait implemented once per concrete scalar keeps the bound where it is
//! nameable. The set of implementors matches what `tensor4all_aci::AciScalar`
//! admits, namely `f64` and `Complex64`.

use num_complex::Complex64;
use tensor4all_simplett::{CompressionMethod, CompressionOptions, SimpleTensorTrain};

/// Scalars the generic Hadamard arms and the HDF5 export accept.
pub trait BenchScalar:
    tensor4all_aci::AciScalar
    + tensor4all_core::TensorElement
    + tensor4all_simplett::EinsumScalar
    + PartialEq
    + Clone
{
    /// SVD-compress `tt` in place to `tol`, capped at `max_bond`.
    fn compress_svd_in_place(
        tt: &mut SimpleTensorTrain<Self>,
        tol: f64,
        max_bond: usize,
    ) -> anyhow::Result<()>;
}

macro_rules! impl_bench_scalar {
    ($t:ty) => {
        impl BenchScalar for $t {
            fn compress_svd_in_place(
                tt: &mut SimpleTensorTrain<Self>,
                tol: f64,
                max_bond: usize,
            ) -> anyhow::Result<()> {
                tt.compress(&CompressionOptions {
                    method: CompressionMethod::SVD,
                    tolerance: tol,
                    max_bond_dim: Some(max_bond),
                    normalize_error: true,
                })?;
                Ok(())
            }
        }
    };
}

impl_bench_scalar!(f64);
impl_bench_scalar!(Complex64);

#[cfg(test)]
mod tests {
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
}
