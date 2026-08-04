//! Band-limited random Fourier series and its exact quantics tensor train.
//!
//! The series is `f(x) = sum_{k=0}^{K} c_k exp(2 pi i k x)` on `x in [0, 1)`,
//! with `re(c_k)`, `im(c_k)` drawn from `U[0, 1]` and the vector normalized so
//! that `sum_k |c_k|^2 = 1` (Ritter, arXiv:2604.00037).
//!
//! With the MSB-first quantics map `x = sum_n s_n 2^{-(n+1)}` each mode
//! factorizes, `exp(2 pi i k x) = prod_n exp(2 pi i k s_n / 2^{n+1})`, so a
//! single mode is exactly rank 1 and the sum of `K+1` modes is an exact QTT of
//! bond dimension `K+1` with mode-diagonal cores.

use num_complex::Complex64;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor4all_simplett::{tensor3_from_data, CompressionMethod, CompressionOptions, TensorTrain};

#[derive(Clone, Debug)]
pub struct FourierSeries {
    /// c_k for k = 0..=K
    pub coeffs: Vec<Complex64>,
}

impl FourierSeries {
    pub fn random(k_max: usize, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut coeffs: Vec<Complex64> = (0..=k_max)
            .map(|_| Complex64::new(rng.random::<f64>(), rng.random::<f64>()))
            .collect();
        let norm = coeffs.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        for c in &mut coeffs {
            *c /= norm;
        }
        Self { coeffs }
    }

    pub fn eval(&self, x: f64) -> Complex64 {
        self.coeffs
            .iter()
            .enumerate()
            .map(|(k, c)| {
                c * (Complex64::new(0.0, 2.0 * std::f64::consts::PI * k as f64 * x)).exp()
            })
            .sum()
    }

    /// Coefficient convolution: exact product series.
    pub fn product(&self, other: &Self) -> Self {
        let n = self.coeffs.len() + other.coeffs.len() - 1;
        let mut coeffs = vec![Complex64::new(0.0, 0.0); n];
        for (i, a) in self.coeffs.iter().enumerate() {
            for (j, b) in other.coeffs.iter().enumerate() {
                coeffs[i + j] += a * b;
            }
        }
        Self { coeffs }
    }

    /// Exact QTT with bond dimension K+1, R sites of dim 2, MSB first.
    pub fn to_qtt(&self, r: usize) -> anyhow::Result<TensorTrain<Complex64>> {
        anyhow::ensure!(r >= 2, "need r >= 2");
        let m = self.coeffs.len(); // K+1
        let phase = |k: usize, s: usize, n: usize| -> Complex64 {
            let theta = 2.0 * std::f64::consts::PI * (k * s) as f64 / (1u64 << (n + 1)) as f64;
            Complex64::new(0.0, theta).exp()
        };
        let mut cores = Vec::with_capacity(r);
        // first core: (1, 2, m), data[0 + 1*(s + 2*k)]
        let mut d0 = vec![Complex64::new(0.0, 0.0); 2 * m];
        for k in 0..m {
            for s in 0..2 {
                d0[s + 2 * k] = self.coeffs[k] * phase(k, s, 0);
            }
        }
        cores.push(tensor3_from_data(d0, 1, 2, m)?);
        // middle cores n = 1..r-1: (m, 2, m), diagonal in k
        for n in 1..r - 1 {
            let mut d = vec![Complex64::new(0.0, 0.0); m * 2 * m];
            for k in 0..m {
                for s in 0..2 {
                    d[k + m * (s + 2 * k)] = phase(k, s, n);
                }
            }
            cores.push(tensor3_from_data(d, m, 2, m)?);
        }
        // last core: (m, 2, 1)
        let mut dl = vec![Complex64::new(0.0, 0.0); m * 2];
        for k in 0..m {
            for s in 0..2 {
                dl[k + m * s] = phase(k, s, r - 1);
            }
        }
        cores.push(tensor3_from_data(dl, m, 2, 1)?);
        Ok(TensorTrain::new(cores)?)
    }
}

pub fn compress_svd(
    tt: &mut TensorTrain<Complex64>,
    tol: f64,
    max_bond: usize,
) -> anyhow::Result<()> {
    tt.compress(&CompressionOptions {
        method: CompressionMethod::SVD,
        tolerance: tol,
        max_bond_dim: max_bond,
        normalize_error: true,
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
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
}
