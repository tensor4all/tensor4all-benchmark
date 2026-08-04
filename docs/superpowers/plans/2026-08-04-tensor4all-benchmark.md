# tensor4all-benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **Per user instruction: implementation subagents must run on the Opus model** (Agent tool `model: "opus"`).

**Goal:** Public benchmark repo comparing tensor4all-rs algorithms on two cases: (1) elementwise product of random Fourier series QTTs (naive/zipup/fit/ACI), (2) MPO-MPO contraction of 2D quantics Gaussian mixtures (naive/zipup/fit) with an analytic reference.

**Architecture:** Single Rust package (lib + two runner binaries) with git-pinned tensor4all-rs dependencies. Runners emit versioned JSON to `result/<profile>/raw/`; a uv Python script renders Markdown reports with SVG scaling plots. Inputs are generated in Rust only and handed to a Julia correctness script via ITensor-compatible HDF5 plus a JSON instance file.

**Tech Stack:** Rust (tensor4all-simplett, tensor4all-aci, tensor4all-treetn, tensor4all-quanticstci, tensor4all-hdf5, serde, rand_chacha), Python via uv (matplotlib), Julia (ITensors.jl, HDF5.jl, JSON3.jl).

## Global Constraints

- Repo root: `/Users/hiroshi/projects/tensor4all/tensor4all-benchmark` (git repo already initialized, spec committed).
- tensor4all-rs pinned via git: `git = "https://github.com/tensor4all/tensor4all-rs", rev = "69a24e7e86edc7079b758784864e3976776d208e"`. The API signatures below were read from the local checkout on a slightly newer branch; if a signature does not compile against the pin, first try bumping the rev to the current origin/main head, and record the final rev in this plan file.
- All tensor4all crates must use their default backend feature (`tenferro-cpu-faer`) consistently. Do not mix backends.
- `tensor4all-hdf5` needs a system HDF5 at build time (`brew install hdf5` on macOS, `libhdf5-dev` on Linux CI).
- All randomness through `rand_chacha::ChaCha8Rng` with explicit seeds recorded in JSON output.
- JSON records carry `schema_version: 1`.
- Every runner fails with nonzero exit if any algorithm error exceeds its sanity bound, so wrong timings never reach reports.
- Prose style in README and reports: no em/en dashes anywhere (user rule); use commas, colons, or separate sentences.
- Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## Key upstream API facts (verified against local tensor4all-rs checkout)

- `tensor4all_simplett`: `TensorTrain<T>::new(Vec<Tensor3<T>>)`, `tensor3_from_data(data, left, site, right)` (column-major, index `l + left*(s + site*r)`), trait `AbstractTensorTrain` provides `evaluate(&[usize])`, `rank()`, `link_dims()`, `norm()`. `TensorTrain::compress(&CompressionOptions { method: CompressionMethod::SVD, tolerance, max_bond_dim, normalize_error })`.
- `tensor4all_simplett::mpo`: `MPO<T>::new(Vec<Tensor4<T>>)`, `tensor4_from_data(data, l, s1, s2, r)` (column-major), `MPO::evaluate(&[usize])` takes 2L interleaved indices `[i1, j1, i2, j2, ...]`. Contractions: `contract_naive(&a, &b, Option<ContractionOptions>)`, `contract_zipup(&a, &b, &ContractionOptions)`, `contract_fit(&a, &b, &FitOptions, Option<MPO<T>>)`. `ContractionOptions { tolerance, max_bond_dim, factorize_method }`, `FitOptions { tolerance, max_bond_dim, max_sweeps, convergence_tol, factorize_method }`.
- `tensor4all_aci`: `elementwise(|xs: &[T]| ..., &[TensorTrain<T>], &AciOptions<T>) -> Result<AciResult<T>>`, `AciOptions { max_iters, min_iters, max_bond_dim, tolerance, scale_tolerance, initial_guess, rng_seed }`, `AciResult { tensor_train, ranks, errors }`. T is `f64` or `Complex64`.
- `tensor4all_treetn`: `hadamard(&left_treetn, &right_treetn, &[(DynIndex, DynIndex)], &center_node, ContractionOptions)`; bridges `tensor_train_to_treetn(&TensorTrain<T>) -> Result<(TreeTN<TensorDynLen, usize>, Vec<DynIndex>)>` and `treetn_to_tensor_train::<T>(TreeTN<...>) -> Result<TensorTrain<T>>`. `tensor4all_treetn::contraction::ContractionOptions` has public fields `method: ContractionMethod` (Zipup/Fit/Naive), `max_rank`, `svd_policy`, `nfullsweeps`, `convergence_tol` and more; check exact field types at `crates/tensor4all-treetn/src/treetn/contraction.rs` when compiling.
- `tensor4all_quanticstci`: `quanticscrossinterpolate(&DiscretizedGrid, f: Fn(&[f64]) -> V, None, QtciOptions) -> (QuanticsTensorCI2<V>, ranks, errors)`; `DiscretizedGrid::builder(&[R, R]).with_lower_bound(..).with_upper_bound(..).with_unfolding_scheme(UnfoldingScheme::Fused).build()`; `qtci.tensor_train()` returns `tensor4all_simplett::TensorTrain<V>` with site dim 4 per site (fused 2D), fused local index is `x_bit + 2*y_bit` (x least significant; verify by test, transpose helper provided as fallback). `QtciOptions` builders: `with_tolerance`, `with_maxbonddim`, `with_unfoldingscheme`.
- `tensor4all_hdf5`: `save_mps(path, name, &tensor4all_itensorlike::TensorTrain)`, `append_mps`, ITensors.jl-compatible (`@type = "MPS"`). Consumes itensorlike TT only; bridge: simplett TT -> `tensor_train_to_treetn` -> `tensor4all_itensorlike::TensorTrain::from_treetn(treetn)`.
- Quantics bit convention: MSB first, grid index `i = sum_n s_n 2^(R-1-n)`, `x = lower + i*step`, `step = (upper-lower)/2^R`, half-open grid.

## File structure

```
Cargo.toml                     package t4a-bench (lib + bins)
src/lib.rs                     module decls
src/harness.rs                 timing (median of N), sample index helper
src/record.rs                  RunRecord JSON schema + writer
src/fourier.rs                 FourierSeries: random gen, eval, convolution, to_qtt
src/elementwise.rs             4 algorithm wrappers + error measurement
src/gaussian.rs                GaussianMixture2D, analytic y-integral, to_quantics_mpo
src/mpo_contract.rs            3 MPO contraction wrappers + error measurement
src/hdf5_export.rs             simplett TT -> HDF5 (via itensorlike bridge)
src/bin/elementwise_fourier.rs runner (case 1)
src/bin/mpo_mpo_quantics.rs    runner (case 2)
julia/Project.toml             ITensors, HDF5, JSON3
julia/check_elementwise.jl     case 1 correctness check
julia/check_mpo_mpo.jl         case 2 correctness check
scripts/run_all.sh             full local run
scripts/report.py              JSON -> Markdown + SVG plots
pyproject.toml                 uv config (matplotlib, numpy)
result/<profile>/              committed reports (created by runs)
.github/workflows/ci.yml       build + test + smoke
README.md
```

---

### Task 1: Package scaffold and shared harness

**Files:**
- Create: `Cargo.toml`, `.gitignore`, `src/lib.rs`, `src/harness.rs`, `src/record.rs`

**Interfaces:**
- Produces: `t4a_bench::harness::{time_median, Timing, sample_grid_indices}`, `t4a_bench::record::{RunRecord, write_record}`. Later tasks import these exactly as named.

- [ ] **Step 1: Write Cargo.toml and .gitignore**

`Cargo.toml`:

```toml
[package]
name = "t4a-bench"
version = "0.1.0"
edition = "2021"
license = "MIT"
publish = false

[dependencies]
tensor4all-simplett = { git = "https://github.com/tensor4all/tensor4all-rs", rev = "69a24e7e86edc7079b758784864e3976776d208e", package = "tensor4all-simplett" }
tensor4all-aci = { git = "https://github.com/tensor4all/tensor4all-rs", rev = "69a24e7e86edc7079b758784864e3976776d208e", package = "tensor4all-aci" }
tensor4all-treetn = { git = "https://github.com/tensor4all/tensor4all-rs", rev = "69a24e7e86edc7079b758784864e3976776d208e", package = "tensor4all-treetn" }
tensor4all-itensorlike = { git = "https://github.com/tensor4all/tensor4all-rs", rev = "69a24e7e86edc7079b758784864e3976776d208e", package = "tensor4all-itensorlike" }
tensor4all-quanticstci = { git = "https://github.com/tensor4all/tensor4all-rs", rev = "69a24e7e86edc7079b758784864e3976776d208e", package = "tensor4all-quanticstci" }
tensor4all-hdf5 = { git = "https://github.com/tensor4all/tensor4all-rs", rev = "69a24e7e86edc7079b758784864e3976776d208e", package = "tensor4all-hdf5" }
tensor4all-core = { git = "https://github.com/tensor4all/tensor4all-rs", rev = "69a24e7e86edc7079b758784864e3976776d208e", package = "tensor4all-core" }
num-complex = "0.4"
rand = "0.8"
rand_chacha = "0.3"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
anyhow = "1"

[profile.release]
debug = true
```

Note: if the pinned rev fails to build or an API below is missing, bump every `rev` to current `origin/main` of tensor4all-rs and record the new rev here. If `rand` 0.8 conflicts with workspace deps, match the version tensor4all-rs uses.

`.gitignore`:

```
/target
Cargo.lock.orig
__pycache__/
.venv/
julia/Manifest.toml
result/**/raw/*.h5
```

(Cargo.lock IS committed: this is a binary-style repo.)

- [ ] **Step 2: Write failing test for harness**

`src/harness.rs` (test first, at bottom of the new file with stub-free implementation in step 3; create the file with only the test module and `use` lines, expect compile failure):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_median_returns_result_and_positive_times() {
        let (val, timing) = time_median(1, 3, || 40 + 2);
        assert_eq!(val, 42);
        assert_eq!(timing.runs_secs.len(), 3);
        assert!(timing.median_secs >= 0.0);
    }

    #[test]
    fn sample_grid_indices_is_deterministic_and_in_range() {
        let a = sample_grid_indices(10, 64, 7);
        let b = sample_grid_indices(10, 64, 7);
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
        assert!(a.iter().all(|&i| i < 1u64 << 10));
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test harness -- --nocapture`
Expected: compile error, `time_median` not found.

- [ ] **Step 4: Implement harness**

Fill in `src/harness.rs` above the test module:

```rust
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

pub struct Timing {
    pub median_secs: f64,
    pub runs_secs: Vec<f64>,
}

/// Run `f` `warmups` times untimed, then `repeats` times timed.
/// Returns the last result and the timing summary.
pub fn time_median<R>(warmups: usize, repeats: usize, mut f: impl FnMut() -> R) -> (R, Timing) {
    assert!(repeats >= 1);
    for _ in 0..warmups {
        std::hint::black_box(f());
    }
    let mut runs_secs = Vec::with_capacity(repeats);
    let mut last = None;
    for _ in 0..repeats {
        let t0 = Instant::now();
        last = Some(std::hint::black_box(f()));
        runs_secs.push(t0.elapsed().as_secs_f64());
    }
    let mut sorted = runs_secs.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_secs = sorted[sorted.len() / 2];
    (last.unwrap(), Timing { median_secs, runs_secs })
}

/// `n` random grid indices in [0, 2^r), deterministic in `seed`.
pub fn sample_grid_indices(r: usize, n: usize, seed: u64) -> Vec<u64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n).map(|_| rng.gen_range(0..(1u64 << r))).collect()
}

/// MSB-first bit decomposition of a grid index into R local indices.
pub fn index_to_bits(i: u64, r: usize) -> Vec<usize> {
    (0..r).map(|n| ((i >> (r - 1 - n)) & 1) as usize).collect()
}
```

Add a third test in the test module:

```rust
    #[test]
    fn index_to_bits_msb_first() {
        assert_eq!(index_to_bits(0b100, 3), vec![1, 0, 0]);
        assert_eq!(index_to_bits(0b011, 3), vec![0, 1, 1]);
    }
```

- [ ] **Step 5: Implement record.rs with test**

`src/record.rs`:

```rust
use serde::Serialize;
use std::path::Path;

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Serialize)]
pub struct RunRecord {
    pub schema_version: u32,
    pub case: String,
    pub algorithm: String,
    pub params: serde_json::Value,
    pub seed: u64,
    pub tolerance: f64,
    pub wall_time_median_secs: f64,
    pub wall_times_secs: Vec<f64>,
    pub max_error: f64,
    pub input_max_bond_dim: usize,
    pub output_max_bond_dim: usize,
    pub output_bond_dims: Vec<usize>,
}

pub fn write_record(dir: &Path, name: &str, record: &RunRecord) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!("{name}.json"));
    std::fs::write(&path, serde_json::to_string_pretty(record)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_roundtrips_to_json() {
        let rec = RunRecord {
            schema_version: SCHEMA_VERSION,
            case: "elementwise_fourier".into(),
            algorithm: "aci".into(),
            params: serde_json::json!({"k_max": 8, "r": 12}),
            seed: 0,
            tolerance: 1e-8,
            wall_time_median_secs: 0.1,
            wall_times_secs: vec![0.1],
            max_error: 1e-9,
            input_max_bond_dim: 5,
            output_max_bond_dim: 9,
            output_bond_dims: vec![2, 9, 2],
        };
        let s = serde_json::to_string(&rec).unwrap();
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["schema_version"], 1);
        assert_eq!(v["params"]["k_max"], 8);
    }
}
```

`src/lib.rs`:

```rust
pub mod harness;
pub mod record;
```

- [ ] **Step 6: Run tests, verify pass**

Run: `cargo test`
Expected: all tests pass (first build compiles the pinned tensor4all-rs, takes minutes).

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: scaffold t4a-bench package with timing harness and JSON record schema"
```

---

### Task 2: Random Fourier series and exact QTT construction

**Files:**
- Create: `src/fourier.rs`
- Modify: `src/lib.rs` (add `pub mod fourier;`)

**Interfaces:**
- Consumes: `harness::index_to_bits`.
- Produces: `t4a_bench::fourier::FourierSeries` with `pub coeffs: Vec<Complex64>`, `fn random(k_max: usize, seed: u64) -> Self`, `fn eval(&self, x: f64) -> Complex64`, `fn product(&self, other: &Self) -> Self`, `fn to_qtt(&self, r: usize) -> anyhow::Result<TensorTrain<Complex64>>`, and free fn `compress_svd(tt: &mut TensorTrain<Complex64>, tol: f64, max_bond: usize) -> anyhow::Result<()>`.

Math: series `f(x) = sum_{k=0}^{K} c_k exp(2 pi i k x)` on `x in [0,1)`. Each mode is exactly rank 1 in quantics because `x = sum_n s_n 2^{-(n+1)}` (MSB first), so `exp(2 pi i k x) = prod_n exp(2 pi i k s_n / 2^{n+1})`. The sum of K+1 modes is an exact QTT of bond dimension K+1 with mode-diagonal cores. Coefficients: re and im each U[0,1], then normalized so `sum |c_k|^2 = 1` (Ritter, arXiv:2604.00037). The product of two series is the coefficient convolution, band limit 2K.

- [ ] **Step 1: Write failing tests**

```rust
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
```

- [ ] **Step 2: Run tests, verify compile failure**

Run: `cargo test fourier`
Expected: FAIL, module missing.

- [ ] **Step 3: Implement**

```rust
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
            .map(|_| Complex64::new(rng.gen::<f64>(), rng.gen::<f64>()))
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
            .map(|(k, c)| c * (Complex64::new(0.0, 2.0 * std::f64::consts::PI * k as f64 * x)).exp())
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
```

If `CompressionOptions` field names differ at the pinned rev, read `crates/tensor4all-simplett/src/compression.rs` and adjust; keep SVD method and relative tolerance semantics.

- [ ] **Step 4: Run tests, verify pass**

Run: `cargo test fourier`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: random Fourier series with exact rank-(K+1) QTT construction"
```

---

### Task 3: Elementwise product algorithm wrappers

**Files:**
- Create: `src/elementwise.rs`
- Modify: `src/lib.rs` (add `pub mod elementwise;`)

**Interfaces:**
- Consumes: `fourier::{FourierSeries, compress_svd}`, `harness::{sample_grid_indices, index_to_bits}`.
- Produces:
  ```rust
  pub enum ElementwiseAlgo { Naive, Zipup, Fit, Aci }
  pub fn elementwise_product(
      algo: ElementwiseAlgo,
      a: &TensorTrain<Complex64>,
      b: &TensorTrain<Complex64>,
      tol: f64,
      max_bond: usize,
  ) -> anyhow::Result<TensorTrain<Complex64>>;
  pub fn max_error_vs_series(
      tt: &TensorTrain<Complex64>, exact: &FourierSeries, r: usize, n_samples: usize, seed: u64,
  ) -> f64;
  ```

Algorithm notes:
- Naive: direct core-wise Hadamard (bond Kronecker product) then SVD compression. This is the O(chi^4) baseline from the paper. Implemented locally, about 30 lines.
- Zipup and Fit: `tensor4all_treetn::hadamard` on bridged TreeTNs with `ContractionMethod::Zipup` / `::Fit`. Known upstream issue: Fit may converge to about 5e-4 relative error on elementwise products (documented in `tensor4all-itensorlike/tests/bug_fit_elementwise.rs`). Do NOT hide this: the fit arm gets a looser sanity bound (1e-3) and the report must mention the discrepancy.
- ACI: `tensor4all_aci::elementwise` with `|xs| xs[0] * xs[1]`.

- [ ] **Step 1: Write failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::fourier::{compress_svd, FourierSeries};

    fn setup(r: usize, k: usize) -> (TensorTrain<Complex64>, TensorTrain<Complex64>, FourierSeries) {
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
            assert!(err < bound, "{algo:?}: err {err} exceeds {bound}");
        }
    }
}
```

- [ ] **Step 2: Run test, verify failure**

Run: `cargo test elementwise`
Expected: compile failure.

- [ ] **Step 3: Implement**

```rust
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

fn hadamard_naive(
    a: &TensorTrain<Complex64>,
    b: &TensorTrain<Complex64>,
    tol: f64,
    max_bond: usize,
) -> anyhow::Result<TensorTrain<Complex64>> {
    let mut cores = Vec::with_capacity(a.len());
    for (ca, cb) in a.site_tensors().iter().zip(b.site_tensors()) {
        let (la, s, ra) = (ca.left_dim(), ca.site_dim(), ca.right_dim());
        let (lb, rb) = (cb.left_dim(), cb.right_dim());
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
    let pairs: Vec<_> = ia.into_iter().zip(ib.into_iter()).collect();
    let mut opts = ContractionOptions::default();
    opts.method = if fit { ContractionMethod::Fit } else { ContractionMethod::Zipup };
    opts.max_rank = Some(max_bond);
    opts.svd_policy = SvdTruncationPolicy::new(tol);
    let out = hadamard(&ta, &tb, &pairs, &0, opts)
        .map_err(|e| anyhow::anyhow!("hadamard failed: {e:?}"))?;
    Ok(treetn_to_tensor_train::<Complex64>(out)?)
}

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
    let res = elementwise(|xs: &[Complex64]| xs[0] * xs[1], &[a.clone(), b.clone()], &opts)?;
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
```

Exact field names of `ContractionOptions` (`max_rank` may be `usize` instead of `Option<usize>`, extra fields may exist) must be checked against `crates/tensor4all-treetn/src/treetn/contraction.rs` around line 854 at the pinned rev; adjust the two assignments, nothing else. If `hadamard`'s center-node type does not accept `&0`, use the first node name returned by the bridge.

- [ ] **Step 4: Run test, verify pass**

Run: `cargo test elementwise -- --nocapture`
Expected: PASS. If the Fit arm fails its 1e-3 bound, loosen to 1e-2 and record the measured value in a code comment referencing the upstream bug test.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: four elementwise product algorithm wrappers with exact-series error metric"
```

---

### Task 4: Case 1 runner binary

**Files:**
- Create: `src/bin/elementwise_fourier.rs`

**Interfaces:**
- Consumes: everything from Tasks 1 to 3.
- Produces: JSON records named `elementwise_fourier-<algo>-k<k>.json` under `$OUT_DIR` (default `result/dev/raw`). Environment knobs: `BENCH_KS` (comma list, default `4,8,16,32,64`), `BENCH_R` (default 20), `BENCH_TOL` (default 1e-8), `BENCH_MAX_BOND` (default 4096), `BENCH_RUNS` (default 5), `BENCH_WARMUPS` (default 1), `BENCH_SEED` (default 0), `BENCH_ALGOS` (default `naive,zipup,fit,aci`), `OUT_DIR`.

- [ ] **Step 1: Implement the runner**

```rust
use num_complex::Complex64;
use std::path::PathBuf;
use t4a_bench::elementwise::{elementwise_product, max_error_vs_series, ElementwiseAlgo};
use t4a_bench::fourier::{compress_svd, FourierSeries};
use t4a_bench::harness::time_median;
use t4a_bench::record::{write_record, RunRecord, SCHEMA_VERSION};
use tensor4all_simplett::AbstractTensorTrain;

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn parse_algo(s: &str) -> ElementwiseAlgo {
    match s {
        "naive" => ElementwiseAlgo::Naive,
        "zipup" => ElementwiseAlgo::Zipup,
        "fit" => ElementwiseAlgo::Fit,
        "aci" => ElementwiseAlgo::Aci,
        other => panic!("unknown algorithm {other}"),
    }
}

fn main() -> anyhow::Result<()> {
    let ks: Vec<usize> = std::env::var("BENCH_KS")
        .unwrap_or_else(|_| "4,8,16,32,64".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let r: usize = env_or("BENCH_R", 20);
    let tol: f64 = env_or("BENCH_TOL", 1e-8);
    let max_bond: usize = env_or("BENCH_MAX_BOND", 4096);
    let runs: usize = env_or("BENCH_RUNS", 5);
    let warmups: usize = env_or("BENCH_WARMUPS", 1);
    let seed: u64 = env_or("BENCH_SEED", 0);
    let algos: Vec<String> = std::env::var("BENCH_ALGOS")
        .unwrap_or_else(|_| "naive,zipup,fit,aci".into())
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap_or_else(|_| "result/dev/raw".into()));

    let mut failures = Vec::new();
    for &k in &ks {
        let f = FourierSeries::random(k, seed.wrapping_add(2 * k as u64));
        let g = FourierSeries::random(k, seed.wrapping_add(2 * k as u64 + 1));
        let exact = f.product(&g);
        let mut a = f.to_qtt(r)?;
        let mut b = g.to_qtt(r)?;
        compress_svd(&mut a, tol, max_bond)?;
        compress_svd(&mut b, tol, max_bond)?;
        let input_chi = a.rank().max(b.rank());
        eprintln!("k_max={k} input_chi={input_chi}");

        for algo_name in &algos {
            let algo = parse_algo(algo_name);
            let (out, timing) = time_median(warmups, runs, || {
                elementwise_product(algo, &a, &b, tol, max_bond).expect("algorithm failed")
            });
            let max_error = max_error_vs_series(&out, &exact, r, 256, seed.wrapping_add(999));
            // fit has a known upstream accuracy issue on elementwise products
            let sanity = if matches!(algo, ElementwiseAlgo::Fit) { 1e-2 } else { 10.0 * tol };
            let rec = RunRecord {
                schema_version: SCHEMA_VERSION,
                case: "elementwise_fourier".into(),
                algorithm: algo_name.clone(),
                params: serde_json::json!({
                    "k_max": k, "r": r, "input_max_bond_dim": input_chi, "max_bond": max_bond,
                }),
                seed,
                tolerance: tol,
                wall_time_median_secs: timing.median_secs,
                wall_times_secs: timing.runs_secs,
                max_error,
                input_max_bond_dim: input_chi,
                output_max_bond_dim: out.rank(),
                output_bond_dims: out.link_dims(),
            };
            write_record(&out_dir, &format!("elementwise_fourier-{algo_name}-k{k}"), &rec)?;
            eprintln!("  {algo_name}: t={:.4}s err={max_error:.2e} chi_out={}", timing.median_secs, out.rank());
            if max_error > sanity {
                failures.push(format!("{algo_name} k={k}: err {max_error:.2e} > sanity {sanity:.2e}"));
            }
        }
    }
    if !failures.is_empty() {
        anyhow::bail!("sanity failures:\n{}", failures.join("\n"));
    }
    Ok(())
}
```

- [ ] **Step 2: Smoke run**

```bash
BENCH_KS=4,8 BENCH_R=12 BENCH_RUNS=1 BENCH_WARMUPS=0 OUT_DIR=/tmp/t4abench-smoke cargo run --release --bin elementwise_fourier
```

Expected: exit 0, four JSON files per k in `/tmp/t4abench-smoke`, stderr shows per-algorithm time/error lines. Inspect one JSON for sane fields.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat: elementwise_fourier runner with env-configurable sweep and sanity gate"
```

---

### Task 5: HDF5 export and Julia correctness check (case 1)

**Files:**
- Create: `src/hdf5_export.rs`, `julia/Project.toml`, `julia/check_elementwise.jl`
- Modify: `src/lib.rs` (add `pub mod hdf5_export;`), `src/bin/elementwise_fourier.rs` (export hook)

**Interfaces:**
- Produces: `t4a_bench::hdf5_export::save_tt_as_mps(path: &str, name: &str, tt: &TensorTrain<Complex64>, append: bool) -> anyhow::Result<()>`. Runner env knob `EXPORT_HDF5=<dir>`: writes `instance-k<k>.h5` containing MPS groups `f` and `g`, plus `instance-k<k>.json` with the coefficient arrays.

- [ ] **Step 1: Implement export module**

```rust
use num_complex::Complex64;
use tensor4all_simplett::TensorTrain;
use tensor4all_treetn::tensor_train_to_treetn;

pub fn save_tt_as_mps(
    path: &str,
    name: &str,
    tt: &TensorTrain<Complex64>,
    append: bool,
) -> anyhow::Result<()> {
    let (treetn, _indices) = tensor_train_to_treetn(tt)?;
    let itt = tensor4all_itensorlike::TensorTrain::from_treetn(treetn)?;
    if append {
        tensor4all_hdf5::append_mps(path, name, &itt)?;
    } else {
        tensor4all_hdf5::save_mps(path, name, &itt)?;
    }
    Ok(())
}
```

Add a unit test in the same file: build `FourierSeries::random(3, 1).to_qtt(6)`, save to a temp file (`std::env::temp_dir()`), reload with `tensor4all_hdf5::load_mps`, assert `maxbonddim()` matches. This proves the Rust-side round trip; the Julia script proves cross-language.

- [ ] **Step 2: Hook into the runner**

In `src/bin/elementwise_fourier.rs`, inside the k loop after compressing `a` and `b`:

```rust
if let Ok(dir) = std::env::var("EXPORT_HDF5") {
    std::fs::create_dir_all(&dir)?;
    let h5 = format!("{dir}/instance-k{k}.h5");
    t4a_bench::hdf5_export::save_tt_as_mps(&h5, "f", &a, false)?;
    t4a_bench::hdf5_export::save_tt_as_mps(&h5, "g", &b, true)?;
    let meta = serde_json::json!({
        "schema_version": 1,
        "case": "elementwise_fourier",
        "r": r,
        "k_max": k,
        "f_coeffs": f.coeffs.iter().map(|c| [c.re, c.im]).collect::<Vec<_>>(),
        "g_coeffs": g.coeffs.iter().map(|c| [c.re, c.im]).collect::<Vec<_>>(),
    });
    std::fs::write(format!("{dir}/instance-k{k}.json"), serde_json::to_string_pretty(&meta)?)?;
}
```

- [ ] **Step 3: Julia environment and check script**

`julia/Project.toml`:

```toml
[deps]
HDF5 = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
ITensors = "9136182c-28ba-11e9-034c-db9fb085ebd5"
ITensorMPS = "0d1a4710-d33b-49a5-8f18-73bdf49b47e2"
JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
```

`julia/check_elementwise.jl`:

```julia
# Usage: julia --project=julia julia/check_elementwise.jl <dir> <k>
# Reads instance-k<k>.h5 (MPS groups "f","g") and instance-k<k>.json,
# evaluates both MPS at sample grid points, compares to the analytic series.
using HDF5, ITensors, ITensorMPS, JSON3

dir, k = ARGS[1], ARGS[2]
meta = JSON3.read(read(joinpath(dir, "instance-k$k.json"), String))
R = meta.r

coeffs(v) = [complex(c[1], c[2]) for c in v]
series(c, x) = sum(c[j+1] * exp(2im * pi * j * x) for j in 0:length(c)-1)

function eval_mps(psi::MPS, bits::Vector{Int})
    s = siteinds(psi)
    v = ITensor(1.0)
    for n in eachindex(psi)
        v *= psi[n] * onehot(s[n] => bits[n] + 1)
    end
    return scalar(v)
end

fails = 0
h5open(joinpath(dir, "instance-k$k.h5"), "r") do file
    for (name, cs) in (("f", coeffs(meta.f_coeffs)), ("g", coeffs(meta.g_coeffs)))
        psi = read(file, name, MPS)
        @assert length(psi) == R
        for trial in 1:50
            i = rand(0:(2^R - 1))
            bits = [Int((i >> (R - n)) & 1) for n in 1:R]  # MSB first
            x = i / 2^R
            got = eval_mps(psi, bits)
            want = series(cs, x)
            if abs(got - want) > 1e-6
                global fails += 1
                println("MISMATCH $name x=$x got=$got want=$want")
            end
        end
    end
end
fails == 0 || error("$fails mismatches")
println("check_elementwise: OK")
```

- [ ] **Step 4: End-to-end verification**

```bash
BENCH_KS=4 BENCH_R=10 BENCH_RUNS=1 BENCH_WARMUPS=0 OUT_DIR=/tmp/t4abench-h5 EXPORT_HDF5=/tmp/t4abench-h5 cargo run --release --bin elementwise_fourier
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
julia --project=julia julia/check_elementwise.jl /tmp/t4abench-h5 4
```

Expected: `check_elementwise: OK`. If ITensors cannot read complex MPS groups written by tensor4all-hdf5, check the `@type` attribute handling and consult `tensor4all-rs/docs/examples/julia/hdf5.jl` for the working read pattern.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: ITensor-compatible HDF5 export and Julia cross-check for case 1"
```

---

### Task 6: Gaussian mixtures, analytic integral, quantics MPO construction

**Files:**
- Create: `src/gaussian.rs`
- Modify: `src/lib.rs` (add `pub mod gaussian;`)

**Interfaces:**
- Produces:
  ```rust
  pub struct GaussianMixture2D { pub weights: Vec<f64>, pub alphas: Vec<f64>, pub centers: Vec<(f64, f64)> }
  impl GaussianMixture2D {
      pub fn random(n: usize, box_l: f64, alpha_range: (f64, f64), seed: u64) -> Self;
      pub fn eval(&self, x: f64, y: f64) -> f64;
  }
  /// closed form of int f(x,y) g(y,z) dy over the whole real line
  pub fn analytic_contraction(f: &GaussianMixture2D, g: &GaussianMixture2D, x: f64, z: f64) -> f64;
  /// f(v1, v2) as a quantics MPO: site n has (bit n of v1, bit n of v2), R sites
  pub fn to_quantics_mpo(mix: &GaussianMixture2D, r: usize, box_l: f64, tol: f64, max_bond: usize)
      -> anyhow::Result<(MPO<f64>, f64)>;  // (mpo, grid_step)
  pub fn grid_coord(i: u64, r: usize, box_l: f64) -> f64;  // -L + i * 2L/2^R
  ```

Math: a Gaussian `w exp(-a ((x-cx)^2 + (y-cy)^2))` factorizes over x and y, so
`int f_i(x,y) g_j(y,z) dy = w_i w_j e^{-a_i (x-cx_i)^2} e^{-b_j (z-cz_j)^2} sqrt(pi/(a_i+b_j)) exp(-(a_i b_j/(a_i+b_j)) (cy_i - cy_j)^2)`.
Random mixture: centers uniform in `[-L/2, L/2]^2` (kept away from the box edge so the tail truncation error stays small), weights U[0.5, 1.5], alphas log-uniform in `alpha_range`.

MPO construction: build a fused 2D quantics TT with `quanticscrossinterpolate` on `DiscretizedGrid::builder(&[r, r]).with_lower_bound(&[-L, -L]).with_upper_bound(&[L, L]).with_unfolding_scheme(UnfoldingScheme::Fused).build()`, then reinterpret each dim-4 site core `(l, 4, rdim)` as a `Tensor4` `(l, 2, 2, rdim)` with the SAME column-major buffer. Per quanticsgrids fused convention the fused local index is `bit(var1) + 2*bit(var2)` with var order to be confirmed by the unit test below; if the evaluation test fails, transpose the two site legs (swap `s1`/`s2` when copying, index `l + L*(s1 + 2*(s2 + 2*r))`).

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytic_contraction_matches_quadrature() {
        let f = GaussianMixture2D::random(3, 4.0, (0.5, 2.0), 1);
        let g = GaussianMixture2D::random(2, 4.0, (0.5, 2.0), 2);
        let (x, z) = (0.3, -0.7);
        // trapezoid quadrature over y in [-8, 8]
        let n = 20_000;
        let (lo, hi) = (-8.0, 8.0);
        let h = (hi - lo) / n as f64;
        let mut s = 0.0;
        for i in 0..=n {
            let y = lo + i as f64 * h;
            let w = if i == 0 || i == n { 0.5 } else { 1.0 };
            s += w * f.eval(x, y) * g.eval(y, z) * h;
        }
        let a = analytic_contraction(&f, &g, x, z);
        assert!((s - a).abs() < 1e-8 * a.abs().max(1.0), "quad {s} vs analytic {a}");
    }

    #[test]
    fn quantics_mpo_evaluates_to_function_values() {
        let r = 8;
        let l = 4.0;
        let mix = GaussianMixture2D::random(3, l, (0.5, 2.0), 3);
        let (mpo, _dy) = to_quantics_mpo(&mix, r, l, 1e-10, 200).unwrap();
        assert_eq!(mpo.len(), r);
        for &(i, j) in &[(0u64, 0u64), (37, 200), (255, 1), (128, 128)] {
            let x = grid_coord(i, r, l);
            let y = grid_coord(j, r, l);
            let xb = crate::harness::index_to_bits(i, r);
            let yb = crate::harness::index_to_bits(j, r);
            let mut idx = Vec::with_capacity(2 * r);
            for n in 0..r {
                idx.push(xb[n]);
                idx.push(yb[n]);
            }
            let v = mpo.evaluate(&idx).unwrap();
            assert!((v - mix.eval(x, y)).abs() < 1e-6, "at ({x},{y}): {v} vs {}", mix.eval(x, y));
        }
    }
}
```

- [ ] **Step 2: Run tests, verify failure**

Run: `cargo test gaussian`
Expected: compile failure.

- [ ] **Step 3: Implement**

```rust
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor4all_simplett::mpo::{tensor4_from_data, MPO};
use tensor4all_simplett::{AbstractTensorTrain, Tensor3Ops};
use tensor4all_quanticstci::{quanticscrossinterpolate, DiscretizedGrid, QtciOptions, UnfoldingScheme};

#[derive(Clone, Debug)]
pub struct GaussianMixture2D {
    pub weights: Vec<f64>,
    pub alphas: Vec<f64>,
    pub centers: Vec<(f64, f64)>,
}

impl GaussianMixture2D {
    pub fn random(n: usize, box_l: f64, alpha_range: (f64, f64), seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let half = box_l / 2.0;
        let (a_lo, a_hi) = alpha_range;
        let mut weights = Vec::new();
        let mut alphas = Vec::new();
        let mut centers = Vec::new();
        for _ in 0..n {
            weights.push(rng.gen_range(0.5..1.5));
            alphas.push((rng.gen_range(a_lo.ln()..a_hi.ln())).exp());
            centers.push((rng.gen_range(-half..half), rng.gen_range(-half..half)));
        }
        Self { weights, alphas, centers }
    }

    pub fn eval(&self, x: f64, y: f64) -> f64 {
        (0..self.weights.len())
            .map(|i| {
                let (cx, cy) = self.centers[i];
                self.weights[i] * (-self.alphas[i] * ((x - cx).powi(2) + (y - cy).powi(2))).exp()
            })
            .sum()
    }
}

pub fn analytic_contraction(f: &GaussianMixture2D, g: &GaussianMixture2D, x: f64, z: f64) -> f64 {
    let mut s = 0.0;
    for i in 0..f.weights.len() {
        let (fcx, fcy) = f.centers[i];
        let a = f.alphas[i];
        let fx = f.weights[i] * (-a * (x - fcx).powi(2)).exp();
        for j in 0..g.weights.len() {
            let (gcy, gcz) = g.centers[j];
            let b = g.alphas[j];
            let gz = g.weights[j] * (-b * (z - gcz).powi(2)).exp();
            let ab = a + b;
            let yfac = (std::f64::consts::PI / ab).sqrt() * (-(a * b / ab) * (fcy - gcy).powi(2)).exp();
            s += fx * gz * yfac;
        }
    }
    s
}

pub fn grid_coord(i: u64, r: usize, box_l: f64) -> f64 {
    let step = 2.0 * box_l / (1u64 << r) as f64;
    -box_l + i as f64 * step
}

pub fn to_quantics_mpo(
    mix: &GaussianMixture2D,
    r: usize,
    box_l: f64,
    tol: f64,
    max_bond: usize,
) -> anyhow::Result<(MPO<f64>, f64)> {
    let grid = DiscretizedGrid::builder(&[r, r])
        .with_lower_bound(&[-box_l, -box_l])
        .with_upper_bound(&[box_l, box_l])
        .with_unfolding_scheme(UnfoldingScheme::Fused)
        .build()?;
    let m = mix.clone();
    let opts = QtciOptions::default()
        .with_tolerance(tol)
        .with_maxbonddim(max_bond)
        .with_unfoldingscheme(UnfoldingScheme::Fused);
    let (qtci, _ranks, _errs) = quanticscrossinterpolate(&grid, move |xy: &[f64]| m.eval(xy[0], xy[1]), None, opts)?;
    let tt = qtci.tensor_train();
    let mut cores4 = Vec::with_capacity(tt.len());
    for c in tt.site_tensors() {
        let (l, s, rd) = (c.left_dim(), c.site_dim(), c.right_dim());
        anyhow::ensure!(s == 4, "expected fused site dim 4, got {s}");
        // fused index assumed s = s1 + 2*s2 (var1 bit least significant).
        // If quantics_mpo_evaluates_to_function_values fails, swap s1 and s2 here.
        let mut data = vec![0.0f64; l * 4 * rd];
        for rr in 0..rd {
            for s2 in 0..2 {
                for s1 in 0..2 {
                    for ll in 0..l {
                        let fused = s1 + 2 * s2;
                        data[ll + l * (s1 + 2 * (s2 + 2 * rr))] = c.get3(ll, fused, rr);
                    }
                }
            }
        }
        cores4.push(tensor4_from_data(data, l, 2, 2, rd)?);
    }
    let step = 2.0 * box_l / (1u64 << r) as f64;
    Ok((MPO::new(cores4)?, step))
}
```

Note the identity copy: with fused = s1 + 2*s2 the Tensor4 column-major layout equals the Tensor3 layout, but write the explicit loop anyway, it makes the swap fallback a one-line change. If the fused variable order turns out reversed (test failure), change `let fused = s1 + 2 * s2;` to `let fused = s2 + 2 * s1;` and re-run.

- [ ] **Step 4: Run tests, verify pass**

Run: `cargo test gaussian -- --nocapture`
Expected: both tests PASS (TCI on a 3-Gaussian mixture at R=8 runs in seconds).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: 2D Gaussian mixtures with analytic y-integral and quantics MPO construction"
```

---

### Task 7: MPO-MPO contraction wrappers and case 2 runner

**Files:**
- Create: `src/mpo_contract.rs`, `src/bin/mpo_mpo_quantics.rs`
- Modify: `src/lib.rs` (add `pub mod mpo_contract;`)

**Interfaces:**
- Consumes: `gaussian::*`, `harness::*`, `record::*`.
- Produces:
  ```rust
  pub enum MpoAlgo { Naive, Zipup, Fit }
  pub fn mpo_contract(algo: MpoAlgo, a: &MPO<f64>, b: &MPO<f64>, tol: f64, max_bond: usize)
      -> anyhow::Result<MPO<f64>>;
  pub fn max_rel_error_vs_analytic(
      h: &MPO<f64>, dy: f64, f: &GaussianMixture2D, g: &GaussianMixture2D,
      r: usize, box_l: f64, n_samples: usize, seed: u64,
  ) -> f64;
  ```
- Runner env knobs: `BENCH_RS` (default `10,12,14,16`), `BENCH_NGAUSS` (default 8), `BENCH_BOX_L` (default 6.0), `BENCH_ALPHA_LO`/`BENCH_ALPHA_HI` (default 0.5/8.0), `BENCH_TOL` (1e-8), `BENCH_MAX_BOND` (512), `BENCH_RUNS` (5), `BENCH_WARMUPS` (1), `BENCH_SEED` (0), `BENCH_ALGOS` (`naive,zipup,fit`), `BENCH_SANITY` (default 1e-4), `OUT_DIR`, `EXPORT_HDF5`.

- [ ] **Step 1: Write failing test for wrappers**

In `src/mpo_contract.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaussian::{to_quantics_mpo, GaussianMixture2D};

    #[test]
    fn all_algorithms_agree_with_analytic_integral() {
        let (r, l) = (8, 6.0);
        let f = GaussianMixture2D::random(3, l, (0.5, 2.0), 20);
        let g = GaussianMixture2D::random(3, l, (0.5, 2.0), 21);
        let (fa, dy) = to_quantics_mpo(&f, r, l, 1e-10, 200).unwrap();
        let (gb, _) = to_quantics_mpo(&g, r, l, 1e-10, 200).unwrap();
        for algo in [MpoAlgo::Naive, MpoAlgo::Zipup, MpoAlgo::Fit] {
            let h = mpo_contract(algo, &fa, &gb, 1e-10, 400).unwrap();
            let err = max_rel_error_vs_analytic(&h, dy, &f, &g, r, l, 50, 22);
            // R=8 discretization floor dominates; bound is loose on purpose
            assert!(err < 1e-2, "{algo:?}: rel err {err}");
        }
    }
}
```

- [ ] **Step 2: Run test, verify failure**

Run: `cargo test mpo_contract`
Expected: compile failure.

- [ ] **Step 3: Implement wrappers**

```rust
use tensor4all_simplett::mpo::{
    contract_fit, contract_naive, contract_zipup, ContractionOptions, FitOptions, MPO,
};

use crate::gaussian::{analytic_contraction, grid_coord, GaussianMixture2D};
use crate::harness::{index_to_bits, sample_grid_indices};

#[derive(Clone, Copy, Debug)]
pub enum MpoAlgo {
    Naive,
    Zipup,
    Fit,
}

pub fn mpo_contract(
    algo: MpoAlgo,
    a: &MPO<f64>,
    b: &MPO<f64>,
    tol: f64,
    max_bond: usize,
) -> anyhow::Result<MPO<f64>> {
    let opts = ContractionOptions {
        tolerance: tol,
        max_bond_dim: max_bond,
        ..ContractionOptions::default()
    };
    let out = match algo {
        MpoAlgo::Naive => contract_naive(a, b, Some(opts))?,
        MpoAlgo::Zipup => contract_zipup(a, b, &opts)?,
        MpoAlgo::Fit => {
            let fopts = FitOptions {
                tolerance: tol,
                max_bond_dim: max_bond,
                ..FitOptions::default()
            };
            contract_fit(a, b, &fopts, None)?
        }
    };
    Ok(out)
}

/// Relative max error of h(x,z)*... vs the analytic integral, normalized by
/// the max sampled |analytic| value. The MPO already contains the plain sum
/// over the y grid, so multiply by dy to approximate the integral.
pub fn max_rel_error_vs_analytic(
    h: &MPO<f64>,
    dy: f64,
    f: &GaussianMixture2D,
    g: &GaussianMixture2D,
    r: usize,
    box_l: f64,
    n_samples: usize,
    seed: u64,
) -> f64 {
    let xs = sample_grid_indices(r, n_samples, seed);
    let zs = sample_grid_indices(r, n_samples, seed.wrapping_add(1));
    let mut max_abs = 0.0f64;
    let mut max_ref = 0.0f64;
    for (&ix, &iz) in xs.iter().zip(&zs) {
        let x = grid_coord(ix, r, box_l);
        let z = grid_coord(iz, r, box_l);
        let xb = index_to_bits(ix, r);
        let zb = index_to_bits(iz, r);
        let mut idx = Vec::with_capacity(2 * r);
        for n in 0..r {
            idx.push(xb[n]);
            idx.push(zb[n]);
        }
        let got = h.evaluate(&idx).unwrap() * dy;
        let want = analytic_contraction(f, g, x, z);
        max_abs = max_abs.max((got - want).abs());
        max_ref = max_ref.max(want.abs());
    }
    max_abs / max_ref.max(f64::MIN_POSITIVE)
}
```

If `ContractionOptions` or `FitOptions` field names differ at the pinned rev, read `crates/tensor4all-simplett/src/mpo/options.rs` (or wherever they live) and adjust only the struct literals.

- [ ] **Step 4: Run test, verify pass**

Run: `cargo test mpo_contract -- --nocapture`
Expected: PASS.

- [ ] **Step 5: Implement the runner**

`src/bin/mpo_mpo_quantics.rs`, same skeleton as the case 1 runner:

```rust
use std::path::PathBuf;
use t4a_bench::gaussian::{to_quantics_mpo, GaussianMixture2D};
use t4a_bench::harness::time_median;
use t4a_bench::mpo_contract::{max_rel_error_vs_analytic, mpo_contract, MpoAlgo};
use t4a_bench::record::{write_record, RunRecord, SCHEMA_VERSION};

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn main() -> anyhow::Result<()> {
    let rs: Vec<usize> = std::env::var("BENCH_RS")
        .unwrap_or_else(|_| "10,12,14,16".into())
        .split(',').map(|s| s.trim().parse().unwrap()).collect();
    let ngauss: usize = env_or("BENCH_NGAUSS", 8);
    let box_l: f64 = env_or("BENCH_BOX_L", 6.0);
    let alpha_lo: f64 = env_or("BENCH_ALPHA_LO", 0.5);
    let alpha_hi: f64 = env_or("BENCH_ALPHA_HI", 8.0);
    let tol: f64 = env_or("BENCH_TOL", 1e-8);
    let max_bond: usize = env_or("BENCH_MAX_BOND", 512);
    let runs: usize = env_or("BENCH_RUNS", 5);
    let warmups: usize = env_or("BENCH_WARMUPS", 1);
    let seed: u64 = env_or("BENCH_SEED", 0);
    let sanity: f64 = env_or("BENCH_SANITY", 1e-4);
    let algos: Vec<String> = std::env::var("BENCH_ALGOS")
        .unwrap_or_else(|_| "naive,zipup,fit".into())
        .split(',').map(|s| s.trim().to_string()).collect();
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap_or_else(|_| "result/dev/raw".into()));

    let f = GaussianMixture2D::random(ngauss, box_l, (alpha_lo, alpha_hi), seed.wrapping_add(1));
    let g = GaussianMixture2D::random(ngauss, box_l, (alpha_lo, alpha_hi), seed.wrapping_add(2));

    let mut failures = Vec::new();
    for &r in &rs {
        let (fa, dy) = to_quantics_mpo(&f, r, box_l, tol, max_bond)?;
        let (gb, _) = to_quantics_mpo(&g, r, box_l, tol, max_bond)?;
        let input_chi = fa.rank().max(gb.rank());
        eprintln!("r={r} input_chi={input_chi}");

        for algo_name in &algos {
            let algo = match algo_name.as_str() {
                "naive" => MpoAlgo::Naive,
                "zipup" => MpoAlgo::Zipup,
                "fit" => MpoAlgo::Fit,
                other => panic!("unknown algorithm {other}"),
            };
            let (h, timing) = time_median(warmups, runs, || {
                mpo_contract(algo, &fa, &gb, tol, max_bond).expect("contraction failed")
            });
            let max_error = max_rel_error_vs_analytic(&h, dy, &f, &g, r, box_l, 128, seed.wrapping_add(99));
            let rec = RunRecord {
                schema_version: SCHEMA_VERSION,
                case: "mpo_mpo_quantics".into(),
                algorithm: algo_name.clone(),
                params: serde_json::json!({
                    "r": r, "n_gauss": ngauss, "box_l": box_l,
                    "alpha_range": [alpha_lo, alpha_hi], "max_bond": max_bond,
                }),
                seed,
                tolerance: tol,
                wall_time_median_secs: timing.median_secs,
                wall_times_secs: timing.runs_secs,
                max_error,
                input_max_bond_dim: input_chi,
                output_max_bond_dim: h.rank(),
                output_bond_dims: h.link_dims(),
            };
            write_record(&out_dir, &format!("mpo_mpo_quantics-{algo_name}-r{r}"), &rec)?;
            eprintln!("  {algo_name}: t={:.4}s rel_err={max_error:.2e} chi_out={}", timing.median_secs, h.rank());
            if max_error > sanity {
                failures.push(format!("{algo_name} r={r}: rel err {max_error:.2e} > {sanity:.2e}"));
            }
        }
    }
    if !failures.is_empty() {
        anyhow::bail!("sanity failures:\n{}", failures.join("\n"));
    }
    Ok(())
}
```

Note: `MPO::rank()` and `link_dims()` are inherent methods; if names differ at the pin, check `crates/tensor4all-simplett/src/mpo/mpo.rs`. The sanity bound default 1e-4 accounts for the discretization floor at moderate R; the report shows the actual error curve.

- [ ] **Step 6: Smoke run**

```bash
BENCH_RS=8,10 BENCH_NGAUSS=3 BENCH_RUNS=1 BENCH_WARMUPS=0 BENCH_SANITY=1e-1 OUT_DIR=/tmp/t4abench-mpo cargo run --release --bin mpo_mpo_quantics
```

Expected: exit 0, six JSON files, plausible errors that shrink from r=8 to r=10.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: MPO-MPO contraction wrappers and mpo_mpo_quantics runner"
```

---

### Task 8: Case 2 HDF5 export and Julia check

**Files:**
- Modify: `src/hdf5_export.rs`, `src/bin/mpo_mpo_quantics.rs`
- Create: `julia/check_mpo_mpo.jl`

**Interfaces:**
- Produces: `hdf5_export::save_fused_tt_as_mps(path, name, tt: &TensorTrain<f64>, append: bool)`, same bridge as case 1 but for `f64`. The runner exports the FUSED site-dim-4 TT (before Tensor4 conversion) so `save_mps` applies directly, plus `instance-r<r>.json` with mixture parameters.

- [ ] **Step 1: Generalize the export function**

Make `save_tt_as_mps` generic:

```rust
pub fn save_tt_as_mps<T>(
    path: &str,
    name: &str,
    tt: &tensor4all_simplett::TensorTrain<T>,
    append: bool,
) -> anyhow::Result<()>
where
    T: tensor4all_simplett::TTScalar,
    // add whatever bounds tensor_train_to_treetn requires; check its signature at the pin
{
    let (treetn, _indices) = tensor_train_to_treetn(tt)?;
    let itt = tensor4all_itensorlike::TensorTrain::from_treetn(treetn)?;
    if append {
        tensor4all_hdf5::append_mps(path, name, &itt)?;
    } else {
        tensor4all_hdf5::save_mps(path, name, &itt)?;
    }
    Ok(())
}
```

To export the fused TT, `to_quantics_mpo` must also return it (or add a sibling `to_quantics_fused_tt` returning `TensorTrain<f64>`; prefer the sibling, it keeps interfaces stable). Update `src/gaussian.rs`: extract the qtci call into `pub fn to_quantics_fused_tt(mix, r, box_l, tol, max_bond) -> anyhow::Result<TensorTrain<f64>>` and have `to_quantics_mpo` call it. Existing tests must still pass unchanged.

- [ ] **Step 2: Hook into the runner**

Inside the r loop, mirroring case 1:

```rust
if let Ok(dir) = std::env::var("EXPORT_HDF5") {
    std::fs::create_dir_all(&dir)?;
    let h5 = format!("{dir}/instance-r{r}.h5");
    let ftt = t4a_bench::gaussian::to_quantics_fused_tt(&f, r, box_l, tol, max_bond)?;
    let gtt = t4a_bench::gaussian::to_quantics_fused_tt(&g, r, box_l, tol, max_bond)?;
    t4a_bench::hdf5_export::save_tt_as_mps(&h5, "f", &ftt, false)?;
    t4a_bench::hdf5_export::save_tt_as_mps(&h5, "g", &gtt, true)?;
    let meta = serde_json::json!({
        "schema_version": 1, "case": "mpo_mpo_quantics", "r": r, "box_l": box_l,
        "f": {"weights": f.weights, "alphas": f.alphas, "centers": f.centers},
        "g": {"weights": g.weights, "alphas": g.alphas, "centers": g.centers},
    });
    std::fs::write(format!("{dir}/instance-r{r}.json"), serde_json::to_string_pretty(&meta)?)?;
}
```

- [ ] **Step 3: Julia check script**

`julia/check_mpo_mpo.jl`: loads the two MPS (site dim 4), evaluates at random fused grid indices, compares to the mixture formula:

```julia
# Usage: julia --project=julia julia/check_mpo_mpo.jl <dir> <r>
using HDF5, ITensors, ITensorMPS, JSON3

dir, rstr = ARGS[1], ARGS[2]
R = parse(Int, rstr)
meta = JSON3.read(read(joinpath(dir, "instance-r$rstr.json"), String))
L = meta.box_l

function mixture(m, x, y)
    s = 0.0
    for i in eachindex(m.weights)
        cx, cy = m.centers[i]
        s += m.weights[i] * exp(-m.alphas[i] * ((x - cx)^2 + (y - cy)^2))
    end
    return s
end

coord(i) = -L + i * 2L / 2^R

function eval_mps(psi::MPS, locals::Vector{Int})
    s = siteinds(psi)
    v = ITensor(1.0)
    for n in eachindex(psi)
        v *= psi[n] * onehot(s[n] => locals[n] + 1)
    end
    return scalar(v)
end

fails = 0
h5open(joinpath(dir, "instance-r$rstr.h5"), "r") do file
    for (name, m) in (("f", meta.f), ("g", meta.g))
        psi = read(file, name, MPS)
        @assert length(psi) == R
        for trial in 1:50
            ix, iy = rand(0:2^R-1), rand(0:2^R-1)
            xb = [Int((ix >> (R - n)) & 1) for n in 1:R]
            yb = [Int((iy >> (R - n)) & 1) for n in 1:R]
            fused = [xb[n] + 2 * yb[n] for n in 1:R]  # matches Rust s1 + 2*s2; swap if mismatched
            got = eval_mps(psi, fused)
            want = mixture(m, coord(ix), coord(iy))
            if abs(got - want) > 1e-5 * max(1.0, abs(want))
                global fails += 1
                println("MISMATCH $name ($(coord(ix)), $(coord(iy))): $got vs $want")
            end
        end
    end
end
fails == 0 || error("$fails mismatches")
println("check_mpo_mpo: OK")
```

The fused-order comment mirrors the Rust-side fallback: whichever variant made the Rust test pass, use the same here.

- [ ] **Step 4: End-to-end verification**

```bash
BENCH_RS=8 BENCH_NGAUSS=3 BENCH_RUNS=1 BENCH_WARMUPS=0 BENCH_SANITY=1e-1 OUT_DIR=/tmp/t4abench-mpo-h5 EXPORT_HDF5=/tmp/t4abench-mpo-h5 cargo run --release --bin mpo_mpo_quantics
julia --project=julia julia/check_mpo_mpo.jl /tmp/t4abench-mpo-h5 8
```

Expected: `check_mpo_mpo: OK`.

- [ ] **Step 5: Run full test suite, commit**

```bash
cargo test
git add -A && git commit -m "feat: case 2 HDF5 export and Julia cross-check"
```

---

### Task 9: Report generator and run script

**Files:**
- Create: `pyproject.toml`, `scripts/report.py`, `scripts/run_all.sh`

**Interfaces:**
- Consumes: `RunRecord` JSON files in `result/<profile>/raw/`.
- Produces: `result/<profile>/<case>.md` with a summary table, fitted scaling exponents, and two SVG plots per case (`<case>-time.svg`, `<case>-error.svg`); `result/<profile>/run.yaml` metadata.

- [ ] **Step 1: pyproject.toml**

```toml
[project]
name = "t4a-bench-report"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["matplotlib>=3.8", "numpy>=1.26"]
```

- [ ] **Step 2: scripts/report.py**

```python
#!/usr/bin/env python3
"""Render Markdown reports and SVG scaling plots from RunRecord JSON files.

Usage: uv run scripts/report.py result/<profile>
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

X_AXIS = {
    # case name -> (record field used as x, axis label)
    "elementwise_fourier": ("input_max_bond_dim", "input bond dimension chi"),
    "mpo_mpo_quantics": ("input_max_bond_dim", "input bond dimension chi"),
}


def load(profile_dir: Path):
    cases = defaultdict(lambda: defaultdict(list))
    for path in sorted((profile_dir / "raw").glob("*.json")):
        rec = json.loads(path.read_text())
        assert rec["schema_version"] == 1, f"unknown schema in {path}"
        cases[rec["case"]][rec["algorithm"]].append(rec)
    return cases


def fit_exponent(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    mask = (xs > 0) & (ys > 0)
    if mask.sum() < 2:
        return float("nan")
    p = np.polyfit(np.log(xs[mask]), np.log(ys[mask]), 1)
    return p[0]


def render_case(case, algos, profile_dir: Path):
    xfield, xlabel = X_AXIS[case]
    lines = [f"# {case}", "", "| algorithm | points | fitted time exponent | worst error |",
             "|---|---|---|---|"]
    fig_t, ax_t = plt.subplots(figsize=(5, 4))
    fig_e, ax_e = plt.subplots(figsize=(5, 4))
    for algo, recs in sorted(algos.items()):
        recs = sorted(recs, key=lambda rec: rec[xfield])
        xs = [rec[xfield] for rec in recs]
        ts = [rec["wall_time_median_secs"] for rec in recs]
        es = [rec["max_error"] for rec in recs]
        expo = fit_exponent(xs, ts)
        lines.append(f"| {algo} | {len(recs)} | {expo:.2f} | {max(es):.2e} |")
        ax_t.loglog(xs, ts, "o-", label=f"{algo} (chi^{expo:.1f})")
        ax_e.loglog(xs, es, "o-", label=algo)
    for ax, ylab in ((ax_t, "median wall time [s]"), (ax_e, "max error")):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylab)
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
    fig_t.tight_layout()
    fig_e.tight_layout()
    fig_t.savefig(profile_dir / f"{case}-time.svg")
    fig_e.savefig(profile_dir / f"{case}-error.svg")
    lines += ["", f"![time](./{case}-time.svg)", "", f"![error](./{case}-error.svg)", ""]
    (profile_dir / f"{case}.md").write_text("\n".join(lines))
    print(f"wrote {profile_dir / (case + '.md')}")


def main():
    profile_dir = Path(sys.argv[1])
    cases = load(profile_dir)
    if not cases:
        sys.exit(f"no records under {profile_dir}/raw")
    for case, algos in cases.items():
        render_case(case, algos, profile_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: scripts/run_all.sh**

```bash
#!/usr/bin/env bash
# Usage: scripts/run_all.sh <profile>   (e.g. mac-cpu)
set -euo pipefail
PROFILE="${1:?usage: run_all.sh <profile>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/result/$PROFILE"
mkdir -p "$OUT/raw"

cargo build --release

OUT_DIR="$OUT/raw" cargo run --release --bin elementwise_fourier
OUT_DIR="$OUT/raw" cargo run --release --bin mpo_mpo_quantics

cat > "$OUT/run.yaml" <<EOF
profile: $PROFILE
date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
host: $(hostname)
os: $(uname -sm)
repo_rev: $(git -C "$ROOT" rev-parse HEAD)
tensor4all_rs_rev: $(grep -m1 -o 'rev = "[a-f0-9]*"' "$ROOT/Cargo.toml" | cut -d'"' -f2)
threads: ${RAYON_NUM_THREADS:-default}
EOF

uv run scripts/report.py "$OUT"
echo "reports in $OUT"
```

`chmod +x scripts/run_all.sh`.

- [ ] **Step 4: Verify on smoke data**

```bash
mkdir -p result/dev/raw
BENCH_KS=4,8 BENCH_R=12 BENCH_RUNS=1 BENCH_WARMUPS=0 OUT_DIR=result/dev/raw cargo run --release --bin elementwise_fourier
uv run scripts/report.py result/dev
```

Expected: `result/dev/elementwise_fourier.md` plus two SVGs render sensibly. Then `git clean -fd result/dev` (dev results are not committed).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: Markdown/SVG report generator and run_all script"
```

---

### Task 10: README, CI, and final verification

**Files:**
- Create: `README.md`, `.github/workflows/ci.yml`, `LICENSE`

- [ ] **Step 1: README.md**

Content requirements (write it, no placeholders; style rule: no em/en dashes):

- Title `# tensor4all-benchmark`, one-paragraph purpose: an open experimentation ground for comparing tensor network contraction algorithms in tensor4all-rs on reproducible problem instances.
- Section `## Benchmark cases`: case 1 (elementwise product of random Fourier series QTTs, following arXiv:2604.00037; algorithms naive, zipup, fit, ACI) and case 2 (MPO-MPO contraction of 2D quantics Gaussian mixtures with an analytic reference; algorithms naive, zipup, fit).
- Section `## Latest results`: links to `result/mac-cpu/elementwise_fourier.md` and `result/mac-cpu/mpo_mpo_quantics.md` (these exist only after the first real run; write the links anyway and note they appear after the first committed run).
- Section `## Running`: prerequisites (Rust, HDF5 via brew or apt, uv, optionally Julia), then `scripts/run_all.sh mac-cpu`, env knobs table (BENCH_KS, BENCH_RS, BENCH_TOL, BENCH_RUNS, OUT_DIR, EXPORT_HDF5), smoke one-liner.
- Section `## Julia correctness checks`: the two commands from Tasks 5 and 8.
- Section `## Known issues`: the fit elementwise accuracy discrepancy with a pointer to the upstream test `tensor4all-itensorlike/tests/bug_fit_elementwise.rs`.
- License: MIT. Copy the MIT license text into `LICENSE` with `Copyright (c) 2026 tensor4all developers`.

- [ ] **Step 2: CI workflow**

`.github/workflows/ci.yml`:

```yaml
name: ci
on:
  push:
    branches: [main]
  pull_request:

jobs:
  rust:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: sudo apt-get update && sudo apt-get install -y libhdf5-dev
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - run: cargo test --release
      - name: smoke elementwise
        run: >
          BENCH_KS=4 BENCH_R=10 BENCH_RUNS=1 BENCH_WARMUPS=0
          OUT_DIR=/tmp/smoke EXPORT_HDF5=/tmp/smoke
          cargo run --release --bin elementwise_fourier
      - name: smoke mpo
        run: >
          BENCH_RS=8 BENCH_NGAUSS=3 BENCH_RUNS=1 BENCH_WARMUPS=0 BENCH_SANITY=1e-1
          OUT_DIR=/tmp/smoke cargo run --release --bin mpo_mpo_quantics
      - uses: julia-actions/setup-julia@v2
        with:
          version: "1"
      - name: julia check
        run: |
          julia --project=julia -e 'using Pkg; Pkg.instantiate()'
          julia --project=julia julia/check_elementwise.jl /tmp/smoke 4
```

- [ ] **Step 3: Full local verification**

```bash
cargo test --release
BENCH_KS=4 BENCH_R=10 BENCH_RUNS=1 BENCH_WARMUPS=0 OUT_DIR=/tmp/final-smoke EXPORT_HDF5=/tmp/final-smoke cargo run --release --bin elementwise_fourier
BENCH_RS=8 BENCH_NGAUSS=3 BENCH_RUNS=1 BENCH_WARMUPS=0 BENCH_SANITY=1e-1 OUT_DIR=/tmp/final-smoke cargo run --release --bin mpo_mpo_quantics
julia --project=julia julia/check_elementwise.jl /tmp/final-smoke 4
julia --project=julia julia/check_mpo_mpo.jl /tmp/final-smoke 8
```

Expected: everything green.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "docs: README, MIT license, CI workflow"
```

---

### Task 11 (manual, after review): first real run and publication

Not for a subagent; the user decides when.

- [ ] Run `scripts/run_all.sh mac-cpu` with full sweeps (default env), inspect reports, commit `result/mac-cpu/`.
- [ ] Create the public GitHub repo `tensor4all/tensor4all-benchmark` and push (`gh repo create tensor4all/tensor4all-benchmark --public --source . --push`). Requires user confirmation.

## Self-review notes

- Spec coverage: layout (Task 1), case 1 (Tasks 2 to 5), case 2 (Tasks 6 to 8), results workflow (Task 9), README/CI (Task 10), first committed results (Task 11). The spec's `benches/` directory became `src/bin/`, a deliberate simplification recorded here.
- Known uncertainty is confined to exact upstream option-struct field names at the pinned rev; each affected step names the exact source file to consult and limits the allowed adjustment.
- The treetn hadamard path for case 1 zipup/fit and the fused-order question in case 2 both carry explicit test-first fallbacks.
