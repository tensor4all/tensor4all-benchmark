# tensor4all-benchmark design

Date: 2026-08-04
Status: draft for review

## Purpose

Public repository (`github.com/tensor4all/tensor4all-benchmark`, MIT) serving as an
open experimentation ground for comparing tensor-network contraction algorithms
implemented in tensor4all-rs. Primary audience: ourselves and third parties who want
reproducible algorithm comparisons (fitting vs zipup vs ACI vs naive) on well-defined
problem instances. It is not a low-level kernel benchmark (that is tenferro-benchmark)
and not a cross-framework showdown.

## Scope of the initial version

Two benchmark cases, implemented in this order:

1. **Elementwise multiplication of random Fourier series (QTT)**, following the setup
   of Ritter, arXiv:2604.00037 (ACI paper).
2. **General MPO-MPO contraction on 2D quantics Gaussians**, with an analytically
   known result.

Out of scope for the initial version: GPU, TreeTN beyond a possible same-problem run
in case 2, cross-language performance comparison (Julia is used for correctness
reference only), CI-based full measurement runs.

## Stack and repository layout

Cargo workspace, standalone repository. tensor4all-rs is consumed as a pinned git
dependency (switch to crates.io versions once published). Report generation in Python
managed by uv. Modeled on tenferro-benchmark but smaller.

```
tensor4all-benchmark/
  README.md              # purpose, latest result summary, links to reports
  Cargo.toml             # workspace, tensor4all-rs pinned via git
  src/                   # shared: problem generators, timing harness, JSON schema
  benches/elementwise_fourier/   # case 1 runner (binary, not criterion)
  benches/mpo_mpo_quantics/      # case 2 runner
  julia/                 # Julia reference implementation for correctness checks
  scripts/               # run_all.sh, uv Python: JSON -> Markdown report + SVG plots
  result/<profile>/      # latest committed reports; history lives in git
  docs/
```

Runners are plain binaries emitting JSON (one file per run, schema versioned), not
criterion benches: we need custom sweeps, error metrics, and rank reporting.

## Case 1: elementwise multiplication, random Fourier series

Problem definition (per arXiv:2604.00037):

- Two band-limited random Fourier series on [0,1), K+1 modes each. Complex
  coefficients with real and imaginary parts drawn independently from U[0,1],
  normalized so that sum_k |g_k|^2 = 1. Fixed RNG seed per instance; the same
  instance is fed to every algorithm.
- Each series is represented as a QTT (R bits, resolution 2^R chosen large enough
  that the series is exactly representable up to numerical precision). Band limit
  implies QTT bond dimension chi in O(sqrt(K)).
- Task: compute the elementwise product as a QTT to tolerance tau (default 1e-8).

Sweep: K over a geometric grid (thus chi from ~10 to ~few hundred, machine
permitting).

Algorithms (all from tensor4all-rs):

- naive contraction + SVD truncation (baseline, O(chi^4))
- zipup (tensor4all-simplett)
- fitting / variational (tensor4all-itensorlike / simplett fit)
- ACI (tensor4all-aci)

Measured per (algorithm, K, seed):

- wall time (median of N repeats, N configurable, warmup separate)
- output bond dimension profile (max and per-bond)
- max error against the exact product, evaluated via the exact Fourier
  coefficients of the product series (convolution of the input coefficient
  vectors), sampled on a fixed set of points
- fitted scaling exponent of time vs chi reported in the summary

## Case 2: MPO-MPO contraction, 2D quantics Gaussians

Problem definition:

- f(x,y) = sum_i a_i exp(-alpha_i |(x,y) - r_i|^2), a random mixture of Gaussians
  (positions, weights, widths drawn with a fixed seed; count and width range are
  sweep parameters controlling rank). g(y,z) is an independent instance.
- f and g are encoded as 2D quantics MPOs on a box [-L, L]^2 with R bits per axis.
- Task: MPO-MPO contraction h(x,z) = sum_y f(x,y) g(y,z) * Delta_y, i.e. the
  discretized integral over the shared variable.
- Exact reference: for L large and R large, h converges to the closed-form Gaussian
  integral int f(x,y) g(y,z) dy (Gaussian x Gaussian in y integrates analytically).
  Error is measured against this closed form on sample points; box truncation and
  discretization error are controlled by L and R and reported alongside.

Sweep: R (bits per axis), number of Gaussians / width range.

Algorithms: naive contraction + SVD, zipup, fitting (TT/MPO forms in
tensor4all-simplett / itensorlike). A TreeTN run of the same problem is a stretch
goal, not required for the initial version.

Measured: wall time, max relative error vs analytic result, output bond dimension.

## Correctness reference (Julia)

Inputs that must be generated on the fly are generated in Rust only; there is no
duplicate instance construction on the Julia side. The Rust runner writes the input
TT/MPO objects (and the computed results where useful) in ITensor-compatible HDF5
format via tensor4all-hdf5. `julia/` contains a small script (ITensors / Quantics.jl
based) that reads those HDF5 files and checks agreement within tolerance (e.g.
contracting or sampling the loaded objects against the analytic reference). This is a
test, run manually or in CI smoke, not a performance comparison.

## Results workflow

- `scripts/run_all.sh <profile>` runs all cases, writes raw JSON under
  `result/<profile>/raw/`, then invokes the Python report generator.
- Reports: `result/<profile>/elementwise_fourier.md` and
  `result/<profile>/mpo_mpo_quantics.md`, each with a summary table and scaling
  plots (time vs chi, error vs chi) as committed SVGs. README links to the latest
  reports. Historical results live in git history only.
- Initial profile: `mac-cpu` (Apple Silicon, Accelerate). Profile metadata (machine,
  commit hashes of this repo and the tensor4all-rs pin, thread count) recorded in a
  `run.yaml` next to the reports.
- CI (GitHub Actions): build + smoke run (smallest instance, 1 repeat) + Julia
  correctness check. No performance numbers from CI are ever committed.

## Error handling and reproducibility

- All randomness behind explicit seeds recorded in the JSON output.
- Runner fails loudly (nonzero exit) if any algorithm's error exceeds a sanity bound
  (10x the requested tolerance) so silently-wrong timings never enter reports.
- JSON schema carries a version field; report generator rejects unknown versions.

## Testing

Test format stays deliberately flexible: no rigid harness or fixed fixture layout is
imposed. Plain `#[test]` functions, per-case smoke binaries, and standalone Julia
scripts are all acceptable; tests may take tolerances and instance sizes from
environment variables or arguments so the same test scales from CI smoke to local
deep checks.

- Unit tests for problem generators (Fourier coefficients normalization, Gaussian
  analytic integral against numerical quadrature at small R).
- One end-to-end smoke test per case (tiny instance, all algorithms, error within
  tolerance).
- Julia cross-check via ITensor-compatible HDF5 as above.

## Implementation order

1. Repo scaffold, workspace, shared harness + JSON schema, report generator skeleton.
2. Case 1 end to end (generator, 4 algorithms, report, smoke test, Julia check).
3. Case 2 end to end.
4. README with first committed mac-cpu results.
