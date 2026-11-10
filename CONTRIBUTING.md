<!-- unified contribution policy: ../../docs/contributing/rust-repositories.md in tensor4all/tensor4all-meta -->
# Contributing to tensor4all-benchmark

The rules that apply to every Rust repository in `tensor4all` are in one place:
**[tensor4all-meta · docs/contributing/rust-repositories.md](https://github.com/tensor4all/tensor4all-meta/blob/main/docs/contributing/rust-repositories.md)**.
Read that first — it covers the issue-first workflow, what a pull request body has to contain, red-first tests, the
gates every repository runs, the review requirement, and the version policy.

This file only records what is specific to `tensor4all-benchmark`:

## Local facts

- **MSRV**: none is declared — no `rust-version` in `Cargo.toml`. CI pins `stable`, so there is no MSRV claim and no
  `msrv` job.
- **Feature combinations CI builds**: one. The package declares no features, and every step uses `--locked` against
  the committed `Cargo.lock`. After the test run, three smoke invocations exercise the binaries themselves
  (`elementwise_fourier`, `elementwise_gauss2d_patched`, `mpo_mpo_aniso_patched`) with small `BENCH_*` settings and
  `OUT_DIR=/tmp/smoke/raw`.
- **Platforms claimed**: `ubuntu-latest` only; the job has a 60-minute timeout.
- **Local gate**: `cargo test --release --locked`. CI installs two system libraries first —
  `sudo apt-get update && sudo apt-get install -y libhdf5-dev liblapack-dev` — and `--locked` is a real part of the
  gate: without it a stale `Cargo.lock` builds here and fails for anyone building with `--locked`.

  The dependencies are git revisions of `tensor4all/tensor4all-rs`, so the first build needs network access.
  `libhdf5-dev` is the one the test target needs; measured on a host with `libhdf5-dev` installed and `liblapack-dev`
  absent, `cargo test --release --locked` was green, so the LAPACK package is not required by the gate above.

## Anything here that departs from the shared rules

Two do:

- There is no `msrv` gate and no `rust-version` to hold one to.
- The benchmark itself is not a gate: the workflow only smokes three binaries with `BENCH_RUNS=1`, so a performance
  claim in a pull request needs its own recorded run, as the shared page's evidence section requires.
