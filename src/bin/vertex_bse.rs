//! Case 4: the Bethe-Salpeter contraction Φ(ν,ν',ω) = Σ_ν₁ Γ(ν,ν₁,ω) χ0(ν₁,ω) F(ν₁,ν',ω) of impurity-model vertices,
//! zip-up and fit. Inputs: the fixed files of the `vertex-bse-inputs-v1` release (layout in its notes), downloaded once
//! into the input cache.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, ensure};
use t4a_bench::harness::time_median;
use t4a_bench::record::{write_record, RunRecord, SCHEMA_VERSION};
use tensor4all_core::SvdTruncationPolicy;
use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor, IndexLike, LinearizationOrder};
use tensor4all_treetn::contraction::ContractionOptions;
use tensor4all_treetn::{hadamard, partial_contract, PartialContractionSpec, TreeTN};

type Tree = TreeTN<IdxTensor, usize>;

const CASE: &str = "vertex_bse_contraction";
const URL: &str =
    "https://github.com/tensor4all/tensor4all-benchmark/releases/download/vertex-bse-inputs-v1";
const FILES: [(f64, &str); 3] = [
    (1e-4, "vertex_bse_tau1e-04.h5"),
    (1e-6, "vertex_bse_tau1e-06.h5"),
    (1e-8, "vertex_bse_tau1e-08.h5"),
];
/// Quantics bits per frequency.
const R: usize = 10;
const MAX_BOND: usize = 4096;

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// The cached input file, downloaded first if missing; returns the path and whether it was cached.
fn fetch(name: &str, cache: &Path) -> anyhow::Result<(PathBuf, bool)> {
    let path = cache.join("vertex_bse").join(name);
    if path.exists() {
        return Ok((path, true));
    }
    std::fs::create_dir_all(path.parent().unwrap())?;
    let partial = path.with_extension("partial");
    let status = std::process::Command::new("curl")
        .args(["-fsSL", "--retry", "3", "-o"])
        .arg(&partial)
        .arg(format!("{URL}/{name}"))
        .status()?;
    ensure!(status.success(), "downloading {name} failed");
    std::fs::rename(&partial, &path)?;
    Ok((path, false))
}

/// One train of the file with its fused sites split into legs of dimension 2 (the first leg fastest).
fn load(path: &str, name: &str, legs: &[&[DynIndex]]) -> anyhow::Result<Tree> {
    let fused = tensor4all_hdf5::load_mps(path, name)?.into_treetn();
    ensure!(fused.node_count() == R, "{name}: expected {R} sites");
    let mut tensors = Vec::with_capacity(R);
    for site in 0..R {
        let index = fused
            .site_space(&site)
            .and_then(|s| s.iter().next())
            .cloned();
        let index = index.ok_or_else(|| anyhow!("{name}: site {site} has no index"))?;
        ensure!(
            index.dim() == 1 << legs.len(),
            "{name}: wrong site dimension"
        );
        let tensor = fused.tensor(fused.node_index(&site).unwrap()).unwrap();
        ensure!(tensor.is_f64(), "{name}: not real double precision");
        let parts: Vec<DynIndex> = legs.iter().map(|series| series[site].clone()).collect();
        tensors.push(tensor.unfuse_index(&index, &parts, LinearizationOrder::ColumnMajor)?);
    }
    Ok(TreeTN::from_tensors(tensors, (0..R).collect())?)
}

fn series() -> Vec<DynIndex> {
    (0..R).map(|_| DynIndex::new_dyn(2)).collect()
}

fn max_bond(tree: &Tree) -> usize {
    tree.link_dims().into_iter().max().unwrap_or(1)
}

fn main() -> anyhow::Result<()> {
    let taus = std::env::var("BENCH_VERTEX_TAUS").unwrap_or_else(|_| "1e-4,1e-6".into());
    let algos = std::env::var("BENCH_ALGOS").unwrap_or_else(|_| "zipup,fit".into());
    let out_dir = PathBuf::from(env_or("OUT_DIR", "result/dev/raw".to_string()));
    let cache = PathBuf::from(env_or("BENCH_INPUT_CACHE_DIR", ".cache/inputs".to_string()));
    let (runs, warmups) = (
        env_or("BENCH_RUNS", 1usize),
        env_or("BENCH_WARMUPS", 0usize),
    );
    for tau in taus.split(',').map(|v| v.trim().parse::<f64>()) {
        let tau = tau?;
        let (_, name) = FILES
            .iter()
            .find(|(t, _)| (t / tau - 1.0).abs() < 1e-9)
            .ok_or_else(|| anyhow!("no input file for tau {tau:e}"))?;
        let (path, cache_hit) = fetch(name, &cache)?;
        let started = Instant::now();
        let file = path.to_str().unwrap();
        // Γ(ν, ν₁, ω) and F(ν₁, ν', ω), fused site index b_ω + 2 b_ν + 4 b_ν'; χ0(ν₁, ω): b_ω + 2 b_ν₁
        let (g_omega, g_nu, g_nu1) = (series(), series(), series());
        let (f_omega, f_nu1, f_nup) = (series(), series(), series());
        let (b_omega, b_nu1) = (series(), series());
        let gamma = load(file, "Gamma", &[&g_omega, &g_nu, &g_nu1])?;
        let full = load(file, "F", &[&f_omega, &f_nu1, &f_nup])?;
        let bubble = load(file, "bubble", &[&b_omega, &b_nu1])?;
        let h5 = hdf5_metno::File::open(&path)?;
        let best: f64 = h5.attr("contraction_error_of_inputs")?.read_scalar()?;
        let reference = |n: &str| h5.dataset(&format!("reference/{n}"))?.read_raw::<i64>();
        let points = [reference("omega")?, reference("nu")?, reference("nup")?];
        let exact: Vec<f64> = h5.dataset("reference/value")?.read_raw()?;
        ensure!(
            points.iter().all(|p| p.len() == exact.len()),
            "inconsistent reference"
        );
        let load_secs = started.elapsed().as_secs_f64();

        let weight_pairs: Vec<_> = (0..R)
            .flat_map(|s| {
                [
                    (g_omega[s].clone(), b_omega[s].clone()),
                    (g_nu1[s].clone(), b_nu1[s].clone()),
                ]
            })
            .collect();
        let join = PartialContractionSpec {
            contract_pairs: (0..R)
                .map(|s| (g_nu1[s].clone(), f_nu1[s].clone()))
                .collect(),
            diagonal_pairs: (0..R)
                .map(|s| (g_omega[s].clone(), f_omega[s].clone()))
                .collect(),
            output_order: None,
        };
        let rtol = 0.1 * tau;
        let policy = SvdTruncationPolicy::new(rtol * rtol / (R - 1) as f64)
            .with_squared_values()
            .with_discarded_tail_sum();
        for algo in algos.split(',').map(str::trim) {
            let options = match algo {
                "zipup" => ContractionOptions::zipup(),
                "fit" => ContractionOptions::fit()
                    .with_nfullsweeps(1)
                    .with_convergence_tol(rtol),
                _ => anyhow::bail!("unknown arm {algo}"),
            }
            .with_svd_policy(policy)
            .with_max_bond_dim(MAX_BOND);
            let (phi, timing) = time_median(warmups, runs, || -> anyhow::Result<Tree> {
                let weighted = hadamard(&gamma, &bubble, &weight_pairs, &0, options.clone())
                    .map_err(|e| anyhow!("{e}"))?;
                Ok(partial_contract(
                    &weighted,
                    &full,
                    &join,
                    &0,
                    options.clone(),
                )?)
            });
            let phi = phi?;

            // relative L2 error against the exact values; the result has legs ω, ν of Γ and ν' of F
            let indices: Vec<DynIndex> = (0..R)
                .flat_map(|s| [g_omega[s].clone(), g_nu[s].clone(), f_nup[s].clone()])
                .collect();
            let values: Vec<usize> = (0..exact.len())
                .flat_map(|p| (0..R).flat_map(move |s| (0..3).map(move |k| (p, s, k))))
                .map(|(p, s, k)| ((points[k][p] as usize) >> (R - 1 - s)) & 1)
                .collect();
            let shape = [indices.len(), exact.len()];
            let got = phi.evaluate(&indices, ColMajorArrayRef::new(&values, &shape)?)?;
            let diff: f64 = got
                .iter()
                .zip(&exact)
                .map(|(g, e)| (g.real() - e).powi(2))
                .sum();
            let error = (diff / exact.iter().map(|e| e * e).sum::<f64>()).sqrt();
            ensure!(
                error <= 10.0 * tau,
                "{algo} at tau {tau:e}: error {error:.3e}"
            );

            let n_params = (0..R)
                .map(|s| {
                    phi.tensor(phi.node_index(&s).unwrap())
                        .unwrap()
                        .dims()
                        .iter()
                        .product::<usize>()
                })
                .sum();
            let record = RunRecord {
                schema_version: SCHEMA_VERSION,
                case: CASE.into(),
                algorithm: algo.into(),
                params: serde_json::json!({
                    "input_tolerance": tau, "input_file": name, "cache_hit": cache_hit,
                    "gamma_chi": max_bond(&gamma), "f_chi": max_bond(&full), "bubble_chi": max_bond(&bubble),
                    "best_attainable_error": best, "external_error_metric": "reference_relative_l2",
                }),
                seed: 0,
                tolerance: rtol,
                wall_time_median_secs: timing.median_secs,
                wall_times_secs: timing.runs_secs,
                max_error: error,
                input_max_bond_dim: max_bond(&gamma).max(max_bond(&full)),
                output_max_bond_dim: max_bond(&phi),
                output_bond_dims: phi.link_dims(),
                n_params: Some(n_params),
                n_patches: None,
                max_patch_bond: None,
                rtol: Some(rtol),
                input_build_secs: Some(load_secs),
                n_allocations: Some(timing.alloc_calls_median),
                allocated_bytes: Some(timing.alloc_bytes_median),
            };
            write_record(&out_dir, &format!("{CASE}-{algo}-tau{tau:e}"), &record)?;
            eprintln!(
                "tau {tau:e} {algo}: {:.2} s, error {error:.3e} (best {best:.3e})",
                timing.median_secs
            );
        }
    }
    Ok(())
}
