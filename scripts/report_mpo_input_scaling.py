#!/usr/bin/env python3
"""Render the case-6 anisotropic MPO input scaling report from raw JSON records."""

import json
import sys
from pathlib import Path


def pair(values, integer=False):
    if integer:
        return f"({values[0]:,}, {values[1]:,})"
    return f"({values[0]:.2e}, {values[1]:.2e})"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: report_mpo_input_scaling.py result/<profile>")
    profile = Path(sys.argv[1])
    paths = sorted(
        (profile / "raw").glob("mpo_mpo_aniso_input-tci-n*-chi*.json"),
        key=lambda path: json.loads(path.read_text())["n_gauss"],
    )
    if not paths:
        raise SystemExit(f"no case-6 input records under {profile / 'raw'}")
    records = [json.loads(path.read_text()) for path in paths]
    expected_caps = {512: 512, 4096: 512, 20480: 1280, 100000: 3072, 125000: 3584}
    assert {record["n_gauss"] for record in records} == set(expected_caps)
    for record in records:
        assert record["case"] == "mpo_mpo_aniso_input"
        assert record["input_generator"] == "tci"
        assert record["sigma"] == 0.05
        assert record["rho_max"] == 8.0
        assert record["spacing"] == 3.0
        assert record["box_padding"] == 1.0
        assert record["r_extra"] == 0
        assert record["seed"] == 0
        assert record["input_tci_cap"] == expected_caps[record["n_gauss"]]
        assert record["input_tci_rtol"] == 1e-8
        assert record["input_tci_local_abs_tol"] == 1e-12
        assert record["input_tci_initial_pivots"] == 8
        assert record["input_svd_l2_rtol"] == 1e-6
        assert record["patch_cap"] == 128
        assert record["max_input_chi"] == 1400
        assert record["input_error_samples"] == 256
        assert not record["input_cache_hit"]
        assert max(record["input_sampled_relative_errors"]) <= record["input_sanity"]

    lines = [
        "# Anisotropic Gaussian MPO input and patch scaling",
        "",
        "This focused case-6 input study measures adaptive patching through compressed input bond dimension `chi_in ≈ 1000`. It uses the unchanged constant-density random anisotropic family (`sigma_minor = 0.05`, log-uniform aspect ratio in `[1, 8]`, independent angle, center, and positive weight per Gaussian).",
        "",
        "## Construction and accuracy",
        "",
        "The input pipeline is:",
        "",
        "1. fused two-variable TCI at relative tolerance `1e-8`;",
        "2. global relative-L2 SVD truncation at `1e-6`;",
        "3. adaptive input truncation with per-patch cap 128.",
        "",
        "TCI evaluates the same random mixture through a spatial index. A component is omitted only outside a radius where its individual Gaussian is below a common exponent threshold. Since all weights are positive, the sum of all omitted tails is bounded pointwise by `1e-12`. The committed sampled input errors include this bound and remain below `1e-5` after global SVD truncation.",
        "",
        "The alternative multiscale constructor builds one randomly rotated Gaussian directly. It marks points every half minor-axis standard deviation along the major-axis ridge as unsafe. The focused test covers `rho = 8`, a 45-degree rotation, and `R = 10`, and requires maximum sampled absolute error below `5e-8` at polynomial degree 28. Mixtures can be formed by adding these single-Gaussian QTTs with intermediate SVD compression. This path preserves the instance family but is not used for the large sweep because its per-Gaussian construction cost scales linearly with `N`; spatially indexed TCI reaches the requested rank regime efficiently.",
        "",
        "## Reproducible settings",
        "",
        "These input-only records were collected separately from the profile's earlier contraction sweep. The settings below, including core 0, are authoritative for these records rather than the contraction affinity in `run.yaml`.",
        "",
        "- Machine: AMD EPYC 7713P 64-Core Processor",
        "- CPU affinity: core 0 (`taskset -c 0`)",
        "- `RAYON_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`",
        "- `BENCH_INPUT_GENERATOR=tci`",
        "- `BENCH_INPUT_TCI_RTOL=1e-8`",
        "- `BENCH_TCI_LOCAL_ABS_TOL=1e-12`",
        "- `BENCH_TCI_INITIAL_PIVOTS=8` (deterministic center-derived pivots)",
        "- `BENCH_INPUT_SVD_RTOL=1e-6`",
        "- `BENCH_PATCH_MAX_BOND=128`",
        "- `BENCH_MAX_INPUT_CHI=1400`",
        "- `BENCH_MAX_BOND=512, 512, 1280, 3072, 3584` for `N=512, 4096, 20480, 100000, 125000`",
        "- one deterministic seed (`BENCH_SEED=0`)",
        "- `BENCH_ERROR_SAMPLES=256` at every size",
        "",
        "Input generation, global SVD truncation, sampled validation, and adaptive patch preparation are all included in `preparation time`; no contraction is run (`BENCH_INPUT_ONLY=1`).",
        "",
        "## Results",
        "",
        "Each pair is `(left, right)` for the two independently drawn mixtures.",
        "",
        "| N | R | raw chi | compressed chi | global parameters | patches | max patch chi | patch parameters | sampled relative error | preparation time |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            f"| {record['n_gauss']:,} | {record['r']} | {record['raw_input_chi']:,} | "
            f"{record['input_chi']:,} | {pair(record['input_params'], True)} | "
            f"{pair(record['input_patch_counts'], True)} | "
            f"{pair(record['input_patch_max_bonds'], True)} | "
            f"{pair(record['input_patch_params'], True)} | "
            f"{pair(record['input_sampled_relative_errors'])} | "
            f"{record['input_build_secs']:,.2f} s |"
        )
    lines += [
        "",
        "The compressed-rank plateau reaches `chi_in = 900`, within 10% of the requested `chi_in ≈ 1000` regime. Patch count grows stepwise from 1 to 6, 30, and 120 while every patch remains below cap 128. The `N = 125,000`, `R = 13` point increases raw rank and parameter count but remains on the same compressed-rank and patch-count plateau, so increasing `N` alone does not produce a smooth patch-count curve.",
        "",
        "Raw JSON records are in [`raw/`](raw/) with filenames `mpo_mpo_aniso_input-tci-n*-chi*.json`.",
        "",
    ]
    (profile / "mpo_mpo_aniso_input_scaling.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
