#!/usr/bin/env python3
"""Generate the padded Gaussian patch-scaling and profiled timing report."""

import json
import math
import re
import sys
from pathlib import Path

DURATION = re.compile(r"([0-9.]+)(ns|µs|ms|s)")
SCALE = {"ns": 1e-9, "µs": 1e-6, "ms": 1e-3, "s": 1.0}


def seconds(text: str) -> float:
    match = DURATION.fullmatch(text)
    if match is None:
        raise ValueError(f"invalid Rust duration: {text}")
    return float(match.group(1)) * SCALE[match.group(2)]


def profile(path: Path) -> dict[str, float]:
    rows: list[dict[str, float | str]] = []
    current = None
    for line in path.read_text().splitlines():
        start = re.match(r"=== (\w+) Profiling ===", line)
        if start:
            if current is not None:
                rows.append(current)
            current = {"kind": start.group(1)}
            continue
        if current is not None:
            item = re.match(r"(zipup init|canonicalize|sweeps total):\s+([^ ]+)", line)
            if item:
                current[item.group(1)] = seconds(item.group(2))
    if current is not None:
        rows.append(current)
    contracts = [row for row in rows if row["kind"] == "contract_fit"]
    sums = [row for row in rows if row["kind"] == "fit_sum"]
    if not contracts:
        raise ValueError(f"no contract_fit profile in {path}")
    global_zipup = float(contracts[0].get("zipup init", 0.0))
    global_sweeps = float(contracts[0].get("sweeps total", 0.0))
    contribution = sum(
        float(row.get("zipup init", 0.0)) + float(row.get("sweeps total", 0.0))
        for row in contracts[1:]
    )
    fit_sum = sum(
        float(row.get("canonicalize", 0.0)) + float(row.get("sweeps total", 0.0))
        for row in sums
    )
    return {
        "global_zipup": global_zipup,
        "global_sweeps": global_sweeps,
        "contribution": contribution,
        "fit_sum": fit_sum,
        "contribution_count": len(contracts) - 1,
        "fit_sum_count": len(sums),
    }


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("result/linux-epyc-7713p-padded-r16")
    records = [json.loads(path.read_text()) for path in (root / "raw").glob("*.json")]
    scaling = sorted(
        (record for record in records if record["case"] == "gaussian_mpo_patch_scaling"),
        key=lambda record: record["params"]["n_gauss"],
    )
    operations = {
        (record["params"]["n_gauss"], record["algorithm"]): record
        for record in records
        if record["case"] == "gaussian_mpo_contraction"
    }
    lines = [
        "# Factor-4 padded Gaussian patch scaling and small-N MPO contraction",
        "",
        "All values use fixed `R=16`, patch cap 128, one pinned EPYC 7713P core, and single-threaded Rayon/BLAS. Gaussian centers occupy the central active box; the computational half-width is four times larger. Patch-scaling rows perform no MPO contraction. Operation timings exclude input generation, patch preparation, reference construction/cache I/O, output conversion, and validation.",
        "",
        "## N versus balanced patch layout",
        "",
        "| N | input χ (L/R) | (Px, PyL, PyR, Pz) | patches (L/R) | compatible contractions | output projectors | patch build (s) | retained output Gaussians | candidate / N² | retained cache estimate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in scaling:
        p = record["params"]
        lines.append(
            f"| {p['n_gauss']:,} | {p['left_chi']}/{p['right_chi']} | "
            f"({p['x_patch_count']}, {p['left_y_patch_count']}, {p['right_y_patch_count']}, {p['z_patch_count']}) | "
            f"{p['left_input_patch_count']}/{p['right_input_patch_count']} | {p['compatible_pair_count']} | "
            f"{p['output_projector_count']} | {p['patch_build_secs']:.3f} | "
            f"{p['integrated_retained_pair_count']:,} | "
            f"{p['integrated_candidate_pair_count'] / p['integrated_total_pair_count']:.3f} | "
            f"{p['integrated_estimated_component_bytes'] / 2**20:.1f} MiB |"
        )
    first, last = scaling[0]["params"], scaling[-1]["params"]
    survivor_exponent = math.log(
        last["integrated_retained_pair_count"] / first["integrated_retained_pair_count"]
    ) / math.log(last["n_gauss"] / first["n_gauss"])
    lines += [
        "",
        f"Across N={first['n_gauss']} to {last['n_gauss']}, retained integrated Gaussians scale empirically as approximately `N^{survivor_exponent:.2f}`. This is closer to the expected fixed-density y-overlap law `N^(3/2)` than to linear scaling; storing every retained component is already about {last['integrated_estimated_component_bytes'] / 2**30:.2f} GiB at N={last['n_gauss']:,}. The cell-list candidate fraction decreases with N, but the rigorous global `1e-12` tail extent is broad at these sizes.",
        "",
        "The patch count is a rank-cap staircase rather than a smooth power law: one patch at N=512, an asymmetric transition at N=1024, 4 patches per operand at N=2048, and 32 regular Cartesian patches per operand from N=4096 through N=12000. Therefore these data do not support a single continuous `N_p ∝ N^α` fit.",
        "",
        "## Profiled small-N contractions",
        "",
        "| N | input χ | global (s) | patched (s) | speedup | global error | patched error | compatible contributions | output patches |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in sorted({key[0] for key in operations}):
        global_record = operations[n, "global_fit"]
        patched_record = operations[n, "patched_fit"]
        p = patched_record["params"]
        lines.append(
            f"| {n:,} | {patched_record['input_max_bond_dim']} | "
            f"{global_record['wall_time_median_secs']:.3f} | {patched_record['wall_time_median_secs']:.3f} | "
            f"{global_record['wall_time_median_secs'] / patched_record['wall_time_median_secs']:.3f}x | "
            f"{global_record['max_error']:.3e} | {patched_record['max_error']:.3e} | "
            f"{p['compatible_pair_count']} | {p['output_projector_count']} |"
        )
    lines += [
        "",
        "## Fit/contraction timing breakdown",
        "",
        "| N | global zipup init (s) | global sweep (s) | contribution contractions (s) | fit_sum (s) | patched remainder (s) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for n in sorted({key[0] for key in operations}):
        values = profile(root / "profiles" / f"n{n}.log")
        patched = operations[n, "patched_fit"]["wall_time_median_secs"]
        remainder = patched - values["contribution"] - values["fit_sum"]
        lines.append(
            f"| {n:,} | {values['global_zipup']:.3f} | {values['global_sweeps']:.3f} | "
            f"{values['contribution']:.3f} ({int(values['contribution_count'])} calls) | "
            f"{values['fit_sum']:.3f} ({int(values['fit_sum_count'])} calls) | {remainder:.3f} |"
        )
    lines += [
        "",
        "At N=4096 the compatible contribution contractions dominate the patched arm; `fit_sum` is a small fraction. The current factor-4 padded input reaches χ≈263 at N=4096. Larger-N contraction timing was intentionally not run: this profile is for patch scaling plus repeatable small-N timing, with every measurement command bounded below ten minutes.",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
