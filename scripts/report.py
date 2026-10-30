#!/usr/bin/env python3
"""Render and validate the three maintained benchmark cases."""

import json
import sys
from collections import defaultdict
from pathlib import Path

# Arms each case measures at the pinned tensor4all-rs revision. The patched arms
# are paused: tensor4all-rs#1005 removed tensor4all-partitionedtt, which they are
# built on, and the port is tracked in
# https://github.com/tensor4all/tensor4all-benchmark/issues/11. A sweep recorded
# before the pause still carries them, so PAUSED arms are accepted but not
# required.
MEASURED = {
    "elementwise_fourier": {"naive", "zipup", "fit", "aci"},
    "gaussian_elementwise": {"global_fit", "global_aci"},
    "gaussian_mpo_contraction": {"global_fit"},
}
PAUSED = {
    "elementwise_fourier": set(),
    "gaussian_elementwise": {"patched_fit", "patched_aci"},
    "gaussian_mpo_contraction": {"patched_fit"},
}


def alloc_calls(record: dict) -> str:
    """Allocation count, blank on records written before the counters existed."""
    value = record.get("n_allocations")
    return "" if value is None else str(value)


def alloc_mib(record: dict) -> str:
    value = record.get("allocated_bytes")
    return "" if value is None else f"{value / 2**20:.2f}"


def metadata(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
    return result


def input_patch_count(params: dict) -> object:
    """Input patch count, blank on records written while the patched arms are paused."""
    if "input_patch_count" in params:
        return params["input_patch_count"]
    if "left_input_patch_count" in params:
        return params.get("left_input_patch_count", 0) + params.get(
            "right_input_patch_count", 0
        )
    return ""


def input_patch_chi(params: dict) -> object:
    """Largest input patch bond dimension, blank while the patched arms are paused."""
    if "input_max_patch_chi" in params:
        return params["input_max_patch_chi"]
    if "left_input_max_patch_chi" in params:
        return max(
            params.get("left_input_max_patch_chi", 0),
            params.get("right_input_max_patch_chi", 0),
        )
    return ""


def main(profile: Path) -> None:
    records = [json.loads(path.read_text()) for path in sorted((profile / "raw").glob("*.json"))]
    assert records, f"no JSON records in {profile / 'raw'}"
    grouped = defaultdict(list)
    for record in records:
        assert record["schema_version"] == 1
        assert record["case"] in MEASURED, record["case"]
        case = record["case"]
        assert record["algorithm"] in MEASURED[case] | PAUSED[case], record["algorithm"]
        grouped[case].append(record)
    assert set(grouped) == set(MEASURED), set(grouped)
    for case, arms in MEASURED.items():
        assert arms <= {record["algorithm"] for record in grouped[case]}, set(grouped[case])
        by_instance = defaultdict(list)
        for record in grouped[case]:
            key = record["params"].get("k_max", record["params"].get("n_gauss"))
            by_instance[key].append(record["algorithm"])
        for key, actual in by_instance.items():
            assert arms <= set(actual), (case, key, actual)
            assert len(actual) == len(set(actual)), (case, key, actual)
    for record in grouped["gaussian_elementwise"] + grouped["gaussian_mpo_contraction"]:
        params = record["params"]
        assert params["patch_cap"] == 128
        assert params["input_l2_rtol"] == 1e-6
        assert params["external_error_metric"] == "sampled_relative_l2"
        assert record["max_error"] <= 1e-4
    run = metadata(profile / "run.yaml")
    lines = [
        "# Tensor4all benchmark results",
        "",
        f"Profile: `{run['profile']}`. CPU: {run['chip']}. Threads: {run['threads']}. CPU affinity: {run['cpu_affinity']}. Source revision: `{run['repo_rev']}`. tensor4all-rs revision: `{run['tensor4all_rs_rev']}`.",
        "",
        "The source revision identifies the clean code that was measured. The commit adding these generated records necessarily follows that revision.",
        "",
        "All timings exclude input construction, cache I/O, format conversion, patch preparation, output conversion and accuracy evaluation. Gaussian inputs use independent two-dimensional interpolative decompositions, balanced pairwise addition, final relative-L2/SVD tolerance `1e-6`, and fixed patch cap 128.",
        "",
        "## Case 1: Fourier elementwise",
        "",
        "| input χ | K | arm | time (s) | sampled relative L2 | output χ | parameters | allocations | allocated (MiB) |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for record in sorted(grouped["elementwise_fourier"], key=lambda item: (item["input_max_bond_dim"], item["params"]["k_max"], item["algorithm"])):
        lines.append(
            f"| {record['input_max_bond_dim']} | {record['params']['k_max']} | {record['algorithm']} | {record['wall_time_median_secs']:.6f} | {record['max_error']:.3e} | {record['output_max_bond_dim']} | {record.get('n_params', '')} | {alloc_calls(record)} | {alloc_mib(record)} |"
        )
    for case, title in [
        ("gaussian_elementwise", "Case 2: Gaussian elementwise, global versus patched"),
        ("gaussian_mpo_contraction", "Case 3: Gaussian MPO-MPO contraction, global versus patched"),
    ]:
        lines += [
            "",
            f"## {title}",
            "",
            "| input χ | raw χ | N | R | arm | time (s) | sampled relative L2 | input patches | input max patch χ | output patches | parameters | allocations | allocated (MiB) | speedup |",
            "|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for record in sorted(grouped[case], key=lambda item: (item["input_max_bond_dim"], item["algorithm"])):
            params = record["params"]
            raw_chi = max(params["raw_left_chi"], params["raw_right_chi"])
            speedup = params.get("speedup_vs_global", 1.0)
            lines.append(
                f"| {record['input_max_bond_dim']} | {raw_chi} | {params['n_gauss']} | {params['r']} | {record['algorithm']} | {record['wall_time_median_secs']:.6f} | {record['max_error']:.3e} | {input_patch_count(params)} | {input_patch_chi(params)} | {record.get('n_patches', 1)} | {record.get('n_params', '')} | {alloc_calls(record)} | {alloc_mib(record)} | {speedup:.3f} |"
            )
    (profile / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: report.py result/<profile>")
    main(Path(sys.argv[1]))
