#!/usr/bin/env python3
"""Render and validate the three maintained benchmark cases."""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED = {
    "elementwise_fourier": {"naive", "zipup", "fit", "aci"},
    "gaussian_elementwise": {"global_fit", "patched_fit", "global_aci", "patched_aci"},
    "gaussian_mpo_contraction": {"global_fit", "patched_fit"},
}


def metadata(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
    return result


def main(profile: Path) -> None:
    records = [json.loads(path.read_text()) for path in sorted((profile / "raw").glob("*.json"))]
    assert records, f"no JSON records in {profile / 'raw'}"
    grouped = defaultdict(list)
    for record in records:
        assert record["schema_version"] == 1
        assert record["case"] in EXPECTED, record["case"]
        assert record["algorithm"] in EXPECTED[record["case"]], record["algorithm"]
        grouped[record["case"]].append(record)
    assert set(grouped) == set(EXPECTED), set(grouped)
    for case, arms in EXPECTED.items():
        assert {record["algorithm"] for record in grouped[case]} == arms
        by_instance = defaultdict(list)
        for record in grouped[case]:
            key = record["params"].get("k_max", record["params"].get("n_gauss"))
            by_instance[key].append(record["algorithm"])
        for key, actual in by_instance.items():
            assert Counter(actual) == Counter(arms), (case, key, actual)
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
        "All timings exclude input construction, cache I/O, format conversion, patch preparation, output conversion and accuracy evaluation. Gaussian inputs use independent two-dimensional interpolative decompositions, balanced pairwise addition, final relative-L2/SVD tolerance `1e-6`, and fixed patch cap 128.",
        "",
        "## Case 1: Fourier elementwise",
        "",
        "| input χ | K | arm | time (s) | sampled relative L2 | output χ | parameters |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for record in sorted(grouped["elementwise_fourier"], key=lambda item: (item["input_max_bond_dim"], item["params"]["k_max"], item["algorithm"])):
        lines.append(
            f"| {record['input_max_bond_dim']} | {record['params']['k_max']} | {record['algorithm']} | {record['wall_time_median_secs']:.6f} | {record['max_error']:.3e} | {record['output_max_bond_dim']} | {record.get('n_params', '')} |"
        )
    for case, title in [
        ("gaussian_elementwise", "Case 2: Gaussian elementwise, global versus patched"),
        ("gaussian_mpo_contraction", "Case 3: Gaussian MPO-MPO contraction, global versus patched"),
    ]:
        lines += [
            "",
            f"## {title}",
            "",
            "| input χ | raw χ | N | R | arm | time (s) | sampled relative L2 | input patches | input max patch χ | output patches | parameters | speedup |",
            "|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for record in sorted(grouped[case], key=lambda item: (item["input_max_bond_dim"], item["algorithm"])):
            params = record["params"]
            raw_chi = max(params["raw_left_chi"], params["raw_right_chi"])
            input_patches = params.get("input_patch_count", params.get("left_input_patch_count", 0) + params.get("right_input_patch_count", 0))
            input_patch_chi = params.get("input_max_patch_chi", max(params.get("left_input_max_patch_chi", 0), params.get("right_input_max_patch_chi", 0)))
            speedup = params.get("speedup_vs_global", 1.0)
            lines.append(
                f"| {record['input_max_bond_dim']} | {raw_chi} | {params['n_gauss']} | {params['r']} | {record['algorithm']} | {record['wall_time_median_secs']:.6f} | {record['max_error']:.3e} | {input_patches} | {input_patch_chi} | {record.get('n_patches', 1)} | {record.get('n_params', '')} | {speedup:.3f} |"
            )
    (profile / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: report.py result/<profile>")
    main(Path(sys.argv[1]))
