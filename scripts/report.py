#!/usr/bin/env python3
"""Render the three maintained benchmark cases from raw JSON records."""

import json
import sys
from collections import defaultdict
from pathlib import Path

EXPECTED = {
    "elementwise_fourier": {"naive", "zipup", "fit", "aci"},
    "gaussian_elementwise": {"global_fit", "patched_fit", "global_aci", "patched_aci"},
    "gaussian_mpo_contraction": {"global_fit", "patched_fit"},
}


def fmt(value):
    return f"{value:.3e}" if isinstance(value, float) else str(value)


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

    lines = [
        "# Tensor4all benchmark results",
        "",
        "All timings exclude input construction, cache I/O, format conversion, patch preparation, output conversion and accuracy evaluation. Gaussian inputs use independent two-dimensional interpolative decompositions, balanced pairwise addition, final relative-L2/SVD tolerance `1e-6`, and fixed patch cap 128.",
        "",
    ]
    for case, title in [
        ("elementwise_fourier", "Case 1: Fourier elementwise"),
        ("gaussian_elementwise", "Case 2: Gaussian elementwise, global versus patched"),
        ("gaussian_mpo_contraction", "Case 3: Gaussian MPO-MPO contraction, global versus patched"),
    ]:
        lines += [f"## {title}", ""]
        if case == "elementwise_fourier":
            lines += ["| input χ | arm | time (s) | error | output χ | parameters |", "|---:|---|---:|---:|---:|---:|"]
        else:
            lines += ["| input χ | arm | time (s) | sampled relative L2 | output χ | patches | parameters | speedup |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
        for record in sorted(grouped[case], key=lambda item: (item["input_max_bond_dim"], item["algorithm"])):
            if case == "elementwise_fourier":
                lines.append(
                    f"| {record['input_max_bond_dim']} | {record['algorithm']} | {record['wall_time_median_secs']:.6f} | {record['max_error']:.3e} | {record['output_max_bond_dim']} | {record.get('n_params', '')} |"
                )
            else:
                speedup = record["params"].get("speedup_vs_global", 1.0)
                lines.append(
                    f"| {record['input_max_bond_dim']} | {record['algorithm']} | {record['wall_time_median_secs']:.6f} | {record['max_error']:.3e} | {record['output_max_bond_dim']} | {record.get('n_patches', 1)} | {record.get('n_params', '')} | {speedup:.3f} |"
                )
        lines.append("")
    (profile / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: report.py result/<profile>")
    main(Path(sys.argv[1]))
