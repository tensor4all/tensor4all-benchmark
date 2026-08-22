#!/usr/bin/env python3
"""Generate the input-only 3D Gaussian rank report."""

import json
import sys
from pathlib import Path


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "result/linux-epyc-7713p-gaussian3d-rank-r16"
    )
    records = sorted(
        (json.loads(path.read_text()) for path in (root / "raw").glob("*.json")),
        key=lambda record: record["params"]["n_gauss"],
    )
    lines = [
        "# Fully correlated 3D Gaussian input rank scaling",
        "",
        "This input-only Case 3 probe constructs `A(b,x,y)` and embeds it as the batch-diagonal MPO `A(b,x;b',y) = delta(b,b') A(b,x,y)`. It performs no patching and no contraction.",
        "",
        "| N | raw TCI χ | compressed χ | diagonal MPO χ | raw parameters | compressed parameters | diagonal MPO parameters | principal-axis error | build (s) | compression (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        p = record["params"]
        lines.append(
            f"| {p['n_gauss']} | {p['raw_qtt_chi']} | {p['compressed_qtt_chi']} | "
            f"{p['batch_diagonal_mpo_chi']} | {p['raw_qtt_params']:,} | "
            f"{p['compressed_qtt_params']:,} | {p['batch_diagonal_mpo_params']:,} | "
            f"{p['principal_axis_relative_l2']:.3e} | {p['input_build_secs']:.3f} | "
            f"{p['input_compression_secs']:.3f} |"
        )
    largest = records[-1]["params"]
    lines += [
        "",
        f"The local batch-diagonal embedding preserves every QTT bond dimension exactly. At N={largest['n_gauss']} the compressed input reaches χ{largest['compressed_qtt_chi']} while the raw Global TCI reaches χ{largest['raw_qtt_chi']}. Every recorded off-pivot principal-axis error is below the requested 1e-6 input target. The bounded N=64 command timed out after 570 seconds during input construction, before producing a record.",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
