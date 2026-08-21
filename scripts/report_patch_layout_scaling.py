#!/usr/bin/env python3
"""Report contraction-free balanced versus shared-y patch scaling."""

import json
import sys
from pathlib import Path


def ratio(a: int | float, b: int | float) -> float:
    return float(a) / float(b)


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "result/linux-epyc-7713p-patch-layout-scaling-r16"
    )
    records = [json.loads(path.read_text()) for path in (root / "raw").glob("*.json")]
    by_key = {
        (record["params"]["n_gauss"], record["params"]["input_l2_rtol"], record["algorithm"]): record
        for record in records
    }
    ns = sorted({key[0] for key in by_key if key[1] == 1e-6})
    lines = [
        "# Balanced versus shared-y-only patch scaling",
        "",
        "These are contraction-free measurements on factor-4 padded, fixed-`R=16` inputs. Both layouts use patch cap 128 and patch reconstruction tolerance `1e-6`. Operation times are not measured here; work proxies are structural sums over actual compatible patch pairs.",
        "",
        "## Default-compression N sweep",
        "",
        "| N | χ | balanced patches L/R | balanced pairs | balanced outputs | y-only patches L/R | y-only pairs | y-only outputs | pair ratio B/Y | parameter-product proxy ratio Y/B |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in ns:
        balanced = by_key[n, 1e-6, "balanced_xyz"]
        y_only = by_key[n, 1e-6, "shared_y_only"]
        b, y = balanced["params"], y_only["params"]
        lines.append(
            f"| {n:,} | {balanced['input_max_bond_dim']} | "
            f"{b['left_input_patch_count']}/{b['right_input_patch_count']} | "
            f"{b['compatible_pair_count']} | {b['output_projector_count']} | "
            f"{y['left_input_patch_count']}/{y['right_input_patch_count']} | "
            f"{y['compatible_pair_count']} | {y['output_projector_count']} | "
            f"{ratio(b['compatible_pair_count'], y['compatible_pair_count']):.2f} | "
            f"{ratio(y['compatible_parameter_product_proxy'], b['compatible_parameter_product_proxy']):.3f} |"
        )
    lines += [
        "",
        "## Local ranks, storage, and exact patch reconstruction error",
        "",
        "| N | layout | max χ L/R | median χ L/R | p90 χ L/R | saturated L/R | parameters L+R | reconstruction error L/R | build (s) | validation (s) |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in ns:
        for algorithm in ("balanced_xyz", "shared_y_only"):
            record = by_key[n, 1e-6, algorithm]
            p = record["params"]
            lines.append(
                f"| {n:,} | {algorithm} | "
                f"{p['left_input_max_patch_chi']}/{p['right_input_max_patch_chi']} | "
                f"{p['left_patch_chi_median']}/{p['right_patch_chi_median']} | "
                f"{p['left_patch_chi_p90']}/{p['right_patch_chi_p90']} | "
                f"{p['left_cap_saturated_patches']}/{p['right_cap_saturated_patches']} | "
                f"{p['left_input_patch_params'] + p['right_input_patch_params']:,} | "
                f"{p['left_patch_relative_error']:.3e}/{p['right_patch_relative_error']:.3e} | "
                f"{record['wall_time_median_secs']:.3f} | {p['patch_validation_secs']:.3f} |"
            )
    chi_records = sorted(
        (
            record
            for record in records
            if record["params"]["n_gauss"] == 12000
        ),
        key=lambda record: (record["input_max_bond_dim"], record["algorithm"]),
    )
    lines += [
        "",
        "## Fixed-N compressed-χ sweep (N=12,000)",
        "",
        "| input rtol | χ | layout | patches L/R | pairs | outputs | max χ L/R | parameter-product proxy | cubed max-bond proxy | reconstruction error max |",
        "|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in chi_records:
        p = record["params"]
        lines.append(
            f"| {p['input_l2_rtol']:.0e} | {record['input_max_bond_dim']} | {record['algorithm']} | "
            f"{p['left_input_patch_count']}/{p['right_input_patch_count']} | "
            f"{p['compatible_pair_count']} | {p['output_projector_count']} | "
            f"{p['left_input_max_patch_chi']}/{p['right_input_max_patch_chi']} | "
            f"{p['compatible_parameter_product_proxy']:,} | "
            f"{p['compatible_max_bond_product_cubed_proxy']:.3e} | "
            f"{max(p['left_patch_relative_error'], p['right_patch_relative_error']):.3e} |"
        )
    b = by_key[12000, 1e-6, "balanced_xyz"]["params"]
    y = by_key[12000, 1e-6, "shared_y_only"]["params"]
    lines += [
        "",
        "## Interpretation",
        "",
        "- At N=512 and N=1,024 the layouts are identical because no x/z refinement is selected; the balanced-versus-y-only comparison becomes nontrivial at N=2,048.",
        "- The observed balanced staircase reaches 32 patches/operand, 128 compatible pairs, and 16 output groups at N=4,096, then remains unchanged through N=12,000.",
        "- Shared-y-only reaches 16 patches/operand at N=4,096 and 32 at N=8,192. From N=8,192 onward it has 32 compatible pairs and one output group, versus balanced 128 and 16.",
        f"- At N=12,000, shared-y-only has {ratio(y['compatible_parameter_product_proxy'], b['compatible_parameter_product_proxy']):.3f}× the balanced parameter-product proxy and {ratio(y['compatible_max_bond_product_cubed_proxy'], b['compatible_max_bond_product_cubed_proxy']):.3f}× the cubed max-bond proxy. This suggests less pairwise contraction work, but it does not include the rank of each contracted contribution or the cost of fitting all y contributions into one global x/z output group.",
        "- Increasing compressed input χ from 282 to 583 at fixed N does not change either patch layout. The patch representation is truncated with its separate fixed `patch_input_rtol=1e-6`, so additional global-input precision is discarded before the hypothetical contraction.",
        "- Exact reconstruction errors are approximately 1.3e-6 to 4.8e-6 and therefore exceed the nominal 1e-6 patch tolerance. The records mark `patch_tolerance_met=false`; no contraction estimate should hide this effective patch-error floor.",
        "- The bounded N=16,384 attempt timed out during padded global-TCI input construction before producing a patch record. Therefore the next N-driven staircase is only bounded as greater than 12,000 in this single-core, 570-second workflow.",
        "",
        "The compatible-pair and structural proxy ratios are not measured contraction speedups.",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
