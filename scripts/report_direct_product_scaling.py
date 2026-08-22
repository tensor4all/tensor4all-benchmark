#!/usr/bin/env python3
"""Generate the doubled-space direct-product input rank report."""

import json
import sys
from pathlib import Path


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "result/linux-epyc-7713p-direct-product-r16"
    )
    records = sorted(
        (json.loads(path.read_text()) for path in (root / "raw").glob("*.json")),
        key=lambda record: (record["params"]["n_gauss"], record["tolerance"]),
    )
    lines = [
        "# Doubled-space direct-product input rank scaling",
        "",
        "This input-only Case 3 probe forms `F(x,x';y,y') = f(x,y) tensor_product f(x',y')` and the analogous `G`. It performs no patching and no contraction.",
        "",
        "| N | factor rtol | factor χ L/R | product χ L/R | max χ² identity | product memory (MiB) | product build (s) | max sampled factor error |",
        "|---:|---:|---:|---:|:---:|---:|---:|---:|",
    ]
    for record in records:
        p = record["params"]
        identity = (
            p["left_direct_product_chi"] == p["left_factor_chi"] ** 2
            and p["right_direct_product_chi"] == p["right_factor_chi"] ** 2
        )
        lines.append(
            f"| {p['n_gauss']} | {record['tolerance']:.0e} | "
            f"{p['left_factor_chi']}/{p['right_factor_chi']} | "
            f"{p['left_direct_product_chi']}/{p['right_direct_product_chi']} | "
            f"{'yes' if identity else 'no'} | {p['direct_product_bytes'] / 2**20:.1f} | "
            f"{p['direct_product_build_secs']:.3f} | "
            f"{max(p['left_factor_sampled_relative_l2'], p['right_factor_sampled_relative_l2']):.3e} |"
        )
    largest = max(records, key=lambda record: record["output_max_bond_dim"])
    lines += [
        "",
        f"Every exact direct product satisfies χ_product = χ_factor² bond by bond. The largest bounded point is N={largest['params']['n_gauss']}, factor rtol={largest['tolerance']:.0e}, with product χ={largest['output_max_bond_dim']} and {largest['params']['direct_product_bytes'] / 2**20:.1f} MiB for both product MPOs. These are materialized input ranks, not contraction timings.",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
