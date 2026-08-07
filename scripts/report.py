#!/usr/bin/env python3
"""Render Markdown reports and SVG scaling plots from RunRecord JSON files.

Usage: uv run scripts/report.py result/<profile>
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

X_AXIS = {
    # case name -> (record field used as x, axis label)
    "elementwise_fourier": ("input_max_bond_dim", "input bond dimension chi"),
    "mpo_mpo_quantics": ("input_max_bond_dim", "input bond dimension chi"),
    "elementwise_gauss2d": ("input_max_bond_dim", "input bond dimension chi"),
}

ERROR_LABEL = {
    # case name -> what `max_error` in the records actually measures
    "elementwise_fourier": "max abs error",
    "mpo_mpo_quantics": "max relative error",
    "elementwise_gauss2d": "max relative error",
}

NOTES = {
    # case name -> caveat emitted under the summary table
    "mpo_mpo_quantics": (
        "Note: every algorithm contracts at the same output budget, its "
        "maximum bond dimension capped at the input rank chi, so the error "
        "column is the discriminator. naive and zipup_simplett run on the "
        "simplett engine, zipup_treetn and fit_treetn on treetn; both engines "
        "truncate relative to the largest singular value at the pinned "
        "revision. The two zipup arms are the same algorithm on the two "
        "engines, so their difference isolates the engine, and it is now "
        "confined to wall time. "
        "The fitted time exponent is measured against input chi "
        "along a sweep of r, where the site count also grows, so it is not "
        "a pure chi power law."
    ),
    "elementwise_gauss2d": (
        "Note: every algorithm forms the product at the same output budget, "
        "its maximum bond dimension capped at the input rank chi, so the error "
        "column is the discriminator. The exact elementwise product has rank up "
        "to chi squared, so this budget is tight: naive, fit_treetn and aci "
        "stay near the working tolerance while zipup_treetn spends the whole "
        "budget and still returns an order-unity relative error. Raising the "
        "budget recovers it, so that is the price of the fixed budget rather "
        "than a broken arm. There is no simplett arm here: simplett exposes no "
        "elementwise product for tensor trains at the pinned revision, so this "
        "case cannot compare the two engines on one algorithm the way case 2 "
        "does. The engine that ran each arm is recorded as engine: local for "
        "naive, treetn for the two hadamard arms, aci for the cross "
        "interpolation. "
        "The fitted time exponent is measured against input chi along a sweep "
        "of r, where the site count also grows, so it is not a pure chi power "
        "law."
    ),
}


def load(profile_dir: Path):
    cases = defaultdict(lambda: defaultdict(list))
    for path in sorted((profile_dir / "raw").glob("*.json")):
        rec = json.loads(path.read_text())
        assert rec["schema_version"] == 1, f"unknown schema in {path}"
        if "algorithm" not in rec:
            continue  # not a RunRecord (e.g. exported instance sidecar)
        cases[rec["case"]][rec["algorithm"]].append(rec)
    return cases


def fit_exponent(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    mask = (xs > 0) & (ys > 0)
    if mask.sum() < 2:
        return float("nan")
    if np.ptp(np.log(xs[mask])) < 1e-9:
        return float("nan")
    p = np.polyfit(np.log(xs[mask]), np.log(ys[mask]), 1)
    return p[0]


def render_case(case, algos, profile_dir: Path):
    xfield, xlabel = X_AXIS[case]
    elabel = ERROR_LABEL.get(case, "max error")
    lines = [f"# {case}", "",
             f"| algorithm | points | fitted time exponent | worst {elabel} |",
             "|---|---|---|---|"]
    fig_t, ax_t = plt.subplots(figsize=(5, 4))
    fig_e, ax_e = plt.subplots(figsize=(5, 4))
    for algo, recs in sorted(algos.items()):
        recs = sorted(recs, key=lambda rec: rec[xfield])
        xs = [rec[xfield] for rec in recs]
        ts = [rec["wall_time_median_secs"] for rec in recs]
        es = [rec["max_error"] for rec in recs]
        expo = fit_exponent(xs, ts)
        lines.append(f"| {algo} | {len(recs)} | {expo:.2f} | {max(es):.2e} |")
        ax_t.loglog(xs, ts, "o-", label=f"{algo} (chi^{expo:.1f})")
        ax_e.loglog(xs, es, "o-", label=algo)
    if case in NOTES:
        lines += ["", NOTES[case]]
    for ax, ylab in ((ax_t, "median wall time [s]"), (ax_e, elabel)):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylab)
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
    fig_t.tight_layout()
    fig_e.tight_layout()
    fig_t.savefig(profile_dir / f"{case}-time.svg")
    fig_e.savefig(profile_dir / f"{case}-error.svg")
    plt.close(fig_t)
    plt.close(fig_e)
    lines += ["", f"![time](./{case}-time.svg)", "", f"![error](./{case}-error.svg)", ""]
    (profile_dir / f"{case}.md").write_text("\n".join(lines))
    print(f"wrote {profile_dir / (case + '.md')}")


def main():
    profile_dir = Path(sys.argv[1])
    cases = load(profile_dir)
    if not cases:
        sys.exit(f"no records under {profile_dir}/raw")
    for case, algos in cases.items():
        render_case(case, algos, profile_dir)


if __name__ == "__main__":
    main()
