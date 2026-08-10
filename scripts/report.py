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
    "elementwise_gauss2d_scaling": ("input_max_bond_dim", "input bond dimension chi"),
}

ERROR_LABEL = {
    # case name -> what `max_error` in the records actually measures
    "elementwise_fourier": "max abs error",
    "mpo_mpo_quantics": "max relative error",
    "elementwise_gauss2d": "max relative error",
    "elementwise_gauss2d_scaling": "max relative error",
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
    "elementwise_gauss2d_scaling": (
        "Note: this case is the density-constant scaling study of "
        "elementwise_gauss2d. The number of Gaussians N is swept while the box "
        "area grows proportionally to N, box half-width L = L0 sqrt(N / N0), so "
        "the Gaussians per unit area stay fixed, and the bit count grows with "
        "the box, R = R0 + round(log2(L / L0)), so the grid spacing and hence "
        "the resolution per Gaussian stay roughly constant. The quantity of "
        "interest is the input rank chi_in as a function of N, reported in the "
        "instance table and the chi plot below. The elementwise product itself "
        "runs at the same fixed output budget chi_out <= chi_in as case 3. The "
        "naive arm of case 3 is excluded here: it forms the full chi_in-squared "
        "bond before truncating, which dominates the sweep at these ranks "
        "without adding a conclusion, since it tracks fit_treetn to the last "
        "reported digit in case 3. As in case 3, zipup_treetn spends the whole "
        "budget and still returns an order-unity relative error. "
        "The fitted time exponent is measured against input chi along a sweep "
        "of N, where the site count also grows, so it is not a pure chi power "
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


def render_gauss2d_scaling_extra(case, algos, profile_dir: Path):
    """Case-4 only: the rank-versus-N answer the generic tables cannot show.

    The generic path plots everything against chi_in, which is the x axis of
    every other case. Here chi_in is the dependent variable and N is the knob,
    so this adds the instance table, the fitted exponent of chi_in against N,
    and a log-log chi_in versus N plot. Returns extra Markdown lines.
    """
    # One instance per N, shared by every arm, so read it off any one arm.
    # Cheap check that this really is one instance per N: every arm must have
    # been run over the same N set, or reading off one arm would misreport.
    n_sets = {name: {rec["params"]["n_gauss"] for rec in rs} for name, rs in algos.items()}
    assert len(set(map(frozenset, n_sets.values()))) == 1, f"arms disagree on N: {n_sets}"
    recs = sorted(next(iter(algos.values())), key=lambda rec: rec["params"]["n_gauss"])
    ns = [rec["params"]["n_gauss"] for rec in recs]
    chis = [rec["input_max_bond_dim"] for rec in recs]
    lines = ["", "## Instances and input rank", "",
             "| N | box half-width L | bits per variable R | input rank chi_in |",
             "|---|---|---|---|"]
    for rec in recs:
        p = rec["params"]
        lines.append(
            f"| {p['n_gauss']} | {p['box_l']:.3f} | {p['r']} | "
            f"{rec['input_max_bond_dim']} |"
        )
    # A single N (the CI smoke, or a probe) has no slope to fit, so say nothing
    # rather than print "x = nan" and draw a one-point log-log plot.
    if len(set(ns)) < 2:
        lines += ["", "Only one N in this sweep, so there is no rank-versus-N "
                      "exponent to fit and no chi plot."]
        return lines

    expo = fit_exponent(ns, chis)
    lines += ["", f"Fitted over this sweep, chi_in grows like N^x with "
                  f"x = {expo:.2f}, against x = 0.5 for the sqrt(N) hypothesis "
                  f"and x = 1 for the linear one."]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.loglog(ns, chis, "o-", label=f"chi_in (N^{expo:.2f})")
    ax.set_xlabel("number of Gaussians N")
    ax.set_ylabel("input bond dimension chi_in")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(profile_dir / f"{case}-chi.svg")
    plt.close(fig)
    lines += ["", f"![chi](./{case}-chi.svg)"]
    return lines


EXTRA_RENDER = {
    # case name -> function(case, algos, profile_dir) -> extra Markdown lines.
    # Case-keyed on purpose: the generic path stays untouched.
    "elementwise_gauss2d_scaling": render_gauss2d_scaling_extra,
}


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
    if case in EXTRA_RENDER:
        lines += EXTRA_RENDER[case](case, algos, profile_dir)
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
