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
    # case name -> (record field used as x, axis label). A field name prefixed
    # with "params." is read out of the record's params object instead of the top
    # level, which is where a case whose knob is not a bond dimension keeps it.
    "elementwise_fourier": ("input_max_bond_dim", "input bond dimension chi"),
    "mpo_mpo_quantics": ("input_max_bond_dim", "input bond dimension chi"),
    "elementwise_gauss2d": ("input_max_bond_dim", "input bond dimension chi"),
    "elementwise_gauss2d_scaling": ("input_max_bond_dim", "input bond dimension chi"),
    "elementwise_gauss2d_patched": ("params.n_gauss", "number of Gaussians N"),
}

X_SYMBOL = {
    # case name -> short symbol of the x axis, used in the fitted exponent
    # labels. Every case sweeps a bond dimension except case 5, which sweeps N.
    "elementwise_gauss2d_patched": "N",
}

# Case 6 has an intentionally hand-maintained crossover table because its x
# points are selected by input TCI caps and every committed run records a failed
# accuracy gate. Do not replace that context with the generic scaling report.
HAND_MAINTAINED_CASES = {"mpo_mpo_aniso_patched"}

ERROR_LABEL = {
    # case name -> what `max_error` in the records actually measures
    "elementwise_fourier": "max abs error",
    "mpo_mpo_quantics": "max relative error",
    "elementwise_gauss2d": "max relative error",
    "elementwise_gauss2d_scaling": "max relative error",
    "elementwise_gauss2d_patched": "max relative error",
}

NOTES = {
    # case name -> caveat emitted under the summary table
    "mpo_mpo_quantics": (
        "Note: every algorithm contracts at the same output budget, its "
        "maximum bond dimension capped at the input rank chi, and the cap is "
        "the only truncation control: the contraction tolerance is pinned "
        "inert at contract_tol, so every arm spends the whole budget unless "
        "its exact rank is smaller and the error "
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
        "its maximum bond dimension capped at the input rank chi, and the cap "
        "is the only truncation control: the product tolerance is pinned inert "
        "at contract_tol and the aci arm runs with a scale-relative stopping "
        "criterion, so every arm spends the whole budget unless its exact rank "
        "is smaller and the error "
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
        "runs at the same fixed output budget chi_out <= chi_in as case 3, "
        "decided by the cap alone with an inert contract_tol. The "
        "naive arm of case 3 is excluded here: it forms the full chi_in-squared "
        "bond before truncating, which dominates the sweep at these ranks "
        "without adding a conclusion, since it tracks fit_treetn to the last "
        "reported digit in case 3. As in case 3, zipup_treetn spends the whole "
        "budget and still returns an order-unity relative error. "
        "The fitted time exponent is measured against input chi along a sweep "
        "of N, where the site count also grows, so it is not a pure chi power "
        "law."
    ),
    "elementwise_gauss2d_patched": (
        "Note: this case runs two instance families, recorded per arm as family "
        "and tabulated separately below, and it is controlled by the accuracy "
        "instead of by a fixed output budget. The default family aniso is N "
        "anisotropic narrow spikes at a fixed spacing-to-width ratio, whose random "
        "orientations and aspect ratios push the global rank toward the geometric "
        "bound of the bit count; smooth is the density-constant isotropic family of "
        "elementwise_gauss2d_scaling. Every arm is asked "
        "for the same global relative tolerance rtol, and what the case measures "
        "is the size and the time each one needs to reach it, so the arms are "
        "comparable only because rtol is the same for all of them and the error "
        "column is a check that they got there rather than the discriminator. "
        "The size metric is n_params, the total number of stored core entries: a "
        "bond dimension says nothing across the two representations here, since "
        "a patched arm holds one train per patch and no single global rank "
        "exists. For the patched arms n_params counts the free sites of each "
        "patch only, the cores at the projected sites being one-hot copy "
        "selectors that carry structure rather than data. The patched arms build "
        "each input as a partitioned tensor train, split until every patch fits "
        "under the per-patch rank cap, form the product patch pair by patch "
        "pair, and budget the result once at the end with volume-proportional "
        "absolute budgets, which is what makes shrinking patch norms harmless. "
        "Which construction split the inputs is recorded per arm as input_path: "
        "the default norm builds one global train per input and splits it by "
        "Frobenius norms, so its input_build_secs includes that global build, "
        "while tci runs a TCI per patch on the function and forms no global "
        "train at all. The two global arms fit_treetn and aci are the case-3 "
        "arms run tolerance-driven at the same rtol with no binding rank cap, and "
        "each has its own N ceiling, since the uncapped global fit is orders of "
        "magnitude more expensive than the interpolating arm. Input construction is "
        "not part of the reported wall time: it is recorded separately as "
        "input_build_secs, since one build is shared by every arm of an "
        "instance. The fitted time exponent is measured against N."
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


def xvalue(rec, xfield):
    """Read the x axis value of one record, from params when asked for."""
    if xfield.startswith("params."):
        return rec["params"][xfield[len("params."):]]
    return rec[xfield]


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


def family_of(rec):
    """Instance family of one case-5 record.

    Records written before the case had two families carry no family field, and
    they are all of the smooth one, so that is the fallback.
    """
    return rec["params"].get("family", "smooth")


# Families of case 5, in report order: the anisotropic spikes are the default and
# the headline, the smooth family is case 4's and comes second.
FAMILY_ORDER = ["aniso", "smooth"]

FAMILY_BLURB = {
    "aniso": (
        "N anisotropic narrow spikes: minor width sigma fixed, aspect ratio "
        "log-uniform in [1, rho_max] and orientation uniform in [0, pi) drawn per "
        "spike, mean spacing held at a fixed number of minor widths so the box "
        "grows like sqrt(N) and R resolves sigma to a quarter step. This is the "
        "family the case defaults to: a field of small hard features whose global "
        "rank climbs like N^0.5 through the sweep and then decelerates toward a "
        "density-set plateau at larger N, while a patched representation is held "
        "at its per-patch cap by construction. The isotropic control of the same "
        "family, rho_max = 1, grows the same way, so the rank comes from the "
        "density of narrow features rather than from the anisotropy."
    ),
    "smooth": (
        "N isotropic Gaussians of log-uniform inverse width at constant density, "
        "case 4's family. Smooth everywhere, so there is no hard region for the "
        "patching to isolate."
    ),
}


def render_gauss2d_patched_extra(case, algos, profile_dir: Path):
    """Case-5 only: the size and patch structure the generic tables cannot show.

    The generic path reports a time exponent and a worst error per arm, which for
    an equal-accuracy case is a check rather than a result. What case 5 measures
    is size against N at fixed accuracy, so this adds the instance table, the
    per-arm size and patch-count table, and a parameter-count plot. All of it is
    grouped by instance family, since the two families of the case are two
    different problems and an arm's row means nothing without knowing which one it
    ran on. Returns extra Markdown lines.
    """
    families = sorted(
        {family_of(rec) for recs in algos.values() for rec in recs},
        key=lambda name: (FAMILY_ORDER.index(name) if name in FAMILY_ORDER else len(FAMILY_ORDER),
                          name),
    )
    lines = []
    for family in families:
        per_family = {
            name: [rec for rec in recs if family_of(rec) == family]
            for name, recs in algos.items()
        }
        per_family = {name: recs for name, recs in per_family.items() if recs}
        lines += ["", f"# Family: {family}", ""]
        if family in FAMILY_BLURB:
            lines += [FAMILY_BLURB[family], ""]
        lines += render_patched_family(case, family, per_family, profile_dir)
    return lines


def render_patched_family(case, family, algos, profile_dir: Path):
    """The instance table, size table and parameter plot of one case-5 family."""
    lines = []
    # One instance per N, shared by every arm, so read it off any one arm. The
    # patched fields come from a patched arm, the global chi_in from a global one.
    patched = {name: recs for name, recs in algos.items() if name.startswith("patched_")}
    globals_ = {name: recs for name, recs in algos.items() if not name.startswith("patched_")}
    if patched:
        recs = sorted(next(iter(patched.values())), key=lambda rec: rec["params"]["n_gauss"])
        # Every global arm of one N ran on the same two global trains, so any of
        # them carries the instance columns. Merged over all of them rather than
        # read off one, since the two baselines have different N ceilings and the
        # cheaper one reaches points the expensive one does not.
        by_n_global = {}
        for recs_g in globals_.values():
            for rec in recs_g:
                by_n_global.setdefault(rec["params"]["n_gauss"], rec)
        lines += ["", "## Instances", "",
                  "| N | box half-width L | bits per variable R | input patches f, g | "
                  "input params, patched | patched build [s] | global chi_in | "
                  "input params, global | global build [s] |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for rec in recs:
            p = rec["params"]
            g = by_n_global.get(p["n_gauss"])
            gchi = str(g["input_max_bond_dim"]) if g else "not run"
            gpar = f"{g['params']['input_n_params']}" if g else "not run"
            gbuild = f"{g['input_build_secs']:.2f}" if g else "not run"
            lines.append(
                f"| {p['n_gauss']} | {p['box_l']:.3f} | {p['r']} | "
                f"{p['n_patches_f']}, {p['n_patches_g']} | {p['input_n_params']} | "
                f"{rec['input_build_secs']:.2f} | {gchi} | {gpar} | {gbuild} |"
            )

    # The patched arms split their wall time between the patch-pair loop and the
    # one final budgeting, and which half dominates is not guessable from the
    # total, so those columns appear whenever a record carries them. A global arm
    # has neither.
    breakdown = any(
        "pairs_secs" in rec["params"] for recs in algos.values() for rec in recs
    )
    header = ["algorithm", "N", "median time [s]", "max relative error", "params",
              "patches", "max patch bond"]
    if breakdown:
        header += ["pairs", "pairs time [s]", "truncate time [s]"]
    lines += ["", "## Size and time at equal accuracy", "",
              "| " + " | ".join(header) + " |",
              "|" + "---|" * len(header)]
    for algo, recs in sorted(algos.items()):
        for rec in sorted(recs, key=lambda rec: rec["params"]["n_gauss"]):
            patches = rec.get("n_patches")
            bond = rec.get("max_patch_bond", rec["output_max_bond_dim"])
            row = (
                f"| {algo} | {rec['params']['n_gauss']} | "
                f"{rec['wall_time_median_secs']:.4f} | {rec['max_error']:.2e} | "
                f"{rec.get('n_params', 'n/a')} | {patches if patches else 'one train'} | "
                f"{bond} |"
            )
            if breakdown:
                p = rec["params"]
                if "pairs_secs" in p:
                    row += (f" {p['n_pairs']} | {p['pairs_secs']:.3f} | "
                            f"{p['truncate_secs']:.3f} |")
                else:
                    row += " not patched | not patched | not patched |"
            lines.append(row)

    fig, ax = plt.subplots(figsize=(5, 4))
    drew = False
    for algo, recs in sorted(algos.items()):
        recs = sorted(recs, key=lambda rec: rec["params"]["n_gauss"])
        ns = [rec["params"]["n_gauss"] for rec in recs]
        params = [rec.get("n_params") for rec in recs]
        if any(value is None for value in params) or len(set(ns)) < 2:
            continue
        expo = fit_exponent(ns, params)
        ax.loglog(ns, params, "o-", label=f"{algo} (N^{expo:.1f})")
        drew = True
    if not drew:
        plt.close(fig)
        return lines
    ax.set_xlabel("number of Gaussians N")
    ax.set_ylabel("stored parameters of the product")
    ax.set_title(f"family: {family}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(profile_dir / f"{case}-params-{family}.svg")
    plt.close(fig)
    lines += ["", f"![params](./{case}-params-{family}.svg)"]
    return lines


EXTRA_RENDER = {
    # case name -> function(case, algos, profile_dir) -> extra Markdown lines.
    # Case-keyed on purpose: the generic path stays untouched.
    "elementwise_gauss2d_scaling": render_gauss2d_scaling_extra,
    "elementwise_gauss2d_patched": render_gauss2d_patched_extra,
}


def series(case, algos):
    """Series of the generic summary table and plots, one per line drawn.

    An arm name is enough everywhere except case 5, whose two instance families
    share their arm names: a profile holding both would otherwise draw one line per
    arm over the union of two different problems. Splitting the series by family
    keeps each line one measurement.
    """
    if case != "elementwise_gauss2d_patched":
        return algos
    families = {family_of(rec) for recs in algos.values() for rec in recs}
    if len(families) < 2:
        return algos
    split = defaultdict(list)
    for algo, recs in algos.items():
        for rec in recs:
            split[f"{algo} ({family_of(rec)})"].append(rec)
    return split


def render_case(case, algos, profile_dir: Path):
    xfield, xlabel = X_AXIS[case]
    xsym = X_SYMBOL.get(case, "chi")
    elabel = ERROR_LABEL.get(case, "max error")
    lines = [f"# {case}", "",
             f"| algorithm | points | fitted time exponent | worst {elabel} |",
             "|---|---|---|---|"]
    fig_t, ax_t = plt.subplots(figsize=(5, 4))
    fig_e, ax_e = plt.subplots(figsize=(5, 4))
    for algo, recs in sorted(series(case, algos).items()):
        recs = sorted(recs, key=lambda rec: xvalue(rec, xfield))
        xs = [xvalue(rec, xfield) for rec in recs]
        ts = [rec["wall_time_median_secs"] for rec in recs]
        es = [rec["max_error"] for rec in recs]
        expo = fit_exponent(xs, ts)
        lines.append(f"| {algo} | {len(recs)} | {expo:.2f} | {max(es):.2e} |")
        ax_t.loglog(xs, ts, "o-", label=f"{algo} ({xsym}^{expo:.1f})")
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
        if case in HAND_MAINTAINED_CASES:
            report = profile_dir / f"{case}.md"
            if not report.is_file():
                sys.exit(f"missing hand-maintained report {report}")
            print(f"kept hand-maintained {report}")
            continue
        render_case(case, algos, profile_dir)


if __name__ == "__main__":
    main()
