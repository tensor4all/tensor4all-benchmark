import json, os, re, statistics, subprocess
bins = {
    "baseline": "/tmp/benchmark_src.pr694-typed-probes-skip-nograd-inspect",
    "candidate": "/tmp/benchmark_src.pr694-validated-derived-wrap-final",
}
env = os.environ | {
    "RAYON_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "T4A_BENCH_ALGORITHM": "src-adaptive",
    "T4A_BENCH_WARMUP": "1",
}
pattern = re.compile(r"case=mpo-mps/src-adaptive .* per_run=([0-9.]+)s .*relative_error=([0-9.eE+-]+)")
rows = []
for bond, reps in [(32, 50), (64, 25), (128, 8)]:
    for pair in range(1, 11):
        row = {"bond": bond, "pair": pair, "reps": reps}
        for name in (["candidate", "baseline"] if pair % 2 else ["baseline", "candidate"]):
            output = subprocess.check_output(
                ["taskset", "-c", "63", bins[name], "10", str(bond), str(reps), "mpo-mps", "3", "false", "32"],
                env=env,
                text=True,
            )
            match = pattern.search(output)
            assert match, output
            row[f"{name}_ms"] = float(match.group(1)) * 1000
            row[f"{name}_error"] = float(match.group(2))
        row["change_pct"] = (row["candidate_ms"] / row["baseline_ms"] - 1.0) * 100.0
        rows.append(row)
        print(json.dumps(row), flush=True)
summary = []
for bond in (32, 64, 128):
    selected = [row for row in rows if row["bond"] == bond]
    baseline = statistics.median(row["baseline_ms"] for row in selected)
    candidate = statistics.median(row["candidate_ms"] for row in selected)
    summary.append({
        "bond": bond,
        "baseline_ms": baseline,
        "candidate_ms": candidate,
        "paired_median_pct": statistics.median(row["change_pct"] for row in selected),
        "paired_mean_pct": statistics.mean(row["change_pct"] for row in selected),
        "max_error": max(max(row["baseline_error"], row["candidate_error"]) for row in selected),
        "pair_values": [row["change_pct"] for row in selected],
    })
print(json.dumps(summary, indent=2))
with open("/tmp/pr694_validated_wrap_ab.json", "w") as handle:
    json.dump({"rows": rows, "summary": summary}, handle, indent=2)
