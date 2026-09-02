import json
import os
import re
import statistics
import subprocess

bonds = [32, 64, 128]
pairs = 5
reps = 10
env = os.environ | {
    "RAYON_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "T4A_BENCH_SKIP_EXACT": "1",
    "T4A_BENCH_ALGORITHM": "src-adaptive",
    "T4A_BENCH_WARMUP": "1",
}
rust_cmd = "/tmp/benchmark_src.pr694-validated-derived-wrap-final"
python_cmd = "/tmp/randommpomps-venv/bin/python"
rows = []

def rust(bond):
    out = subprocess.check_output(
        ["taskset", "-c", "63", rust_cmd, "10", str(bond), str(reps), "mpo-mps", "3", "false", "32"],
        env=env,
        text=True,
    )
    match = re.search(r"case=mpo-mps/src-adaptive .*?per_run=([0-9.]+)s", out)
    assert match, out
    return float(match.group(1)), out.strip().splitlines()[-1]

def python(bond):
    out = subprocess.check_output(
        ["taskset", "-c", "63", python_cmd, "/tmp/bench_randommpomps_matched.py", str(bond), str(reps)],
        env=env,
        text=True,
    )
    data = json.loads(out.strip().splitlines()[-1])
    return statistics.mean(data["times_s"]), data["output_bonds"]

for bond in bonds:
    for pair in range(1, pairs + 1):
        order = ("rust", "python") if pair % 2 else ("python", "rust")
        got = {}
        for implementation in order:
            got[implementation] = rust(bond) if implementation == "rust" else python(bond)
        row = {
            "bond": bond,
            "pair": pair,
            "rust_s": got["rust"][0],
            "python_s": got["python"][0],
            "python_output_bonds": got["python"][1],
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

with open("/tmp/pr694_final_matched_python.json", "w") as f:
    json.dump(rows, f, indent=2)

for bond in bonds:
    rs = [r["rust_s"] * 1000 for r in rows if r["bond"] == bond]
    ps = [r["python_s"] * 1000 for r in rows if r["bond"] == bond]
    r = statistics.median(rs)
    p = statistics.median(ps)
    print(json.dumps({
        "bond": bond,
        "rust_median_ms": r,
        "python_median_ms": p,
        "rust_over_python": r / p,
        "gap_ms": r - p,
        "rust_range_ms": [min(rs), max(rs)],
        "python_range_ms": [min(ps), max(ps)],
    }))
