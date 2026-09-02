#!/usr/bin/env python3
import argparse
import json
import os
import re
import resource
import statistics
import subprocess
import time

RUST_PATTERN = re.compile(
    r"case=mpo-mps/src-adaptive .*?per_run=([0-9.]+)s .*?relative_error=([0-9.eE+-]+|n/a)"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", required=True)
    parser.add_argument("--merged-bcgs2", required=True)
    parser.add_argument("--merged-householder", required=True)
    parser.add_argument("--python", default="/tmp/randommpomps-venv/bin/python")
    parser.add_argument("--python-script", default="/tmp/bench_randommpomps_matched.py")
    parser.add_argument("--output", default="/tmp/pr694_post_1751.json")
    parser.add_argument("--pairs", type=int, default=7)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--cpu", default="63")
    parser.add_argument("--min-cpu-utilization", type=float, default=0.95)
    parser.add_argument("--min-python-cpu-utilization", type=float, default=0.75)
    parser.add_argument("--max-attempts", type=int, default=28)
    parser.add_argument("--bonds", default="32,64,128")
    return parser.parse_args()


def run_output(command, env):
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    output = subprocess.check_output(command, env=env, text=True)
    wall_seconds = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_seconds = (
        after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
    )
    return output, cpu_seconds / wall_seconds


def run_rust(binary, bond, args, env):
    output, cpu_utilization = run_output(
        [
            "taskset", "-c", args.cpu, binary, "10", str(bond),
            str(args.reps), "mpo-mps", "3", "false", "32",
        ],
        env,
    )
    match = RUST_PATTERN.search(output)
    if match is None:
        raise RuntimeError(output)
    error = None if match.group(2) == "n/a" else float(match.group(2))
    return {
        "seconds": float(match.group(1)),
        "relative_error": error,
        "cpu_utilization": cpu_utilization,
    }


def run_python(bond, args, env):
    output, cpu_utilization = run_output(
        ["taskset", "-c", args.cpu, args.python, args.python_script, str(bond), str(args.reps)],
        env,
    )
    data = json.loads(output.strip().splitlines()[-1])
    return {
        "seconds": statistics.mean(data["times_s"]),
        "output_bonds": data["output_bonds"],
        "cpu_utilization": cpu_utilization,
    }


def main():
    args = parse_args()
    env = os.environ | {
        "RAYON_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "T4A_BENCH_SKIP_EXACT": "1",
        "T4A_BENCH_ALGORITHM": "src-adaptive",
        "T4A_BENCH_WARMUP": "1",
    }
    binaries = {
        "current": args.current,
        "merged_bcgs2": args.merged_bcgs2,
        "merged_householder": args.merged_householder,
    }
    lanes = ["current", "merged_bcgs2", "merged_householder", "python"]
    bonds = tuple(int(value) for value in args.bonds.split(","))
    rows = []
    for bond in bonds:
        accepted = 0
        for attempt in range(1, args.max_attempts + 1):
            order = lanes[(attempt - 1) % len(lanes):] + lanes[:(attempt - 1) % len(lanes)]
            row = {"bond": bond, "attempt": attempt, "order": order}
            for lane in order:
                if lane == "python":
                    row[lane] = run_python(bond, args, env)
                else:
                    row[lane] = run_rust(binaries[lane], bond, args, env)
            row["accepted"] = (
                all(
                    row[lane]["cpu_utilization"] >= args.min_cpu_utilization
                    for lane in ("current", "merged_bcgs2", "merged_householder")
                )
                and row["python"]["cpu_utilization"] >= args.min_python_cpu_utilization
            )
            if row["accepted"]:
                accepted += 1
                row["pair"] = accepted
            rows.append(row)
            print(json.dumps(row), flush=True)
            if accepted == args.pairs:
                break
        if accepted != args.pairs:
            raise RuntimeError(
                f"bond {bond}: only {accepted}/{args.pairs} uncontended pairs "
                f"within {args.max_attempts} attempts"
            )

    summary = []
    for bond in bonds:
        selected = [row for row in rows if row["bond"] == bond and row["accepted"]]
        medians = {
            lane: statistics.median(row[lane]["seconds"] for row in selected) * 1000.0
            for lane in lanes
        }
        summary.append({
            "bond": bond,
            "median_ms": medians,
            "merged_bcgs2_vs_current": medians["merged_bcgs2"] / medians["current"],
            "merged_householder_vs_current": medians["merged_householder"] / medians["current"],
            "merged_householder_vs_merged_bcgs2": medians["merged_householder"] / medians["merged_bcgs2"],
            "merged_householder_vs_python": medians["merged_householder"] / medians["python"],
            "merged_householder_gap_python_ms": medians["merged_householder"] - medians["python"],
            "max_rust_relative_error": max(
                (row[lane]["relative_error"]
                 for row in selected
                 for lane in ("current", "merged_bcgs2", "merged_householder")
                 if row[lane]["relative_error"] is not None),
                default=None,
            ),
        })
    artifact = {
        "protocol": {
            "pairs": args.pairs,
            "reps": args.reps,
            "cpu": args.cpu,
            "threads": 1,
            "min_rust_cpu_utilization": args.min_cpu_utilization,
            "min_python_cpu_utilization": args.min_python_cpu_utilization,
            "max_attempts_per_bond": args.max_attempts,
            "sites": 10,
            "physical_dim": 2,
            "rank_increment": 3,
            "max_rank": 32,
        },
        "binaries": binaries,
        "rows": rows,
        "summary": summary,
    }
    with open(args.output, "w") as handle:
        json.dump(artifact, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
