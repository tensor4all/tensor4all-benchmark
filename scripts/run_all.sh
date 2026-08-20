#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: scripts/run_all.sh <profile>" >&2
  exit 2
fi

profile=$1
out="result/$profile"
raw="$out/raw"
core=${BENCH_CPU_CORE:-0}
mkdir -p "$raw"
rm -f "$raw"/*.json

export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OUT_DIR="$raw"
cases=${BENCH_CASES:-all}
if [[ $cases != all && $cases != mpo ]]; then
  echo "BENCH_CASES must be 'all' or 'mpo'" >&2
  exit 2
fi

run() {
  if command -v taskset >/dev/null 2>&1; then
    taskset -c "$core" "$@"
  else
    "$@"
  fi
}

repo_rev=$(git rev-parse HEAD)
if [[ -n $(git status --porcelain --untracked-files=no -- . ':!result') ]]; then
  repo_rev="${repo_rev}-dirty"
fi
chip=$(awk -F: '/model name/{sub(/^[[:space:]]+/, "", $2); print $2; exit}' /proc/cpuinfo 2>/dev/null || true)
if [[ -z $chip ]] && command -v sysctl >/dev/null 2>&1; then
  chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)
fi
memory_gb=$(awk '/MemTotal/{printf "%.1f", $2/1024/1024}' /proc/meminfo 2>/dev/null || true)
cat > "$out/run.yaml" <<EOF
profile: $profile
date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)
os: $(uname -s)
chip: ${chip:-unknown}
memory_gb: ${memory_gb:-unknown}
repo_rev: $repo_rev
tensor4all_rs_rev: 9e9aedaebe0d3918b34dd399ff0981e337f3835b
threads: 1
cpu_affinity: $core
EOF

if [[ $cases == all ]]; then
  run cargo run --release --locked --bin elementwise_fourier
  run cargo run --release --locked --bin elementwise_gauss2d_patched
fi
run cargo run --release --locked --bin mpo_mpo_aniso_patched
python3 scripts/report.py "$out"
