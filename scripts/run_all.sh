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
threads=${BENCH_THREADS:-1}
if ((threads < 1)); then
  echo "BENCH_THREADS must be at least 1, got $threads" >&2
  exit 2
fi
# One pinned core for the source-of-truth sweep, a contiguous range for the
# optional `<machine>-<n>t` scaling profile.
if ((threads == 1)); then
  affinity="$core"
else
  affinity="$core-$((core + threads - 1))"
fi
mkdir -p "$raw"
rm -f "$raw"/*.json

export RAYON_NUM_THREADS="$threads"
export OMP_NUM_THREADS="$threads"
export OPENBLAS_NUM_THREADS="$threads"
export MKL_NUM_THREADS="$threads"
export VECLIB_MAXIMUM_THREADS="$threads"
export TENFERRO_NUM_THREADS="$threads"
export OUT_DIR="$raw"

run() {
  if command -v taskset >/dev/null 2>&1; then
    taskset -c "$affinity" "$@"
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
tensor4all_rs_rev: a5157385f58048355a28a6a003d4cb63d3b1349b
threads: $threads
cpu_affinity: $affinity
EOF

run cargo run --release --locked --bin elementwise_fourier
run cargo run --release --locked --bin elementwise_gauss2d_patched
run cargo run --release --locked --bin mpo_mpo_aniso_patched
python3 scripts/report.py "$out"
