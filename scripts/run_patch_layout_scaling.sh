#!/usr/bin/env bash
set -euo pipefail

profile=${1:-linux-epyc-7713p-patch-layout-scaling-r16}
out="result/$profile"
raw="$out/raw"
cache=${BENCH_INPUT_CACHE_DIR:-.cache/inputs}
core=${BENCH_CPU_CORE:-0}
mkdir -p "$raw" "$out/attempts" "$cache"
rm -f "$raw"/*.json

repo_rev=$(git rev-parse HEAD)
local_tensor4all="../tensor4all-rs/.worktrees/adaptive-contract-scheduling"
tensor4all_rev=${TENSOR4ALL_RS_REV:-}
if [[ -z $tensor4all_rev && -d $local_tensor4all/.git ]]; then
  tensor4all_rev=$(git -C "$local_tensor4all" rev-parse HEAD)
fi
tensor4all_rev=${tensor4all_rev:-9e9aedaebe0d3918b34dd399ff0981e337f3835b}
cat >"$out/run.yaml" <<EOF
profile: $profile
date: $(date -I)
repository_revision: $repo_rev
tensor4all_rs_revision: $tensor4all_rev
input_cache_compatibility_revision: 9e9aedaebe0d3918b34dd399ff0981e337f3835b
cpu: $(lscpu | awk -F: '/Model name/{sub(/^[[:space:]]+/, "", $2); print $2; exit}')
cpu_affinity: $core
threads: {rayon: 1, omp: 1, openblas: 1, mkl: 1, blis: 1}
build: cargo release locked
quantics_bits_per_axis: 16
padding_factor: 4
patch_cap: 128
patch_input_rtol: 1e-6
patch_local_sweep_rtol: 1.8257418583505538e-7
patch_local_svd_cutoff: 3.333333333333333e-14
patch_svd_visit_budget_count: 30
contraction_performed: false
patch_build_repetitions: 1
layouts: [balanced_xyz, shared_y_only]
default_n_points: [512, 1024, 2048, 4096, 8192, 12000]
fixed_n_chi_checks: {n: 8192, input_l2_rtols: [1e-6, 3e-10], achieved_chi: [283, 418]}
measurement_command_timeout_seconds: 570
notes:
  - The cache-key revision is the input-generator compatibility baseline; tensor4all_rs_revision is the actual patched worktree used for this run.
  - Raw padded input tensors are gitignored and are not part of this artifact; exact regeneration rebuilds or supplies the v4 padded caches.
  - Patch build values are single measurements; the report does not label them medians.
  - The requested patch-input squared budget is divided over two visits to each of 15 chain edges; exact residual validation remains authoritative.
EOF

export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export BENCH_PATCH_ONLY=1 BENCH_SKIP_REFERENCE_COUNT=1

run_point() {
  local n=$1
  local input_rtol=$2
  local layouts=${3:-balanced_xyz,shared_y_only}
  timeout 570s env BENCH_NS="$n" BENCH_INPUT_L2_RTOL="$input_rtol" \
    BENCH_PATCH_LAYOUTS="$layouts" BENCH_INPUT_CACHE_DIR="$cache" \
    OUT_DIR="$raw" taskset -c "$core" target/release/mpo_mpo_aniso_patched
}

cargo build --release --locked --bin mpo_mpo_aniso_patched
for n in 512 1024 2048 4096 8192 12000; do
  run_point "$n" 1e-6
done
for layout in balanced_xyz shared_y_only; do
  run_point 8192 3e-10 "$layout"
done
python3 scripts/report_patch_layout_scaling.py "$out"
