#!/usr/bin/env bash
set -euo pipefail

profile=${1:-linux-epyc-7713p-padded-r16}
out="result/$profile"
raw="$out/raw"
profiles="$out/profiles"
cache=${BENCH_INPUT_CACHE_DIR:-.cache/inputs}
core=${BENCH_CPU_CORE:-0}
mkdir -p "$raw" "$profiles" "$cache"
rm -f "$raw"/*.json "$profiles"/*.log

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
cpu: $(lscpu | awk -F: '/Model name/{sub(/^[[:space:]]+/, "", $2); print $2; exit}')
cpu_affinity: $core
threads:
  rayon: 1
  omp: 1
  openblas: 1
  mkl: 1
  blis: 1
build: cargo release locked
fit_profile: T4A_PROFILE_FIT=1
quantics_bits_per_axis: 16
padding_factor: 4
patch_cap: 128
operation_points: [512, 4096, 8192]
near_chi_418_input_l2_rtol: 1e-9
patch_scaling_points: [512, 1024, 2048, 4096, 8192, 12000]
measurement_command_timeout_seconds: 570
EOF

export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export T4A_PROFILE_FIT=1 BENCH_RUNS=1 BENCH_WARMUPS=0

run_point() {
  local n=$1
  shift
  timeout 570s env BENCH_NS="$n" BENCH_INPUT_CACHE_DIR="$cache" OUT_DIR="$raw" \
    taskset -c "$core" target/release/mpo_mpo_aniso_patched "$@"
}

cargo build --release --locked --bin mpo_mpo_aniso_patched
for n in 512 1024 2048 4096 8192 12000; do
  BENCH_PATCH_ONLY=1 run_point "$n"
done
for n in 512 4096; do
  run_point "$n" >"$profiles/n$n.log" 2>&1
done
: >"$profiles/n8192.log"
for arm in global patched; do
  BENCH_ARM=$arm BENCH_INPUT_L2_RTOL=1e-9 run_point 8192 \
    >>"$profiles/n8192.log" 2>&1
done
python3 scripts/report_padded_scaling.py "$out"
