#!/usr/bin/env bash
set -euo pipefail

profile=${1:-linux-epyc-7713p-direct-product-r16}
out="result/$profile"
raw="$out/raw"
cache=${BENCH_INPUT_CACHE_DIR:-.cache/inputs}
core=${BENCH_CPU_CORE:-0}
mkdir -p "$raw" "$cache"
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
n_points: [1, 2, 4, 8]
factor_rtols: [3e-5, 1e-5, 1e-6]
direct_product_max_bytes: 2147483648
patching_performed: false
contraction_performed: false
EOF

export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cargo build --release --locked --bin mpo_mpo_aniso_patched
for n in 1 2 4 8; do
  for rtol in 3e-5 1e-5 1e-6; do
    if [[ $n == 8 && $rtol == 3e-5 ]]; then
      continue
    fi
    timeout 570s env BENCH_DIRECT_PRODUCT_INPUT_ONLY=1 BENCH_NS="$n" \
      BENCH_INPUT_L2_RTOL="$rtol" BENCH_INPUT_CACHE_DIR="$cache" \
      OUT_DIR="$raw" taskset -c "$core" target/release/mpo_mpo_aniso_patched
  done
done
python3 scripts/report_direct_product_scaling.py "$out"
