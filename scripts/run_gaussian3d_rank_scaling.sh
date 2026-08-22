#!/usr/bin/env bash
set -euo pipefail

profile=${1:-linux-epyc-7713p-gaussian3d-rank-r16}
out="result/$profile"
raw="$out/raw"
cache=${BENCH_INPUT_CACHE_DIR:-.cache/inputs}
core=${BENCH_CPU_CORE:-0}
mkdir -p "$raw" "$out/attempts" "$cache"
rm -f "$raw"/*.json
attempts="$out/attempts/status.tsv"
printf 'n\tstatus\telapsed_seconds\n' >"$attempts"

repo_rev=$(git rev-parse HEAD)
local_tensor4all="../tensor4all-rs/.worktrees/adaptive-contract-scheduling"
tensor4all_rev=${TENSOR4ALL_RS_REV:-}
if [[ -z $tensor4all_rev && -d $local_tensor4all/.git ]]; then
  tensor4all_rev=$(git -C "$local_tensor4all" rev-parse HEAD)
fi
tensor4all_rev=${tensor4all_rev:-9e9aedaebe0d3918b34dd399ff0981e337f3835b}
local_compression_rtol=$(python3 - <<'PY'
import math
print(repr(1e-6 / math.sqrt(15)))
PY
)
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
dimensions: [batch, x, y]
batch_diagonal_embedding: "A(b,x;b_prime,y)=delta(b,b_prime)A(b,x,y)"
padding_factor: 4
input_l2_rtol: 1e-6
local_compression_rtol: $local_compression_rtol
compression_bond_budget_count: 15
n_points: [1, 2, 4, 8, 16, 32, 64]
measurement_command_timeout_seconds: 570
patching_performed: false
contraction_performed: false
notes:
  - The requested squared compression budget is distributed across 15 QTT bonds; off-pivot principal-axis validation remains authoritative.
EOF

export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cargo build --release --locked --bin mpo_mpo_aniso_patched
for n in 1 2 4 8 16 32 64; do
  start=$(date +%s)
  set +e
  timeout 570s env BENCH_3D_INPUT_ONLY=1 BENCH_NS="$n" \
    BENCH_INPUT_CACHE_DIR="$cache" OUT_DIR="$raw" taskset -c "$core" \
    target/release/mpo_mpo_aniso_patched
  status=$?
  set -e
  elapsed=$(($(date +%s) - start))
  printf '%s\t%s\t%s\n' "$n" "$status" "$elapsed" >>"$attempts"
  if [[ $status -ne 0 && $status -ne 124 ]]; then
    exit "$status"
  fi
done
python3 scripts/report_gaussian3d_rank_scaling.py "$out"
