#!/usr/bin/env bash
set -euo pipefail

profile=${1:-linux-epyc-7713p-padded-r16}
out="result/$profile"
raw="$out/raw"
profiles="$out/profiles"
cache=${BENCH_INPUT_CACHE_DIR:-.cache/inputs}
core=${BENCH_CPU_CORE:-0}
mkdir -p "$raw" "$profiles" "$cache"

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
python3 scripts/report_padded_scaling.py "$out"
