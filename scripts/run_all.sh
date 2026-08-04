#!/usr/bin/env bash
# Usage: scripts/run_all.sh <profile>   (e.g. mac-cpu)
set -euo pipefail
PROFILE="${1:?usage: run_all.sh <profile>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/result/$PROFILE"
mkdir -p "$OUT/raw"
# cargo, uv and the relative report.py path below all resolve against the repo root.
cd "$ROOT"

cargo build --release

echo "== running elementwise_fourier (full sweep, this takes a while)"
OUT_DIR="$OUT/raw" cargo run --release --bin elementwise_fourier
echo "== running mpo_mpo_quantics (full sweep, this takes a while)"
OUT_DIR="$OUT/raw" cargo run --release --bin mpo_mpo_quantics

cat > "$OUT/run.yaml" <<EOF
profile: $PROFILE
date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
host: $(hostname)
os: $(uname -sm)
repo_rev: $(git -C "$ROOT" rev-parse HEAD)
tensor4all_rs_rev: $(grep -m1 -o 'rev = "[a-f0-9]*"' "$ROOT/Cargo.toml" | cut -d'"' -f2)
threads: ${RAYON_NUM_THREADS:-default}
EOF

uv run scripts/report.py "$OUT"
echo "reports in $OUT"
