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
echo "== running elementwise_gauss2d (full sweep, this takes a while)"
OUT_DIR="$OUT/raw" cargo run --release --bin elementwise_gauss2d
echo "== running mpo_mpo_quantics (full sweep, this takes a while)"
OUT_DIR="$OUT/raw" cargo run --release --bin mpo_mpo_quantics

# repo_rev must name the source revision that actually produced this sweep, so a
# run from a modified tree is stamped with a -dirty suffix rather than a clean
# looking hash. Everything under result/ is excluded from that check: this script
# has just rewritten it, so it is dirty by construction and says nothing about
# which code ran.
DIRTY="$(git -C "$ROOT" diff --quiet -- ':(exclude)result' \
  && git -C "$ROOT" diff --cached --quiet -- ':(exclude)result' \
  || echo "-dirty")"

cat > "$OUT/run.yaml" <<EOF
profile: $PROFILE
date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
host: $(hostname)
os: $(uname -sm)
repo_rev: $(git -C "$ROOT" rev-parse HEAD)$DIRTY
tensor4all_rs_rev: $(grep -m1 -o 'rev = "[a-f0-9]*"' "$ROOT/Cargo.toml" | cut -d'"' -f2)
threads: ${RAYON_NUM_THREADS:-default}
EOF

uv run scripts/report.py "$OUT"
echo "reports in $OUT"
