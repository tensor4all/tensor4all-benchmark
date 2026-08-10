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
echo "== running elementwise_gauss2d_scaling (full sweep, this takes a while)"
OUT_DIR="$OUT/raw" cargo run --release --bin elementwise_gauss2d_scaling
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

# Wall times are machine bound for the memory heavy arms (README known issue 10),
# so the hardware that produced a sweep is part of the record, not decoration.
if [ "$(uname -s)" = "Darwin" ]; then
  CHIP="$(sysctl -n machdep.cpu.brand_string)"
  MEM_GB="$(( $(sysctl -n hw.memsize) / 1073741824 ))"
else
  CHIP="$(lscpu 2>/dev/null | sed -n 's/^Model name: *//p' | head -1)"
  CHIP="${CHIP:-unknown}"
  MEM_GB="$(awk '/MemTotal/ {printf "%d", $2 / 1048576}' /proc/meminfo 2>/dev/null || echo unknown)"
fi

# No hostname: on a public repository a DHCP name leaks the operator's
# institution and location over time, and the machine identity is better
# carried by the profile name, the label below and the hardware fields.
# Set BENCH_MACHINE to override the label.
cat > "$OUT/run.yaml" <<EOF
profile: $PROFILE
date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
machine: ${BENCH_MACHINE:-$PROFILE}
os: $(uname -sm)
chip: $CHIP
memory_gb: $MEM_GB
repo_rev: $(git -C "$ROOT" rev-parse HEAD)$DIRTY
tensor4all_rs_rev: $(grep -m1 -o 'rev = "[a-f0-9]*"' "$ROOT/Cargo.toml" | cut -d'"' -f2)
threads: ${RAYON_NUM_THREADS:-default}
EOF

# REPORT_PYTHON overrides the report runner on machines without uv, for example
# REPORT_PYTHON="$HOME/miniforge3/envs/foo/bin/python". The default stays uv.
${REPORT_PYTHON:-uv run} scripts/report.py "$OUT"
echo "reports in $OUT"
