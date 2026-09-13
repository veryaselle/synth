#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$BENCHMARK_ROOT/logs"
cd "$BENCHMARK_ROOT"
exec sbatch --export=ALL,BENCHMARK_ROOT="$BENCHMARK_ROOT" "$SCRIPT_DIR/tradeoff_analysis.sbatch"
