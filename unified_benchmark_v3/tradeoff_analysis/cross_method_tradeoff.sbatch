#!/bin/bash
#SBATCH --job-name=cross_tradeoff
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/cross_method_%j.out
#SBATCH --error=logs/cross_method_%j.err

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
TRADEOFF_DIR="$BENCHMARK_ROOT/tradeoff_analysis"
PY="${CORE_PY:-python}"

cd "$TRADEOFF_DIR"

echo "PWD=$(pwd)"
echo "Python=$PY"
echo "Starting cross-method trade-off analysis..."

$PY analyze_cross_method_tradeoffs.py

echo "DONE"
