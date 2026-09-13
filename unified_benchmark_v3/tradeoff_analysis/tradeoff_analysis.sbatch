#!/bin/bash
#SBATCH --job-name=tradeoff_analysis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/tradeoff_analysis_%j.out
#SBATCH --error=logs/tradeoff_analysis_%j.err

set -eo pipefail

# Prefer: bash tradeoff_analysis/submit_tradeoff_analysis.sh
# The submit wrapper exports BENCHMARK_ROOT explicitly.

BENCHMARK_ROOT="${BENCHMARK_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PROJECT_ROOT="$BENCHMARK_ROOT"
SCRIPT_DIR="$BENCHMARK_ROOT/tradeoff_analysis"

mkdir -p "$PROJECT_ROOT/logs"

PY="${CORE_PY:-python}"

echo "Project root:    $PROJECT_ROOT"
echo "Analysis bundle: $SCRIPT_DIR"

if [ ! -d "$PROJECT_ROOT/results" ]; then
  echo "ERROR: results/ not found under project root:"
  echo "  $PROJECT_ROOT"
  exit 2
fi

if [ ! -f "$SCRIPT_DIR/analyze_tradeoffs.py" ]; then
  echo "ERROR: analyze_tradeoffs.py not found:"
  echo "  $SCRIPT_DIR/analyze_tradeoffs.py"
  echo
  echo "Expected launch command from project root:"
  echo "  bash tradeoff_analysis/submit_tradeoff_analysis.sh"
  exit 2
fi

cd "$PROJECT_ROOT"

$PY "$SCRIPT_DIR/analyze_tradeoffs.py" \
  --manifest "$SCRIPT_DIR/gan_tradeoff_manifest.csv" \
  --frozen_root "$PROJECT_ROOT/results/main_benchmark" \
  --outdir "$PROJECT_ROOT/results/tradeoff_analysis"
