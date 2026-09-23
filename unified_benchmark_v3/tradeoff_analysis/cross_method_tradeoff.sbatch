#!/bin/bash
#SBATCH --job-name=cross_tradeoff
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="${CORE_PY:-python}"

echo "Project root=$PROJECT_ROOT"
echo "Python=$PY"
echo "Starting cross-method trade-off analysis..."

"$PY" "$SCRIPT_DIR/analyze_cross_method_tradeoffs.py"

echo "DONE"
