#!/bin/bash
#SBATCH --job-name=splitrob_final
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:25:00
#SBATCH --output=robustness_split_realization/logs/final_%j.out
#SBATCH --error=robustness_split_realization/logs/final_%j.err
set -euo pipefail
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}}"
ROBUST_ROOT="${ROBUST_ROOT:-$BENCHMARK_ROOT/robustness_split_realization}"
cd "$BENCHMARK_ROOT"
PY="${CORE_PY:-python}"
SEED="${ROBUSTNESS_SEED:-2026}"
"$PY" "$ROBUST_ROOT/finalize_split_realization.py" --seed "$SEED"
