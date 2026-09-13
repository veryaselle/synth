#!/bin/bash
#SBATCH --job-name=ctgan_sens_final
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/ctgan_sens_final_%j.out
#SBATCH --error=logs/ctgan_sens_final_%j.err
set -eo pipefail
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$BENCHMARK_ROOT"
PY="${CORE_PY:-python}"
$PY summarize_ctgan_sensitivity.py --datasets cleveland --strict
