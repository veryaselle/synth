#!/bin/bash
#SBATCH --job-name=ckd_final
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/ckd_final_%j.out
#SBATCH --error=logs/ckd_final_%j.err
set -eo pipefail
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$BENCHMARK_ROOT"
PY="${CORE_PY:-python}"
$PY finalize_ckd.py --results_root results/main_benchmark --tables_out results/main_benchmark/tables --assembler assemble_main_tables.py
