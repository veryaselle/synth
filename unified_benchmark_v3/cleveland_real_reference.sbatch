#!/bin/bash
#SBATCH --job-name=clev_real
#SBATCH --array=0-4%2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/clev_real_s%A_%a.out
#SBATCH --error=logs/clev_real_s%A_%a.err

set -eo pipefail
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$BENCHMARK_ROOT"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PY="${CORE_PY:-python}"
SPLIT=$SLURM_ARRAY_TASK_ID

echo "========================================"
echo "Cleveland REAL split_${SPLIT}"
echo "host=$(hostname)"
echo "========================================"

$PY evaluate_real_reference.py \
  --dataset cleveland \
  --real_train frozen_data/cleveland/split_${SPLIT}/real_train.csv \
  --real_test frozen_data/cleveland/split_${SPLIT}/real_test.csv \
  --target target \
  --numeric_cols age,trestbps,chol,thalach,oldpeak \
  --categorical_cols sex,cp,fbs,restecg,exang,slope,ca,thal \
  --seed 42 \
  --outdir results/main_benchmark/cleveland/split_${SPLIT}/REAL/evaluation

cat results/main_benchmark/cleveland/split_${SPLIT}/REAL/evaluation/main_results_row.csv
