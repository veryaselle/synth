#!/bin/bash
#SBATCH --job-name=pima_real
#SBATCH --array=0-4%2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/pima_real_s%A_%a.out
#SBATCH --error=logs/pima_real_s%A_%a.err
set -euo pipefail
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$BENCHMARK_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY="${CORE_PY:-python}"
SPLIT=$SLURM_ARRAY_TASK_ID
"$PY" evaluate_real_reference.py \
  --dataset pima \
  --real_train frozen_data/pima/split_${SPLIT}/real_train.csv \
  --real_test frozen_data/pima/split_${SPLIT}/real_test.csv \
  --target Outcome \
  --numeric_cols Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age \
  --seed 42 \
  --outdir results/main_benchmark/pima/split_${SPLIT}/REAL/evaluation
cat results/main_benchmark/pima/split_${SPLIT}/REAL/evaluation/main_results_row.csv
