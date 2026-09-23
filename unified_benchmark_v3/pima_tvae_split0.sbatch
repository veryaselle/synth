#!/bin/bash
#SBATCH --job-name=pima_tvae_s0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/pima_tvae_s0_%j.out
#SBATCH --error=logs/pima_tvae_s0_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHON="${CORE_PY:-python}"

echo "=== PIMA TVAE split 0 ==="
echo "Host: $(hostname)"
echo "Job:  $SLURM_JOB_ID"

$PYTHON generate_sdv_arf_safe_cpu.py \
  --method tvae \
  --real_train frozen_data/pima/split_0/real_train.csv \
  --schema frozen_data/pima/split_0/schema.json \
  --outdir results/main_benchmark/pima/split_0/TVAE \
  --seed 42 \
  --epochs 300 \
  --batch_size 500 \
  --verbose \
  --joblib_n_jobs 1

echo "=== GENERATION COMPLETE ==="

$PYTHON evaluate_release.py \
  --dataset pima \
  --method TVAE \
  --real_train frozen_data/pima/split_0/real_train.csv \
  --real_test frozen_data/pima/split_0/real_test.csv \
  --synthetic results/main_benchmark/pima/split_0/TVAE/synthetic_1x.csv \
  --target Outcome \
  --numeric_cols Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age \
  --outdir results/main_benchmark/pima/split_0/TVAE/evaluation

echo "=== EVALUATION COMPLETE ==="

cat results/main_benchmark/pima/split_0/TVAE/evaluation/main_results_row.csv
