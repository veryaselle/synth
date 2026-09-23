#!/bin/bash
#SBATCH --job-name=pima_tvae
#SBATCH --array=1-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/pima_tvae_s%A_%a.out
#SBATCH --error=logs/pima_tvae_s%A_%a.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHON="${CORE_PY:-python}"
SPLIT=$SLURM_ARRAY_TASK_ID
SEED=$((42 + SPLIT))

echo "=== PIMA split_${SPLIT}, seed=${SEED} ==="

# REAL reference
$PYTHON evaluate_real_reference.py \
  --dataset pima \
  --real_train frozen_data/pima/split_${SPLIT}/real_train.csv \
  --real_test frozen_data/pima/split_${SPLIT}/real_test.csv \
  --target Outcome \
  --numeric_cols Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age \
  --outdir results/main_benchmark/pima/split_${SPLIT}/REAL/evaluation

# TVAE
$PYTHON generate_sdv_arf_safe_cpu.py \
  --method tvae \
  --real_train frozen_data/pima/split_${SPLIT}/real_train.csv \
  --schema frozen_data/pima/split_${SPLIT}/schema.json \
  --outdir results/main_benchmark/pima/split_${SPLIT}/TVAE \
  --seed ${SEED} \
  --epochs 300 \
  --batch_size 500 \
  --verbose \
  --joblib_n_jobs 1

# Unified evaluation
$PYTHON evaluate_release.py \
  --dataset pima \
  --method TVAE \
  --real_train frozen_data/pima/split_${SPLIT}/real_train.csv \
  --real_test frozen_data/pima/split_${SPLIT}/real_test.csv \
  --synthetic results/main_benchmark/pima/split_${SPLIT}/TVAE/synthetic_1x.csv \
  --target Outcome \
  --numeric_cols Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age \
  --outdir results/main_benchmark/pima/split_${SPLIT}/TVAE/evaluation

echo "=== split_${SPLIT} COMPLETE ==="
