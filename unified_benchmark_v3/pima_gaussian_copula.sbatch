#!/bin/bash
#SBATCH --job-name=pima_gc
#SBATCH --array=0-4%2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/pima_gc_s%A_%a.out
#SBATCH --error=logs/pima_gc_s%A_%a.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHON="${CORE_PY:-python}"

SPLIT=$SLURM_ARRAY_TASK_ID
SEED=$((42 + SPLIT))

$PYTHON generate_sdv_arf_safe_cpu.py \
  --method gaussian_copula \
  --real_train frozen_data/pima/split_${SPLIT}/real_train.csv \
  --schema frozen_data/pima/split_${SPLIT}/schema.json \
  --outdir results/main_benchmark/pima/split_${SPLIT}/GaussianCopula \
  --seed ${SEED} \
  --joblib_n_jobs 1

$PYTHON evaluate_release.py \
  --dataset pima \
  --method "Gaussian Copula" \
  --real_train frozen_data/pima/split_${SPLIT}/real_train.csv \
  --real_test frozen_data/pima/split_${SPLIT}/real_test.csv \
  --synthetic results/main_benchmark/pima/split_${SPLIT}/GaussianCopula/synthetic_1x.csv \
  --target Outcome \
  --numeric_cols Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age \
  --outdir results/main_benchmark/pima/split_${SPLIT}/GaussianCopula/evaluation
