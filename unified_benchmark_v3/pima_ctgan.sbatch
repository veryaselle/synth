#!/bin/bash
#SBATCH --job-name=pima_ctgan
#SBATCH --array=0-4%2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pima_ctgan_s%A_%a.out
#SBATCH --error=logs/pima_ctgan_s%A_%a.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHON="${CORE_PY:-python}"

SPLIT=$SLURM_ARRAY_TASK_ID
SEED=$((42 + SPLIT))

echo "================================"
echo "PIMA CTGAN split_${SPLIT}"
echo "seed=${SEED}"
echo "host=$(hostname)"
echo "================================"

$PYTHON generate_sdv_arf_safe_cpu.py \
  --method ctgan \
  --real_train frozen_data/pima/split_${SPLIT}/real_train.csv \
  --schema frozen_data/pima/split_${SPLIT}/schema.json \
  --outdir results/main_benchmark/pima/split_${SPLIT}/CTGAN \
  --seed ${SEED} \
  --epochs 300 \
  --batch_size 500 \
  --verbose \
  --joblib_n_jobs 1

echo "=== GENERATION COMPLETE ==="

$PYTHON evaluate_release.py \
  --dataset pima \
  --method CTGAN \
  --real_train frozen_data/pima/split_${SPLIT}/real_train.csv \
  --real_test frozen_data/pima/split_${SPLIT}/real_test.csv \
  --synthetic results/main_benchmark/pima/split_${SPLIT}/CTGAN/synthetic_1x.csv \
  --target Outcome \
  --numeric_cols Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age \
  --outdir results/main_benchmark/pima/split_${SPLIT}/CTGAN/evaluation

echo "=== CTGAN split_${SPLIT} COMPLETE ==="
cat results/main_benchmark/pima/split_${SPLIT}/CTGAN/evaluation/main_results_row.csv
