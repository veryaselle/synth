#!/bin/bash
#SBATCH --job-name=pima_ddpm
#SBATCH --array=0-4%2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/pima_ddpm_s%A_%a.out
#SBATCH --error=logs/pima_ddpm_s%A_%a.err

set -eo pipefail
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$BENCHMARK_ROOT"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHON="${CORE_PY:-python}"
SPLIT=$SLURM_ARRAY_TASK_ID
SEED=$((42 + SPLIT))

echo "================================"
echo "PIMA Conditional DDPM split_${SPLIT}"
echo "seed=${SEED}"
echo "host=$(hostname)"
echo "================================"

$PYTHON generate_conditional_ddpm.py \
  --real_train frozen_data/pima/split_${SPLIT}/real_train.csv \
  --schema frozen_data/pima/split_${SPLIT}/schema.json \
  --outdir results/main_benchmark/pima/split_${SPLIT}/ConditionalDDPM \
  --seed ${SEED} \
  --device cpu \
  --epochs 300 \
  --timesteps 100 \
  --batch_size 64 \
  --hidden_dim 128 \
  --time_dim 32 \
  --label_dim 8 \
  --lr 0.001 \
  --weight_decay 0.00001 \
  --verbose

echo "=== GENERATION COMPLETE ==="

$PYTHON evaluate_release.py \
  --dataset pima \
  --method "Conditional DDPM" \
  --real_train frozen_data/pima/split_${SPLIT}/real_train.csv \
  --real_test frozen_data/pima/split_${SPLIT}/real_test.csv \
  --synthetic results/main_benchmark/pima/split_${SPLIT}/ConditionalDDPM/synthetic_1x.csv \
  --target Outcome \
  --numeric_cols Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age \
  --outdir results/main_benchmark/pima/split_${SPLIT}/ConditionalDDPM/evaluation

echo "=== CONDITIONAL DDPM split_${SPLIT} COMPLETE ==="

cat results/main_benchmark/pima/split_${SPLIT}/ConditionalDDPM/evaluation/main_results_row.csv
