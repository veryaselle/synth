#!/bin/bash
#SBATCH --job-name=clev_ddpm
#SBATCH --array=0-4%2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/clev_ddpm_s%A_%a.out
#SBATCH --error=logs/clev_ddpm_s%A_%a.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PY="${CORE_PY:-python}"
SPLIT=$SLURM_ARRAY_TASK_ID
SEED=$((42 + SPLIT))

echo "========================================"
echo "Cleveland Conditional DDPM split_${SPLIT}"
echo "seed=${SEED}"
echo "host=$(hostname)"
echo "========================================"

$PY generate_conditional_ddpm.py \
  --real_train frozen_data/cleveland/split_${SPLIT}/real_train.csv \
  --schema frozen_data/cleveland/split_${SPLIT}/schema.json \
  --outdir results/main_benchmark/cleveland/split_${SPLIT}/ConditionalDDPM \
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

$PY evaluate_release.py \
  --dataset cleveland \
  --method "Conditional DDPM" \
  --real_train frozen_data/cleveland/split_${SPLIT}/real_train.csv \
  --real_test frozen_data/cleveland/split_${SPLIT}/real_test.csv \
  --synthetic results/main_benchmark/cleveland/split_${SPLIT}/ConditionalDDPM/synthetic_1x.csv \
  --target target \
  --numeric_cols age,trestbps,chol,thalach,oldpeak \
  --categorical_cols sex,cp,fbs,restecg,exang,slope,ca,thal \
  --aia_k 1 \
  --seed 42 \
  --outdir results/main_benchmark/cleveland/split_${SPLIT}/ConditionalDDPM/evaluation

echo "=== CONDITIONAL DDPM split_${SPLIT} COMPLETE ==="
cat results/main_benchmark/cleveland/split_${SPLIT}/ConditionalDDPM/evaluation/main_results_row.csv
