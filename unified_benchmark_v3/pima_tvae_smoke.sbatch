#!/bin/bash
#SBATCH --job-name=pima_tvae_smoke
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/pima_tvae_smoke_%j.out
#SBATCH --error=logs/pima_tvae_smoke_%j.err

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHON="${CORE_PY:-python}"

echo "================================"
echo "PIMA TVAE SMOKE TEST"
echo "host: $(hostname)"
echo "job:  $SLURM_JOB_ID"
echo "cpus: $SLURM_CPUS_PER_TASK"
echo "================================"

$PYTHON - <<'PY'
import sys
import sdv
import torch

print("Python:", sys.executable)
print("SDV:", sdv.__version__)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY

echo "=== Starting TVAE ==="

$PYTHON generate_sdv_arf_safe_cpu.py \
  --method tvae \
  --real_train frozen_data/pima/split_0/real_train.csv \
  --schema frozen_data/pima/split_0/schema.json \
  --outdir results/main_benchmark/pima/split_0/TVAE \
  --seed 42 \
  --epochs 1 \
  --verbose \
  --joblib_n_jobs 1

echo "=== TVAE SMOKE COMPLETED SUCCESSFULLY ==="
