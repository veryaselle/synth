#!/bin/bash
#SBATCH --job-name=ctgan_sens_clev
#SBATCH --array=0-24%2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ctgan_sens_clev_%A_%a.out
#SBATCH --error=logs/ctgan_sens_clev_%A_%a.err
set -eo pipefail
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$BENCHMARK_ROOT"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PY="${CORE_PY:-python}"

CONFIG_NAMES=(
  A_baseline_b500_d2e4_s1
  B_b100_d2e4_s1
  C_b100_d1e4_s1
  D_b100_d2e4_s2
  E_b100_d2e5_s1
)
BATCH=(500 100 100 100 100)
GLR=(0.0002 0.0002 0.0002 0.0002 0.0002)
DLR=(0.0002 0.0002 0.0001 0.0002 0.00002)
DSTEPS=(1 1 1 2 1)

TASK=$SLURM_ARRAY_TASK_ID
CONFIG_INDEX=$((TASK / 5))
SPLIT=$((TASK % 5))
SEED=$((42 + SPLIT))
CFG=${CONFIG_NAMES[$CONFIG_INDEX]}
OUT=results/gan_sensitivity/ctgan/cleveland/split_${SPLIT}/${CFG}

$PY generate_ctgan_sensitivity.py \
  --real_train frozen_data/cleveland/split_${SPLIT}/real_train.csv \
  --schema frozen_data/cleveland/split_${SPLIT}/schema.json \
  --outdir "${OUT}" \
  --config_name "${CFG}" \
  --seed "${SEED}" \
  --epochs 300 \
  --batch_size "${BATCH[$CONFIG_INDEX]}" \
  --generator_lr "${GLR[$CONFIG_INDEX]}" \
  --discriminator_lr "${DLR[$CONFIG_INDEX]}" \
  --discriminator_steps "${DSTEPS[$CONFIG_INDEX]}" \
  --pac 10 \
  --verbose

$PY evaluate_release.py \
  --dataset cleveland \
  --method "CTGAN::${CFG}" \
  --real_train frozen_data/cleveland/split_${SPLIT}/real_train.csv \
  --real_test frozen_data/cleveland/split_${SPLIT}/real_test.csv \
  --synthetic "${OUT}/synthetic_1x.csv" \
  --target target \
  --numeric_cols age,trestbps,chol,thalach,oldpeak \
  --categorical_cols sex,cp,fbs,restecg,exang,slope,ca,thal \
  --aia_k 1 \
  --seed 42 \
  --outdir "${OUT}/evaluation"

echo "=== COMPLETE ${CFG} split_${SPLIT} ==="
cat "${OUT}/evaluation/main_results_row.csv"
