#!/bin/bash
#SBATCH --job-name=ckd_ctgan_b200
#SBATCH --array=0-4%2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ckd_ctgan_b200_%A_%a.out
#SBATCH --error=logs/ckd_ctgan_b200_%A_%a.err

set -eo pipefail
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$BENCHMARK_ROOT"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

PY="${CORE_PY:-python}"

SPLIT=$SLURM_ARRAY_TASK_ID
SEED=$((42 + SPLIT))

echo "============================================================"
echo "ckd CTGAN batch=200 split_${SPLIT}"
echo "seed=${SEED}"
echo "host=$(hostname)"
echo "============================================================"

$PY generate_sdv_arf_safe_cpu.py \
  --method ctgan \
  --real_train frozen_data/ckd/split_${SPLIT}/real_train.csv \
  --schema frozen_data/ckd/split_${SPLIT}/schema.json \
  --outdir results/gan_sensitivity/b200/ckd/split_${SPLIT}/CTGAN \
  --seed ${SEED} \
  --epochs 300 \
  --batch_size 200 \
  --joblib_n_jobs 1 \
  --verbose

echo "=== GENERATION COMPLETE ==="

$PY evaluate_release.py \
  --dataset ckd \
  --method "CTGAN" \
  --real_train frozen_data/ckd/split_${SPLIT}/real_train.csv \
  --real_test frozen_data/ckd/split_${SPLIT}/real_test.csv \
  --synthetic results/gan_sensitivity/b200/ckd/split_${SPLIT}/CTGAN/synthetic_1x.csv \
  --target target \
  --numeric_cols age,bp,bgr,bu,sc,sod,pot,hemo,pcv,wc,rc \
  --categorical_cols sg,al,su,rbc,pc,pcc,ba,htn,dm,cad,appet,pe,ane \
  --aia_k 1 \
  --seed 42 \
  --outdir results/gan_sensitivity/b200/ckd/split_${SPLIT}/CTGAN/evaluation

echo "=== CTGAN split_${SPLIT} COMPLETE ==="
cat results/gan_sensitivity/b200/ckd/split_${SPLIT}/CTGAN/evaluation/main_results_row.csv
