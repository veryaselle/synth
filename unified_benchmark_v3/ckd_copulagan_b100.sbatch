#!/bin/bash
#SBATCH --job-name=ckd_copula_b100
#SBATCH --array=0-4%2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ckd_copula_b100_%A_%a.out
#SBATCH --error=logs/ckd_copula_b100_%A_%a.err

set -eo pipefail
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$BENCHMARK_ROOT"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PY="${CORE_PY:-python}"
SPLIT=$SLURM_ARRAY_TASK_ID
SEED=$((42 + SPLIT))
OUT=results/gan_sensitivity/ckd/split_${SPLIT}/CopulaGAN_b100

$PY generate_sdv_arf_safe_cpu.py \
  --method copulagan \
  --real_train frozen_data/ckd/split_${SPLIT}/real_train.csv \
  --schema frozen_data/ckd/split_${SPLIT}/schema.json \
  --outdir "$OUT" \
  --seed ${SEED} \
  --epochs 300 \
  --batch_size 100 \
  --discriminator_lr 0.0002 \
  --joblib_n_jobs 1 \
  --verbose

$PY evaluate_release.py \
  --dataset ckd \
  --method CopulaGAN \
  --real_train frozen_data/ckd/split_${SPLIT}/real_train.csv \
  --real_test frozen_data/ckd/split_${SPLIT}/real_test.csv \
  --synthetic "$OUT/synthetic_1x.csv" \
  --target target \
  --numeric_cols age,bp,bgr,bu,sc,sod,pot,hemo,pcv,wc,rc \
  --categorical_cols sg,al,su,rbc,pc,pcc,ba,htn,dm,cad,appet,pe,ane \
  --aia_k 1 \
  --seed 42 \
  --outdir "$OUT/evaluation"

cat "$OUT/evaluation/main_results_row.csv"
