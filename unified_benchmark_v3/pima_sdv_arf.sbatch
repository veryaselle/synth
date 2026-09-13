#!/bin/bash
#SBATCH --job-name=pima_sdv_arf
#SBATCH --array=0-24%2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=logs/pima_sdv_%A_%a.out
#SBATCH --error=logs/pima_sdv_%A_%a.err
set -euo pipefail
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$BENCHMARK_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
PY="${CORE_PY:-python}"
METHOD_KEYS=(tvae ctgan copulagan gaussian_copula arf)
METHOD_NAMES=("TVAE" "CTGAN" "CopulaGAN" "Gaussian Copula" "ARF")
METHOD_DIRS=(TVAE CTGAN CopulaGAN GaussianCopula ARF)
TASK=$SLURM_ARRAY_TASK_ID
METHOD_INDEX=$((TASK / 5))
SPLIT=$((TASK % 5))
SEED=$((42 + SPLIT))
KEY=${METHOD_KEYS[$METHOD_INDEX]}
NAME=${METHOD_NAMES[$METHOD_INDEX]}
DIR=${METHOD_DIRS[$METHOD_INDEX]}
"$PY" generate_sdv_arf_safe_cpu.py \
  --method "$KEY" \
  --real_train frozen_data/pima/split_${SPLIT}/real_train.csv \
  --schema frozen_data/pima/split_${SPLIT}/schema.json \
  --outdir results/main_benchmark/pima/split_${SPLIT}/${DIR} \
  --seed ${SEED} --epochs 300 --batch_size 500 --joblib_n_jobs 1 --verbose
"$PY" evaluate_release.py \
  --dataset pima --method "$NAME" \
  --real_train frozen_data/pima/split_${SPLIT}/real_train.csv \
  --real_test frozen_data/pima/split_${SPLIT}/real_test.csv \
  --synthetic results/main_benchmark/pima/split_${SPLIT}/${DIR}/synthetic_1x.csv \
  --target Outcome \
  --numeric_cols Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age \
  --aia_k 1 --seed 42 \
  --outdir results/main_benchmark/pima/split_${SPLIT}/${DIR}/evaluation
cat results/main_benchmark/pima/split_${SPLIT}/${DIR}/evaluation/main_results_row.csv
