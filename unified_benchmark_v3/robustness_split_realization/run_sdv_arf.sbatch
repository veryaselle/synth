#!/bin/bash
#SBATCH --job-name=splitrob_sdv
#SBATCH --array=0-74%3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:45:00
#SBATCH --output=robustness_split_realization/logs/sdv_%A_%a.out
#SBATCH --error=robustness_split_realization/logs/sdv_%A_%a.err
set -euo pipefail
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}}"
ROBUST_ROOT="${ROBUST_ROOT:-$BENCHMARK_ROOT/robustness_split_realization}"
cd "$BENCHMARK_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
PY="${CORE_PY:-python}"
SEED="${ROBUSTNESS_SEED:-2026}"
DATASETS=(pima cleveland ckd)
METHODS=(tvae ctgan copulagan gaussian_copula arf)
TASK="$SLURM_ARRAY_TASK_ID"
DATASET_INDEX=$((TASK / 25))
REM=$((TASK % 25))
METHOD_INDEX=$((REM / 5))
SPLIT=$((REM % 5))
DATASET="${DATASETS[$DATASET_INDEX]}"
METHOD="${METHODS[$METHOD_INDEX]}"
"$PY" "$ROBUST_ROOT/run_stage.py" --stage sdv_arf --dataset "$DATASET" --split "$SPLIT" --method "$METHOD" --realization_seed "$SEED"
