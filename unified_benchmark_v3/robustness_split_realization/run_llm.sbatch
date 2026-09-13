#!/bin/bash
#SBATCH --job-name=splitrob_llm
#SBATCH --partition=clara
#SBATCH --gpus=rtx2080ti:1
#SBATCH --array=0-14%1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:30:00
#SBATCH --output=robustness_split_realization/logs/llm_%A_%a.out
#SBATCH --error=robustness_split_realization/logs/llm_%A_%a.err
set -euo pipefail
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}}"
ROBUST_ROOT="${ROBUST_ROOT:-$BENCHMARK_ROOT/robustness_split_realization}"
cd "$BENCHMARK_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
PY="${CORE_PY:-python}"
SEED="${ROBUSTNESS_SEED:-2026}"
DATASETS=(pima cleveland ckd)
TASK="$SLURM_ARRAY_TASK_ID"
DATASET_INDEX=$((TASK / 5))
SPLIT=$((TASK % 5))
DATASET="${DATASETS[$DATASET_INDEX]}"
echo "dataset=$DATASET split=$SPLIT realization_seed=$SEED host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
"$PY" "$ROBUST_ROOT/run_stage.py" --stage llm --dataset "$DATASET" --split "$SPLIT" --realization_seed "$SEED"
