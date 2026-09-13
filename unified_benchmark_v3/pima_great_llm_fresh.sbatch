#!/bin/bash
#SBATCH --job-name=pima_great_fresh
#SBATCH --partition=clara
#SBATCH --gpus=rtx2080ti:1
#SBATCH --array=0-4%1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pima_great_fresh_s%A_%a.out
#SBATCH --error=logs/pima_great_fresh_s%A_%a.err

set -euo pipefail
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$BENCHMARK_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
LLM_PY="${LLM_PY:-python}"
CORE_PY="${CORE_PY:-python}"
SPLIT=$SLURM_ARRAY_TASK_ID
SEED=$((42 + SPLIT))
OUT="results/main_benchmark/pima/split_${SPLIT}/LLM"

# Fresh training with the same declared 10-epoch PIMA LLM configuration.
# This is a portability/reproduction route, not byte-identical recreation of
# the preserved checkpoint used for the final thesis release.
"$LLM_PY" generate_great_llm_final.py \
  --real_train frozen_data/pima/split_${SPLIT}/real_train.csv \
  --schema frozen_data/pima/split_${SPLIT}/schema.json \
  --outdir "$OUT" \
  --seed ${SEED} --epochs 10 --batch_size 1 --gradient_accumulation_steps 8 \
  --dataloader_num_workers 2 --sampling_batch_size 50 --chunk_size 250 \
  --max_length 512 --temperature 0.7 --catastrophic_numeric_factor 100 \
  --max_resample_rounds 20 --fp16

"$CORE_PY" evaluate_release.py \
  --dataset pima --method LLM \
  --real_train frozen_data/pima/split_${SPLIT}/real_train.csv \
  --real_test frozen_data/pima/split_${SPLIT}/real_test.csv \
  --synthetic "$OUT/synthetic_1x.csv" \
  --target Outcome \
  --numeric_cols Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age \
  --outdir "$OUT/evaluation"
