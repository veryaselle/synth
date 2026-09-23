#!/bin/bash
#SBATCH --job-name=pima_great
#SBATCH --partition=clara
#SBATCH --gpus=rtx2080ti:1
#SBATCH --array=0-4%1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pima_great_s%A_%a.out
#SBATCH --error=logs/pima_great_s%A_%a.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

LLM_PY="${LLM_PY:-python}"
CORE_PY="${CORE_PY:-python}"

SPLIT=$SLURM_ARRAY_TASK_ID
SEED=$((42 + SPLIT))

echo "========================================"
echo "PIMA GReaT/Qwen split_${SPLIT}"
echo "seed=${SEED}"
echo "host=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"
echo "========================================"

$LLM_PY - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

$LLM_PY generate_great_llm.py \
  --real_train frozen_data/pima/split_${SPLIT}/real_train.csv \
  --schema frozen_data/pima/split_${SPLIT}/schema.json \
  --outdir results/main_benchmark/pima/split_${SPLIT}/LLM \
  --seed ${SEED} \
  --epochs 10 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --dataloader_num_workers 2 \
  --sampling_batch_size 50 \
  --chunk_size 250 \
  --max_length 512 \
  --temperature 0.7 \
  --fp16


echo "=== GENERATION COMPLETE ==="

$CORE_PY evaluate_release.py \
  --dataset pima \
  --method LLM \
  --real_train frozen_data/pima/split_${SPLIT}/real_train.csv \
  --real_test frozen_data/pima/split_${SPLIT}/real_test.csv \
  --synthetic results/main_benchmark/pima/split_${SPLIT}/LLM/synthetic_1x.csv \
  --target Outcome \
  --numeric_cols Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age \
  --outdir results/main_benchmark/pima/split_${SPLIT}/LLM/evaluation

echo "=== LLM split_${SPLIT} COMPLETE ==="

cat results/main_benchmark/pima/split_${SPLIT}/LLM/evaluation/main_results_row.csv
