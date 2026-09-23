#!/bin/bash
#SBATCH --job-name=pima_great_final
#SBATCH --partition=clara
#SBATCH --gpus=rtx2080ti:1
#SBATCH --array=0-4%1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pima_great_final_s%A_%a.out
#SBATCH --error=logs/pima_great_final_s%A_%a.err

set -euo pipefail
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
ROOT="results/main_benchmark/pima/split_${SPLIT}"
ARCHIVE="${ROOT}/LLM_unconstrained_epoch10"
OUT="${ROOT}/LLM"

if [[ ! -d "${ARCHIVE}/saved_model" ]]; then
  echo "ERROR: ${ARCHIVE}/saved_model not found." >&2
  echo "Run archive_pima_llm_unconstrained.sh once before submitting this job." >&2
  exit 1
fi

# Reuse the exact already-trained 10-epoch model. We change only the release
# validity rule; no retraining and no outcome-driven hyperparameter tuning.
mkdir -p "${OUT}"
if [[ ! -d "${OUT}/saved_model" ]]; then
  cp -a "${ARCHIVE}/saved_model" "${OUT}/saved_model"
fi

# Preserve training provenance beside the final validated release when present.
for F in trainer_log_history.json precision_report.json run_configuration.json; do
  if [[ -f "${ARCHIVE}/${F}" && ! -f "${OUT}/training_${F}" ]]; then
    cp "${ARCHIVE}/${F}" "${OUT}/training_${F}"
  fi
done

echo "========================================"
echo "PIMA final GReaT/Qwen split_${SPLIT}"
echo "seed=${SEED}"
echo "host=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-not-set}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "========================================"

$LLM_PY - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

$LLM_PY generate_great_llm_final.py \
  --real_train frozen_data/pima/split_${SPLIT}/real_train.csv \
  --schema frozen_data/pima/split_${SPLIT}/schema.json \
  --outdir "${OUT}" \
  --seed ${SEED} \
  --epochs 10 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --dataloader_num_workers 2 \
  --sampling_batch_size 50 \
  --chunk_size 250 \
  --max_length 512 \
  --temperature 0.7 \
  --catastrophic_numeric_factor 100 \
  --max_resample_rounds 20 \
  --fp16 \
  --reuse_saved_model

echo "=== GENERATION COMPLETE ==="

$CORE_PY evaluate_release.py \
  --dataset pima \
  --method LLM \
  --real_train frozen_data/pima/split_${SPLIT}/real_train.csv \
  --real_test frozen_data/pima/split_${SPLIT}/real_test.csv \
  --synthetic "${OUT}/synthetic_1x.csv" \
  --target Outcome \
  --numeric_cols Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age \
  --outdir "${OUT}/evaluation"

echo "=== FINAL LLM split_${SPLIT} COMPLETE ==="
cat "${OUT}/evaluation/main_results_row.csv"
echo "=== REJECTION METADATA ==="
$CORE_PY - <<PY
import json
from pathlib import Path
p = Path("${OUT}/generation_metadata.json")
m = json.loads(p.read_text())
for k in [
    "generated_rows_attempted_this_run",
    "n_rejected_rows_this_run",
    "rejection_rate_this_run",
    "rejection_reason_counts",
    "rejection_reason_by_feature",
    "numeric_support_violation_counts",
]:
    print(f"{k}: {m.get(k)}")
PY
