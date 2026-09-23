#!/bin/bash
#SBATCH --job-name=ckd_great
#SBATCH --partition=clara
#SBATCH --gpus=rtx2080ti:1
#SBATCH --array=0-4%1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ckd_great_s%A_%a.out
#SBATCH --error=logs/ckd_great_s%A_%a.err
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
LLM_PY="${LLM_PY:-python}"
CORE_PY="${CORE_PY:-python}"
SPLIT=$SLURM_ARRAY_TASK_ID
SEED=$((42 + SPLIT))
echo "CKD GReaT/Qwen split_${SPLIT} seed=${SEED} host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
$LLM_PY - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available(): print("GPU:", torch.cuda.get_device_name(0))
PY
# Fresh 10-epoch training; use the fixed generator implementation that completed Cleveland.
$LLM_PY generate_great_llm_final.py --real_train frozen_data/ckd/split_${SPLIT}/real_train.csv --schema frozen_data/ckd/split_${SPLIT}/schema.json --outdir results/main_benchmark/ckd/split_${SPLIT}/LLM --seed ${SEED} --epochs 10 --batch_size 1 --gradient_accumulation_steps 8 --dataloader_num_workers 2 --sampling_batch_size 50 --chunk_size 250 --max_length 512 --temperature 0.7 --catastrophic_numeric_factor 100 --max_resample_rounds 20 --fp16
echo "=== GENERATION COMPLETE ==="
$CORE_PY evaluate_release.py --dataset ckd --method LLM --real_train frozen_data/ckd/split_${SPLIT}/real_train.csv --real_test frozen_data/ckd/split_${SPLIT}/real_test.csv --synthetic results/main_benchmark/ckd/split_${SPLIT}/LLM/synthetic_1x.csv --target target --numeric_cols age,bp,bgr,bu,sc,sod,pot,hemo,pcv,wc,rc --categorical_cols sg,al,su,rbc,pc,pcc,ba,htn,dm,cad,appet,pe,ane --aia_k 1 --seed 42 --outdir results/main_benchmark/ckd/split_${SPLIT}/LLM/evaluation
echo "=== LLM split_${SPLIT} COMPLETE ==="
cat results/main_benchmark/ckd/split_${SPLIT}/LLM/evaluation/main_results_row.csv
