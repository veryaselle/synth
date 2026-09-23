#!/bin/bash
#SBATCH --job-name=ckd_sdv_arf
#SBATCH --array=0-24%2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=logs/ckd_sdv_%A_%a.out
#SBATCH --error=logs/ckd_sdv_%A_%a.err
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
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
echo "CKD ${NAME} split_${SPLIT} seed=${SEED} host=$(hostname)"
$PY generate_sdv_arf_safe_cpu.py --method "${KEY}" --real_train frozen_data/ckd/split_${SPLIT}/real_train.csv --schema frozen_data/ckd/split_${SPLIT}/schema.json --outdir results/main_benchmark/ckd/split_${SPLIT}/${DIR} --seed ${SEED} --epochs 300 --batch_size 500 --joblib_n_jobs 1 --verbose
echo "=== GENERATION COMPLETE ==="
$PY evaluate_release.py --dataset ckd --method "${NAME}" --real_train frozen_data/ckd/split_${SPLIT}/real_train.csv --real_test frozen_data/ckd/split_${SPLIT}/real_test.csv --synthetic results/main_benchmark/ckd/split_${SPLIT}/${DIR}/synthetic_1x.csv --target target --numeric_cols age,bp,bgr,bu,sc,sod,pot,hemo,pcv,wc,rc --categorical_cols sg,al,su,rbc,pc,pcc,ba,htn,dm,cad,appet,pe,ane --aia_k 1 --seed 42 --outdir results/main_benchmark/ckd/split_${SPLIT}/${DIR}/evaluation
echo "=== ${NAME} split_${SPLIT} COMPLETE ==="
cat results/main_benchmark/ckd/split_${SPLIT}/${DIR}/evaluation/main_results_row.csv
