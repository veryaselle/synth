#!/bin/bash
#SBATCH --job-name=ckd_ddpm
#SBATCH --array=0-4%2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/ckd_ddpm_s%A_%a.out
#SBATCH --error=logs/ckd_ddpm_s%A_%a.err
set -eo pipefail
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$BENCHMARK_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY="${CORE_PY:-python}"
SPLIT=$SLURM_ARRAY_TASK_ID
SEED=$((42 + SPLIT))
echo "CKD Conditional DDPM split_${SPLIT} seed=${SEED} host=$(hostname)"
$PY generate_conditional_ddpm.py --real_train frozen_data/ckd/split_${SPLIT}/real_train.csv --schema frozen_data/ckd/split_${SPLIT}/schema.json --outdir results/main_benchmark/ckd/split_${SPLIT}/ConditionalDDPM --seed ${SEED} --device cpu --epochs 300 --timesteps 100 --batch_size 64 --hidden_dim 128 --time_dim 32 --label_dim 8 --lr 0.001 --weight_decay 0.00001 --verbose
echo "=== GENERATION COMPLETE ==="
$PY evaluate_release.py --dataset ckd --method "Conditional DDPM" --real_train frozen_data/ckd/split_${SPLIT}/real_train.csv --real_test frozen_data/ckd/split_${SPLIT}/real_test.csv --synthetic results/main_benchmark/ckd/split_${SPLIT}/ConditionalDDPM/synthetic_1x.csv --target target --numeric_cols age,bp,bgr,bu,sc,sod,pot,hemo,pcv,wc,rc --categorical_cols sg,al,su,rbc,pc,pcc,ba,htn,dm,cad,appet,pe,ane --aia_k 1 --seed 42 --outdir results/main_benchmark/ckd/split_${SPLIT}/ConditionalDDPM/evaluation
echo "=== CONDITIONAL DDPM split_${SPLIT} COMPLETE ==="
cat results/main_benchmark/ckd/split_${SPLIT}/ConditionalDDPM/evaluation/main_results_row.csv
