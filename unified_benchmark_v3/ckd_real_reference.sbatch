#!/bin/bash
#SBATCH --job-name=ckd_real
#SBATCH --array=0-4%2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/ckd_real_s%A_%a.out
#SBATCH --error=logs/ckd_real_s%A_%a.err
set -eo pipefail
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$BENCHMARK_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY="${CORE_PY:-python}"
SPLIT=$SLURM_ARRAY_TASK_ID
$PY evaluate_real_reference.py --dataset ckd --real_train frozen_data/ckd/split_${SPLIT}/real_train.csv --real_test frozen_data/ckd/split_${SPLIT}/real_test.csv --target target --numeric_cols age,bp,bgr,bu,sc,sod,pot,hemo,pcv,wc,rc --categorical_cols sg,al,su,rbc,pc,pcc,ba,htn,dm,cad,appet,pe,ane --seed 42 --outdir results/main_benchmark/ckd/split_${SPLIT}/REAL/evaluation
echo "=== REAL split_${SPLIT} COMPLETE ==="
cat results/main_benchmark/ckd/split_${SPLIT}/REAL/evaluation/main_results_row.csv
