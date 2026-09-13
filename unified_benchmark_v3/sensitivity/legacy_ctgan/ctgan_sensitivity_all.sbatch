#!/bin/bash
#SBATCH --job-name=ctgan_sens_all
#SBATCH --array=0-74%3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:30:00
#SBATCH --output=logs/ctgan_sens_all_%A_%a.out
#SBATCH --error=logs/ctgan_sens_all_%A_%a.err
# Optional full confirmation grid: 3 datasets x 5 configs x 5 splits = 75.
set -eo pipefail
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$BENCHMARK_ROOT"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PY="${CORE_PY:-python}"
DATASETS=(pima cleveland ckd)
CONFIG_NAMES=(A_baseline_b500_d2e4_s1 B_b100_d2e4_s1 C_b100_d1e4_s1 D_b100_d2e4_s2 E_b100_d2e5_s1)
BATCH=(500 100 100 100 100)
GLR=(0.0002 0.0002 0.0002 0.0002 0.0002)
DLR=(0.0002 0.0002 0.0001 0.0002 0.00002)
DSTEPS=(1 1 1 2 1)

TASK=$SLURM_ARRAY_TASK_ID
DATASET_INDEX=$((TASK / 25))
INNER=$((TASK % 25))
CONFIG_INDEX=$((INNER / 5))
SPLIT=$((INNER % 5))
SEED=$((42 + SPLIT))
DS=${DATASETS[$DATASET_INDEX]}
CFG=${CONFIG_NAMES[$CONFIG_INDEX]}
OUT=results/gan_sensitivity/ctgan/${DS}/split_${SPLIT}/${CFG}

case "$DS" in
  pima)
    TARGET=Outcome
    NUMERIC="Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age"
    CATEGORICAL=""
    ;;
  cleveland)
    TARGET=target
    NUMERIC="age,trestbps,chol,thalach,oldpeak"
    CATEGORICAL="sex,cp,fbs,restecg,exang,slope,ca,thal"
    ;;
  ckd)
    TARGET=target
    NUMERIC="age,bp,bgr,bu,sc,sod,pot,hemo,pcv,wc,rc"
    CATEGORICAL="sg,al,su,rbc,pc,pcc,ba,htn,dm,cad,appet,pe,ane"
    ;;
esac

$PY generate_ctgan_sensitivity.py \
  --real_train frozen_data/${DS}/split_${SPLIT}/real_train.csv \
  --schema frozen_data/${DS}/split_${SPLIT}/schema.json \
  --outdir "${OUT}" --config_name "${CFG}" \
  --seed "${SEED}" --epochs 300 \
  --batch_size "${BATCH[$CONFIG_INDEX]}" \
  --generator_lr "${GLR[$CONFIG_INDEX]}" \
  --discriminator_lr "${DLR[$CONFIG_INDEX]}" \
  --discriminator_steps "${DSTEPS[$CONFIG_INDEX]}" \
  --pac 10 --verbose

EVAL=(
 "$PY" evaluate_release.py
 --dataset "$DS"
 --method "CTGAN::${CFG}"
 --real_train "frozen_data/${DS}/split_${SPLIT}/real_train.csv"
 --real_test "frozen_data/${DS}/split_${SPLIT}/real_test.csv"
 --synthetic "${OUT}/synthetic_1x.csv"
 --target "$TARGET"
 --numeric_cols "$NUMERIC"
 --aia_k 1
 --seed 42
 --outdir "${OUT}/evaluation"
)
if [[ -n "$CATEGORICAL" ]]; then
  EVAL+=(--categorical_cols "$CATEGORICAL")
fi
"${EVAL[@]}"
