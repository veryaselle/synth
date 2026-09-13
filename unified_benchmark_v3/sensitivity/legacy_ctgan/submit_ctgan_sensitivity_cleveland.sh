#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")"
mkdir -p logs

PY="${CORE_PY:-python}"

for f in generate_sdv_arf_safe_cpu.py evaluate_release.py generate_ctgan_sensitivity.py summarize_ctgan_sensitivity.py; do
  test -f "$f" || { echo "MISSING: $f"; exit 1; }
done

$PY -m py_compile generate_ctgan_sensitivity.py summarize_ctgan_sensitivity.py

for SPLIT in 0 1 2 3 4; do
  test -f "frozen_data/cleveland/split_${SPLIT}/real_train.csv" || exit 1
done

$PY generate_ctgan_sensitivity.py \
  --real_train frozen_data/cleveland/split_0/real_train.csv \
  --schema frozen_data/cleveland/split_0/schema.json \
  --outdir /tmp/ctgan_sens_dry \
  --config_name B_b100_d2e4_s1 \
  --seed 42 --epochs 300 --batch_size 100 \
  --generator_lr 0.0002 --discriminator_lr 0.0002 \
  --discriminator_steps 1 --pac 10 --dry_run

JOB=$(sbatch --parsable ctgan_sensitivity_cleveland.sbatch)
FINAL=$(sbatch --parsable --dependency=afterany:${JOB} ctgan_sensitivity_finalize.sbatch)

echo "CTGAN sensitivity array: ${JOB}"
echo "Strict summary job:       ${FINAL}"
echo "sacct -j ${JOB} --format=JobID,JobName,State,Elapsed,ExitCode"
echo "Final log: logs/ctgan_sens_final_${FINAL}.out"
