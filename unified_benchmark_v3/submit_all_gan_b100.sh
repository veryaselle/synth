#!/bin/bash
set -eo pipefail
mkdir -p logs
for f in \
  pima_ctgan_b100.sbatch \
  pima_copulagan_b100.sbatch \
  cleveland_ctgan_b100.sbatch \
  cleveland_copulagan_b100.sbatch \
  ckd_ctgan_b100.sbatch \
  ckd_copulagan_b100.sbatch
do
  echo "Submitting $f"
  sbatch "$f"
done
