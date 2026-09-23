#!/bin/bash
#SBATCH --job-name=gan_b100_final
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/gan_b100_finalize_%j.out
#SBATCH --error=logs/gan_b100_finalize_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

PY="${CORE_PY:-python}"

$PY finalize_gan_exceptions.py \
  --results_root results/gan_sensitivity \
  --tables_out results/gan_sensitivity/tables \
  --assembler assemble_main_tables.py
