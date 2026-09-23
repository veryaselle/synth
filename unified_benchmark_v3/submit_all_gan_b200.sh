#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")"
mkdir -p logs

echo "Submitting GAN batch=200 sensitivity runs..."

J1=$(sbatch --parsable pima_ctgan_b200.sbatch)
J2=$(sbatch --parsable pima_copulagan_b200.sbatch)
J3=$(sbatch --parsable cleveland_ctgan_b200.sbatch)
J4=$(sbatch --parsable cleveland_copulagan_b200.sbatch)
J5=$(sbatch --parsable ckd_ctgan_b200.sbatch)
J6=$(sbatch --parsable ckd_copulagan_b200.sbatch)

echo "PIMA CTGAN:        $J1"
echo "PIMA CopulaGAN:    $J2"
echo "Cleveland CTGAN:   $J3"
echo "Cleveland Copula:  $J4"
echo "CKD CTGAN:         $J5"
echo "CKD CopulaGAN:     $J6"

echo
echo "After all six arrays finish, run:"
echo "python finalize_gan_exceptions.py --results_root results/gan_sensitivity/lr5 --tables_out results/gan_sensitivity/lr5/tables --assembler assemble_main_tables.py"
