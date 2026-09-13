#!/bin/bash
set -eo pipefail
BENCHMARK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCHMARK_ROOT"
export BENCHMARK_ROOT
mkdir -p logs

J1=$(sbatch --parsable --export=ALL,BENCHMARK_ROOT="$BENCHMARK_ROOT" pima_ctgan_lr5e4_b100.sbatch)
J2=$(sbatch --parsable --export=ALL,BENCHMARK_ROOT="$BENCHMARK_ROOT" pima_copulagan_lr5e4_b100.sbatch)
J3=$(sbatch --parsable --export=ALL,BENCHMARK_ROOT="$BENCHMARK_ROOT" cleveland_ctgan_lr5e4_b100.sbatch)
J4=$(sbatch --parsable --export=ALL,BENCHMARK_ROOT="$BENCHMARK_ROOT" cleveland_copulagan_lr5e4_b100.sbatch)
J5=$(sbatch --parsable --export=ALL,BENCHMARK_ROOT="$BENCHMARK_ROOT" ckd_ctgan_lr5e4_b100.sbatch)
J6=$(sbatch --parsable --export=ALL,BENCHMARK_ROOT="$BENCHMARK_ROOT" ckd_copulagan_lr5e4_b100.sbatch)

echo "PIMA CTGAN:        $J1"
echo "PIMA CopulaGAN:    $J2"
echo "Cleveland CTGAN:   $J3"
echo "Cleveland Copula:  $J4"
echo "CKD CTGAN:         $J5"
echo "CKD CopulaGAN:     $J6"

echo
echo "After all jobs finish:"
echo "python finalize_gan_exceptions.py --results_root results/gan_sensitivity/lr5e4_b100 --tables_out results/gan_sensitivity/lr5e4_b100/tables --assembler assemble_main_tables.py"
