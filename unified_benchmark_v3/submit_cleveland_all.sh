#!/bin/bash
set -eo pipefail
BENCHMARK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCHMARK_ROOT"
export BENCHMARK_ROOT
mkdir -p logs

echo "Running Cleveland preflight..."
bash cleveland_preflight.sh

echo
echo "Submitting Cleveland benchmark..."
REAL_JOB=$(sbatch --parsable --export=ALL,BENCHMARK_ROOT="$BENCHMARK_ROOT" cleveland_real_reference.sbatch)
SDV_JOB=$(sbatch --parsable --export=ALL,BENCHMARK_ROOT="$BENCHMARK_ROOT" cleveland_sdv_arf.sbatch)
DDPM_JOB=$(sbatch --parsable --export=ALL,BENCHMARK_ROOT="$BENCHMARK_ROOT" cleveland_conditional_ddpm.sbatch)
LLM_JOB=$(sbatch --parsable --export=ALL,BENCHMARK_ROOT="$BENCHMARK_ROOT" cleveland_great_llm.sbatch)

FINAL_JOB=$(sbatch --parsable --export=ALL,BENCHMARK_ROOT="$BENCHMARK_ROOT" \
  --dependency=afterok:${REAL_JOB}:${SDV_JOB}:${DDPM_JOB}:${LLM_JOB} \
  cleveland_finalize.sbatch)

echo
echo "Submitted:"
echo "  REAL:       ${REAL_JOB}"
echo "  SDV + ARF:  ${SDV_JOB}"
echo "  DDPM:       ${DDPM_JOB}"
echo "  LLM:        ${LLM_JOB}"
echo "  FINALIZE:   ${FINAL_JOB} (afterok dependency)"
echo
echo "Monitor:"
echo "  squeue -u \$USER"
echo
echo "Final log:"
echo "  logs/clev_final_${FINAL_JOB}.out"
