#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")"
mkdir -p logs

echo "Running CKD preflight..."
bash ckd_preflight.sh

echo
echo "Submitting CKD benchmark..."
REAL_JOB=$(sbatch --parsable ckd_real_reference.sbatch)
SDV_JOB=$(sbatch --parsable ckd_sdv_arf.sbatch)
DDPM_JOB=$(sbatch --parsable ckd_conditional_ddpm.sbatch)
LLM_JOB=$(sbatch --parsable ckd_great_llm.sbatch)
FINAL_JOB=$(sbatch --parsable --dependency=afterany:${REAL_JOB}:${SDV_JOB}:${DDPM_JOB}:${LLM_JOB} ckd_finalize.sbatch)

echo
echo "Submitted:"
echo "  REAL:       ${REAL_JOB}"
echo "  SDV + ARF:  ${SDV_JOB}"
echo "  DDPM:       ${DDPM_JOB}"
echo "  LLM:        ${LLM_JOB}"
echo "  FINALIZE:   ${FINAL_JOB} (afterany; strict 40/40 gate)"
echo
echo "Monitor:"
echo "  squeue -u \$USER"
echo
echo "Final log:"
echo "  logs/ckd_final_${FINAL_JOB}.out"
