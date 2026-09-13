#!/bin/bash
set -euo pipefail
BENCHMARK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCHMARK_ROOT"
export BENCHMARK_ROOT
mkdir -p logs
bash pima_preflight.sh
REAL_JOB=$(sbatch --parsable --export=ALL,BENCHMARK_ROOT="$BENCHMARK_ROOT" pima_real_reference.sbatch)
SDV_JOB=$(sbatch --parsable --export=ALL,BENCHMARK_ROOT="$BENCHMARK_ROOT" pima_sdv_arf.sbatch)
DDPM_JOB=$(sbatch --parsable --export=ALL,BENCHMARK_ROOT="$BENCHMARK_ROOT" pima_conditional_ddpm.sbatch)
LLM_JOB=$(sbatch --parsable --export=ALL,BENCHMARK_ROOT="$BENCHMARK_ROOT" pima_great_llm_fresh.sbatch)
FINAL_JOB=$(sbatch --parsable --export=ALL,BENCHMARK_ROOT="$BENCHMARK_ROOT" --dependency=afterok:${REAL_JOB}:${SDV_JOB}:${DDPM_JOB}:${LLM_JOB} pima_finalize.sbatch)
printf 'PIMA fresh-reproduction jobs: REAL=%s SDV=%s DDPM=%s LLM=%s FINAL=%s\n' "$REAL_JOB" "$SDV_JOB" "$DDPM_JOB" "$LLM_JOB" "$FINAL_JOB"
