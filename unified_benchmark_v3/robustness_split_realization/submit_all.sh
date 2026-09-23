#!/bin/bash
set -euo pipefail
ROBUST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="$(cd "$ROBUST_ROOT/.." && pwd)"
cd "$BENCHMARK_ROOT"
SEED="${ROBUSTNESS_SEED:-2026}"
export ROBUSTNESS_SEED="$SEED"
export BENCHMARK_ROOT ROBUST_ROOT
mkdir -p "$ROBUST_ROOT/logs"

echo "Independent split-realization robustness experiment"
echo "realization_seed=$SEED"
echo "generator seed schedule remains 42 + split_id"
echo "evaluator seed remains 42"
echo

bash "$ROBUST_ROOT/preflight.sh"

echo
echo "Submitting second-realization jobs..."
REAL_JOB=$(sbatch --parsable --chdir="$BENCHMARK_ROOT" --export=ALL,ROBUSTNESS_SEED="$SEED",BENCHMARK_ROOT="$BENCHMARK_ROOT",ROBUST_ROOT="$ROBUST_ROOT" "$ROBUST_ROOT/run_real.sbatch")
SDV_JOB=$(sbatch --parsable --chdir="$BENCHMARK_ROOT" --export=ALL,ROBUSTNESS_SEED="$SEED",BENCHMARK_ROOT="$BENCHMARK_ROOT",ROBUST_ROOT="$ROBUST_ROOT" "$ROBUST_ROOT/run_sdv_arf.sbatch")
DDPM_JOB=$(sbatch --parsable --chdir="$BENCHMARK_ROOT" --export=ALL,ROBUSTNESS_SEED="$SEED",BENCHMARK_ROOT="$BENCHMARK_ROOT",ROBUST_ROOT="$ROBUST_ROOT" "$ROBUST_ROOT/run_ddpm.sbatch")
LLM_JOB=$(sbatch --parsable --chdir="$BENCHMARK_ROOT" --export=ALL,ROBUSTNESS_SEED="$SEED",BENCHMARK_ROOT="$BENCHMARK_ROOT",ROBUST_ROOT="$ROBUST_ROOT" "$ROBUST_ROOT/run_llm.sbatch")
FINAL_JOB=$(sbatch --parsable --chdir="$BENCHMARK_ROOT" --export=ALL,ROBUSTNESS_SEED="$SEED",BENCHMARK_ROOT="$BENCHMARK_ROOT",ROBUST_ROOT="$ROBUST_ROOT" \
  --dependency=afterok:${REAL_JOB}:${SDV_JOB}:${DDPM_JOB}:${LLM_JOB} \
  "$ROBUST_ROOT/finalize.sbatch")

echo "Submitted:"
echo "  REAL:      $REAL_JOB (15 tasks)"
echo "  SDV+ARF:   $SDV_JOB (75 tasks)"
echo "  DDPM:      $DDPM_JOB (15 tasks)"
echo "  LLM:       $LLM_JOB (15 tasks)"
echo "  FINALIZE:  $FINAL_JOB (afterok all four groups)"
echo
echo "Monitor: squeue -u $USER"
echo "Final comparison: $ROBUST_ROOT/results/seed_${SEED}/tables/primary_comparison/SUMMARY.md"
