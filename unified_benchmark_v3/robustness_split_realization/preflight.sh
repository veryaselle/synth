#!/bin/bash
set -euo pipefail
ROBUST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="$(cd "$ROBUST_ROOT/.." && pwd)"
cd "$BENCHMARK_ROOT"
CORE="${CORE_PY:-python}"
SEED="${ROBUSTNESS_SEED:-2026}"
TMP_ROOT="${TMPDIR:-/tmp}/synth_split_realization_preflight_${USER:-user}_${SEED}_$$"
trap 'rm -rf "$TMP_ROOT"' EXIT
mkdir -p "$TMP_ROOT"

echo "=== PREPARE SECOND SPLIT REALIZATION (seed=$SEED) ==="
"$CORE" "$ROBUST_ROOT/prepare_second_realization.py" --seed "$SEED"

echo
echo "=== PYTHON / SHELL SYNTAX ==="
"$CORE" -m py_compile \
  "$ROBUST_ROOT/prepare_second_realization.py" \
  "$ROBUST_ROOT/run_stage.py" \
  "$ROBUST_ROOT/finalize_second_realization.py" \
  "$ROBUST_ROOT/compare_with_primary.py"
for f in "$ROBUST_ROOT"/*.sh "$ROBUST_ROOT"/*.sbatch; do
  bash -n "$f"
done

echo
echo "=== REAL-REFERENCE EVALUATOR SMOKE TEST ==="
"$CORE" "$ROBUST_ROOT/run_stage.py" \
  --stage real --dataset pima --split 0 --realization_seed "$SEED" \
  --results_root "$TMP_ROOT/real"

echo
echo "=== GENERATOR CONTRACT DRY RUNS ==="
for DS in pima cleveland ckd; do
  "$CORE" "$ROBUST_ROOT/run_stage.py" \
    --stage sdv_arf --dataset "$DS" --split 0 --method tvae \
    --realization_seed "$SEED" --results_root "$TMP_ROOT/dry" --dry_run
  "$CORE" "$ROBUST_ROOT/run_stage.py" \
    --stage ddpm --dataset "$DS" --split 0 \
    --realization_seed "$SEED" --results_root "$TMP_ROOT/dry" --dry_run
done

# One LLM dry-run on CKD (the widest schema) validates conditioning and adapter input
# without model fitting. Use LLM_PY if explicitly provided.
"$CORE" "$ROBUST_ROOT/run_stage.py" \
  --stage llm --dataset ckd --split 0 \
  --realization_seed "$SEED" --results_root "$TMP_ROOT/dry" --dry_run

echo
echo "SECOND-REALIZATION PREFLIGHT PASS"
echo "Data root: $ROBUST_ROOT/data/seed_${SEED}"
echo "Primary results remain untouched."
