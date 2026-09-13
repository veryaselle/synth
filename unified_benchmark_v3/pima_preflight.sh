#!/bin/bash
set -euo pipefail
BENCHMARK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCHMARK_ROOT"
CORE_PY="${CORE_PY:-python}"
LLM_PY="${LLM_PY:-python}"
echo "=== PIMA FROZEN DATA ==="
for s in 0 1 2 3 4; do
  test -f "frozen_data/pima/split_${s}/real_train.csv"
  test -f "frozen_data/pima/split_${s}/real_test.csv"
  test -f "frozen_data/pima/split_${s}/schema.json"
done
"$CORE_PY" -m py_compile evaluate_release.py evaluate_real_reference.py generate_sdv_arf_safe_cpu.py generate_conditional_ddpm.py finalize_pima.py
"$LLM_PY" -m py_compile generate_great_llm_final.py
echo "=== SDV/ARF DRY RUN ==="
for METHOD in tvae ctgan copulagan gaussian_copula arf; do
  "$CORE_PY" generate_sdv_arf_safe_cpu.py --method "$METHOD" --real_train frozen_data/pima/split_0/real_train.csv --schema frozen_data/pima/split_0/schema.json --outdir "${TMPDIR:-/tmp}/pima_dry_${METHOD}" --seed 42 --dry_run
done
echo "=== DDPM DRY RUN ==="
"$CORE_PY" generate_conditional_ddpm.py --real_train frozen_data/pima/split_0/real_train.csv --schema frozen_data/pima/split_0/schema.json --outdir "${TMPDIR:-/tmp}/pima_dry_ddpm" --seed 42 --device cpu --dry_run
echo "=== LLM DRY RUN ==="
"$LLM_PY" generate_great_llm_final.py --real_train frozen_data/pima/split_0/real_train.csv --schema frozen_data/pima/split_0/schema.json --outdir "${TMPDIR:-/tmp}/pima_dry_llm" --seed 42 --epochs 10 --catastrophic_numeric_factor 100 --dry_run
echo "PIMA PRE-FLIGHT COMPLETE"
