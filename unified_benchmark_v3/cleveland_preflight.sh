#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")"

CORE_PY="${CORE_PY:-python}"
LLM_PY="${LLM_PY:-python}"

echo "=== FILE / FROZEN-DATA CHECK ==="
$CORE_PY check_cleveland_frozen.py

echo
echo "=== REQUIRED SCRIPT CHECK ==="
for f in \
  evaluate_release.py \
  evaluate_real_reference.py \
  generate_sdv_arf_safe_cpu.py \
  generate_conditional_ddpm.py \
  generate_great_llm_final.py \
  assemble_main_tables.py
do
  test -f "$f" || { echo "MISSING: $f"; exit 1; }
  echo "OK: $f"
done

echo
echo "=== SDV/ARF DRY RUN ==="
for METHOD in tvae ctgan copulagan gaussian_copula arf; do
  $CORE_PY generate_sdv_arf_safe_cpu.py \
    --method "$METHOD" \
    --real_train frozen_data/cleveland/split_0/real_train.csv \
    --schema frozen_data/cleveland/split_0/schema.json \
    --outdir /tmp/cleveland_dry_${METHOD} \
    --seed 42 \
    --dry_run
done

echo
echo "=== DDPM DRY RUN ==="
$CORE_PY generate_conditional_ddpm.py \
  --real_train frozen_data/cleveland/split_0/real_train.csv \
  --schema frozen_data/cleveland/split_0/schema.json \
  --outdir /tmp/cleveland_dry_ddpm \
  --seed 42 \
  --device cpu \
  --dry_run

echo
echo "=== LLM DRY RUN ==="
$LLM_PY generate_great_llm_final.py \
  --real_train frozen_data/cleveland/split_0/real_train.csv \
  --schema frozen_data/cleveland/split_0/schema.json \
  --outdir /tmp/cleveland_dry_llm \
  --seed 42 \
  --epochs 10 \
  --catastrophic_numeric_factor 100 \
  --dry_run

echo
echo "CLEVELAND PRE-FLIGHT COMPLETE"
