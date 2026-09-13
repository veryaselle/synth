#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PY="${CORE_PY:-python}"

echo "Project root:    $PROJECT_ROOT"
echo "Analysis bundle: $SCRIPT_DIR"

if [ ! -d "$PROJECT_ROOT/results" ]; then
  echo "ERROR: results/ not found under $PROJECT_ROOT"
  echo "Run this command from the project root."
  exit 2
fi

if [ ! -f "$SCRIPT_DIR/analyze_tradeoffs.py" ]; then
  echo "ERROR: analyze_tradeoffs.py not found under $SCRIPT_DIR"
  exit 2
fi

$PY "$SCRIPT_DIR/analyze_tradeoffs.py" \
  --manifest "$SCRIPT_DIR/gan_tradeoff_manifest.csv" \
  --frozen_root "$PROJECT_ROOT/results/main_benchmark" \
  --outdir "$PROJECT_ROOT/results/tradeoff_analysis"
