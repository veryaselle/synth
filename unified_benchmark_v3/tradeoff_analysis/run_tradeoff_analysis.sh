#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="${CORE_PY:-python}"

echo "Project root: $PROJECT_ROOT"
echo "Trade-off dir: $SCRIPT_DIR"
echo "Python:        $PY"

if [ ! -d "$PROJECT_ROOT/results" ]; then
    echo "ERROR: results/ not found under $PROJECT_ROOT"
    exit 2
fi

"$PY" "$SCRIPT_DIR/analyze_tradeoffs.py"
