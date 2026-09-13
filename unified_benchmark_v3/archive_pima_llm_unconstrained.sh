#!/bin/bash
set -euo pipefail

BENCHMARK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCHMARK_ROOT"
ROOT="results/main_benchmark/pima"
LLM_PY="${LLM_PY:-python}"

# Preflight first: do not move anything unless all five source runs exist and
# none of the archive destinations already exists.
for SPLIT in 0 1 2 3 4; do
  SRC="${ROOT}/split_${SPLIT}/LLM"
  DST="${ROOT}/split_${SPLIT}/LLM_unconstrained_epoch10"

  if [[ ! -d "${SRC}" ]]; then
    echo "ERROR: missing source directory: ${SRC}" >&2
    exit 1
  fi
  if [[ -e "${DST}" ]]; then
    echo "ERROR: archive destination already exists: ${DST}" >&2
    exit 1
  fi
  if [[ ! -d "${SRC}/saved_model" ]]; then
    echo "ERROR: saved_model missing in ${SRC}; refusing to archive." >&2
    exit 1
  fi
  if [[ ! -f "${SRC}/generation_metadata.json" ]]; then
    echo "ERROR: generation_metadata.json missing in ${SRC}; refusing to archive." >&2
    exit 1
  fi
  EPOCHS=$($LLM_PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["epochs"])' "${SRC}/generation_metadata.json")
  if [[ "${EPOCHS}" != "10" && "${EPOCHS}" != "10.0" ]]; then
    echo "ERROR: ${SRC} reports epochs=${EPOCHS}, expected 10." >&2
    exit 1
  fi
done

for SPLIT in 0 1 2 3 4; do
  SRC="${ROOT}/split_${SPLIT}/LLM"
  DST="${ROOT}/split_${SPLIT}/LLM_unconstrained_epoch10"
  mv "${SRC}" "${DST}"
  echo "Archived split_${SPLIT}: ${DST}"
done

echo "All five unconstrained 10-epoch LLM runs archived safely."
