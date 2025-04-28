#!/usr/bin/env bash
# Simple wrapper to run the factor analysis pipeline.
# Usage: ./run_factor_analysis.sh <input-data-file> [output-prefix]

if [ $# -lt 1 ]; then
  echo "Usage: $0 <input-data-file> [output-prefix]"
  exit 1
fi

INPUT=$1
PREFIX=${2:-results/run1}

python factor_analysis_pipeline.py \
  --input "$INPUT" \
  --id-col Year \
  --output-prefix "$PREFIX"
