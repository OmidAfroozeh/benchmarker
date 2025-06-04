#!/usr/bin/env bash
set -euo pipefail

# Full-coverage run of benchmark_zipf_cli.py
poetry run python aggregate_variable_length.py \
  --systems DUCK_DB_MAIN UnifiedStringDictionary_lock_free_16mB \
  --length-specs 8 16 32 \
  --total-rows-list 1000000 \
  --n-unique-list 100 500 1000 2000 \
  --s-values 0.0 \
  --chunk-rows 2000000 \
  --parquet-codec zstd \
  --seed-base 42 \
  --n-parallel 1 \
  --n-runs 5 \
  --n-threads 8
