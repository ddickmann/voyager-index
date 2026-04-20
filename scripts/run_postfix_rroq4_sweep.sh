#!/bin/bash
# One process per dataset so the OS reclaims memory cleanly between runs.
# Each per-dataset jsonl is concatenated into the combined output at the end.
set -euo pipefail

cd /workspace/voyager-index
mkdir -p reports/postfix_rroq4_riem_per_dataset

OUT_COMBINED=reports/beir_postfix_rroq4_riem_cpu.jsonl
LOG_COMBINED=reports/beir_postfix_rroq4_riem_cpu.log
: > "$OUT_COMBINED"
: > "$LOG_COMBINED"

for ds in arguana fiqa nfcorpus quora scidocs scifact; do
  per_out=reports/postfix_rroq4_riem_per_dataset/${ds}.jsonl
  per_log=reports/postfix_rroq4_riem_per_dataset/${ds}.log
  echo "[$(date -Iseconds)] starting $ds" | tee -a "$LOG_COMBINED"
  PYTHONPATH=. python benchmarks/beir_benchmark.py \
      --datasets "$ds" \
      --modes cpu \
      --compression rroq4_riem \
      --n-workers 8 \
      --output "$per_out" >> "$per_log" 2>&1
  if [[ -s "$per_out" ]]; then
    cat "$per_out" >> "$OUT_COMBINED"
    echo "[$(date -Iseconds)] finished $ds OK" | tee -a "$LOG_COMBINED"
  else
    echo "[$(date -Iseconds)] $ds produced no jsonl, see $per_log" | tee -a "$LOG_COMBINED"
  fi
done

echo "[$(date -Iseconds)] all datasets done; combined: $OUT_COMBINED" | tee -a "$LOG_COMBINED"
