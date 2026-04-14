#!/usr/bin/env bash
# Run the hybrid RRF vs Tabu benchmark with system Python (not the repo .venv).
# Use this when CUDA PyTorch and deps are installed for /usr/bin/python3 (or similar)
# and the project venv would shadow them with a CPU-only torch.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

# Prefer common system interpreters; avoids activated venv's python on PATH.
if [[ -x /usr/bin/python3 ]]; then
  PYTHON=/usr/bin/python3
elif [[ -x /usr/local/bin/python3 ]]; then
  PYTHON=/usr/local/bin/python3
else
  PYTHON="$(command -v python3 2>/dev/null || true)"
fi

if [[ -z "${PYTHON}" ]]; then
  echo "error: no python3 found (tried /usr/bin/python3, /usr/local/bin/python3, PATH)" >&2
  exit 1
fi

exec "$PYTHON" "$SCRIPT_DIR/benchmark_hybrid_rrf_vs_tabu.py" "$@"
