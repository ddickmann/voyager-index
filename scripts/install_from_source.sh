#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# install_from_source.sh — one-command source install of colsearch
#
# Installs system dependencies, Rust toolchain, the Python package in
# editable mode, the optional graph dependency, and the supported native packages:
#   - latence_shard_engine (Rust shard CPU fast-path)
#   - latence_solver       (Tabu Search knapsack solver)
#
# Usage:
#   bash scripts/install_from_source.sh          # full install
#   bash scripts/install_from_source.sh --skip-system-deps  # skip apt
#   bash scripts/install_from_source.sh --cpu     # CPU PyTorch only
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SKIP_SYSTEM_DEPS=0
CPU_TORCH=0

for arg in "$@"; do
  case "$arg" in
    --skip-system-deps) SKIP_SYSTEM_DEPS=1 ;;
    --cpu)              CPU_TORCH=1 ;;
    *)                  echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

echo "══════════════════════════════════════════════════"
echo "  colsearch: install from source"
echo "══════════════════════════════════════════════════"

# ── 1. System dependencies ───────────────────────────────────────────
if [ "$SKIP_SYSTEM_DEPS" -eq 0 ]; then
  echo ""
  echo "▶ Installing system dependencies..."
  if command -v apt-get &>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq --no-install-recommends \
      build-essential curl pkg-config libssl-dev
  elif command -v dnf &>/dev/null; then
    sudo dnf install -y gcc gcc-c++ make curl openssl-devel pkg-config
  elif command -v brew &>/dev/null; then
    brew install openssl pkg-config
  else
    echo "  ⚠ Unknown package manager — skipping system deps."
    echo "  Make sure you have: gcc, make, curl, openssl headers."
  fi
fi

# ── 2. Rust toolchain ────────────────────────────────────────────────
if ! command -v cargo &>/dev/null; then
  echo ""
  echo "▶ Installing Rust toolchain via rustup..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
  # shellcheck source=/dev/null
  source "$HOME/.cargo/env"
else
  echo ""
  echo "▶ Rust already installed: $(cargo --version)"
fi

# ── 3. Python dependencies ───────────────────────────────────────────
echo ""
echo "▶ Upgrading pip, setuptools, maturin..."
python -m pip install --upgrade pip setuptools wheel maturin

# ── 4. CPU PyTorch (optional) ────────────────────────────────────────
if [ "$CPU_TORCH" -eq 1 ]; then
  echo ""
  echo "▶ Installing CPU-only PyTorch..."
  python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
fi

# ── 5. Main Python package (editable) ────────────────────────────────
echo ""
echo "▶ Installing colsearch in editable mode with the public full CPU surface..."
cd "$REPO_ROOT"
python -m pip install -e ".[server,shard,multimodal,preprocessing,latence-graph,dev,native-build]"

# ── 6. Native Rust extensions ────────────────────────────────────────
echo ""
echo "▶ Building supported native packages..."

NATIVE_CRATES=(
  "src/kernels/shard_engine"
  "src/kernels/knapsack_solver"
)

for crate in "${NATIVE_CRATES[@]}"; do
  crate_path="$REPO_ROOT/$crate"
  if [ -d "$crate_path" ]; then
    echo "  → Building $crate ..."
    python -m pip install "$crate_path"
  else
    echo "  ⚠ Skipping $crate (directory not found)"
  fi
done

# ── 7. Verify imports ────────────────────────────────────────────────
echo ""
echo "▶ Verifying installation..."
python -c "
import colsearch
print(f'  ✓ colsearch loaded ({colsearch.__file__})')

try:
    import latence_shard_engine
    print(f'  ✓ latence_shard_engine loaded ({latence_shard_engine.__file__})')
except ImportError as e:
    print(f'  ✗ latence_shard_engine: {e}')

try:
    import latence_solver
    print(f'  ✓ latence_solver loaded ({latence_solver.__file__})')
except ImportError as e:
    print(f'  ✗ latence_solver: {e}')
"

echo ""
echo "══════════════════════════════════════════════════"
echo "  ✓ colsearch installed successfully"
echo "══════════════════════════════════════════════════"
