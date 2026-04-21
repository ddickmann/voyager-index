.PHONY: install install-dev install-native install-cpu build-native \
       test test-rust test-python lint clean help

SHELL := /bin/bash
REPO_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Full installs ────────────────────────────────────────────────────

install: ## Full install from source (editable + all native crates)
	bash scripts/install_from_source.sh --skip-system-deps

install-cpu: ## Full install with CPU-only PyTorch
	bash scripts/install_from_source.sh --skip-system-deps --cpu

install-dev: ## Editable install with dev extras (no native crates)
	python -m pip install -e ".[server,multimodal,preprocessing,dev]"

install-system-deps: ## Install system packages (apt/dnf/brew)
	bash scripts/install_from_source.sh 2>&1 | head -20 || true
	@echo "Run the full script for complete install: bash scripts/install_from_source.sh"

# ── Native crates ────────────────────────────────────────────────────

build-native: ## Build the supported Rust/PyO3 native crate
	python -m pip install maturin
	python -m pip install ./src/kernels/knapsack_solver

install-native: build-native ## Alias for build-native

build-solver: ## Build only the knapsack solver crate
	python -m pip install ./src/kernels/knapsack_solver

# ── Testing ──────────────────────────────────────────────────────────

test: test-rust test-python ## Run all tests

test-rust: ## Run Rust unit tests for supported crates
	cd src/kernels/knapsack_solver && cargo test

test-python: ## Run Python test suite
	python -m pytest tests/ -v

# ── Maintenance ──────────────────────────────────────────────────────

lint: ## Run linters
	python -m pip install ruff 2>/dev/null || true
	ruff check colsearch/ tests/

clean: ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info
	rm -rf src/kernels/knapsack_solver/target
	rm -rf src/kernels/shard_engine/target
	rm -rf research/gem_index/target
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

verify: ## Verify all native modules are importable
	python -c "import colsearch; print('colsearch OK')"
	python -c "import importlib.util; print('latence_solver OK' if importlib.util.find_spec('latence_solver') else 'latence_solver not installed (optional)')"

benchmark: ## Run the supported benchmark smoke
	python benchmarks/oss_reference_benchmark.py --device cpu --points 32 --top-k 5
