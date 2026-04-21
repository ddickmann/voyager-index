## latence-solver

`latence-solver` is the Tabu Search Quadratic Knapsack solver used by `colsearch`
for both local refinement and the canonical OSS `/reference/optimize` contract.

The default build is **CPU fallback** (deterministic Rust). Optional Cargo features enable
experimental **CUDA** (NVIDIA) and **wgpu** hooks; when unavailable or disabled, the solver
falls back to the CPU backend. Python builds expose `cuda_available()` / `gpu_available()`
and `backend_status()` for runtime reporting.

Build locally (Python extension) with:

```bash
python -m pip install ./src/kernels/knapsack_solver
```

This repo currently ships the solver through local source install rather than a
separate PyPI release.

CUDA-enabled wheels require building with `--features cuda` (see `Cargo.toml`) and a suitable
CUDA toolkit/runtime on the build machine.
