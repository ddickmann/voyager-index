# Installation

## pip (recommended)

```bash
pip install voyager-index                # pure Python
pip install voyager-index[native]        # + Rust accelerated kernels
pip install voyager-index[native,server] # + FastAPI reference server
```

## From Source

Requires Python >= 3.10 and a Rust toolchain:

```bash
git clone https://github.com/ddickmann/voyager-index.git
cd voyager-index
bash scripts/install_from_source.sh --cpu
```

## Native Crates

The Rust crates are built with [maturin](https://github.com/PyO3/maturin):

| Crate | Description |
|---|---|
| `latence-gem-index` | Native GEM graph index (core) |
| `latence-gem-router` | Codebook, clustering, qCH scoring |
| `latence-hnsw` | HNSW segment wrapper (legacy) |
| `latence-solver` | Tabu Search knapsack solver |

Build individually:

```bash
cd src/kernels/gem_index && maturin develop --release
cd src/kernels/gem_router && maturin develop --release
```

## Verify

```python
import latence_gem_index
import latence_gem_router
print("Native modules OK")
```
