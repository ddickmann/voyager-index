# Benchmarks

## GEM vs HNSW — 1024 Token Sequence Length

Configuration:
- 100 documents, 1024 tokens/doc, 128 dimensions
- 100 search queries, k=10
- GEM: n_fine=64, n_coarse=8, max_degree=16, ef_construction=100
- HNSW: m=16, ef_construct=100, per-token search + MaxSim aggregation (32 tokens)

| Metric | GEM | HNSW | GEM Speedup |
|---|---|---|---|
| Build time | 2,298 ms | 103 ms | — |
| **Search p50** | **1,288 us** | 67,218 us | **52.2x** |
| Search p95 | 1,471 us | 71,227 us | 48.4x |
| **Search p99** | **1,517 us** | 73,958 us | **48.8x** |
| Search mean | 1,297 us | 67,458 us | 52.0x |

### Why GEM is faster

GEM operates natively at the **document level**: a single beam search over the
proximity graph scores entire documents using quantized Chamfer distance (qCH).
HNSW is a single-vector index that must search per-token and aggregate results
via MaxSim. At 1024 tokens per document, HNSW performs ~32 independent graph
traversals per query.

### Build time trade-off

GEM's build is slower because it trains a two-stage codebook (k-means) and
constructs a diversity-pruned proximity graph. This is a one-time cost that
pays for itself at query time. For read-heavy workloads (the common case),
the 52x search speedup far outweighs the build overhead.

## GPU-Accelerated qCH Scoring

When a CUDA GPU is available, the optional `GpuQchScorer` offloads proxy
scoring to the GPU via a Triton-autotuned max-gather kernel. This is most
beneficial for large candidate sets (1000+ documents) where the GPU can
amortize the kernel launch overhead.

| Backend | Typical latency (1000 docs, 64 queries, 256 codes) |
|---|---|
| CPU (Rust AVX2) | Baseline |
| GPU (Triton) | Faster for large batches |
| GPU (PyTorch fallback) | Faster for large batches, no Triton needed |

The GPU path is fully optional and does not affect the CPU search path.
Documents up to 2048 tokens are scored without truncation; longer documents
automatically fall back to the PyTorch path.

Run the innovation benchmarks:

```bash
python benchmarks/bench_elite_innovations.py
```

## Methodology

Benchmarks use random data with `np.random.RandomState(42)` for reproducibility.
CPU measurements are wall-clock time on a single core. GPU measurements include
kernel launch overhead.

Run the core benchmark:

```bash
python benchmarks/gem_vs_hnsw_benchmark.py
```

Results are saved to `benchmarks/results/gem_vs_hnsw_1024.json`.
