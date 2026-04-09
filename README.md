# voyager-index

[![CI](https://github.com/ddickmann/voyager-index/actions/workflows/ci.yml/badge.svg)](https://github.com/ddickmann/voyager-index/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/voyager-index)](https://pypi.org/project/voyager-index/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**The first open-source native multi-vector index.**
Built for ColBERT, ColPali, and late-interaction retrieval at scale.

`voyager-index` ships a Rust-native **GEM graph index** — a proximity graph
built directly over vector sets — that replaces traditional HNSW for
multi-vector workloads. Search is **52x faster** than HNSW at 1024-token
sequence length, with native support for insert, delete, upsert, compaction,
WAL-based crash recovery, and sealed/active segment lifecycle.

```python
from voyager_index import Index

idx = Index("my_index", dim=128, engine="gem", seed_batch_size=64)
idx.add(embeddings, ids=[1, 2, 3], payloads=[{"title": "doc1"}, ...])
results = idx.search(query_vectors, k=10)
```

## Key Features

- **Native multi-vector graph index** — GEM proximity graph over document vector sets, not single-vector HNSW
- **52x faster search** — at 1024 tokens/doc vs HNSW with per-token MaxSim aggregation
- **Full CRUD** — insert, delete, upsert with soft-delete + compaction
- **WAL + checkpointing** — crash-safe durability for the mutable active segment
- **Sealed/active segments** — automatic segment lifecycle management
- **qCH proxy scoring** — quantized Chamfer distance for sub-millisecond candidate ranking
- **Filter-aware routing** — Roaring bitmap filter index for cluster-level payload pruning at search time
- **GPU-accelerated scoring** — optional Triton-autotuned qCH kernel for GPU-native proxy scoring (up to 2048 tokens/doc)
- **Self-healing graphs** — automatic drift detection and local graph repair for mutable segments under heavy deletes
- **Multi-modal ensembles** — per-modality codebooks with Reciprocal Rank Fusion across independent graph searches
- **Semantic shortcuts** — optional learned shortcut edges from query-positive training pairs
- **Hybrid dense+sparse** — BM25 + vector fusion via `SearchPipeline`
- **Triton MaxSim** — exact late-interaction scoring with FP16/INT8/FP8/RoQ profiles
- **Tabu Search solver** — constraint-aware context packing as an alternative to RRF
- **Docker + FastAPI** — production reference server with CRUD, preprocessing, and optimization endpoints

## Why It Is Fast

- **Document-level graph**: GEM scores entire documents in a single beam search
  traversal, avoiding the per-token search overhead of HNSW
- **qCH proxy scoring**: quantized Chamfer distance uses pre-computed centroid
  distance tables for sub-microsecond candidate scoring
- **Cluster-guided entry points**: coarse cluster routing seeds beam search
  from the most relevant graph regions
- **AVX2 kernels**: vectorized proxy scoring gathers 8 scores per instruction
- **GPU qCH path**: optional Triton-autotuned kernel scores thousands of documents
  in a single GPU launch, with PyTorch GEMM for query-centroid scores
- **Filter pruning at cluster level**: Roaring bitmap masks skip irrelevant
  clusters before graph traversal, not after
- **Sealed segments**: read-only graphs with cache-friendly flat code layouts
- **Triton MaxSim**: exact late-interaction reranking in FP16/INT8 on GPU

## Native GEM Graph Index

The core innovation is a Rust-native implementation of the GEM algorithm:

> Yao Tian, Zhoujin Tian, Xi Zhao, Ruiyuan Zhang, Xiaofang Zhou.
> "[GEM: A Native Graph-based Index for Multi-Vector Retrieval](https://arxiv.org/abs/2603.20336)."
> arXiv:2603.20336, March 2026.

GEM constructs a proximity graph directly over vector sets with set-level
clustering, qCH proxy scoring for graph construction and search, and
optional semantic shortcuts. Reference: https://github.com/sigmod26gem/sigmod26gem.

### Benchmark: GEM vs HNSW (1024 tokens, 128 dim, 100 docs)

| Metric | GEM | HNSW | Speedup |
|---|---|---|---|
| Search p50 | 1,288 us | 67,218 us | **52.2x** |
| Search p99 | 1,517 us | 73,958 us | **48.8x** |
| Build | 2,298 ms | 103 ms | — |

GEM's advantage grows with sequence length because it scores at the
document level (via quantized Chamfer), while HNSW must search per-token
and aggregate.

### Architecture

A standalone Rust crate (`src/kernels/gem_index/`) implementing the full
GEM graph index with PyO3 bindings:

- **GemSegment** — sealed read-only graph with build, search, save/load, filter-aware routing
- **PyMutableGemSegment** — writable graph with insert, delete, upsert, compact, heal, quality metrics
- **PyEnsembleGemSegment** — multi-modal ensemble with per-modality codebooks and RRF fusion
- **Diversity-heuristic neighbor selection** — HNSW-style pruning for graph quality
- **Cluster-guided entry points** — multi-entry beam search for better recall
- **Self-healing** — automatic drift detection and local repair for mutable segments
- **Semantic shortcuts** — optional learned edges from training pairs
- **Atomic persistence** — bincode + CRC32 with crash-safe atomic rename
- **GpuQchScorer** (Python) — optional GPU-native proxy scoring with Triton autotune

### Python API (`voyager_index.Index`)

```python
from voyager_index import Index, IndexBuilder

# Simple creation
idx = Index("my_index", dim=128, engine="gem", seed_batch_size=64)

# Builder pattern
idx = (IndexBuilder("my_index", dim=128)
       .with_gem(seed_batch_size=64, n_fine=128)
       .with_wal(enabled=True)
       .build())

# CRUD
idx.add(embeddings, ids=[1, 2, 3], payloads=[{"cat": "A"}, ...])
results = idx.search(query, k=10)          # -> List[SearchResult]
idx.update_payload(1, {"cat": "B"})
idx.delete([2])
page = idx.scroll(limit=100, offset=0)     # pagination
stats = idx.stats()                        # IndexStats
idx.snapshot("backup.tar.gz")              # full snapshot
```

### Implemented GEM Paper Features

- **qEMD graph construction** (Section 4.2): Graph edges built using quantized Earth Mover's Distance via Sinkhorn-regularized OT with histogram reduction, providing metric-decoupled construction for stable navigation
- **Dual-graph construction** (Section 4.3): Per-cluster local graph build with explicit cross-cluster bridge enforcement (Algorithms 1-3)
- **Adaptive cluster cutoff** (Section 4.4.2): Decision tree predictor for per-document cluster membership, trained from query-positive pairs

### Beyond the Paper: Elite Innovation Layer

- **Filter-aware routing**: Roaring bitmap filter index on sealed segments; `set_doc_payloads()` declares per-document metadata, and filtered search prunes at the cluster level via bitmap intersection before graph traversal
- **GPU-accelerated qCH scoring**: Optional `GpuQchScorer` with Triton-autotuned max-gather kernel (32–128 docs/block), PyTorch GEMM fallback, and support for document lengths up to 2048 tokens
- **Self-healing graphs**: `heal()` on mutable segments detects stale cluster representatives, reconnects isolated nodes, and cleans up edges to deleted documents; background healing thread in the segment manager
- **Multi-modal ensemble**: `PyEnsembleGemSegment` builds per-modality codebooks and graphs, searches each independently, and fuses results via 1-based Reciprocal Rank Fusion

### Future Work

- Full-scale benchmarks at 1M+ documents
- Learned search policy for adaptive ef/n_probes (trained model, deferred)

## Additional Components

- `SearchPipeline` — hybrid dense+sparse (BM25 + vector) retrieval pipeline
- `ColbertIndex` — late-interaction multivector text retrieval
- `ColPaliEngine` — multimodal multivector retrieval (PDF/image pages)
- `latence_solver` — Tabu Search knapsack solver for context packing

## Documentation

- **Quickstart**: `docs/getting-started/quickstart.md`
- **API Reference**: `docs/api/python.md`
- **Benchmarks**: `docs/benchmarks.md`
- **GEM Guide**: `docs/guides/gem-native.md`
- **ColBERT Guide**: `docs/guides/colbert.md`
- **ColPali Guide**: `docs/guides/colpali.md`

## Install

```bash
pip install voyager-index                # pure Python
pip install voyager-index[native]        # + prebuilt Rust kernels
pip install voyager-index[native,server] # + FastAPI reference server
```

### Build from source

```bash
git clone https://github.com/ddickmann/voyager-index.git
cd voyager-index
bash scripts/install_from_source.sh --cpu
```

Native Rust crates:

| Crate | Description |
|---|---|
| `latence-gem-index` | Native GEM graph index (the core) |
| `latence-gem-router` | Codebook, clustering, qCH scoring |
| `latence-hnsw` | HNSW segment wrapper (legacy) |
| `latence-solver` | Tabu Search knapsack solver |

## Docker

```bash
docker build -f deploy/reference-api/Dockerfile -t voyager-index .
docker run -p 8080:8080 -v "$(pwd)/data:/data" voyager-index
```

## Reference Server

```bash
voyager-index-server
# OpenAPI docs: http://127.0.0.1:8080/docs
```

Collection types: `dense`, `late_interaction`, `multimodal`

## Multimodal Support

Supported ColPali models:

- `collfm2` — `VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1`
- `colqwen3` — `VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1`
- `nemotron_colembed` — `nvidia/nemotron-colembed-vl-4b-v2`

## Precision Profiles

| Profile | Default | Notes |
|---|---|---|
| Exact (FP16) | Yes | Triton MaxSim, truthful baseline |
| Fast (INT8) | Opt-in | Fused Triton MaxSim |
| FP8 | No | Experimental |
| RoQ4 | No | Memory saver |

## License

The OSS foundation is Apache-2.0. See `LICENSE`.

Vendored Qdrant code under `src/kernels/vendor/qdrant/` remains Apache-2.0 under
its upstream terms. See `QDRANT_VENDORING.md` and `THIRD_PARTY_NOTICES.md`.
