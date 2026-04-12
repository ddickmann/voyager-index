# voyager-index

[![CI](https://github.com/ddickmann/voyager-index/actions/workflows/ci.yml/badge.svg)](https://github.com/ddickmann/voyager-index/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/voyager-index)](https://pypi.org/project/voyager-index/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**The first open-source, completely multi-vector native index.**
Built for ColBERT, ColPali, and late-interaction retrieval at scale.

`voyager-index` ships a Rust-native **GEM graph index** — a proximity graph
built directly over vector sets — that replaces traditional HNSW for
multi-vector workloads. Search is **52x faster** than HNSW at 1024-token
sequence length, with native support for insert, delete, upsert, compaction,
WAL-based crash recovery, and sealed/active segment lifecycle.

### Who is this for?

Teams building **retrieval systems with token-level or patch-level embeddings**:
ColBERT, ColPali, ColQwen, or any late-interaction model. If you store documents
as sets of vectors and need sub-100ms search at 100K+ scale — with full CRUD,
durability, and GPU acceleration — this is the engine.

### Why does it exist?

Every other open-source vector database treats multi-vector retrieval as a
**two-stage hack**: build a single-vector ANN index (HNSW/IVF), retrieve
candidates, then rerank with MaxSim. This works at small scale but breaks
down as corpora grow — the single-vector proxy discards the token-level
signal that makes late interaction powerful.

`voyager-index` is built differently. The index structure itself is set-aware:
built and traversed on set-level proxies (qCH), with MaxSim as the final truth
scorer. For small corpora the system can optionally use GPU brute-force qCH
as a fast path; at larger scales, traversal is graph-native.

### What "multi-vector native" means

We call it **multi-vector native** because all four layers operate on vector sets,
not pooled single vectors:

1. **Storage**: documents are stored and indexed as vector sets (token/patch vectors)
2. **ANN structure**: the graph is built over set-level distances (qCH/qEMD), not pooled vectors
3. **Traversal**: search uses set-aware proxies (qCH) and returns candidates without
   requiring a separate single-vector index
4. **Exact ranking**: final scoring uses late-interaction MaxSim (GPU Triton kernel),
   optionally with ROQ 4-bit quantization

This differentiates `voyager-index` from:
- "vector DB + rerank" — single-vector index with MaxSim bolted on
- "ColBERT reranker on top of single-vector ANN" — pooled proxy, non-native traversal
- "multi-vector in API only" — stores token vectors but indexes centroids

```python
from voyager_index import Index

idx = Index("my_index", dim=128, engine="gem", seed_batch_size=64)
idx.add(embeddings, ids=[1, 2, 3], payloads=[{"title": "doc1"}, ...])
results = idx.search(query_vectors, k=10)
```

## Core Innovations

| Innovation | What it does |
|---|---|
| **GEM graph index** | Proximity graph over document vector sets — single beam search scores entire documents ([paper](https://arxiv.org/abs/2603.20336)) |
| **Set-aware graph construction** | qCH/qEMD-based edge selection ensures neighbors are semantically close at the set level, not just centroid level |
| **Metric decoupling** | Graph built with qEMD (metric, stable navigation), traversed with qCH (fast proxy), reranked with MaxSim (exact) |
| **Triton Flash MaxSim** | Fused GPU kernel for exact late-interaction scoring — FP16/INT8/FP8/ROQ profiles with autotune |
| **ROQ 4-bit quantization** | Rotational quantization with fused Triton kernel — ~8x memory reduction vs FP32, p50 2.8ms rerank |
| **GPU qCH scorer** | Triton-autotuned brute-force proxy scoring on GPU for small-corpus fast path |
| **Tabu Search solver** | Constraint-aware context packing as an alternative to RRF — knapsack optimization over retrieved chunks |
| **Full CRUD + WAL** | Insert, delete, upsert with crash-safe write-ahead log and checkpoint recovery |
| **Self-healing graphs** | Automatic drift detection and local repair under heavy deletes |
| **Multimodal preprocessing** | Built-in PDF/DOCX/XLSX/image → page-image pipeline for ColPali-family models |
| **vLLM-style kernel warmup** | Pre-compilation of all Triton kernels at init for zero cold-start latency |

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
- **ROQ 4-bit reranking** — rotational quantization with fused Triton kernel (~8x memory reduction, p50 2.8ms)
- **Pre-autotune kernel warmup** — vLLM-style Triton kernel pre-compilation at init for zero cold-start latency
- **Tabu Search solver** — constraint-aware context packing as an alternative to RRF
- **Fully multimodal** — built-in preprocessing for PDF, DOCX, XLSX, and images via ColPali-family models
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

# Builder pattern with GPU reranking
idx = (IndexBuilder("my_index", dim=128)
       .with_gem(seed_batch_size=64, n_fine=128)
       .with_gpu_rerank(device="cuda")      # FP32 MaxSim reranking
       .with_wal(enabled=True)
       .build())

# Builder with ROQ 4-bit compressed reranking (~8x less memory)
idx = (IndexBuilder("my_index", dim=128)
       .with_gem(seed_batch_size=64)
       .with_roq(bits=4)                    # ROQ 4-bit + CUDA rerank
       .build())

# CRUD
idx.add(embeddings, ids=[1, 2, 3], payloads=[{"cat": "A"}, ...])
idx.upsert(new_vecs, ids=[2, 3], payloads=[{"cat": "X"}, ...])  # insert or replace
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

### Reranking Profiles

GEM's two-stage architecture uses fast qCH proxy scoring for graph traversal,
then reranks the top candidates with exact late-interaction scoring.  Three
reranking profiles are available:

| Profile | Latency (p50) | Memory | How to enable |
|---|---|---|---|
| Proxy only (qCH) | ~0.3 ms | baseline | default (no `rerank_device`) |
| FP32 MaxSim (Triton) | ~1.4 ms | 512 B/token | `rerank_device="cuda"` |
| ROQ 4-bit (Triton) | ~2.8 ms | ~68 B/token | `roq_rerank=True, roq_bits=4` |

Latencies measured on a single A100 (500 docs, 32 tokens/doc, dim=128).
The FP32 and ROQ paths include vectorized NumPy batch stacking, bulk GPU
transfer, and shape-bucketed Triton autotune for stable kernel selection.
Triton kernels are pre-compiled at initialization (vLLM-style warmup) so
the first query sees no compilation overhead.

### Benchmark Modes

The benchmark suite (`benchmarks/eval_100k.py`) evaluates six pipelines that
prove the "multi-vector native" claim at scale:

| # | Pipeline | Candidate generation | Reranking | Proves |
|---|---|---|---|---|
| 1 | CPU graph search | GEM qCH traversal | — (proxy only) | Graph navigability |
| 2 | GPU qCH → FP16 MaxSim | Brute-force qCH (GPU) | Triton MaxSim FP16 | Quality ceiling |
| 3 | GPU qCH → ROQ 4-bit | Brute-force qCH (GPU) | Triton ROQ 4-bit | Production target |
| 4 | Graph → ROQ 4-bit | GEM qCH traversal | Triton ROQ 4-bit | Hybrid sweet spot |
| 5 | **FAISS baseline** | Single-vector HNSW (mean-pooled) | Triton MaxSim FP16 | Non-native comparison |
| 6 | Fixed-recall QPS | — | — | Honest throughput at recall targets |

The benchmark reports: recall@K, p50/p95/p99 latency, QPS, nodes visited,
proxy comparisons, MaxSim dot products, and GPU memory reads — preventing
"you're just doing more work" accusations.

### Scaling Crossover Story

| Corpus | Winner | Why |
|---|---|---|
| **7.5K docs** | Graph traversal (88ms p50 @ R@10=1.0) | Already faster than brute-force (141ms) at perfect recall |
| **75K–100K** | Graph traversal (sub-linear growth) | Graph visits a fraction of corpus; brute-force grows linearly |
| **1M+** | Graph (only viable path) | Brute-force cost grows O(N); graph is sub-linear |

The single-vector FAISS baseline loses recall at the same latency budget
because mean-pooling discards the token-level information that GEM's set-aware
qCH proxy preserves. See `benchmarks/results/benchmark_report.md` for full
data once the benchmark completes.

### Benchmark Results (Placeholder)

> Full Pareto curves, cost accounting tables, and fixed-recall QPS numbers at
> 75K and 100K scale are in progress. Raw data: `benchmarks/results/eval_100k.json`.
> Report: `benchmarks/results/benchmark_report.md`.

### Future Work

- Full-scale benchmarks at 1M+ documents (v1.2 streaming build)
- Learned search policy for adaptive ef/n_probes (trained model, deferred)

### Scale Path — When to Use What

| Corpus size | Recommended mode | Default `ef` | Default `n_probes` |
|---|---|---|---|
| < 10K | Brute-force qCH → MaxSim (`GpuQchScorer`) | N/A | N/A |
| 10K – 100K | Graph → ROQ 4-bit rerank | 2000–5000 | 4 |
| 100K – 1M | Graph → ROQ 4-bit rerank (aggressive ef) | 5000–10000 | 4–8 |
| > 1M | Graph → ROQ 4-bit rerank (v1.2 streaming build) | tuned per corpus | 4–8 |

See [`docs/guides/scaling.md`](docs/guides/scaling.md) for memory formulas,
hard limits, and the v1.2/v2.0 roadmap.

## Shard Engine — LEMUR-Routed Late Interaction

For corpora that don't require a full graph index, `voyager-index` ships a
**shard engine** — a brutally simple CPU-backed / GPU-routed architecture
built on three ideas from recent retrieval research:

1. **LEMUR** (Learned Multi-Vector Retrieval) reduces multi-vector candidate
   generation to single-vector MIPS in a learned latent space, replacing
   centroid-based routing with a small MLP + FAISS ANN index.
2. **Col-Bandit** (Zero-Shot Query-Time Pruning) adaptively prunes query
   tokens at search time, cutting MaxSim compute without sacrificing recall.
3. **GPU-resident corpus** — when VRAM allows, the entire corpus lives on
   GPU as a contiguous FP16 tensor. Candidate scoring becomes a single
   gather + Triton MaxSim launch with zero CPU→GPU transfer.

The result: **sub-5ms latency at 100K scale** on a single GPU, with full
CRUD, WAL-based crash recovery, compaction, payload filters, scroll, and
ROQ 4-bit quantization — all production-wired through the same `Index` API
and reference HTTP server.

```python
from voyager_index import Index

idx = Index("shard_index", dim=128, engine="shard")
idx.add(embeddings, ids=[1, 2, 3], payloads=[{"title": "doc1"}, ...])
results = idx.search(query_vectors, k=10)
```

### When to use GEM vs Shard

| | GEM graph | Shard engine |
|---|---|---|
| **Architecture** | Rust proximity graph over vector sets | Python + LEMUR router + Triton MaxSim |
| **Best for** | Complex traversal, 1M+ scale, graph-native features | Simple deployment, <500K, GPU-resident fast path |
| **Build time** | Slower (graph construction) | Faster (LEMUR MLP + FAISS index) |
| **Dependencies** | Requires native Rust crate | Pure Python + PyTorch + FAISS |
| **CRUD** | Full (insert/delete/upsert/compact/heal) | Full (WAL + memtable + compaction) |
| **Hybrid search** | BM25 + dense + Tabu solver | BM25 + shard-routed + RRF/Tabu |

### Shard Admin (HTTP API)

```
POST   /collections/{name}/compact       # trigger memtable flush
GET    /collections/{name}/shards        # list shards with metadata
GET    /collections/{name}/shards/{id}   # shard detail
GET    /collections/{name}/wal/status    # WAL entries + memtable state
POST   /collections/{name}/checkpoint    # force WAL checkpoint
```

## Additional Components

- `SearchPipeline` — hybrid dense+sparse (BM25 + vector) retrieval pipeline
- `ColbertIndex` — late-interaction multivector text retrieval
- `ColPaliEngine` — multimodal multivector retrieval (PDF/image pages)
- `ShardSegmentManager` — LEMUR-routed shard engine with GPU MaxSim
- `latence_solver` — Tabu Search knapsack solver for context packing

## Documentation

- **[Quickstart](docs/getting-started/quickstart.md)** — 5-minute install to first search
- **[API Reference](docs/api/python.md)** — `Index`, `IndexBuilder`, `GemSegment`, GPU scorers
- **[GEM Guide](docs/guides/gem-native.md)** — production config and hyperparameter tuning
- **[Shard Engine Guide](docs/guides/shard-engine.md)** — LEMUR-routed retrieval, admin, tuning
- **[Scaling Guide](docs/guides/scaling.md)** — memory formulas, hard limits, v1.2/v2.0 roadmap
- **[ColBERT Guide](docs/guides/colbert.md)** — text-only late-interaction retrieval
- **[ColPali Guide](docs/guides/colpali.md)** — multimodal document retrieval
- **[Benchmarks](docs/benchmarks.md)** — methodology and results

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

Collection types: `dense`, `late_interaction`, `multimodal`, `shard`

## Multimodal Support

`voyager-index` ships a complete multimodal pipeline: ingest any document
format, render to page images, embed with ColPali-family models, and index
natively in GEM — all in one system.

**Preprocessing** (`voyager_index.preprocessing`):
- PDF → page images via PyMuPDF
- DOCX → rendered text pages via python-docx + Pillow
- XLSX → per-sheet rendered tables via openpyxl + Pillow
- Images passed through directly (PNG, JPG, WebP, GIF)

**Embedding models** (via vLLM pooling):

| Alias | Model | Architecture |
|---|---|---|
| `collfm2` | `VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1` | LFM2-VL + ColPali pooling |
| `colqwen3` | `VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1` | Qwen3-VL + ColPali pooling |
| `nemotron_colembed` | `nvidia/nemotron-colembed-vl-4b-v2` | Qwen3-VL bidirectional + ColBERT output |

**Multi-modal ensemble** (`PyEnsembleGemSegment`): builds per-modality
codebooks and graphs, searches each independently, and fuses results via
Reciprocal Rank Fusion.

```python
from voyager_index.preprocessing import enumerate_renderable_documents, render_documents

docs = enumerate_renderable_documents("./my_docs/")
pages = render_documents(docs["documents"], "./rendered/")
# → pages["rendered"] = list of {image_path, page_number, doc_id, ...}
# embed page images with ColPali, then idx.add(embeddings)
```

## Architecture Pipeline

```
Documents (PDF/DOCX/XLSX/images)
  → Preprocessing (page rendering)
  → Embedding (ColBERT / ColPali via vLLM)
  → Two-stage codebook (K-means quantization)
  → GEM graph index (set-aware proximity graph)
  → Search: qCH proxy traversal → candidate set
  → Rerank: Triton MaxSim (FP16 / ROQ 4-bit)
  → Results with payload filters
```

## Precision Profiles

| Profile | Kernel | Memory per token | Notes |
|---|---|---|---|
| Exact (FP16) | `fast_colbert_scores` | 256 B | Truthful baseline |
| Fast (INT8) | `fast_colbert_scores` | 128 B | Fused Triton MaxSim |
| FP8 | `fast_colbert_scores` | 128 B | Experimental |
| ROQ 8-bit | `roq_maxsim_8bit` | ~136 B | Full ROQ ladder |
| ROQ 4-bit | `roq_maxsim_4bit` | ~68 B | **Production recommended** — ~8x reduction vs FP32 |
| ROQ 2-bit | `roq_maxsim_2bit` | ~36 B | Aggressive compression |
| ROQ 1-bit | `roq_maxsim_1bit` | ~20 B | Maximum compression |

## License

Apache-2.0. See `LICENSE`.

Vendored Qdrant code under `research/vendor/qdrant/` remains Apache-2.0 under
its upstream terms. See `THIRD_PARTY_NOTICES.md`.
