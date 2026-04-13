# Production Integration: Shard-Routed Late-Interaction Engine

## Executive Summary

The shard-routed prototype (`benchmarks/shard_bench/`) demonstrated that a graph-free,
centroid-routed, CPU-backed / GPU-scored architecture can achieve **5.7ms p50 latency**
with **perfect Recall@10** at 10k docs, using only **9MB GPU memory** regardless of
corpus size. This document describes how to promote it into the main `voyager_index`
package as a first-class index engine alongside the existing GEM and HNSW paths.

The shard-routed engine becomes the **recommended path for large corpora** (100k+ docs)
where graph construction cost and GPU memory are concerns. GEM and HNSW remain available
for workloads that benefit from graph traversal (small corpora, very high recall targets,
or pre-built graphs).

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Module Structure](#2-module-structure)
3. [Phase 1: Core Engine](#3-phase-1-core-engine)
4. [Phase 2: CRUD, WAL, and Recovery](#4-phase-2-crud-wal-and-recovery)
5. [Phase 3: Kernel Integration](#5-phase-3-kernel-integration)
6. [Phase 4: Quantization and Compression](#6-phase-4-quantization-and-compression)
7. [Phase 5: Hybrid Search](#7-phase-5-hybrid-search)
8. [Phase 6: Public API Integration](#8-phase-6-public-api-integration)
9. [Phase 7: Storage Backend](#9-phase-7-storage-backend)
10. [Phase 8: Distributed and Scaling](#10-phase-8-distributed-and-scaling)
11. [Phase 9: HTTP API and Server](#11-phase-9-http-api-and-server)
12. [Migration from Benchmark to Production](#12-migration-from-benchmark-to-production)
13. [Configuration](#13-configuration)
14. [Testing Strategy](#14-testing-strategy)
15. [Performance Targets](#15-performance-targets)
16. [Risk Register](#16-risk-register)

---

## 1. Architecture Overview

### Current Benchmark Architecture

```
Query → GPU Centroid Router → Top-K Shards → CPU Fetch (safetensors) → Per-Shard GPU MaxSim → Merge Top-K
         (0.2ms)               (8 shards)     (1.5ms warm)              (4ms)                  (0.1ms)
```

GPU-resident: centroid table (256KB) + per-query working set (~32MB).
CPU-resident: sealed safetensors shard files, memory-mapped / page-cached.
Disk-resident: same shard files, NVMe-backed for cold reads.

### Production Architecture

```
                                    ┌─────────────────────┐
                                    │   GPU (fixed ~32MB)  │
                                    │  ┌─────────────────┐ │
             ┌──────────────────┐   │  │ Centroid Table   │ │
 Query ──────│ Query Encoder    │───│  │ (1024×128×FP16)  │ │
             │ (ColBERT/ColPali)│   │  ├─────────────────┤ │
             └──────────────────┘   │  │ Working Set Buf  │ │
                                    │  │ (per-shard)      │ │
                                    │  ├─────────────────┤ │
                                    │  │ Triton MaxSim    │ │
                                    │  │ / ROQ Kernel     │ │
                                    │  └─────────────────┘ │
                                    └──────────┬──────────┘
                                               │ scores
                  ┌────────────────────────────┼────────────────────┐
                  │            CPU / Disk                           │
                  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
                  │  │ WAL      │  │ Sealed   │  │ MemTable     │  │
                  │  │ (append) │  │ Shards   │  │ (in-memory   │  │
                  │  │          │  │ (.shard) │  │  buffer)     │  │
                  │  └──────────┘  └──────────┘  └──────────────┘  │
                  │  ┌──────────┐  ┌──────────┐                    │
                  │  │ Manifest │  │ BM25     │                    │
                  │  │ (JSON)   │  │ Index    │                    │
                  │  └──────────┘  └──────────┘                    │
                  └────────────────────────────────────────────────┘
```

---

## 2. Module Structure

### Current layout (benchmark)

```
benchmarks/shard_bench/
├── config.py            → BuildConfig, SearchConfig
├── shard_store.py       → ShardStore (safetensors read/write)
├── centroid_router.py   → CentroidRouter (k-means, GPU routing)
├── fetch_pipeline.py    → FetchPipeline (per-shard fetch, pipelined)
├── maxsim_scorer.py     → score_shards_and_topk (per-shard GPU scoring)
├── baselines.py         → GPU-only MaxSim, dense FAISS
├── build_index.py       → Offline build pipeline
├── run_benchmark.py     → Benchmark harness
├── profiler.py          → Timing + memory instrumentation
├── metrics.py           → Recall, MRR, NDCG
└── report.py            → Markdown report generation
```

### Target layout (production)

```
voyager_index/_internal/inference/shard_engine/
├── __init__.py
├── config.py                  ← from config.py (production defaults)
├── segment.py                 ← NEW: ShardSegment (sealed immutable shard)
├── shard_store.py             ← from shard_store.py (safetensors backend)
├── centroid_router.py         ← from centroid_router.py (GPU routing)
├── fetch_pipeline.py          ← from fetch_pipeline.py (per-shard fetch)
├── scorer.py                  ← from maxsim_scorer.py (per-shard scoring)
├── memtable.py                ← NEW: in-memory write buffer
├── wal.py                     ← NEW: write-ahead log (modeled on gem_wal.py)
├── compaction.py              ← NEW: background shard compaction
├── manager.py                 ← NEW: ShardSegmentManager (lifecycle, CRUD)
└── builder.py                 ← NEW: ShardIndexBuilder (offline bulk build)

voyager_index/_internal/inference/index_core/
├── shard_manager.py           ← UPDATE: register ShardSegmentManager as engine
```

---

## 3. Phase 1: Core Engine

**Goal**: Promote benchmark code into a production-quality segment manager that the
existing `Index` class can use as an engine.

### 3.1 ShardSegment (sealed immutable shard)

Modeled on the Rust `GemSegment` concept but implemented in Python.

```python
class ShardSegment:
    """One sealed, immutable shard containing documents grouped by centroid."""

    def __init__(self, shard_id: int, path: Path, meta: ShardMeta):
        self.shard_id = shard_id
        self.path = path
        self.meta = meta
        self._data = None  # lazy-loaded

    def load(self, device: str = "cpu") -> Tuple[torch.Tensor, List[Tuple[int,int]], List[int]]:
        """Load shard embeddings. Uses safetensors torch loader."""
        ...

    def doc_ids(self) -> List[int]:
        ...

    @property
    def num_docs(self) -> int:
        return self.meta.num_docs
```

**Source**: Refactor from `ShardStore.load_shard()`.

### 3.2 ShardSegmentManager

This is the main integration point with the existing `Index` class.

```python
class ShardSegmentManager:
    """
    Manages the lifecycle of shard segments: routing, fetching, scoring,
    and CRUD operations.

    Implements the same interface contract as GemNativeSegmentManager
    and HnswSegmentManager so it can be registered as an engine in
    voyager_index.index.Index.
    """

    def __init__(self, config: ShardEngineConfig, device: str = "cuda"):
        self.router: CentroidRouter = ...
        self.store: ShardStore = ...
        self.pipeline: FetchPipeline = ...
        self.memtable: MemTable = ...       # Phase 2
        self.wal: WalWriter = ...           # Phase 2
        self.compactor: Compactor = ...     # Phase 2

    def search_multivector(self, query: np.ndarray, k: int, ...) -> SearchResult:
        """Route → fetch → per-shard MaxSim → merge top-k."""
        ...

    def add_multidense(self, doc_id: int, vectors: np.ndarray):
        """Write to WAL + memtable. Flush when memtable is full."""
        ...

    def delete(self, doc_id: int):
        """Tombstone in WAL + memtable. Purged on compaction."""
        ...

    def build(self, all_vectors, doc_offsets, doc_ids, ...):
        """Offline bulk build: k-means → shard assignment → pack shards."""
        ...
```

### 3.3 Engine Registration

In `voyager_index/index.py`, the `Index` class currently dispatches to `GemNativeSegmentManager`
or `HnswSegmentManager` based on the `engine` parameter. Add `"shard"` as a third option:

```python
class IndexBuilder:
    def __init__(self, engine: str = "auto", ...):
        # engine: "gem" | "hnsw" | "shard" | "auto"
        ...

    def build(self) -> Index:
        if self.engine == "shard":
            manager = ShardSegmentManager(config=self._shard_config, ...)
        elif self.engine == "gem":
            manager = GemNativeSegmentManager(...)
        ...
```

**Auto-selection heuristic**: `engine="auto"` selects `"shard"` when `corpus_size > 50_000`
(configurable threshold), `"gem"` or `"hnsw"` for smaller corpora where graph traversal
is fast and build cost is acceptable.

---

## 4. Phase 2: CRUD, WAL, and Recovery

The shard-routed architecture is naturally suited to LSM-tree-style mutability because
sealed shards are immutable.

### 4.1 Write-Ahead Log

Modeled on the existing `gem_wal.py` (`WalOp`, `WalEntry`, `WalWriter`, `WalReader`,
`CheckpointManager`). The same WAL format can be reused with minimal changes.

```
WAL operations:
  INSERT(doc_id, vectors)    → append to WAL + memtable
  DELETE(doc_id)             → append tombstone to WAL + memtable
  UPDATE(doc_id, vectors)    → DELETE + INSERT (compound op)
  CHECKPOINT(manifest_hash)  → marks a consistent snapshot
```

**Recovery**: On startup, read the WAL from the last checkpoint forward, replay
INSERT/DELETE ops into the memtable. All sealed shards are already consistent
(immutable files).

### 4.2 MemTable (in-memory write buffer)

```python
class MemTable:
    """
    In-memory buffer for recent writes. Searched alongside sealed shards.

    When the memtable exceeds a size threshold, it is flushed to a new
    sealed shard and the WAL is checkpointed.
    """

    def __init__(self, capacity: int = 10_000):
        self.docs: Dict[int, np.ndarray] = {}    # doc_id → token embeddings
        self.tombstones: Set[int] = set()         # deleted doc_ids
        self._dirty = False

    def add(self, doc_id: int, vectors: np.ndarray):
        ...

    def delete(self, doc_id: int):
        self.tombstones.add(doc_id)
        self.docs.pop(doc_id, None)

    def search(self, query: torch.Tensor, k: int) -> Tuple[List[int], List[float]]:
        """Brute-force MaxSim over the small memtable on GPU."""
        ...

    def flush(self) -> ShardSegment:
        """Seal the memtable into a new shard. Returns the new segment."""
        ...
```

### 4.3 Compaction

Background process that merges small shards, purges tombstoned documents,
and optionally re-clusters documents for better routing locality.

```python
class Compactor:
    """
    LSM-style compaction for shard segments.

    Levels:
      L0: freshly flushed memtable shards (small, many)
      L1: merged shards (medium, fewer)
      L2: re-clustered shards (large, centroid-grouped)

    Compaction is incremental and non-blocking. Readers see a consistent
    snapshot via the manifest. Writers create new shards atomically
    (write to temp file, rename, update manifest).
    """

    def compact_l0_to_l1(self):
        """Merge several small L0 shards into one L1 shard."""
        ...

    def compact_l1_to_l2(self, router: CentroidRouter):
        """Re-cluster L1 shards by centroid for optimal fetch locality."""
        ...

    def purge_tombstones(self):
        """Remove deleted docs from shards during compaction."""
        ...
```

### 4.4 Search with CRUD State

```python
def search_multivector(self, query, k):
    # 1. Route to sealed shards
    shard_ids = self.router.route(query, top_shards=...)

    # 2. Fetch + score sealed shards (per-shard MaxSim)
    sealed_ids, sealed_scores = score_shards_and_topk(query, shard_chunks, k=k)

    # 3. Score memtable (brute-force, small)
    mem_ids, mem_scores = self.memtable.search(query, k=k)

    # 4. Merge results, filter tombstones
    merged = merge_topk(sealed_ids + mem_ids, sealed_scores + mem_scores, k)
    merged = [r for r in merged if r.doc_id not in self.memtable.tombstones]

    return merged[:k]
```

---

## 5. Phase 3: Kernel Integration

### 5.1 Triton MaxSim (already used)

The benchmark already calls `fast_colbert_scores` from
`voyager_index._internal.kernels.maxsim`. In production, this remains the primary
scoring kernel. The per-shard scoring wrapper (`score_shards_and_topk`) stays as-is
but moves to `shard_engine/scorer.py`.

**Files involved**:
- `voyager_index/_internal/kernels/maxsim.py` → `fast_colbert_scores`
- `voyager_index/_internal/kernels/triton_maxsim.py` → fused Triton kernel (FP16, INT8, FP8)

### 5.2 ROQ Triton Kernels

ROQ 4-bit scoring is the largest bandwidth optimization available. Instead of
transferring FP16 embeddings (2 bytes/value) from CPU to GPU per query, transfer
4-bit codes (0.5 bytes/value) — a **4x bandwidth reduction**.

**Integration path**:

1. At **build time**, quantize shard embeddings to ROQ 4-bit codes using the existing
   `RotationalQuantizer` from `voyager_index/_internal/inference/quantization/rotational.py`.
   Store the codes and codebook in the shard safetensors file.

2. At **query time**, transfer ROQ codes (4x smaller) to GPU and score using
   `roq_maxsim_4bit` from `voyager_index/_internal/kernels/triton_roq.py`.

**Changes to `ShardStore`**:
```python
def _pack_shard(self, vectors, ..., compression):
    if compression == Compression.ROQ4:
        quantizer = RotationalQuantizer(config=RoQConfig(nbits=4))
        codes, meta = quantizer.encode(vectors)
        return {
            "roq_codes": codes,       # actual key used in shard_store.py
            "roq_meta": meta,          # actual key used in shard_store.py
            "embeddings": vectors_fp16, # FP16 fallback alongside codes
            "doc_offsets": offsets_arr,
            "doc_ids": ids_arr,
        }
```

**Changes to `scorer.py`**:
```python
def score_shards_and_topk(query, shard_chunks, k, compression="fp16"):
    for flat_emb, offsets, doc_ids in shard_chunks:
        if compression == "roq4":
            scores = roq_maxsim_4bit(query, codes, codebook, ...)
        else:
            scores = fast_colbert_scores(query, padded_emb, mask)
```

**Files involved**:
- `voyager_index/_internal/kernels/roq.py` → `roq_maxsim_4bit`
- `voyager_index/_internal/kernels/triton_roq.py` → Triton ROQ kernels
- `voyager_index/_internal/inference/quantization/rotational.py` → `RotationalQuantizer`

### 5.3 Rust MaxSim (fallback)

The Rust `maxsim_score` in `src/kernels/gem_index/src/lib.rs` uses BLAS `sgemm` and
serves as a CPU fallback when Triton is unavailable. The shard engine should fall back
to this when `TRITON_AVAILABLE is False`.

### 5.4 GPU qCH Scoring (optional)

The existing `GpuQchScorer` and Triton qCH kernels
(`triton_qch_kernel.py`) provide proxy scoring for graph construction. These are not
needed for the shard engine's query path but could be used during compaction to
compute cheap shard-level relevance bounds for routing optimization.

---

## 6. Phase 4: Quantization and Compression

### Compression modes (in shard files)

| Mode | Bytes/value | H2D bandwidth | Kernel | Status |
|------|-------------|---------------|--------|--------|
| FP16 | 2.0 | Baseline | `fast_colbert_scores` | ✅ Working |
| INT8 | 1.0 | 2x reduction | `fast_colbert_scores(quantization_mode="int8")` | ✅ Working |
| FP8 | 1.0 | 2x reduction | `fast_colbert_scores(quantization_mode="fp8")` | ✅ Available |
| ROQ 4-bit | 0.5 | 4x reduction | `roq_maxsim_4bit` | 🔧 Wire up |
| Binary | 0.125 | 16x reduction | `BinaryQuantizer` | 🔧 Wire up (coarse filter) |

### Integration with existing quantizers

All quantizers already exist in `voyager_index/_internal/inference/quantization/`:
- `ScalarQuantizer` → INT8 mode in shard files
- `RotationalQuantizer` → ROQ 4-bit mode
- `BinaryQuantizer` → optional coarse-filter pre-screening
- `ProductQuantizer` → optional, for extremely large corpora

**Build-time quantization pipeline**:
```
Raw FP32 embeddings → (optional) RotationalQuantizer.encode() → ROQ 4-bit codes
                    → (alternative) ScalarQuantizer.quantize() → INT8 packed
                    → Pack into safetensors shard file
```

**Query-time dequantization**: Happens inside the Triton kernel, not in Python.
ROQ codes are sent to GPU and the kernel dequantizes on-the-fly during MaxSim.

---

## 7. Phase 5: Hybrid Search

### 7.1 BM25 + Shard-Routed Late Interaction

The existing `HybridSearchManager` combines HNSW dense search with BM25 sparse search
and optional Tabu refinement. The shard-routed engine replaces the HNSW component:

```
                ┌───────────────────────┐
    Query ──────│ BM25 Sparse Search    │──── sparse_ids, sparse_scores
       │        │ (bm25s / InvertedIdx) │
       │        └───────────────────────┘
       │
       │        ┌───────────────────────┐
       └────────│ Shard-Routed Dense    │──── dense_ids, dense_scores
                │ (centroid → MaxSim)   │
                └───────────────────────┘
                            │
                    ┌───────┴───────┐
                    │ Fusion        │
                    │ (RRF / WSum)  │
                    └───────┬───────┘
                            │
                    ┌───────┴───────┐
                    │ Tabu Refine   │  (optional)
                    │ (knapsack)    │
                    └───────────────┘
```

**Changes to `hybrid_manager.py`**:
```python
class HybridSearchManager:
    def __init__(self, ..., dense_engine: str = "auto"):
        if dense_engine == "shard":
            self.dense_manager = ShardSegmentManager(...)
        else:
            self.dense_manager = HnswSegmentManager(...)
        self.sparse_engine = BM25sEngine(...)
        self.solver = TabuSearchSolver(...)  # optional
```

### 7.2 Tabu Search Solver Integration

The existing `TabuSearchSolver` (`latence_solver`, Rust knapsack solver) refines
hybrid search results. It works on a candidate set and optimizes for a combined
objective. The shard engine provides the dense candidate set; BM25 provides the
sparse candidate set; the solver refines the union.

No changes needed to the solver itself — it operates on (doc_id, score) pairs
regardless of how they were produced.

### 7.3 BM25 Index Co-location

The BM25 inverted index lives alongside the shard store on disk:

```
index_dir/
├── manifest.json          ← shard manifest
├── shards/                ← safetensors shard files
│   ├── shard_00000.safetensors
│   ├── shard_00001.safetensors
│   └── ...
├── router/                ← centroid router state
│   └── router_state.json
├── bm25/                  ← BM25 inverted index (bm25s format)
│   ├── vocab.json
│   ├── postings.bin
│   └── ...
├── wal/                   ← write-ahead log
│   ├── wal_000001.log
│   └── ...
└── build_meta.json
```

---

## 8. Phase 6: Public API Integration

### 8.1 Index Class

The public `Index` class in `voyager_index/index.py` is the primary user-facing API.
It already supports multiple engines via the `engine` parameter.

**Addition**: Register `"shard"` as an engine option.

```python
from voyager_index import Index, IndexBuilder

# Build a shard-routed index
builder = IndexBuilder(
    engine="shard",              # NEW: shard-routed engine
    dim=128,
    n_centroids=1024,            # routing centroids
    n_shards=256,                # storage shards
    compression="roq4",          # ROQ 4-bit compression
    uniform_shard_tokens=True,   # zero-copy view() at query time
)

# Add documents (writes to WAL + memtable, auto-flushes to shards)
for doc_id, embeddings in corpus:
    builder.add(doc_id, embeddings)

index = builder.build()

# Search
results = index.search(query_embeddings, k=10)

# CRUD
index.add(new_doc_id, new_embeddings)
index.delete(old_doc_id)
index.upsert(doc_id, updated_embeddings)
```

### 8.2 ColbertIndex Integration

The existing `ColbertIndex` in `index_core/index.py` selects strategies based on
corpus size thresholds. Add a shard-routed strategy for large corpora:

```python
class ColbertIndex:
    def _select_strategy(self):
        n = self.stats.total_docs
        if n < 5_000:
            return "triton_cache"          # everything on GPU
        elif n < 50_000:
            return "triton_mmap"           # mmap + Triton scoring
        elif n < 500_000:
            return "shard_routed"          # NEW: centroid routing + per-shard MaxSim
        else:
            return "shard_routed_roq4"     # NEW: ROQ 4-bit shards for bandwidth
```

### 8.3 SearchPipeline Integration

The `SearchPipeline` wraps `HybridSearchManager`. It already supports dense + sparse
fusion. The shard engine plugs in as the dense backend:

```python
from voyager_index import SearchPipeline

pipeline = SearchPipeline(
    dense_engine="shard",     # shard-routed dense search
    sparse_engine="bm25s",   # BM25 sparse search
    fusion="rrf",             # reciprocal rank fusion
    refinement=True,          # Tabu solver refinement
)
```

### 8.4 Existing API Compatibility

The shard engine must implement the same method signatures as the existing managers:

| Method | GemNativeSegmentManager | ShardSegmentManager |
|--------|------------------------|---------------------|
| `search_multivector(query, k)` | Graph beam search + rerank | Route → fetch → MaxSim |
| `add_multidense(doc_id, vectors)` | Graph insert | WAL + memtable |
| `delete(doc_id)` | Graph remove | Tombstone |
| `build(vectors, ...)` | K-means + graph construction | K-means + shard packing |
| `save(path)` / `load(path)` | Persist graph + router | Persist shards + manifest |
| `stats()` | Segment statistics | Shard statistics |

---

## 9. Phase 7: Storage Backend

### 9.1 Safetensors as Primary Format

The benchmark uses safetensors for shard files. This becomes the production format.

**Why safetensors over HDF5**:
- Zero-copy tensor access (mmap-friendly)
- No GIL contention on reads (unlike h5py)
- Safe (no arbitrary code execution on load)
- Fast: `safetensors.torch.load_file` directly returns torch tensors
- Concurrent reads from multiple threads (each opens independently)

**Why not the existing `Storage` / `AsyncPipelineStorage`**:
- `Storage` uses HDF5 with gzip/lzf compression and mmap. It's designed for the
  ColBERT monolithic storage pattern (one big file, mmap slices).
- The shard engine needs many small files (one per shard) for:
  - Atomic writes (create new shard without touching existing ones)
  - Independent page caching (hot shards stay cached, cold shards evict)
  - Lock-free concurrent reads
  - Simple deletion (delete shard file on compaction)

**Migration**: The existing `PinnedMemoryPool` from `async_storage.py` can be reused
for the fetch pipeline's pinned buffer pool if needed for the legacy merged-tensor
path or double-buffered mode.

### 9.2 Shard File Format

Each shard is a safetensors file with these tensors:

```
FP16 mode:
  "embeddings": float16[total_tokens, dim]
  "doc_offsets": int64[num_docs, 2]       # (start, end) pairs
  "doc_ids": int64[num_docs]

INT8 mode:
  "embeddings": int8[total_tokens, dim]
  "scales": float32[total_tokens]
  "doc_offsets": int64[num_docs, 2]
  "doc_ids": int64[num_docs]

ROQ4 mode:
  "roq_codes": uint8[total_tokens, NB]    # packed 4-bit codes
  "roq_meta": float32[total_tokens, 4]    # per-token rotation metadata
  "embeddings": float16[total_tokens, dim] # FP16 fallback for non-Triton decode
  "doc_offsets": int64[num_docs, 2]
  "doc_ids": int64[num_docs]
```

With `uniform_shard_tokens=True`, all docs have the same token count, so the
offsets are implicit (`doc_i starts at i * shard_max_tokens`). The offsets tensor
is still stored for generality but the reader uses `view()` instead of slicing.

### 9.3 Manifest Format

```json
{
  "version": 2,
  "num_shards": 250,
  "num_docs": 1000000,
  "dim": 128,
  "total_tokens": 122000000,
  "compression": "roq4",
  "uniform_shard_tokens": true,
  "shards": [
    {
      "shard_id": 0,
      "num_docs": 4200,
      "total_tokens": 504000,
      "shard_max_tokens": 120,
      "centroid_ids": [12, 45, 89, 234],
      "level": 2,
      "byte_size": 32505600,
      "file_name": "shard_00000.safetensors"
    }
  ],
  "tombstones": [1042, 5891, 23456],
  "memtable_doc_count": 347,
  "wal_sequence": 89234,
  "last_checkpoint": 89100
}
```

---

## 10. Phase 8: Distributed and Scaling

### 10.1 Single-Node Scaling

At 1M docs with ROQ 4-bit:
- Disk: 1M × 120 tokens × 128 dim × 0.5 bytes = **~7.6GB** on NVMe
- CPU RAM: page cache handles hot shards, RSS scales with working set
- GPU: ~32MB fixed (centroids + working set buffer)

At 10M docs: 76GB disk, same GPU memory. Feasible on a single node with NVMe.

### 10.2 Multi-Node (Scatter-Gather)

The existing `DistributedRouter` in `distributed/router.py` provides a scatter-gather
skeleton. For shard-routed search:

```
Client → Coordinator → [Node 0: shards 0-99]   ← scatter query
                      → [Node 1: shards 100-199] ← scatter query
                      → [Node 2: shards 200-299] ← scatter query
                      ← gather top-k, merge      ← RRF / merge
```

Each node runs a `ShardSegmentManager` over its local shard subset. The coordinator
holds a copy of the centroid router and pre-routes queries to the relevant nodes
(only nodes holding relevant shards receive the query).

**Key design decision**: Route at the coordinator level (send query only to nodes
with matching shards) vs. broadcast to all nodes. Routing at the coordinator saves
network bandwidth but requires the coordinator to know the centroid→node mapping.

### 10.3 Shard Rebalancing

When nodes are added/removed, shards are redistributed. Since shards are immutable
files, rebalancing is just file copy + manifest update. No graph repair needed.

---

## 11. Phase 9: HTTP API and Server

### 11.1 Existing Server

The existing HTTP server in `voyager_index/_internal/server/` provides REST endpoints
for collections, search, and management. It already supports multiple search strategies.

**Changes**: Add `"shard_routed"` as a search strategy option in the API.

```python
# In api/routes.py
@router.post("/collections/{name}/search")
async def search(name: str, request: SearchRequest):
    # request.strategy can now be "shard_routed"
    ...
```

### 11.2 Admin Endpoints

Add shard management endpoints:

```
POST   /collections/{name}/compact          → trigger compaction
GET    /collections/{name}/shards           → list shards with metadata
GET    /collections/{name}/shards/{id}      → shard detail (docs, size, centroids)
DELETE /collections/{name}/shards/{id}      → drop a shard (with tombstoning)
GET    /collections/{name}/wal/status       → WAL sequence, checkpoint info
POST   /collections/{name}/checkpoint       → force checkpoint
```

---

## 12. Migration from Benchmark to Production

### Step-by-step file migration

| Benchmark file | Production destination | Changes needed |
|----------------|----------------------|----------------|
| `config.py` | `shard_engine/config.py` | Add CRUD config, compaction thresholds, WAL config |
| `shard_store.py` | `shard_engine/shard_store.py` | Add ROQ pack/unpack, manifest v2 with tombstones |
| `centroid_router.py` | `shard_engine/centroid_router.py` | Add incremental centroid update for new shards |
| `fetch_pipeline.py` | `shard_engine/fetch_pipeline.py` | Minor cleanup, keep per-shard + pipelined modes |
| `maxsim_scorer.py` | `shard_engine/scorer.py` | Add ROQ scoring path, rename for clarity |
| `build_index.py` | `shard_engine/builder.py` | Integrate with `IndexBuilder` API |
| `profiler.py` | `shard_engine/profiler.py` | Keep as-is, add metrics hook integration |
| — | `shard_engine/segment.py` | NEW: ShardSegment class |
| — | `shard_engine/memtable.py` | NEW: in-memory write buffer |
| — | `shard_engine/wal.py` | NEW: reuse patterns from gem_wal.py |
| — | `shard_engine/compaction.py` | NEW: LSM-style compaction |
| — | `shard_engine/manager.py` | NEW: ShardSegmentManager |

### Code that stays in benchmarks

| File | Reason |
|------|--------|
| `baselines.py` | Comparison baselines, not production code |
| `run_benchmark.py` | Benchmark harness |
| `metrics.py` | Eval metrics (already in main test infra) |
| `report.py` | Report generation |

---

## 13. Configuration

### ShardEngineConfig

```python
@dataclass
class ShardEngineConfig:
    # Routing
    n_centroids: int = 1024           # GPU routing centroids
    n_shards: int = 256               # initial shard count
    top_shards: int = 8               # shards to fetch per query
    max_docs_exact: int = 10_000      # cap on docs scored per query

    # Storage
    dim: int = 128
    compression: str = "fp16"         # "fp16" | "int8" | "roq4"
    uniform_shard_tokens: bool = True # zero-copy view() at query time

    # CRUD
    memtable_capacity: int = 10_000   # docs before flush
    wal_dir: str = "wal/"
    wal_sync_mode: str = "fdatasync"  # "none" | "fdatasync" | "fsync"

    # Compaction
    compaction_trigger: int = 10      # L0 shards before compaction
    compaction_threads: int = 2       # background compaction workers
    recluster_on_l2: bool = True      # re-cluster during L1→L2 compaction

    # Build
    kmeans_sample_fraction: float = 0.1
    max_kmeans_iter: int = 50
    seed: int = 42
```

### Integration with existing IndexConfig

The existing `IndexConfig` in `voyager_index/_internal/inference/config.py` has
fields for PLAID, ColBERT strategies, and GEM router. Add shard engine fields:

```python
@dataclass
class IndexConfig:
    # ... existing fields ...

    # Shard engine (when strategy == "shard_routed")
    shard_n_centroids: int = 1024
    shard_n_shards: int = 256
    shard_top_shards: int = 8
    shard_compression: str = "fp16"
    shard_uniform_tokens: bool = True
```

---

## 14. Testing Strategy

### Unit Tests

| Component | Test focus |
|-----------|-----------|
| `ShardStore` | Build + load round-trip, compression modes, uniform tokens, manifest |
| `CentroidRouter` | Train + route, save/load, empty shards handling |
| `FetchPipeline` | Per-shard fetch, pipelined mode, doc count limits |
| `scorer.py` | Per-shard scoring, top-k merge, zero-copy view path |
| `MemTable` | Add/delete, search, flush to shard |
| `WAL` | Write/read, checkpoint, recovery replay |
| `Compactor` | L0→L1 merge, tombstone purge, L1→L2 re-cluster |

### Integration Tests

| Test | Validates |
|------|-----------|
| Build → search → verify recall | End-to-end correctness |
| Add → search → delete → search | CRUD lifecycle |
| Kill during write → recover → verify | WAL recovery |
| Concurrent reads during compaction | Snapshot isolation |
| 100k corpus benchmark | Performance regression gate |

### Benchmark Regression Gate

Keep the existing `benchmarks/shard_bench/run_benchmark.py` as a CI-runnable
regression gate. Fail the build if:
- Recall@10 drops below 0.95 at 10k docs
- p50 latency exceeds 15ms at 10k docs
- GPU memory exceeds 100MB at 10k docs

---

## 15. Performance Targets

### Query Latency (p50, warm cache)

| Corpus size | Target p50 | Target p95 | GPU memory |
|-------------|-----------|-----------|------------|
| 10k docs | < 6ms | < 10ms | < 50MB |
| 100k docs | < 10ms | < 20ms | < 50MB |
| 1M docs | < 15ms | < 30ms | < 50MB |
| 10M docs | < 25ms | < 50ms | < 50MB |

### Build Time

| Corpus size | Target |
|-------------|--------|
| 100k docs | < 5 min |
| 1M docs | < 30 min |
| 10M docs | < 4 hours |

### CRUD Throughput

| Operation | Target |
|-----------|--------|
| Single insert | < 1ms (WAL + memtable) |
| Single delete | < 0.5ms (tombstone) |
| Memtable flush (10k docs) | < 2s |
| L0→L1 compaction (100k docs) | < 30s |

### Storage Footprint

| Compression | Bytes/token | 1M docs (120 avg tokens) |
|-------------|-------------|--------------------------|
| FP16 | 256 | 30.7 GB |
| INT8 | 132 | 15.8 GB |
| ROQ 4-bit | 68 | 8.2 GB |

---

## 16. Risk Register

### R1: Routing quality degradation at scale

**Risk**: At 1M+ docs, centroid routing may not reduce the candidate set sharply
enough, requiring more shards (higher fetch cost) or hierarchical routing.

**Mitigation**: Benchmark at 100k, 300k, 1M, 3M to find the knee. If routing
degrades, implement two-level routing (centroids of centroids) or adaptive budget
prediction.

### R2: Cold-read latency on NVMe

**Risk**: When shard data is not in page cache, NVMe reads add 5-10ms per shard.
With 8 shards, cold p50 could be 40-80ms.

**Mitigation**: Prefetch hot shards on startup. Use mmap with `MADV_WILLNEED` for
predicted shards. Track page cache hit rate as a metric.

### R3: Compaction during high query load

**Risk**: Compaction reads and writes shard files, potentially evicting hot pages
from the page cache and causing query latency spikes.

**Mitigation**: Rate-limit compaction I/O. Use `ionice` for compaction threads.
Schedule heavy compaction during low-traffic windows. Compaction creates new files
(doesn't modify existing ones), so readers are not blocked.

### R4: MemTable search overhead for large buffers

**Risk**: Brute-force MaxSim over a 10k-doc memtable could add 5-10ms per query.

**Mitigation**: Keep memtable small (flush at 5-10k docs). For the memtable search,
use the same per-shard GPU MaxSim path (treat memtable as one virtual shard).

### R5: ROQ quantization quality loss

**Risk**: 4-bit quantization may reduce recall, especially for documents with high
token diversity.

**Mitigation**: Benchmark ROQ4 recall against FP16 baseline at each scale. Use
FP16 for small corpora where bandwidth is not the bottleneck. ROQ4 is opt-in.

### R6: Centroid staleness after heavy CRUD

**Risk**: After many inserts/deletes, the centroid table no longer represents the
actual document distribution. New documents may be poorly routed.

**Mitigation**: Track centroid assignment quality during compaction. When quality
degrades below a threshold, trigger centroid re-training (offline, non-blocking).
New centroids are swapped atomically via manifest update.

---

## 17. Feature Parity Checklist

Every feature in the existing `voyager_index` codebase, mapped to its shard engine
integration status. This is the acceptance gate: the shard engine is production-ready
when all **Required** items are ✅.

### Legend

- ✅ = Already works or trivially inherited
- 🔧 = Needs implementation (with phase number)
- ⏭️ = Deferred (not required for production launch)
- 🚫 = Not applicable to shard engine architecture

---

### 17.1 Core Index Operations (`voyager_index/index.py`)

| # | Feature | Source | Shard Engine Status |
|---|---------|--------|---------------------|
| 1 | `add` / `add_batch` multivector ingest | `Index.add` | 🔧 Phase 2 — WAL + memtable |
| 2 | `upsert` multivector | `Index.upsert` | 🔧 Phase 2 — DELETE + INSERT in WAL |
| 3 | `delete` by doc_id | `Index.delete` | 🔧 Phase 2 — tombstone in WAL |
| 4 | `update_payload` metadata update | `Index.update_payload` | 🔧 Phase 2 — payload store alongside shards |
| 5 | `search` with k, ef, n_probes | `Index.search` | 🔧 Phase 1 — route → fetch → MaxSim |
| 6 | `search_batch` batched queries | `Index.search_batch` | 🔧 Phase 1 — batch over queries, share shard fetches |
| 7 | `get` payloads by doc_id | `Index.get` | 🔧 Phase 2 — payload store lookup |
| 8 | `scroll` with pagination and filters | `Index.scroll` | 🔧 Phase 6 — iterate shards with filter |
| 9 | `snapshot` tarball backup | `Index.snapshot` | ✅ Shards are files — tar the index directory |
| 10 | `stats` (doc count, token count, memory) | `Index.stats` | 🔧 Phase 1 — from manifest metadata |
| 11 | `set_metrics_hook` (latency, candidates, nodes) | `Index.set_metrics_hook` | 🔧 Phase 1 — emit from profiler |
| 12 | `flush` / `close` / context manager | `Index` lifecycle | 🔧 Phase 2 — flush memtable, sync WAL |
| 13 | Engine selection: `"gem"` / `"hnsw"` / `"shard"` / `"auto"` | `IndexBuilder` | 🔧 Phase 6 — register `"shard"` |
| 14 | Mode hints `colbert` / `colpali` | `IndexBuilder` | ✅ Shard engine is mode-agnostic (stores token embeddings) |
| 15 | `embedding_fn` for `add_texts` / `search_text` | `Index` | ✅ Inherited — embedding hook is at `Index` level |
| 16 | Payload filters (Qdrant-style nested ops) | `GemNativeSegmentManager` | 🔧 Phase 6 — filter during top-k merge |
| 17 | Token attribution / `_explain_score` | `GemNativeSegmentManager` | 🔧 Phase 6 — per-token MaxSim decomposition |
| 18 | `IndexBuilder` fluent API (`with_gem`, `with_shard`, etc.) | `IndexBuilder` | 🔧 Phase 6 — add `with_shard()` |

### 17.2 Write-Ahead Log and Recovery

| # | Feature | Source | Shard Engine Status |
|---|---------|--------|---------------------|
| 19 | WAL operations: INSERT, DELETE, UPSERT, UPDATE_PAYLOAD | `gem_wal.py` | 🔧 Phase 2 — reuse WalOp enum |
| 20 | CRC32 integrity on WAL entries | `WalWriter` | 🔧 Phase 2 — reuse existing format |
| 21 | Corruption-tolerant replay | `WalReader.replay` | 🔧 Phase 2 — reuse skip logic |
| 22 | Atomic checkpoint save/load | `CheckpointManager` | 🔧 Phase 2 — checkpoint = manifest snapshot |
| 23 | Crash recovery: replay WAL from last checkpoint | `GemNativeSegmentManager` | 🔧 Phase 2 — replay into memtable |

### 17.3 Concurrency and I/O Safety

| # | Feature | Source | Shard Engine Status |
|---|---------|--------|---------------------|
| 24 | Reader-writer lock for concurrent search + write | `io_utils.RWLock` | 🔧 Phase 2 — lock manifest swaps |
| 25 | File lock for cross-process safety | `io_utils.FileLock` | 🔧 Phase 2 — lock index directory |
| 26 | Atomic JSON writes (manifest updates) | `io_utils.atomic_json_write` | 🔧 Phase 2 — for manifest.json |
| 27 | Snapshot isolation (readers see consistent state during compaction) | `GemNativeSegmentManager` | 🔧 Phase 2 — manifest versioning |

### 17.4 Kernels and Scoring

| # | Feature | Source | Shard Engine Status |
|---|---------|--------|---------------------|
| 28 | Triton MaxSim FP16 | `fast_colbert_scores` | ✅ Already used in benchmark |
| 29 | Triton MaxSim INT8 mode | `fast_colbert_scores(quantization_mode="int8")` | ✅ Available, wire to shard compression |
| 30 | Triton MaxSim FP8 mode | `fast_colbert_scores(quantization_mode="fp8")` | ✅ Available (experimental) |
| 31 | ROQ 4-bit MaxSim (1/2/4/8-bit ladder) | `roq_maxsim_4bit` | 🔧 Phase 3 — wire to shard scoring |
| 32 | Rust BLAS MaxSim fallback (no GPU) | `gem_index/lib.rs maxsim_score` | 🔧 Phase 3 — CPU fallback path |
| 33 | PyTorch fallback MaxSim (no Triton, no Rust) | `_fallback_maxsim` | ✅ Already in benchmark scorer |
| 34 | Triton kernel warmup (cold-start mitigation) | `kernel_warmup.py` | 🔧 Phase 1 — call on first query |
| 35 | GPU qCH proxy scoring | `GpuQchScorer` | 🚫 Not needed (no graph construction) |
| 36 | Triton qCH pairwise kernels | `triton_qch_kernel.py` | 🚫 Not needed (no graph construction) |

### 17.5 Quantization and Compression

| # | Feature | Source | Shard Engine Status |
|---|---------|--------|---------------------|
| 37 | Scalar INT8 quantization | `ScalarQuantizer` | ✅ Already in shard store |
| 38 | ROQ rotational quantization (1-8 bit) | `RotationalQuantizer` | 🔧 Phase 3 — build-time encode, Triton decode |
| 39 | Binary quantization + Hamming search | `BinaryQuantizer` | ⏭️ Coarse pre-filter for ultra-large corpora |
| 40 | Product quantization (PQ) | `ProductQuantizer` | ⏭️ Alternative compression for 10M+ scale |
| 41 | Walsh-Hadamard transform for ROQ | `FastWalshHadamard` | 🔧 Phase 3 — part of ROQ pipeline |

### 17.6 Hybrid Search, BM25, and Fusion

| # | Feature | Source | Shard Engine Status |
|---|---------|--------|---------------------|
| 42 | BM25s sparse search (Cython engine) | `BM25sEngine` / `bm25s` | 🔧 Phase 5 — co-locate BM25 index with shards |
| 43 | Pure-Python BM25 engine (fallback) | `BM25Engine` | ✅ Available as fallback |
| 44 | BM25 stemmed tokenization (PyStemmer) | `HybridSearchManager` | ✅ Inherited from existing sparse pipeline |
| 45 | Reciprocal Rank Fusion (RRF) | `fusion/strategies.py` | ✅ Reuse existing `reciprocal_rank_fusion` |
| 46 | Weighted sum fusion | `fusion/strategies.py` | ✅ Reuse existing `weighted_sum_fusion` |
| 47 | Max / min fusion | `fusion/strategies.py` | ✅ Reuse existing |
| 48 | Score normalization (min-max, z-score, softmax) | `fusion/strategies.py` | ✅ Reuse existing normalizers |
| 49 | **Tabu Search Solver refinement** | `latence_solver` / `TabuSearchSolver` | 🔧 Phase 5 — replace RRF with solver when available |
| 50 | `GpuFulfilmentPipeline` (GPU precompute for solver) | `stateless_optimizer.py` | 🔧 Phase 5 — optional GPU features for solver |
| 51 | Rich payload feature scoring (ontology, recency, density) | `HybridSearchManager` | 🔧 Phase 5 — pass shard doc features to solver |
| 52 | Sparse index rebuild with staged swap | `HybridSearchManager` | 🔧 Phase 5 — atomic BM25 index update |
| 53 | `HybridSearchManager` dense + sparse fan-out | `hybrid_manager.py` | 🔧 Phase 5 — shard engine as dense backend |
| 54 | `SearchPipeline` wrapping hybrid manager | `search_pipeline.py` | 🔧 Phase 6 — register shard dense backend |

### 17.7 Knowledge Graph Integration

| # | Feature | Source | Shard Engine Status |
|---|---------|--------|---------------------|
| 55 | `Neo4jEngine` (in-memory stub, production needs `neo4j` driver) | `engines/neo4j.py` | ⏭️ Stub — production wiring deferred |
| 56 | `Neo4jConfig` for connection settings | `config.py` | ✅ Config exists, engine is stub |
| 57 | Optional KG entity enrichment in hybrid pipeline | Architectural | ⏭️ Design: KG results as additional signal to fusion/solver |

**Knowledge graph integration path**: The Tabu solver's feature scoring already
supports arbitrary payload features (ontology, entity type, etc.). The integration
point is:

1. At **ingest time**: if a KG is available, annotate documents with entity IDs,
   categories, and relationship types. Store these as payload metadata in the shard
   manifest or a sidecar payload store.

2. At **query time**: optionally query the KG for entities related to the query.
   Pass entity matches as boosting signals to the Tabu solver's feature vector.

3. At **fusion time**: KG-derived results can be a third signal alongside dense
   (shard-routed MaxSim) and sparse (BM25), fused via RRF or the solver.

This is architecturally clean because the shard engine doesn't need to know about
the KG — it just stores payloads and returns scored candidates. The KG integration
happens at the `SearchPipeline` / `HybridSearchManager` level.

### 17.8 Multimodal and ColPali

| # | Feature | Source | Shard Engine Status |
|---|---------|--------|---------------------|
| 58 | `ColBERTEngine` wrapping `ColbertIndex` | `engines/colbert.py` | 🔧 Phase 6 — add shard-routed strategy |
| 59 | `ColPaliEngine` + screening backends | `engines/colpali.py` | 🔧 Phase 6 — add shard-routed screening |
| 60 | `MultiModalEngine` (ColBERT + ColPali fusion) | `engines/colpali.py` | 🔧 Phase 6 — shard engine supports any token embeddings |
| 61 | `MultimodalModelSpec` registry | `multimodal.py` | ✅ Model-agnostic — shard engine stores embeddings |
| 62 | `VllmPoolingProvider` (vLLM `/v1/pooling` HTTP) | `multimodal.py` | ✅ Embedding provider, not index-level |
| 63 | Document preprocessing (PDF/DOCX/images) | `preprocessing.py` | ✅ Preprocessing pipeline, not index-level |

### 17.9 Screening and Routing

| # | Feature | Source | Shard Engine Status |
|---|---------|--------|---------------------|
| 64 | `GemScreeningIndex` (Rust `PyGemRouter`) | `gem_screening.py` | 🚫 Replaced by centroid router |
| 65 | `PrototypeScreeningIndex` (HNSW-style) | `prototype_screening.py` | 🚫 Replaced by centroid router |
| 66 | `CentroidScreeningIndex` | `centroid_screening.py` | ✅ Conceptually identical to shard centroid router |
| 67 | `ScreeningSidecar` protocol + calibration | `screening_sidecar.py` | ⏭️ Calibration for adaptive budget |

### 17.10 Storage and Persistence

| # | Feature | Source | Shard Engine Status |
|---|---------|--------|---------------------|
| 68 | Safetensors shard files | `shard_store.py` | ✅ Already implemented |
| 69 | JSON manifest with shard metadata | `shard_store.py` | ✅ Already implemented |
| 70 | HDF5 + mmap storage | `storage.py` | 🚫 Replaced by safetensors shards |
| 71 | Async HDF5 pipeline + triple buffer | `async_storage.py` | 🚫 Replaced — shard writes are sequential |
| 72 | `PinnedMemoryPool` for GPU staging | `async_storage.py` | ✅ Available if needed for double-buffered mode |
| 73 | Sharded storage / shard manager | `sharded_storage.py` | ✅ Superseded by shard engine's own ShardStore |

### 17.11 Graph Construction (NOT needed for shard engine)

| # | Feature | Source | Shard Engine Status |
|---|---------|--------|---------------------|
| 74 | GEM HNSW multi-level graph | `gem_index graph.rs` | 🚫 No graph needed |
| 75 | NN-Descent graph construction | `gem_index graph.rs` | 🚫 No graph needed |
| 76 | RNN-Descent (MRNG + navigating node) | `gem_index graph.rs` | 🚫 No graph needed |
| 77 | Dual-graph / bridge repair | `gem_index graph.rs` | 🚫 No graph needed |
| 78 | Semantic shortcuts | `gem_manager.py` | 🚫 No graph needed |
| 79 | qEMD / Sinkhorn OT distances | `gem_index emd.rs` | 🚫 No graph needed |
| 80 | Background self-healing thread | `gem_manager.py` | 🚫 No graph to heal — compaction serves this role |

### 17.12 HTTP API and Server

| # | Feature | Source | Shard Engine Status |
|---|---------|--------|---------------------|
| 81 | `GET /health` (version, collections, GPU) | `server/api/routes.py` | ✅ Server-level, engine-agnostic |
| 82 | `GET /ready` readiness report | `server/api/routes.py` | ✅ Server-level |
| 83 | `GET /metrics` Prometheus text | `server/api/routes.py` | ✅ Server-level |
| 84 | Collections CRUD (list, create, delete, info) | `server/api/routes.py` | 🔧 Phase 9 — register shard engine as collection kind |
| 85 | Points add / delete | `server/api/routes.py` | 🔧 Phase 9 — delegates to ShardSegmentManager |
| 86 | `POST .../search` | `server/api/routes.py` | 🔧 Phase 9 — add `"shard_routed"` strategy |
| 87 | `POST /reference/optimize` (stateless optimizer) | `server/api/routes.py` | ✅ Optimizer is engine-agnostic |
| 88 | `POST /reference/preprocess/documents` | `server/api/routes.py` | ✅ Preprocessing, engine-agnostic |
| 89 | Collection kinds: `dense`, `late_interaction`, `multimodal` | `server/api/models.py` | 🔧 Phase 9 — shard engine serves `late_interaction` |
| 90 | Screening modes in server models | `server/api/models.py` | 🔧 Phase 9 — add `"centroid_routed"` mode |
| 91 | Shard admin endpoints (list, compact, checkpoint) | NEW | 🔧 Phase 9 |
| 92 | `SearchService` journal recovery + mutation backup | `server/api/service.py` | 🔧 Phase 9 — adapt for WAL-based recovery |

### 17.13 Distributed

| # | Feature | Source | Shard Engine Status |
|---|---------|--------|---------------------|
| 93 | `DistributedRouter` scatter-gather skeleton | `distributed/router.py` | 🔧 Phase 8 — coordinator-level shard routing |
| 94 | Cross-node top-k merge (RRF / score merge) | `distributed/router.py` | 🔧 Phase 8 |
| 95 | Shard rebalancing on node add/remove | NEW | 🔧 Phase 8 — file copy + manifest update |

### 17.14 Configuration

| # | Feature | Source | Shard Engine Status |
|---|---------|--------|---------------------|
| 96 | `IndexConfig` (PLAID, ColBERT, GEM fields) | `config.py` | 🔧 Phase 6 — add shard engine fields |
| 97 | `BM25Config` | `config.py` | ✅ Reused as-is for hybrid search |
| 98 | `FusionConfig` | `config.py` | ✅ Reused as-is for hybrid fusion |
| 99 | `Neo4jConfig` | `config.py` | ✅ Reused as-is (stub integration) |
| 100 | `ShardEngineConfig` | NEW | 🔧 Phase 1 |

### 17.15 Observability and Debugging

| # | Feature | Source | Shard Engine Status |
|---|---------|--------|---------------------|
| 101 | Per-query profiler (routing/fetch/h2d/maxsim/total) | `shard_bench/profiler.py` | ✅ Already implemented |
| 102 | Memory snapshot (CPU RSS, GPU allocated/reserved) | `shard_bench/profiler.py` | ✅ Already implemented |
| 103 | Metrics hook (search_latency_us, candidates_scored) | `Index.set_metrics_hook` | 🔧 Phase 1 — emit from profiler |
| 104 | Page cache hit rate tracking | NEW | 🔧 Phase 9 — `mincore()` or `/proc/meminfo` |

### 17.16 Experimental / Research (Low Priority)

| # | Feature | Source | Shard Engine Status |
|---|---------|--------|---------------------|
| 105 | `VectorGym` RL environment | `gym/vector_env.py` | ⏭️ Experimental, not production |
| 106 | `TabuPotentialField` control manifold | `control/potential.py` | ⏭️ Research utility |
| 107 | `FeatureBridge` (OSS stub) | `feature_bridge.py` | ⏭️ Removed from OSS |

---

### 17.17 Summary Counts

| Status | Count | Meaning |
|--------|-------|---------|
| ✅ Already works / inherited | 37 | No action needed |
| 🔧 Needs implementation | 49 | Covered by Phases 1-9 |
| 🚫 Not applicable | 11 | Graph-specific, replaced by shard architecture |
| ⏭️ Deferred | 10 | Post-launch, experimental, or stubs |
| **Total** | **107** | |

### 17.18 Critical Path Features (must ship)

These are the features that **must** work before the shard engine can be used as a
drop-in replacement for GEM/HNSW in production:

1. **Search**: route → fetch → per-shard MaxSim → top-k merge (Phase 1)
2. **CRUD**: add / delete / upsert via WAL + memtable (Phase 2)
3. **Recovery**: WAL replay on crash (Phase 2)
4. **ROQ 4-bit**: bandwidth reduction for scale (Phase 3)
5. **Hybrid search**: BM25 + shard-routed + Tabu solver (Phase 5)
6. **Public API**: `Index(engine="shard")` registration (Phase 6)

Everything else (scroll, filters, explain, distributed, admin endpoints) is
important for feature parity but not blocking for initial production use.

---

## Implementation Priority

| Phase | Deliverable | Dependencies | Effort |
|-------|------------|-------------|--------|
| **1** | Core engine (ShardSegmentManager, engine registration) | Benchmark code | 1 week |
| **2** | CRUD + WAL + recovery | Phase 1, gem_wal.py patterns | 1 week |
| **3** | ROQ 4-bit kernel integration | Phase 1, existing ROQ kernels | 3 days |
| **4** | Hybrid search (BM25 + shard-routed) | Phase 1, existing HybridSearchManager | 3 days |
| **5** | Public API (Index, ColbertIndex, SearchPipeline) | Phases 1-4 | 3 days |
| **6** | Compaction | Phase 2 | 1 week |
| **7** | HTTP API + server | Phase 5 | 2 days |
| **8** | Distributed scatter-gather | Phase 5, existing DistributedRouter | 1 week |
| **9** | Scale testing (1M, 3M, 10M) | Phases 1-6 | 1 week |

**Total estimated effort**: ~6 weeks for full production integration.

Phases 1-3 are the critical path. After Phase 3, the shard engine is usable as a
standalone index with ROQ compression. Phases 4-8 integrate it with the broader
voyager-index ecosystem.
