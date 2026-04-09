# GEM Native Index Guide

## Overview

The GEM (Graph-based index for Multi-vector retrieval) native index builds a
proximity graph directly over document vector sets. Unlike HNSW which indexes
individual vectors, GEM treats each document as a set of token embeddings and
constructs graph edges using **metric-decoupled** distance functions.

### Key Architecture (GEM Paper)

1. **qEMD for graph construction**: Edges are built using quantized Earth Mover's
   Distance (qEMD), which satisfies the triangle inequality. This prevents
   attractor basins and local traps during greedy graph traversal.
2. **qCH for search**: At query time, the faster quantized Chamfer distance (qCH)
   is used for scoring. This metric decoupling is a core theoretical contribution
   of the GEM paper (Section 4.2).
3. **Dual-graph construction**: Instead of HNSW-style sequential insertion,
   GEM builds per-cluster local graphs then merges them via bridge sets. This
   gives better intra-cluster locality and explicit cross-cluster connectivity.
4. **Adaptive cluster cutoff**: A decision tree predicts the optimal number of
   top clusters per document, reducing index redundancy for focused documents
   and preventing recall loss for diverse ones.

## How It Works

1. **Codebook training**: Two-stage k-means clusters all token vectors into
   fine centroids, then groups fine centroids into coarse clusters
2. **Document profiling**: Each document gets centroid code assignments and
   a coarse cluster profile (C_top). When adaptive cutoff is enabled, a
   decision tree determines the per-document cluster count instead of a fixed r.
3. **Graph construction** (dual-graph mode):
   - For each cluster, process member documents
   - First encounter: beam search within cluster members, select neighbors via
     diversity heuristic using qEMD distances
   - Subsequent encounters (bridge update): merge old/new neighbors, enforce
     at least one neighbor from each assigned cluster
   - Safety net: bridge repair for remaining disconnected components
4. **Search**: Cluster-based entry points → beam search → qCH proxy scoring →
   top-k results

## Using the Index API

```python
from voyager_index import Index

idx = Index(
    "my_index",
    dim=128,
    engine="gem",
    seed_batch_size=64,
    n_fine=128,
    n_coarse=16,
    max_degree=32,
    ef_construction=200,
)

# Add documents
idx.add(embeddings, ids=doc_ids, payloads=payloads)

# Search
results = idx.search(query_vectors, k=10, ef=100, n_probes=4)

# Lifecycle
idx.flush()
idx.close()
```

## Low-Level API

For direct control over sealed and mutable segments:

```python
from latence_gem_index import GemSegment, PyMutableGemSegment
import numpy as np

# Sealed segment (batch build, read-only)
# use_emd=True enables qEMD for graph construction (default)
# dual_graph=True enables per-cluster dual-graph build (default)
seg = GemSegment()
seg.build(all_vectors, doc_ids, doc_offsets,
          n_fine=256, n_coarse=32, max_degree=32, ef_construction=200,
          use_emd=True, dual_graph=True)
results = seg.search(query, k=10, ef=100, n_probes=4)
seg.save("segment.gem")

# Mutable segment (incremental, use_emd=False by default for insert latency)
mut = PyMutableGemSegment()
mut.build(seed_vectors, seed_ids, seed_offsets,
          n_fine=256, n_coarse=32, max_degree=32, ef_construction=200,
          n_probes=4, use_emd=False)
mut.insert(new_vectors, doc_id=999)
mut.delete(old_doc_id)
mut.upsert(updated_vectors, doc_id=999)
mut.compact()  # remove soft-deleted nodes
```

## Adaptive Cluster Cutoff

Train a decision tree to predict per-document cluster counts:

```python
from latence_gem_router import PyGemRouter

router = PyGemRouter(dim=128)
router.build(all_vectors, doc_ids, doc_offsets, n_fine=256, n_coarse=32)

# Train from query-positive pairs
tree_bytes = router.train_adaptive_cutoff(
    training_queries,       # (Q_total, D) query vectors
    n_query_vecs=[5, 3],    # tokens per query
    training_positives=[0, 7],  # positive doc indices
    t=3, r_max=8, max_depth=6,
)
```

## Construction Metrics

| Metric | Property | Use |
|---|---|---|
| **qEMD** | True metric (triangle inequality) | Graph construction — stable navigation |
| **qCH** | Non-metric (violates triangle inequality) | Search scoring — fast and effective |
| **Sinkhorn** | Regularized OT approximation | qEMD approximation, inherently symmetric |

## Thread Safety

GemSegment and PyMutableGemSegment release the Python GIL during all
compute-intensive operations (build, search, insert) via `py.allow_threads()`.
This enables true parallelism from Python threads.

**Sealed segments (`GemSegment`)**: Concurrent `search()` and `search_batch()`
calls are safe. The underlying Rust graph and codebook are immutable after build.

**Mutable segments (`PyMutableGemSegment`)**: Concurrent `search()` calls are safe.
However, `insert()`, `delete()`, `upsert()`, and `compact()` mutate internal state
and must not be called concurrently with each other or with search. Use external
synchronization (e.g., `threading.Lock`) if mixing reads and writes from multiple
threads.

**Memory efficiency**: The search hot path uses a generation-based visited-set pool
(`VisitedPool`) that reuses allocations across searches, avoiding per-query heap
allocation. Combined with prefetch hints for document codes, this keeps tail
latency predictable under concurrent load.

## Search Architecture

voyager-index supports two search modes:

### GEM-Direct (screening_mode="none")
Queries go directly to the GEM graph index. The router computes cluster
assignments, then beam search traverses the graph using quantized Chamfer
distance (qCH) as the proxy score. Best for single-modal text retrieval
where the router codebook captures token-level similarity well.

### Screening-First (screening_mode="gem") — Default
A two-stage pipeline for multimodal retrieval:
1. **Screening**: The GEM router scores all clusters, returning candidate
   document IDs from the top clusters.
2. **Reranking**: Full MaxSim (or late-interaction) scores are computed on
   the candidate set using raw vectors.

This mode is used by the ColPali engine for vision-language retrieval,
where screening narrows the candidate set before expensive cross-modal
scoring.

## Filter-Aware Routing

Sealed segments support filtered search via per-cluster Roaring bitmap indexes.
Instead of post-filtering results, filters prune entire clusters before graph
traversal, giving sub-linear scaling with selectivity.

```python
seg = GemSegment()
seg.build(all_vectors, doc_ids, doc_offsets)

seg.set_doc_payloads([
    (1, [("category", "science"), ("lang", "en")]),
    (2, [("category", "sports"), ("lang", "de")]),
    (3, [("category", "science"), ("lang", "de")]),
])

results = seg.search(query, k=10, filter=[("category", "science")])
```

Filters use AND semantics: all field-value pairs must match. If no filter index
has been built (`set_doc_payloads()` not called), a warning is logged and the
filter is ignored.

## GPU-Accelerated qCH Scoring

For large-scale scoring on GPU, the optional `GpuQchScorer` bypasses the CPU
path entirely. It uploads the codebook and flat codes to the GPU, computes
query-centroid scores via PyTorch GEMM, and runs a Triton-autotuned max-gather
kernel over packed document codes.

```python
from voyager_index._internal.inference.index_core.gpu_qch import GpuQchScorer

scorer = GpuQchScorer.from_gem_segment(segment, device="cuda")
scores = scorer.score_query(query_vecs)   # (n_docs,) float32 on GPU

mask = torch.tensor([True, False, True, ...], device="cuda")
filtered = scorer.score_query_filtered(query_vecs, mask)
```

Key properties:

- Documents up to **2048 tokens** are scored without truncation
- Documents longer than 2048 automatically fall back to the PyTorch path
- Five autotuned Triton configurations (block sizes 32–128, warps 2–8)
- Fully optional: the CPU Rust path is the default and requires no GPU

## Self-Healing Mutable Graphs

Mutable segments degrade under heavy deletes as cluster representatives become
stale and isolated nodes appear. The `heal()` method performs local repair:

```python
mut = PyMutableGemSegment()
mut.build(seed_vectors, seed_ids, seed_offsets)

# After many deletes...
for doc_id in deleted_ids:
    mut.delete(doc_id)

# Check drift
metrics = mut.graph_quality_metrics()
# → (delete_ratio, avg_degree, isolated_ratio, stale_rep_ratio)

if mut.needs_healing():
    mut.heal()
```

The three repair phases:

1. **Medoid recomputation**: Clusters with deleted or low-degree representatives
   elect a new medoid from live members
2. **Isolated node reconnection**: Live nodes with no edges are connected to
   the nearest cluster representative (capped scan to avoid O(n²))
3. **Edge cleanup**: Edges to deleted nodes are removed; adjacency lists are
   deduplicated

`GemNativeSegmentManager` runs a background healing thread that calls
`needs_healing()` → `heal()` on a configurable interval (default: 30s).

## Multi-Modal Ensemble with RRF Fusion

For workloads with multiple modalities (text + image, text + code, etc.), the
ensemble segment builds a separate codebook and graph per modality, searches
each independently, and fuses results via Reciprocal Rank Fusion.

```python
from latence_gem_index import PyEnsembleGemSegment
import numpy as np

ens = PyEnsembleGemSegment()

modality_tags = np.array([0, 0, 0, 1, 1], dtype=np.uint8)

ens.build(
    all_vectors, doc_ids, doc_offsets,
    modality_tags=modality_tags,
    n_modalities=2,
    n_fine=64, n_coarse=8,
)

query_tags = np.array([0, 0, 1], dtype=np.uint8)
results = ens.search(query_vectors, query_modality_tags=query_tags, k=10)
```

RRF uses 1-based ranking with k=60 (standard), consistent with the project's
`hybrid_manager.py` fusion convention. The ensemble is compatible with the Tabu
Search solver for constraint-aware result fusion.

## Tuning Parameters

| Parameter | Default | Description |
|---|---|---|
| `seed_batch_size` | 256 | Documents before codebook training. Lower = faster start |
| `n_fine` | 256 | Fine centroids. More = better precision, slower build |
| `n_coarse` | 32 | Coarse clusters. More = better routing at scale |
| `max_degree` | 32 | Max graph neighbors. Higher = better recall, more memory |
| `ef_construction` | 200 | Build beam width. Higher = better graph, slower build |
| `n_probes` | 4 | Search clusters to probe. Higher = better recall |
| `ctop_r` | 3 | Top coarse clusters per document (fixed mode) |
| `use_emd` | True | Use qEMD for graph construction (metric decoupling) |
| `dual_graph` | True | Per-cluster dual-graph construction with bridge sets |
| `healing_interval_s` | 30 | Background healing check interval (segment manager) |
| `compaction_threshold` | 0.3 | Delete ratio triggering automatic compaction |
