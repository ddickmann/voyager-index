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
