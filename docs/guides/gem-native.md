# GEM Native Index Guide

## Overview

The GEM (Graph-based index for Multi-vector retrieval) native index builds a
proximity graph directly over document vector sets. Unlike HNSW which indexes
individual vectors, GEM treats each document as a set of token embeddings and
constructs graph edges based on quantized Chamfer distance.

## How It Works

1. **Codebook training**: Two-stage k-means clusters all token vectors into
   fine centroids, then groups fine centroids into coarse clusters
2. **Document profiling**: Each document gets centroid code assignments and
   a coarse cluster profile (C_top)
3. **Graph construction**: Sequential insertion with cluster-guided entry
   points, beam search for candidates, and diversity-heuristic neighbor selection
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
seg = GemSegment()
seg.build(all_vectors, doc_ids, doc_offsets,
          n_fine=256, n_coarse=32, max_degree=32, ef_construction=200)
results = seg.search(query, k=10, ef=100, n_probes=4)
seg.save("segment.gem")

# Mutable segment (incremental)
mut = PyMutableGemSegment()
mut.build(seed_vectors, seed_ids, seed_offsets,
          n_fine=256, n_coarse=32, max_degree=32, ef_construction=200, n_probes=4)
mut.insert(new_vectors, doc_id=999)
mut.delete(old_doc_id)
mut.upsert(updated_vectors, doc_id=999)
mut.compact()  # remove soft-deleted nodes
```

## Tuning Parameters

| Parameter | Default | Description |
|---|---|---|
| `seed_batch_size` | 256 | Documents before codebook training. Lower = faster start |
| `n_fine` | 256 | Fine centroids. More = better precision, slower build |
| `n_coarse` | 32 | Coarse clusters. More = better routing at scale |
| `max_degree` | 32 | Max graph neighbors. Higher = better recall, more memory |
| `ef_construction` | 200 | Build beam width. Higher = better graph, slower build |
| `n_probes` | 4 | Search clusters to probe. Higher = better recall |
| `ctop_r` | 3 | Top coarse clusters per document |
