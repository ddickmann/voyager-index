# voyager-index

**The first open-source native multi-vector index.**

voyager-index ships a Rust-native GEM graph index that replaces traditional HNSW
for multi-vector workloads like ColBERT and ColPali. Search is **52x faster**
than HNSW at 1024-token sequence length.

## Why voyager-index?

| Feature | voyager-index | Traditional HNSW |
|---|---|---|
| Multi-vector native | Yes (document-level graph) | No (single-vector) |
| Search @ 1024 tokens | **1.3 ms** | 67 ms |
| CRUD operations | Insert, delete, upsert, compact | Varies |
| Crash recovery | WAL + checkpointing | Varies |
| Segment lifecycle | Sealed + active | Varies |

## Quick Example

```python
from voyager_index import Index

idx = Index("my_index", dim=128, engine="gem", seed_batch_size=64)
idx.add(embeddings, ids=[1, 2, 3])
results = idx.search(query_vectors, k=10)
for r in results:
    print(f"Doc {r.doc_id}: score={r.score:.4f}")
idx.close()
```

## Architecture

```
┌─────────────────────────────────────────────┐
│                 Index (Python)               │
│  add() / search() / delete() / scroll()     │
├─────────────────────────────────────────────┤
│          GemNativeSegmentManager            │
│  WAL / checkpoint / compaction / sealing    │
├──────────────────┬──────────────────────────┤
│  Active Mutable  │  Sealed GEM Segments     │
│  (insert/delete) │  (read-only, optimized)  │
├──────────────────┴──────────────────────────┤
│         latence_gem_index (Rust/PyO3)       │
│  graph.rs / search.rs / mutable.rs / ...    │
├─────────────────────────────────────────────┤
│         latence_gem_router (Rust/PyO3)      │
│  codebook / qCH scoring / cluster routing   │
└─────────────────────────────────────────────┘
```
