# Python API Reference

## `voyager_index.Index`

The primary interface for creating and querying indexes.

### Constructor

```python
Index(
    path: str,
    dim: int,
    *,
    engine: str = "auto",        # "gem", "hnsw", or "auto"
    n_fine: int = 256,           # fine centroids (codebook size)
    n_coarse: int = 32,          # coarse clusters
    max_degree: int = 32,        # max graph neighbors (M)
    ef_construction: int = 200,  # beam width during build
    n_probes: int = 4,           # clusters to probe during search
    enable_wal: bool = True,     # write-ahead log
    seed_batch_size: int = 256,  # docs before codebook training
    **kwargs,
)
```

### Methods

| Method | Description |
|---|---|
| `add(vectors, ids=None, payloads=None)` | Add documents (3D array or list of 2D) |
| `search(query, k=10, ef=100, n_probes=4, filters=None)` | Multi-vector search → `List[SearchResult]` |
| `delete(ids)` | Delete documents by ID |
| `update_payload(id, payload)` | Update document metadata |
| `get(ids)` | Retrieve payloads by ID |
| `scroll(limit=100, offset=0, filters=None)` | Paginated iteration → `ScrollPage` |
| `stats()` | Summary statistics → `IndexStats` |
| `snapshot(output_path)` | Create tarball backup |
| `flush()` | Force pending writes to disk |
| `close()` | Release resources |

### Properties

| Property | Type | Description |
|---|---|---|
| `path` | `Path` | Index directory |
| `dim` | `int` | Vector dimension |
| `engine` | `str` | Backend engine name |

---

## `voyager_index.IndexBuilder`

Fluent builder for custom Index configuration.

```python
idx = (IndexBuilder("my_index", dim=128)
       .with_gem(seed_batch_size=64, n_fine=128)
       .with_wal(enabled=True)
       .with_quantization(n_fine=256, n_coarse=32)
       .build())
```

---

## `voyager_index.SearchResult`

```python
@dataclass
class SearchResult:
    doc_id: int
    score: float
    payload: Optional[Dict[str, Any]] = None
```

## `voyager_index.ScrollPage`

```python
@dataclass
class ScrollPage:
    results: List[SearchResult]
    next_offset: Optional[int] = None
```

## `voyager_index.IndexStats`

```python
@dataclass
class IndexStats:
    total_documents: int
    sealed_segments: int
    active_documents: int
    dim: int
    engine: str
```

---

## Low-Level: `latence_gem_index`

### `GemSegment`

Sealed (read-only) GEM graph segment.

| Method | Signature |
|---|---|
| `build()` | `(all_vectors, doc_ids, doc_offsets, n_fine=256, n_coarse=32, max_degree=32, ef_construction=200, max_kmeans_iter=30, ctop_r=3)` |
| `search()` | `(query_vectors, k=10, ef=100, n_probes=4, enable_shortcuts=False)` → `List[(doc_id, score)]` |
| `save(path)` | Save to disk |
| `load(path)` | Load from disk |
| `inject_shortcuts()` | `(training_pairs, max_shortcuts_per_node=4)` |
| `n_docs()` | Number of documents |
| `n_nodes()` | Number of graph nodes |
| `n_edges()` | Total directed edges |
| `dim()` | Vector dimension |
| `is_ready()` | Whether segment is built/loaded |
| `total_shortcuts()` | Number of shortcut edges |

### `PyMutableGemSegment`

Writable GEM segment with CRUD support.

| Method | Signature |
|---|---|
| `build()` | `(all_vectors, doc_ids, doc_offsets, n_fine=256, n_coarse=32, max_degree=32, ef_construction=200, max_kmeans_iter=30, ctop_r=3, n_probes=4)` |
| `search()` | `(query_vectors, k=10, ef=100, _n_probes=4)` → `List[(doc_id, score)]` |
| `insert(vectors, doc_id)` | Insert one document |
| `delete(doc_id)` | Soft-delete → `bool` |
| `upsert(vectors, doc_id)` | Delete old + insert new |
| `compact()` | Remove soft-deleted, rebuild |
| `n_live()` | Live (non-deleted) count |
| `n_nodes()` | Total nodes including deleted |
| `n_edges()` | Total directed edges |
| `quality_score()` | 1.0 = fresh, degrades with deletes |
| `delete_ratio()` | Fraction of deleted nodes |
| `avg_degree()` | Average neighbors per node |
| `memory_bytes()` | Estimated memory usage |
