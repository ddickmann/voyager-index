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
| `build()` | `(all_vectors, doc_ids, doc_offsets, n_fine=256, n_coarse=32, max_degree=32, ef_construction=200, max_kmeans_iter=30, ctop_r=3, use_emd=False, dual_graph=True)` |
| `search()` | `(query_vectors, k=10, ef=100, n_probes=4, enable_shortcuts=False, filter=None, min_cluster_ratio=0.01)` → `List[(doc_id, score)]` |
| `search_with_stats()` | `(query_vectors, k=10, ef=100, n_probes=4, enable_shortcuts=False)` → `(results, (nodes_visited, distance_computations))` |
| `search_batch()` | `(queries, k=10, ef=100, n_probes=4, enable_shortcuts=False)` → `List[List[(doc_id, score)]]` |
| `brute_force_proxy()` | `(query_vectors, k=10)` → `List[(doc_id, score)]` — exhaustive qCH ranking (oracle baseline) |
| `save(path)` | Save to disk with CRC32 integrity |
| `load(path)` | Load from disk with integrity verification |
| `set_doc_payloads()` | `(payloads: List[(doc_id, [(field, value)])])` — build filter index for filtered search |
| `inject_shortcuts()` | `(training_pairs, max_shortcuts_per_node=4)` |
| `prune_stale_shortcuts()` | `(deleted_flags, max_age=None, current_generation=0)` |
| `load_cutoff_tree()` | `(tree_bytes)` — load adaptive cluster cutoff tree |
| `get_codebook_centroids()` | → `ndarray (n_fine, dim)` |
| `get_idf()` | → `ndarray (n_fine,)` |
| `get_flat_codes()` | → `(codes, offsets, lengths)` |
| `graph_connectivity_report()` | → `(n_components, giant_component_frac)` — BFS connectivity analysis |
| `n_docs()` | Number of documents |
| `n_nodes()` | Number of graph nodes |
| `n_edges()` | Total directed edges |
| `dim()` | Vector dimension |
| `is_ready()` | Whether segment is built/loaded |
| `total_shortcuts()` | Number of shortcut edges |

**`search()` parameters:**
- `filter`: list of `(field, value)` pairs for filter-aware routing (AND semantics)
- `min_cluster_ratio`: when filtering, skip clusters where fewer than this fraction of docs match (selectivity-aware pruning, default 0.01)

### `PyMutableGemSegment`

Writable GEM segment with CRUD and self-healing support.

| Method | Signature |
|---|---|
| `build()` | `(all_vectors, doc_ids, doc_offsets, n_fine=256, n_coarse=32, max_degree=32, ef_construction=200, max_kmeans_iter=30, ctop_r=3, n_probes=4, use_emd=False)` |
| `search()` | `(query_vectors, k=10, ef=100, n_probes=4)` → `List[(doc_id, score)]` |
| `search_batch()` | `(queries, k=10, ef=100)` → `List[List[(doc_id, score)]]` |
| `insert(vectors, doc_id)` | Insert one document |
| `insert_batch(vectors_list, doc_ids)` | Batch insert multiple documents |
| `delete(doc_id)` | Soft-delete → `bool` |
| `upsert(vectors, doc_id)` | Delete old + insert new |
| `compact()` | Remove soft-deleted, rebuild |
| `heal()` | Local graph repair: fix stale reps, reconnect isolated nodes, clean edges |
| `needs_healing()` | → `bool` — drift detection based on quality thresholds |
| `graph_quality_metrics()` | → `(delete_ratio, avg_degree, isolated_ratio, stale_rep_ratio)` |
| `graph_connectivity_report()` | → `(n_components, giant_frac, cross_cluster_edge_ratio)` — BFS connectivity with cross-cluster analysis |
| `n_live()` | Live (non-deleted) count |
| `n_nodes()` | Total nodes including deleted |
| `n_edges()` | Total directed edges |
| `quality_score()` | 1.0 = fresh, degrades with deletes |
| `delete_ratio()` | Fraction of deleted nodes |
| `avg_degree()` | Average neighbors per node |
| `memory_bytes()` | Estimated memory usage |
| `dim()` | Vector dimension |
| `is_ready()` | Whether segment is built |

### `PyEnsembleGemSegment`

Multi-modal ensemble with per-modality codebooks and RRF fusion.

| Method | Signature |
|---|---|
| `build()` | `(all_vectors, doc_ids, doc_offsets, modality_tags, n_modalities, n_fine=256, n_coarse=32, max_degree=32, ef_construction=200, max_kmeans_iter=30, ctop_r=3)` |
| `search()` | `(query_vectors, query_modality_tags, k=10, ef=100, n_probes=4)` → `List[(doc_id, score)]` |
| `n_docs()` | Number of documents |
| `n_modalities()` | Number of modality types |
| `is_ready()` | Whether ensemble is built |

`modality_tags` is a per-token `u8` array mapping each token to its modality
(0 = text, 1 = image, etc.). The ensemble builds a separate codebook and graph
per modality, searches each independently, and fuses results via Reciprocal
Rank Fusion.

---

## GPU Scoring: `GpuQchScorer`

Optional GPU-native qCH proxy scorer. Requires PyTorch; uses Triton when available.

```python
from voyager_index._internal.inference.index_core.gpu_qch import GpuQchScorer

scorer = GpuQchScorer.from_gem_segment(segment, device="cuda")
scores = scorer.score_query(query_vecs)           # (n_docs,) lower = closer
scores = scorer.score_query_filtered(query_vecs, mask)  # masked scoring
```

| Method | Description |
|---|---|
| `from_gem_segment(segment, device)` | Construct from a built `GemSegment` |
| `score_query(query_vecs)` | Score all docs → `(n_docs,)` float32 |
| `score_query_filtered(query_vecs, doc_mask)` | Score masked docs; unmasked get `inf` |

The GPU path is fully optional. CPU scoring via the Rust `qch_proxy_score_u16`
function remains the default and requires no additional dependencies.
