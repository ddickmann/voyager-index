# Rust Crates

## `latence-gem-index`

Source: `src/kernels/gem_index/`

Native GEM graph index for multi-vector retrieval. Implements the proximity
graph from the GEM paper with qCH proxy scoring, diversity-based neighbor
selection, cluster-guided beam search, self-healing, and multi-modal ensemble
fusion.

### Modules

| Module | Description |
|---|---|
| `lib.rs` | PyO3 bindings: `GemSegment`, `PyMutableGemSegment`, `PyEnsembleGemSegment` |
| `graph.rs` | Graph construction (standard, dual-graph, payload-aware), neighbor selection, shortcut injection and pruning |
| `search.rs` | Beam search with prefetching, shortcut traversal, and search stats |
| `mutable.rs` | Mutable segment: insert, delete, upsert, compact, heal, graph quality metrics |
| `ensemble.rs` | Multi-modal ensemble: per-modality segments with RRF fusion |
| `persistence.rs` | Bincode + CRC32 atomic save/load with mmap for large files and version migration |
| `id_tracker.rs` | External ↔ internal ID mapping with deletion tracking and compaction |
| `visited.rs` | Generation-based visited set with thread-local pooling (zero-alloc per search) |
| `emd.rs` | qCH and qEMD distance functions with chunked AVX-friendly inner loops |
| `network_simplex.rs` | Network simplex EMD solver and Sinkhorn-regularized OT approximation |
| `score_cache.rs` | LRU-style pairwise score cache for construction |

### Dependencies

- `latence-gem-router` (codebook, qCH scoring, flat codes, filter index)
- `pyo3` + `numpy` (Python bindings)
- `rayon` (parallel construction and batch search)
- `parking_lot` (concurrency)
- `bincode` + `crc32fast` (persistence)
- `memmap2` (memory-mapped I/O for large segments)
- `log` (structured logging for filter warnings and diagnostics)

---

## `latence-gem-router`

Source: `src/kernels/gem_router/`

GEM codebook, cluster routing, qCH proxy scoring engine, and filter index.

### Key Types

- `TwoStageCodebook` — fine + coarse centroids with IDF weighting and centroid refinement
- `FlatDocCodes` — contiguous u16 centroid codes for cache-friendly scoring
- `ClusterPostings` — coarse cluster → document posting lists with medoid tracking
- `FilterIndex` — per-cluster Roaring bitmap index for payload-based filtering
- `CutoffTree` — decision tree for adaptive per-document cluster cutoff prediction
- `GemRouter` — full routing pipeline: build, route, score, persist

### Performance

- `matrixmultiply::sgemm` with AVX2+FMA for query-centroid scores
- AVX2 gather-based proxy scoring (`_mm256_i32gather_ps`)
- Pre-allocated score buffer reuse across queries
- Roaring bitmap intersection for sub-microsecond filter evaluation
