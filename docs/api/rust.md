# Rust Crates

## `latence-gem-index`

Source: `src/kernels/gem_index/`

Native GEM graph index for multi-vector retrieval. Implements the proximity
graph from the GEM paper with qCH proxy scoring, diversity-based neighbor
selection, and cluster-guided beam search.

### Modules

| Module | Description |
|---|---|
| `lib.rs` | PyO3 bindings: `GemSegment`, `PyMutableGemSegment` |
| `graph.rs` | Graph construction, neighbor selection, shortcuts |
| `search.rs` | Beam search for construction and query |
| `mutable.rs` | Mutable segment: insert, delete, upsert, compact |
| `persistence.rs` | Bincode + CRC32 atomic save/load |
| `id_tracker.rs` | External ↔ internal ID mapping with deletion tracking |
| `visited.rs` | Compact bit-set for search traversal |
| `emd.rs` | IP-based Chamfer distance between document code sets |

### Dependencies

- `latence-gem-router` (codebook, qCH scoring, flat codes)
- `pyo3` + `numpy` (Python bindings)
- `rayon` (parallel construction)
- `parking_lot` (concurrency)
- `bincode` + `crc32fast` (persistence)

---

## `latence-gem-router`

Source: `src/kernels/gem_router/`

GEM codebook, cluster routing, and qCH proxy scoring engine.

### Key Types

- `TwoStageCodebook` — fine + coarse centroids with IDF weighting
- `FlatDocCodes` — contiguous u16 centroid codes for cache-friendly scoring
- `ClusterPostings` — coarse cluster → document posting lists
- `GemRouter` — full routing pipeline: build, route, score, persist

### Performance

- `matrixmultiply::sgemm` with AVX2+FMA for query-centroid scores
- AVX2 gather-based proxy scoring (`_mm256_i32gather_ps`)
- Pre-allocated score buffer reuse across queries
