# Changelog

## 0.1.0 — GEM Native Index Release

First official release with the Rust-native GEM graph index.

### Core: Native GEM Graph Index (`latence-gem-index`)
- Rust-native proximity graph over document vector sets (ColBERT/ColPali)
- `GemSegment` (sealed, read-only) and `PyMutableGemSegment` (writable) via PyO3
- HNSW-style beam search with cluster-guided multi-entry points
- Diversity-heuristic neighbor selection for graph quality
- Sequential graph construction reusing `latence-gem-router` codebook/qCH
- Atomic persistence: bincode + CRC32 with crash-safe rename
- Optional semantic shortcut edge injection from training pairs

### Python API
- `Index` class with `engine="gem"` for native multi-vector indexing
- `IndexBuilder` fluent builder pattern
- Full CRUD: `add()`, `delete()`, `update_payload()`, `search()`, `get()`, `scroll()`
- `SearchResult`, `ScrollPage`, `IndexStats` dataclasses
- `snapshot()` for tarball backups

### Segment Manager
- `GemNativeSegmentManager`: fully native mutable active + sealed segments
- `GemHybridSegmentManager`: active HNSW + sealed GEM (transitional)
- WAL-based crash recovery with `WalWriter` / `WalReader`
- `CheckpointManager` for atomic active segment state snapshots
- Background compaction thread with configurable threshold
- Payload filtering with Qdrant-compatible operators

### Performance
- **52x faster search** at 1024 tokens/doc vs HNSW with MaxSim aggregation
- AVX2 vectorized proxy scoring in `latence-gem-router`
- Cache-friendly flat `u16` code layout (`FlatDocCodes`)
- Pre-allocated score buffer reuse across queries

### Documentation
- Updated README with GEM-native focus and benchmark results
- mkdocs site: quickstart, API reference, ColBERT/ColPali guides, benchmarks
- Full Python and Rust API documentation

- created the installable `voyager_index` package and root OSS packaging surface
- exposed exact MaxSim and RoQ kernel exports through the public package
- added CPU-safe MaxSim fallback behavior when Triton is unavailable or rejects the shape
- replaced the placeholder in-memory API with a durable reference FastAPI service
- added dense, late-interaction, and multimodal collection kinds with restart-safe persistence
- shipped durable CRUD semantics with point upserts, point deletes, and restart-safe reloads
- added hybrid dense+BM25 retrieval plus optional local solver-backed dense refinement
- exposed the canonical OSS `/reference/optimize` solver API backed by `latence_solver` when installed
- documented multimodal support for `colqwen3`, `collfm2`, and `nemotron_colembed`
- added Docker packaging, examples, notebooks, validation docs, and OSS foundation documents
- published the Apache-2.0 licensing surface and vendor-boundary docs
