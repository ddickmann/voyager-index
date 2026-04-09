# Changelog

## 0.2.0 — Elite Innovation Layer

Production-hardened release adding four innovations that extend the GEM graph
index beyond both the original paper and existing multi-vector engines.

### Filter-Aware Multi-Vector Routing
- Roaring bitmap filter index built per-cluster from document payloads
- `set_doc_payloads()` API on sealed segments for declaring field-value metadata
- Bitmap-guided cluster pruning at search time: filtered queries skip irrelevant
  clusters entirely instead of post-filtering results
- AND semantics across multiple field-value pairs

### GPU-Accelerated qCH Scoring (Optional)
- `GpuQchScorer` class for GPU-native qCH proxy scoring via PyTorch GEMM
- Triton autotuned max-gather kernel with vectorized document blocks (32–128 docs/block)
- Supports document lengths up to 2048 tokens without truncation; auto-fallback
  to PyTorch path for longer documents
- `score_query_filtered()` for combined GPU scoring with boolean document masks
- Fully optional: CPU path remains the default and is unaffected

### Self-Healing Mutable Graphs
- `heal()` method on `PyMutableGemSegment` for periodic local graph repair
- `needs_healing()` drift detection based on delete ratio, isolated nodes,
  and stale cluster representatives
- `graph_quality_metrics()` returns `(delete_ratio, avg_degree, isolated_ratio,
  stale_rep_ratio)` for observability
- Background healing thread in `GemNativeSegmentManager` with configurable interval
- Three-phase repair: medoid recomputation → isolated node reconnection → edge cleanup

### Multi-Index Ensemble with RRF Fusion
- `PyEnsembleGemSegment` for modality-specific codebooks with shared document IDs
- Per-modality graph construction and independent beam search
- Reciprocal Rank Fusion (1-based) merges ranked lists across modalities
- Compatible with the existing Tabu Search solver for constraint-aware fusion
- Validation: query modality tag length must match query token count

### Audit and Hardening
- Zero ruff errors, zero clippy errors across all Python and Rust code
- 235 tests (62 Rust + 173 Python) covering all public API methods
- Quantitative benchmarks for GPU vs CPU scoring, filter overhead, heal
  effectiveness, and ensemble recall
- Shortcut generation pruning synchronized with shortcut vector to prevent
  index drift during age-based pruning
- `prune_stale_shortcuts()` keeps `shortcuts` and `shortcut_generations` in lockstep

### Performance
- `search_multivector()` avoids payload dict copy when no filter is applied
- Query vectors are converted to contiguous float32 once and reused across segments
- Ensemble out-of-range modality tags clamped to modality 0 instead of silent drop
- Triton fallback exceptions now logged at warning level with full error context

---

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
