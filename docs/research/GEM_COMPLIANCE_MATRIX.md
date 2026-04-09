# GEM Paper Compliance Matrix

Maps every numbered algorithm and key equation from the GEM paper to the
corresponding voyager-index implementation module, with notes on any
intentional deviations.

Paper reference: *GEM: Graph-Enhanced Multi-Vector Retrieval* (SIGMOD 2026).

---

## Algorithms

| Paper Ref | Description | Implementation | Status | Notes |
|-----------|-------------|----------------|--------|-------|
| **Algorithm 1** | Per-cluster local graph construction | [`graph.rs` L741–860 `build_graph_dual`](../../src/kernels/gem_index/src/graph.rs) | Complete | Processes each cluster's member documents, builds local adjacency with diversity heuristic |
| **Algorithm 2** | Neighbor selection with diversity heuristic (metric-decoupled) | [`graph.rs` L277 `select_neighbors_heuristic_cached`](../../src/kernels/gem_index/src/graph.rs) | Complete | Uses construction_distance (qEMD or qCH) per the metric decoupling principle |
| **Algorithm 3** | Cross-cluster bridge enforcement | [`graph.rs` L1002 `update_bridges`](../../src/kernels/gem_index/src/graph.rs) | Complete | Ensures at least one neighbor per assigned cluster; evicts worst by construction metric |
| **Algorithm 4** | Semantic shortcut injection | [`graph.rs` L187 `inject_shortcuts`](../../src/kernels/gem_index/src/graph.rs) | Complete | Inject from (query, positive_doc) pairs; degree-capped; age-based pruning via `prune_stale_shortcuts` |
| **Algorithm 5** | Multi-entry shared-heap beam search | [`search.rs` L83 `beam_search`](../../src/kernels/gem_index/src/search.rs) | Complete | Single visited bitset + single candidate/result heap across all entry points; cluster-guided entries from `compute_ctop` |

## Key Equations

| Paper Ref | Description | Implementation | Status | Notes |
|-----------|-------------|----------------|--------|-------|
| **Eq. 3** | IDF-weighted cluster scoring | [`codebook.rs` L296 `apply_idf_weights`](../../src/kernels/gem_router/src/codebook.rs) | Complete | Smoothed IDF: `ln((N+1)/(df+1))`; applied during both build and search |
| **Eq. 14** | Quantized EMD (qEMD) via histogram reduction | [`emd.rs` `qemd_between_docs`](../../src/kernels/gem_index/src/emd.rs), [`network_simplex.rs` `emd_sinkhorn`](../../src/kernels/gem_index/src/network_simplex.rs) | Complete | Sinkhorn regularization (lambda=20, max_iter=50, early exit at residual < 1e-6); exact network simplex available for validation |
| **Eq. 16** | Quantized Chamfer (qCH) proxy score | [`codebook.rs` L365 `qch_proxy_score_u16`](../../src/kernels/gem_router/src/codebook.rs) | Complete | Per-query-vector max over document codes; u16 quantized centroid scores |
| **Section 4.1.1** | Two-stage codebook (fine + coarse) | [`codebook.rs` L51 `TwoStageCodebook::build`](../../src/kernels/gem_router/src/codebook.rs) | Complete | Hierarchical k-means: fine centroids, then coarse over fine centers |
| **Section 4.1.2** | TF-IDF cluster assignment pruning | [`codebook.rs` L520 `compute_ctop`](../../src/kernels/gem_router/src/codebook.rs), [`codebook.rs` L551 `compute_ctop_adaptive`](../../src/kernels/gem_router/src/codebook.rs) | Complete | Static top-r and adaptive via decision tree (`CutoffTree`) |
| **Section 4.2** | Metric decoupling (EMD for build, CH for search) | [`emd.rs` `construction_distance`](../../src/kernels/gem_index/src/emd.rs) | Complete | Toggle via `use_emd` flag; mutable segments default to qCH for insert latency |
| **Section 4.3** | Dual-graph structure (intra + global) | [`graph.rs` L732 `build_graph_dual`](../../src/kernels/gem_index/src/graph.rs) | Complete | Per-cluster local build + bridge enforcement + global entry point |
| **Section 4.4.2** | Adaptive cutoff (learned ctop_r) | [`adaptive_cutoff.rs` `CutoffTree`](../../src/kernels/gem_router/src/adaptive_cutoff.rs) | Complete | CART decision tree on TF-IDF + doc-length features |

## Architecture Components

| GEM Concept | Module | File(s) |
|-------------|--------|---------|
| Document profiles (C_top assignment) | `DocProfile` | `router.rs` |
| Cluster posting lists | `ClusterPostings` | `router.rs` |
| Flat quantized codes | `FlatDocCodes` | `router.rs` |
| ID tracking + soft deletes | `IdTracker` | `id_tracker.rs` |
| Persistence with versioning | `save_segment / load_segment` | `persistence.rs` |
| Write-ahead log | `GemWAL` | `gem_wal.py` |
| Self-healing mutable graph | `MutableGemSegment::heal` | `mutable.rs` |
| Filter-aware routing | `FilterIndex` (Roaring bitmaps) | `router.rs` |
| GPU-accelerated qCH | Triton kernel + PyTorch fallback | `triton_qch_kernel.py`, `gpu_qch.py` |
| Multi-index ensemble with RRF | `EnsembleSegment` | `ensemble.rs` |

## Intentional Deviations

| Area | Paper Approach | Our Approach | Rationale |
|------|---------------|--------------|-----------|
| Mutable insert | Full qEMD construction | qCH construction (`use_emd=False`) for hot inserts; background seal uses qEMD | O(1) insert latency for streaming workloads; sealed rebuild restores metric quality |
| Sinkhorn regularization | Paper implies exact OT | Sinkhorn with lambda=20, 50 iters | 5-10x faster than exact network simplex; validated correlation with exact EMD |
| Shortcut training | Paper uses query logs | Pair injection API + offline file | No embedded training loop; user provides (query, positive) pairs |
| Graph connectivity | Paper does not specify post-delete health | `graph_connectivity_report()` + `heal()` with drift detection | Production necessity: deletes/compaction degrade navigability |

## Coverage Gaps (Documented)

These paper aspects are acknowledged but not yet implemented:

1. **GEM C++ parity runner** -- deterministic recall/traversal comparison against reference C++ implementation
2. **Adaptive per-query probe scheduling** -- paper shows `t` (probe count) has a sweet spot; we use fixed `n_probes` per query
3. **Hardware-native qCH kernels (AVX512/VNNI)** -- CPU path uses scalar Rust; GPU path uses Triton
