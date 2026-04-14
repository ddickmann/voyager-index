# GEM-Native Graph Index: Research Branch Specification

> **Status**: Deferred. Prerequisites: the Rust set-native router must pass
> promotion gates at 100K+ documents with measured recall and latency evidence.

## 1. Motivation

The current implementation adopts GEM-inspired ideas at the **routing and
screening layer** while retaining the vendored Qdrant HNSW graph as the
underlying ANN backend. Specifically, the production code already implements:

- Two-stage codebook (C_quant, C_index) via k-means (GEM Section 4.1.1)
- TF-IDF guided cluster pruning for per-document C_top profiles (GEM Section 4.1.2)
- Cluster-based posting lists with coarse candidate retrieval
- qCH proxy scoring using centroid codebook lookups (GEM Equation 16)
- Multi-entry HNSW traversal from cluster representative entry points
  (partial implementation of GEM Algorithm 5)
- AVX2-optimized scoring kernels (sgemm, gather-based proxy)

These features accelerate candidate generation (measured: 249x vs brute-force
MaxSim at identical recall on 10K clustered documents). However, the HNSW graph
itself remains a single-vector L2 graph — it does not reason about vector sets
natively. The GEM paper demonstrates that building the graph over vector sets
directly yields up to 16x additional speedup over methods like PLAID, DESSERT,
and MUVERA while matching or improving accuracy.

This specification describes what a full GEM-native graph backend would require
to replace the current Qdrant HNSW backend.

## 2. Reference Material

**Paper**: Yao Tian, Zhoujin Tian, Xi Zhao, Ruiyuan Zhang, Xiaofang Zhou.
"GEM: A Native Graph-based Index for Multi-Vector Retrieval."
arXiv:2603.20336, March 2026.
https://arxiv.org/abs/2603.20336

**Reference implementation** (C++, Apache-2.0 compatible):
https://github.com/sigmod26gem/sigmod26gem

Key files in the reference repository:

| File | Contents |
|------|----------|
| `hnswlib/hnswlib/hnswalg.h` | Modified HNSW with `searchKnnClusterEntries`, `addClusterPointEntry`, `searchNodesForFix`, `mutuallyConnectTwoInterElement`, `canAddEdgeinter`, `entry_map`, `search_set` |
| `hnswlib/hnswlib/vectorset.h` | `vectorset` struct: `float* data`, `int* codes`, `dim`, `vecnum` |
| `hnswlib/hnswlib/space_l2.h` | Distance functions: `L2SqrCluster4Search` (qCH proxy), `L2SqrVecCF` (Chamfer), `L2SqrVecSet` (full set distance), `fast_dot_product_blas` (BLAS matmul), EMD-based distance via `EMD_wrap_self` |
| `hnswlib/otlib/network_simplex_simple.h` | Network simplex solver for EMD computation |
| `hnswlib/otlib/EMD.h` | C wrapper: `EMD_wrap_self(n1, n2, X, Y, D, maxIter)` returns float EMD |
| `hnswlib/otlib/EMD_wrapper.cpp` | Implementation of `EMD_wrap_self` using network simplex |
| `hnswlib/examples/cpp/example_vecset_search_gem.cpp` | Full end-to-end: data loading, `build_fine_cluster`, `search_fine_cluster`, per-cluster graph construction, multi-entry search with `searchKnnClusterEntries`, MaxSim reranking via Eigen |

**Datasets used in the paper** (for validation):

| Dataset | Documents | Vectors | Dim | Embedding model |
|---------|-----------|---------|-----|-----------------|
| MS MARCO | 8.8M passages | ~280M | 128 | ColBERT |
| LoTTE | 2.4M passages | ~339M | 128 | ColBERT |
| OKVQA | 114K docs | ~14M | 128 | PreFLMR ViT-L |
| EVQA | 51K docs | ~9.7M | 128 | PreFLMR ViT-L |

## 3. Architecture: What Changes

### 3.1. Dual Graph Structure (GEM Section 4.3, Algorithm 2)

Replace the single flat HNSW graph (currently `graph_layers.rs` in vendored
Qdrant) with GEM's dual structure:

**Intra-cluster subgraphs**: For each coarse cluster C_i, build a local
proximity graph over the vector sets assigned to C_i. Neighbor selection uses
qEMD (Section 3.3 below) instead of single-vector L2. This produces high-quality
within-cluster connectivity where neighbors share fine-grained semantic overlap.

The reference implementation builds these in `Solution::build_fine_cluster`
(`example_vecset_search_gem.cpp` lines 131-165), calling
`addClusterPointEntry` which inserts each vector set into the graph with a
designated cluster entry point.

**Inter-cluster bridge edges** (GEM Algorithm 3): Vector sets assigned to
multiple clusters serve as natural bridges. When a bridge set P appears in
clusters C_i and C_j, its neighbor list is constructed to contain at least one
neighbor from each cluster, preventing new edges from overwriting cross-cluster
links. The reference implementation enforces this in
`mutuallyConnectTwoInterElement` and the bridge repair pass in
`searchNodesForFix` (`hnswalg.h` lines 2965-3080 and 5174-5230).

**Global graph**: The result is a single physical graph where each vector set
is stored once, logically present in multiple clusters but never duplicated.
Global connectivity is maintained through the bridge edges.

### 3.2. Metric Decoupling (GEM Section 4.2)

GEM uses different distance functions for graph construction vs search:

- **Graph construction**: Earth Mover's Distance (EMD), which satisfies the
  triangle inequality. This ensures the graph has stable, monotonic structure.
  The critical property is `CH(Q,P) <= EMD(Q,P)` (GEM Equation 10), meaning
  EMD-near neighbors are guaranteed to be at least as close under Chamfer/MaxSim.

- **Search traversal**: Quantized Chamfer (qCH) for cheap candidate ordering
  during beam search, with exact Chamfer/MaxSim reranking of the final top-k.

This decoupling resolves the fundamental problem with naive set-level graphs:
Chamfer/MaxSim lacks triangle inequality, so a graph built on it has fragmented
neighborhoods and unstable greedy search (GEM Section 3.2).

### 3.3. Quantized EMD for Graph Construction (GEM Section 4.2.2, Equation 14)

Full EMD is O(n^3) via network simplex. GEM makes it practical by quantizing:

1. Map each token vector to its nearest centroid in C_quant (already done in our
   router's `assign_vectors`).
2. Replace token-level distances with centroid-level lookups from the
   precomputed k1 x k1 codebook distance matrix (already stored as
   `centroid_dists` in our `TwoStageCodebook`).
3. Solve the transport problem on the smaller centroid histogram, not the raw
   token vectors.

The reference implementation uses `EMD_wrap_self` from
`hnswlib/otlib/EMD_wrapper.cpp`, which calls the network simplex solver from
`hnswlib/otlib/network_simplex_simple.h` (a C++ port of the solver by
Nicolas Bonneel, originally from
http://people.seas.harvard.edu/~nbonneel/FastTransport/).

**Rust porting approach**: Port `network_simplex_simple.h` (~600 lines of
templated C++) to a Rust module. The solver operates on `Node`, `Arc` structs
with a cost/capacity network. The interface is:
`fn emd_wrap_self(n1: usize, n2: usize, weights_a: &[f64], weights_b: &[f64], cost_matrix: &[f64], max_iter: u64) -> f32`

For uniform histograms (which GEM uses: each token has weight 1/m), the
transport plan is simpler and admits further optimization.

### 3.4. Semantic Shortcuts (GEM Section 4.4.1, Algorithm 4)

After graph construction, inject directed edges using training query-positive
pairs. For each pair (Q, P_positive):

1. Run beam search (Algorithm 5) on the current graph to find top-f' results.
2. If P_positive is not in the result set, add an edge from the current top-1
   result to P_positive (subject to degree capacity M).

This bridges the gap between EMD-based graph structure and Chamfer-based
relevance: two sets that are close under Chamfer but distant under EMD get
a direct connection.

The reference implementation is visible in the graph construction flow
(`hnswalg.h` lines 3700-3760) where `entry_map` is updated after shortcut
insertion.

**Training pairs**: The paper uses existing supervised training data (e.g.,
MS MARCO qrels). No additional labeling is required. For our use case, query
logs or synthetic positive pairs from the embedding model would serve the same
purpose.

### 3.5. Cluster-Guided Beam Search (GEM Algorithm 5)

The full search algorithm (already partially implemented in our
`search_multi_entry`):

1. **Cluster filtering**: Compute query-centroid relevance matrix
   S = Q @ C_index^T, select top-t centroids per query token, union into
   C_query (GEM Section 4.5.1). Already implemented in our `compute_ctop`.

2. **Multi-entry initialization**: Select one entry point per cluster in
   C_query. Already implemented via `representatives_for_clusters`.

3. **Parallel beam search with shared state**: Each entry point gets a local
   priority queue W_ep. A global visited set V and global result heap R (size
   ef) are shared across all paths. Per-step:
   - Pop best candidate P from W_ep by qCH.
   - If qCH(Q,P) > tau (current furthest in R), terminate this path.
   - For each neighbor P' of P:
     - Skip if P' is in a cluster not in C_query (cluster-aware pruning).
     - Skip if P' is already in V.
     - Compute qCH(Q,P'), update W_ep, R, V.
   - If R changes, update tau for all paths.

   The reference uses `searchKnnClusterEntries` with `search_set` (a
   `vector<bool>` marking which nodes belong to relevant clusters) and
   shared `visited_list_pool`.

4. **Final reranking**: Top-k from R are reranked by exact Chamfer/MaxSim.
   The reference uses Eigen for this: `C = A_mat * B_mat.transpose()`,
   `C.rowwise().maxCoeff().sum() / query.vecnum` (`example_vecset_search_gem.cpp`
   lines 209-215).

Our current `search_multi_entry` in `graph_layers.rs` implements steps 1-3 but
uses single-vector L2 as the scoring function (Qdrant's `points_scorer`), not
qCH or qEMD. The GEM-native version would replace the scorer with set-level
distance and use the graph topology described in Section 3.1.

### 3.6. Adaptive Cluster Cutoff (GEM Section 4.4.2)

GEM trains a lightweight decision tree to predict per-document r (number of
clusters to retain) instead of using a fixed value. The features are the top-r_max
TF-IDF scores and the number of vectors in the set. This is a refinement, not a
prerequisite — a fixed r (e.g., 3) works adequately as the starting point.

## 4. Implementation Plan

### 4.1. New Rust Modules

| Module | Purpose | Port from |
|--------|---------|-----------|
| `src/kernels/gem_router/src/emd.rs` | Network simplex EMD solver | `hnswlib/otlib/network_simplex_simple.h` + `EMD_wrapper.cpp` |
| `src/kernels/gem_router/src/graph.rs` | Dual graph construction: intra-cluster build, bridge management, shortcut injection | `hnswalg.h` (`addClusterPointEntry`, `mutuallyConnectTwoInterElement`, `searchNodesForFix`) |
| `src/kernels/gem_router/src/search.rs` | GEM-native beam search (Algorithm 5) with qCH scoring | `hnswalg.h` (`searchKnnClusterEntries`) |
| `src/kernels/gem_router/src/shortcuts.rs` | Shortcut edge training from query-positive pairs | Algorithm 4 logic |

### 4.2. Modified Existing Files

| File | Change |
|------|--------|
| `src/kernels/gem_router/src/codebook.rs` | Add `qemd_distance(doc_a_codes, doc_b_codes)` using `emd.rs` and precomputed centroid distance matrix |
| `src/kernels/gem_router/src/router.rs` | Add `GemGraphState` holding the dual graph adjacency, integrate with existing `GemRouterState` |
| `src/kernels/gem_router/src/persistence.rs` | Extend serialization format for graph adjacency lists, shortcut edges |
| `src/kernels/gem_router/src/lib.rs` | PyO3 bindings for graph build, search, shortcut training |

### 4.3. What Gets Replaced

The vendored Qdrant HNSW backend (`src/kernels/vendor/qdrant/lib/segment/`)
and the `latence_hnsw` crate (`src/kernels/hnsw_indexer/`) would become
optional fallbacks. The GEM-native graph would be the primary index path for
multi-vector retrieval, while the Qdrant HNSW path remains available for
single-vector dense collections.

This is not a delete-and-replace. The transition would be:

1. Build the GEM-native graph as a new index backend behind a configuration flag.
2. Run A/B benchmarks against the current router + HNSW pipeline.
3. Promote only when benchmark gates pass (Section 5).
4. Deprecate the HNSW multi-entry path for multi-vector workloads.

### 4.4. Porting Strategy

The reference C++ implementation is research-grade code (~5200 lines in
`hnswalg.h`, ~1200 lines in `example_vecset_search_gem.cpp`). The porting
approach:

1. **Port the EMD solver first** (`network_simplex_simple.h`, ~600 lines).
   This is self-contained and can be unit-tested against the C++ version
   with known inputs/outputs.

2. **Port the graph construction** (`addClusterPointEntry` + bridge logic).
   The core is ~300 lines of neighbor selection and degree-limited insertion.
   Adapt to Rust's ownership model: the graph adjacency list is a
   `Vec<Vec<u32>>` with a `parking_lot::RwLock` per node for concurrent
   insertion.

3. **Port the beam search** (`searchKnnClusterEntries`). This reuses the
   existing visited-list pattern from `graph_layers.rs` but replaces the
   scorer with qCH and adds the cluster-aware pruning condition.

4. **Port the shortcut injection** (Algorithm 4). This is straightforward
   once search works: run search, check if positive is found, add edge.

5. **Do not port**: The Eigen-based MaxSim reranking (our existing Triton
   MaxSim kernel is faster), the OpenMP parallelism (use rayon), or the
   data loading infrastructure (use our existing Python/NumPy pipeline).

## 5. Benchmark Gates

The following gates must be satisfied before merging the GEM-native graph
into the default multi-vector search path:

| Gate | Threshold | Rationale |
|------|-----------|-----------|
| Recall@10 after rerank | >= 0.95 | Must match or exceed current router + MaxSim pipeline |
| Recall@100 after rerank | >= 0.98 | High-recall regime for downstream reranking |
| End-to-end search latency (1M docs) | Competitive with router + MaxSim | No latency regression at scale |
| Build time (1M docs, 16 cores) | <= 2 hours | Practical for production rebuild cycles |
| Storage overhead | < 2x current HNSW footprint | Graph adjacency should not dominate storage |
| Index update (insert 1K docs) | < 60 seconds | Must support incremental updates |

Validation datasets:

- Synthetic clustered data (current `benchmark_audit.py` scaled to 100K-1M)
- MS MARCO passage embeddings (ColBERT, dim=128, 8.8M passages) using the
  data format from the reference repo
- At least one multimodal dataset (OKVQA or EVQA with PreFLMR embeddings)

## 6. Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| EMD computation cost during build | High — O(n^3) per pair even with quantization | Use qEMD exclusively; batch with rayon; profile and compare against reference C++ timings |
| Storage format migration | High — breaks existing persisted indexes | Versioned format header; old HNSW format remains loadable; migration tool for existing collections |
| Graph quality regression on out-of-domain data | Medium — dual graph tuned on training distribution | A/B gates on held-out queries; no promotion without OOD evidence |
| Shortcut overfitting to training queries | Medium — shortcuts may not generalize | Use diverse training sample; measure recall with and without shortcuts; cap shortcut degree |
| Memory overhead from dual graph adjacency | Medium — more edges per node than single HNSW | Configurable M per sub-graph; budget-aware edge pruning; measure RSS at 1M scale |
| Porting fidelity vs C++ reference | Low-medium — Rust float semantics may differ | Unit tests against C++ reference outputs for EMD, qCH, graph construction on same input data |

## 7. Decision Criteria

Start this research branch only when ALL of the following hold:

1. The Rust router passes promotion gates at 100K+ documents.
2. Multi-entry HNSW shows measurable recall improvement over single-entry
   on production data (not just synthetic).
3. There is a concrete user need for sub-millisecond candidate generation
   that the router + HNSW pipeline cannot meet.
4. Engineering capacity is available for a 4-8 week focused effort.

## 8. Non-Goals for This Branch

- Replacing the Triton MaxSim reranking kernel (it stays as the final scorer).
- Supporting GEM's adaptive cluster cutoff via decision tree (use fixed r first).
- Multi-GPU or distributed graph construction.
- Changing the public Python API contract (`ColbertIndex`, `ColPaliEngine`).
