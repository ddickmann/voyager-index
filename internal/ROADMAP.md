# colsearch Roadmap

This document captures the forward-looking plan for colsearch, organized
by priority tier.  Items here are **acknowledged gaps** in production readiness
or differentiation that require external resources (datasets, compute, larger
engineering effort) beyond what can be addressed in a single session.

Last updated: 2026-04-10 (post v0.2.0 + scaling hardening).

> **Scaling to millions?** See [`docs/guides/scaling.md`](docs/guides/scaling.md)
> for memory formulas, hyperparameter recommendations, and the v1.2/v2.0
> streaming + sharding roadmap.

---

## What v0.2.0 Already Ships

Before looking forward, here is what the current release delivers and has been
verified through the QA Savage Test Suite (113 tests), promotion gates, and
the unified eval harness:

- Native multi-vector index with GEM-paper-aligned algorithms (compliance matrix in `research/legacy/docs/research/GEM_COMPLIANCE_MATRIX.md`)
- Shared-heap multi-entry beam search (Algorithm 5)
- Two-stage codebook with IDF-weighted cluster assignment
- Metric decoupling: qEMD for construction, qCH for search
- Dual-graph construction with cross-cluster bridge enforcement (Algorithms 1-3)
- Semantic shortcuts with degree caps and age-based pruning (Algorithm 4)
- Adaptive cutoff via decision tree (Section 4.4.2)
- Self-healing mutable graph with drift detection and connectivity reporting
- Filter-aware routing with Roaring bitmaps and selectivity-aware cluster pruning
- GPU-accelerated qCH via Triton kernel (no truncation, autotune, up to 2048 tokens)
- Multi-index ensemble with 1-based RRF fusion
- Persistence with CRC32, versioning, and crash recovery via WAL
- Prometheus /metrics endpoint with search observability counters
- Unified eval harness outputting recall@K, MRR@10, p50/p95/p99, nodes_visited, memory, build_time, fixed-recall QPS
- Filter selectivity sweep benchmark
- CI quality gates: promotion gates, eval harness assertions, Rust clippy

---

## Tier 0: Trust and Credibility (Next Priority)

These items convert "looks impressive" into "proven, reproducible, and
trustworthy."  They are the single highest-leverage investments for adoption.

### R1. Public Full Evaluation Harness on Real Datasets

**Problem:** Current benchmarks run on synthetic data.  Claims like "30x
faster" need to be attached to reproducible configs on public datasets at
fixed recall targets.

**Deliverables:**
- Download and preprocess ColBERT embeddings for MS MARCO passage (8.8M docs)
  and LoTTE (500K docs per domain)
- Run `research/legacy/benchmarks/gem_paper_suite/runner.py` with `--config`
  pointing to real dataset configs (already defined in
  `research/legacy/benchmarks/gem_paper_suite/configs/`)
- Publish results as JSON + markdown tables in `research/legacy/benchmarks/results/`
- Add fixed-recall QPS columns: "at recall@10 >= 0.95, what QPS do we
  achieve?"
- Add scale curves: 100K, 500K, 1M, 5M docs

**Acceptance criteria:**
- Results JSON committed to repo with git SHA and timestamp
- At least 3 datasets: MS MARCO, LoTTE-Writing, one multimodal (OKVQA or EVQA)
- Each row includes: recall@K, MRR@10, p50/p95/p99 latency, nodes_visited,
  distance_comps, build_time, memory_bytes

**Resources needed:** ~50GB disk for embeddings; 8-16 CPU cores for build;
~4 hours compute.

---

### R2. GEM C++ Parity Runner

**Problem:** Users and reviewers will ask "is it really GEM or GEM-like?"
The only way to silence this is a deterministic comparison against the
reference C++ implementation.

**Deliverables:**
- Build script for `sigmod26gem/sigmod26gem` C++ reference repo
- Shared data loader for their `doclens/encoding` format
- Side-by-side comparison script outputting:
  - Recall@K parity (within 2%)
  - Nodes visited / distance computations parity
  - Latency breakdown comparison
  - Build time comparison
- Produce a signed "parity report" artifact per run

**Acceptance criteria:**
- On MS MARCO subset (10K docs), recall@10 within 2% of C++ reference
- Traversal stats (nodes visited) within 20% of C++ reference
- Report artifact committed to `research/legacy/benchmarks/results/parity/`

**Resources needed:** C++ compiler with OpenMP; GEM C++ repo build; same
dataset embeddings as R1.

---

### R3. Graph Health Dashboard

**Problem:** `graph_quality_metrics()` and `graph_connectivity_report()`
provide the raw signals, but there's no integrated dashboard or CI policy
that uses them to enforce health invariants over time.

**Deliverables:**
- Add `graph_health_report()` Python API that combines quality_metrics +
  connectivity_report into a single structured dict
- CI assertion in `gem-quality-gates` job: after build, assert
  `giant_component_frac >= 0.8` on clustered (non-random) synthetic data
- Log graph health at seal time and after compaction in `gem_manager.py`
- Document health metric interpretation in `research/legacy/docs/guides/gem-native.md`

**Acceptance criteria:**
- Health dict includes: delete_ratio, avg_degree, isolated_ratio,
  stale_rep_ratio, n_components, giant_component_frac, cross_cluster_edge_ratio
- CI gate fails if graph health degrades below thresholds on structured data

---

## Tier 1: Competitive Features (Win Users vs Qdrant/GEM)

### R4. Selectivity-Aware Query Planning for Filters

**Problem:** `min_cluster_ratio` is now wired into search, but there is no
automatic selectivity estimation.  Users must manually tune the parameter.

**Deliverables:**
- At `set_doc_payloads` time, compute per-field cardinality and selectivity
  statistics
- During search, estimate filter selectivity from field stats and auto-tune:
  - If selectivity < 5%: expand `n_probes` (more clusters needed to find
    enough matching docs)
  - If selectivity > 80%: standard search (filter barely restricts)
  - Between: adjust `ef` proportionally
- Publish selectivity sweep results on real datasets (from R1)

**Resources needed:** Real filtered dataset (MS MARCO with metadata, or
custom payload annotations); benchmark infrastructure from R1.

---

### R5. Million-Scale Multimodal Benchmarks (ColPali/Patch Embeddings)

**Problem:** Multimodal docs mention ColPali/ColQwen and patch vectors, but
production trust requires million-scale benchmarks with realistic patch counts
(512-2048 vectors/doc).

**Deliverables:**
- Build a "multimodal scale lane" benchmark:
  - Realistic patch counts (512-2048 vectors/doc)
  - Large corpora (>= 1M docs)
  - Report: memory, build time, QPS at fixed recall
- Provide modality-aware cluster profile validation (text vs patch distributions)

**Acceptance criteria:**
- 1M-doc benchmark with 1024 patches/doc completes successfully
- Build time, memory, and recall reported
- Comparison with per-token HNSW baseline

**Resources needed:** ~200GB disk for patch embeddings; GPU for embedding
generation; 32GB+ RAM for index build.

---

### R6. Shortcut Lifecycle Tooling and Ablation

**Problem:** Shortcuts can create hub nodes and regress recall if unmanaged.
The current API provides injection and pruning but no validation pipeline.

**Deliverables:**
- Offline shortcut training tool:
  - Input: query-positive pairs file
  - Output: shortcut edges file
  - Validation: held-out query latency reduction without recall drop
- Ablation results: shortcuts on/off, degree caps (4/8/16), hard-query
  uplift measurement
- Publish results in `research/legacy/benchmarks/results/shortcut_ablation.json`

**Resources needed:** Query log dataset (or synthetic query-positive pairs);
base index from R1 benchmarks.

---

## Tier 2: Differentiators (Things Incumbents Can't Do)

### R7. Adaptive Per-Query Compute Scheduling

**Problem:** GEM paper shows probe count `t` has a sweet spot per query.
We use a fixed `n_probes` for all queries.

**Deliverables:**
- Train a tiny router (logistic regression or small MLP) to predict optimal
  `n_probes` and `ef` based on:
  - Query entropy (how spread the query vectors are across clusters)
  - Cluster score distribution (concentrated vs flat)
  - IDF-weighted salience variance
- Evaluate: "adaptive probes" vs fixed probes on recall-at-fixed-latency

**Resources needed:** Training data from query logs; model training
infrastructure; evaluation on R1 datasets.

---

### R8. Multi-Objective Retrieval Outputs (Frontier API)

**Problem:** Retrieval today returns topK by similarity.  For RAG systems,
what matters is coverage/utility, not just similarity.

**Deliverables:**
- Expose a "search frontier" API that returns:
  - TopK docs
  - Per-doc coverage vectors (which query tokens they satisfy)
  - Cluster frontier (promising unexplored regions)
  - Novelty scores (how different each result is from the rest)
- Allow external controllers (Voyager-Zero optimizer) to steer exploration

**Resources needed:** API design; integration with the existing solver API;
evaluation framework for coverage metrics.

---

### R9. Hardware-Native qCH/Pruning Kernels

**Problem:** CPU path uses scalar Rust; GPU path uses Triton.  Neither
exploits AVX512/VNNI on modern x86 CPUs.

**Deliverables:**
- AVX512/VNNI fast path for `qch_proxy_score_u16` (u16 table lookups)
- Fused "cluster prune + proxy score + topK heap" kernel
- Benchmark: AVX512 vs scalar Rust vs Triton GPU

**Resources needed:** AVX512-capable hardware for testing; SIMD intrinsics
expertise.

---

## Tier 3: Category-Leader Moves (Longer Term)

### R10. Learned Micro-Router (Tiny Model)

**Problem:** Hard-coded routing parameters (n_probes, ef, ctop_r) leave
performance on the table for diverse query distributions.

**Deliverables:**
- Train a tiny model (< 1MB) to predict compute knobs per query
- Integrate as optional `use_learned_router=True` flag
- Validate: no recall regression, latency reduction on hard queries

**Status:** Deferred from v0.2.0 Elite Innovation Layer. Infrastructure
exists (adaptive cutoff tree is a precursor).

---

### R11. Streaming Build for Million-Scale Corpora (v1.2)

**Problem:** The monolithic builder requires all float32 vectors in RAM
simultaneously.  At 1M documents (~128M tokens, dim=128), this costs
~131 GB — impractical on commodity hardware.  The persistent index is only
~1.2 GB; the bottleneck is build-time memory.

**Deliverables:**
- Codebook training on a random sample (1-5% of tokens, < 1 GB)
- Streaming centroid code assignment in batches of 10K-50K docs
- Integration with `MutableGemSegment` for incremental graph insertion
  using only centroid codes (qCH scoring, no raw vectors needed)
- On-the-fly ROQ 4-bit quantization written to memory-mapped storage
  for query-time MaxSim reranking

**Working set:** < 5 GB regardless of corpus size.

**Acceptance criteria:**
- 1M-document index built with < 8 GB peak RAM
- Recall@10 within 2% of monolithic build on held-out queries
- Build time < 2x monolithic (graph insertion is incremental, not batch)

**Prerequisites:** v1.0 release proven at 75K scale; MutableGemSegment
already exists and is tested.

See [`docs/guides/scaling.md`](docs/guides/scaling.md) for full architecture.

---

### R12. Distributed / Sharded Deployment (v2.0)

**Problem:** Single-node only.  For datasets > 10M docs, sharding across
multiple index segments is necessary for both build parallelism and
query-time throughput.

**Deliverables:**
- `GemCollection` managing multiple `GemSegment` shards
- Parallel query dispatch with async result collection
- Cross-shard score normalization (shared codebook or z-normalization)
- Shard balancing by cluster affinity
- Optional cross-shard graph edges for hard queries

**Scaling target:** 10M docs on a single 64 GB machine (10 shards of 1M).

**Prerequisites:** R11 (streaming build), R1 (single-node benchmarks),
production adoption feedback.

---

### R13. Index-Agnostic Optimizer Interface (Voyager-Zero Hooks)

**Problem:** The solver API exists (`/reference/optimize`) but is not
integrated with the index's search frontier.

**Deliverables:**
- Frontier API (R8) feeding into solver
- Novelty anchors for exploration
- Visited/heap sharing across walkers for multi-step retrieval

---

## The Guiding Principle

Features don't buy trust. **Benchmark rigor + parity + observability** do.

Every item in Tier 0 exists to make claims undeniable.  Every item in Tier 1
exists to win users.  Everything beyond is differentiation built on a trusted
foundation.
