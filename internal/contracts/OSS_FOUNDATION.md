# Voyager Index OSS Foundation

This document defines the supported OSS surface for `voyager-index`.

## Public Python Surface

Supported public imports live under `voyager_index`:

- `voyager_index.BM25Config`
- `voyager_index.FusionConfig`
- `voyager_index.IndexConfig`
- `voyager_index.Neo4jConfig`
- `voyager_index.ColbertIndex`
- `voyager_index.SearchPipeline`
- `voyager_index.ColPaliEngine`
- `voyager_index.ColPaliConfig`
- `voyager_index.MultiModalEngine`
- `voyager_index.TRITON_AVAILABLE`
- `voyager_index.DEFAULT_MULTIMODAL_MODEL`
- `voyager_index.DEFAULT_MULTIMODAL_MODEL_SPEC`
- `voyager_index.MultimodalModelSpec`
- `voyager_index.fast_colbert_scores`
- `voyager_index.roq_maxsim_1bit`
- `voyager_index.roq_maxsim_2bit`
- `voyager_index.roq_maxsim_4bit`
- `voyager_index.roq_maxsim_8bit`
- `voyager_index.SUPPORTED_MULTIMODAL_MODELS`
- `voyager_index.VllmPoolingProvider`

Imports through `src.*`, `inference.*`, `kernels.*`, or `latence.*` should be
treated as internal or legacy paths, not the OSS contract.

## Namespace And Packaging Boundary

- `voyager_index.*` is the supported public namespace
- `voyager_index._internal.*` is packaged implementation detail and may change
- `src.*` is not a supported or shipped Python namespace
- the reference service CLI is `voyager-index-server`, exposed through `voyager_index.server`
- the package is published to PyPI as `voyager-index`; `pip install voyager-index` is the recommended install path
- source checkout plus local install is the contributor and development alternative

## Supported Foundation Capabilities

- durable local CRUD through the reference API, including point upserts and point deletes
- late-interaction indexing through `ColbertIndex`
- vector-first local hybrid retrieval through `SearchPipeline`
- disk-backed local collections with restart-safe reload semantics
- MaxSim scoring with Triton when available and a PyTorch fallback when not
- RoQ kernel exports when Triton is available
- multimodal late-interaction retrieval through `ColPaliEngine`
- precomputed-embedding query flows for `late_interaction` and `multimodal` collections
- public multimodal provider metadata and request construction through `SUPPORTED_MULTIMODAL_MODELS` and `VllmPoolingProvider`
- the canonical public `/reference/optimize` solver API backed by the OSS `latence_solver` package when installed
- multimodal model integration points for:
  - `collfm2`
  - `colqwen3`
  - `nemotron_colembed`
- reference FastAPI service under `voyager_index.server`, with deploy assets under `deploy/reference-api/`
- the `voyager-index-server` CLI entrypoint under `voyager_index.server`
- filter-aware multi-vector routing via `GemSegment.set_doc_payloads()` and `search(filter=...)`
- optional GPU-accelerated qCH proxy scoring via `GpuQchScorer` (requires PyTorch; Triton optional)
- self-healing mutable graph segments via `PyMutableGemSegment.heal()` and `needs_healing()`
- multi-modal ensemble indexing via `PyEnsembleGemSegment` with Reciprocal Rank Fusion
- background healing and compaction threads in `GemNativeSegmentManager`

## Default OSS Production Path

The supported OSS default is:

- local and restart-safe, with collections persisted under `VOYAGER_INDEX_PATH`
- exact-by-default for late-interaction and multimodal scoring through Triton FP16 MaxSim
- optional INT8 as the main speed-oriented profile where the fused Triton path is already mature
- FP8 as experimental until it is native end-to-end rather than dequantized back to FP16
- RoQ4 as an optional memory-saving profile, not the default latency path

This keeps the public contract truthful for prototypes and small-to-mid-scale
local production deployments.

## Querying Contract

- `SearchPipeline.search()` expects either a single dense query vector or sparse query text
- multi-vector late-interaction queries should use `ColbertIndex` or the reference API directly
- `late_interaction` and `multimodal` collection kinds expect precomputed embeddings, not raw text embedding inside the index package

## Optional Local Refinement Lane

- `SearchPipeline.search(enable_refinement=True)` may use a locally installed `latence_solver`
- this is an optional local/native enhancement, not the baseline OSS retrieval contract
- the wheel under `src/kernels/knapsack_solver/` backs both in-process refinement and the public `/reference/optimize` contract
- that solver is the in-tree OSS Tabu Search knapsack implementation with CPU fallback and optional accelerated execution
- the stable public serving surface is the reference API, not direct imports from the native package source tree

## Multimodal Contract

Phase-1 multimodal support assumes:

- page/document images or precomputed visual embeddings
- multivector embeddings shaped like `(documents, patches_or_tokens, dim)`
- explicit document IDs and optional page-level metadata
- manifest + chunk based local persistence for multimodal embeddings
- optional vLLM pooling endpoints as embedding providers for the supported plugin
  matrix

## Data Residency

Each collection is persisted on disk under the configured local root:

- `collection.json`: collection metadata, payloads, and ID mappings
- `hybrid/`: dense collection HNSW and BM25 state
- `colbert/`: late-interaction HDF5 embeddings and metadata
- `colpali/`: multimodal manifest and chunk files

In-memory and VRAM caches are runtime accelerators layered on top of these
files, not separate persistence modes.

The public mutation/update story is point-oriented:

- `POST /collections/{name}/points` adds, replaces, or upserts points
- `DELETE /collections/{name}/points` removes selected points
- these operations update the persisted collection state under the configured local root

## Deliberately Deferred

The following areas are not part of the supported OSS foundation surface:

- direct public solver APIs under `src/kernels/knapsack_solver/`
- premium solver backends and packaged Python OR modules now live outside OSS
- response scoring
- Voyager research code and manifests, now moved to the separate `latence-voyager` repo
- productized sidecar variants beyond the canonical OSS `/reference/optimize` endpoint

See `internal/contracts/ADAPTER_CONTRACTS.md` for the intended documentation-level seams between
OSS, commercial sidecars, and future compute productization.

## Vendor Boundary

The hybrid/HNSW layer depends on a large vendored Qdrant subtree under
`src/kernels/vendor/qdrant/`. See `internal/contracts/QDRANT_VENDORING.md` and
`THIRD_PARTY_NOTICES.md` for details.

The optional native dense path is built from `src/kernels/hnsw_indexer/`; the
reference package falls back to a pure Python local segment when that wheel is
not installed.
