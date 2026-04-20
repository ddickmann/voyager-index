# Full-Feature Cookbook

This guide is the complete shard-first OSS exploration path for `voyager-index`.

It is designed so a user can walk the whole stack step by step, or skip directly
to the sections they care about.

The cookbook is intentionally honest about three different kinds of features:

- reference HTTP API features you can run directly against `voyager-index-server`
- optional local/native upgrades that change behavior after you install extra packages
- external or library-side integration seams that are documented and supported as boundaries, but are not standalone HTTP endpoints in the OSS reference API

If you want the short first-run path, start with `docs/reference_api_tutorial.md`.

If you want the fastest CPU/GPU deployment path, base64-first request examples,
and shard-specific production tuning, also read
`docs/guides/max-performance-reference-api.md`.

If you want an executable companion to this cookbook, run:

```bash
python examples/reference_api_feature_tour.py --output-json feature-tour-report.json
```

That script logs each step, prints progress as it goes, and writes a JSON report
with the results, skips, and boundary checks.

## Feature Map

| Area | Where It Lives | Status In This Cookbook |
| --- | --- | --- |
| Collection CRUD | reference HTTP API | runnable |
| Dense vector search | reference HTTP API | runnable |
| BM25-only sparse search | reference HTTP API | runnable |
| Dense + BM25 hybrid search | reference HTTP API | runnable |
| Dense hybrid mode selection (`rrf`, `tabu`) | reference HTTP API | runnable |
| Dense optimized refinement with `latence_solver` | reference HTTP API + optional native package | runnable when installed |
| Optional Latence graph-aware retrieval | reference HTTP API + optional commercial dependency | runnable when installed |
| Stateless `/reference/optimize` solver API | reference HTTP API + optional native package | runnable when installed |
| Late-interaction retrieval | reference HTTP API | runnable |
| Multimodal retrieval | reference HTTP API | runnable |
| Shard collection routing + ColBANDIT + quantized scoring | reference HTTP API | runnable |
| Multimodal `strategy="optimized"` screening | reference HTTP API | runnable, with backend selection limits explained |
| Health, readiness, metrics, persistence | reference HTTP API | runnable |
| Multimodal precision profiles (`INT8`, `FP8`, `RoQ`) | OSS guidance + library / validation surface | documented with boundaries |
| vLLM pooling provider | public provider seam | runnable as optional integration |
| Document-processing / `PageBundle` preprocessing | reference HTTP API + public helper | runnable |
| Ontology / `OntologySidecar` seam | adapter contract | documented with workflow |
| Latence graph sidecar boundary | adapter contract + search metadata surface | documented with workflow and API knobs |

## Step 0. Install The Profile You Need

Install from PyPI (recommended):

```bash
pip install "voyager-index[server,shard]"
```

That install includes the supported PDF, DOCX, XLSX, and image rendering stack
used by `/reference/preprocess/documents`.

For the full install matrix and product overview, start with `README.md`.

Optional extras:

```bash
pip install "voyager-index[full]"                     # + full public CPU surface
pip install "voyager-index[full,gpu]"                 # + Triton GPU kernels on CUDA hosts
pip install "voyager-index[server,shard,multimodal]"  # + multimodal helpers
pip install "voyager-index[server,shard,solver]"      # + Tabu Search solver only
pip install "voyager-index[server,shard,native]"      # + both public native wheels
pip install "voyager-index[server,shard,latence-graph]"  # + optional Latence graph lane
```

What the native extras add:

- `latence_shard_engine`: fused Rust shard CPU fast-path for the shard production lane
- `latence_solver`: canonical OSS solver package for dense refinement and `/reference/optimize`
- `latence`: optional LatenceAI SDK used by the premium graph sidecar

### Install from source (contributors)

```bash
git clone https://github.com/ddickmann/voyager-index.git
cd voyager-index
bash scripts/install_from_source.sh --cpu
```
- multimodal extras: optional provider-side helpers and example flows

This source-checkout install flow is mainly for contributors. The supported user
story is the published PyPI package.

## Step 1. Start The Reference Service

```bash
HOST=0.0.0.0 WORKERS=4 voyager-index-server
```

Then open the interactive API docs:

```text
http://127.0.0.1:8080/docs
http://127.0.0.1:8080/redoc
```

Then verify the runtime:

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/ready
curl http://127.0.0.1:8080/metrics
```

What these endpoints are for:

- `/health`: liveness plus version, collection count, and GPU visibility
- `/ready`: degraded-state reporting, including collection load failures and scan-limit warnings
- `/metrics`: Prometheus-friendly counters and gauges
- `/reference/preprocess/documents`: source-doc to page-image preprocessing before embedding

Single-host worker scaling is supported as long as all workers share the same
`VOYAGER_INDEX_PATH`.

Skip ahead if:

- you already have the service running and want to jump directly to search flows

## Step 2. Learn The CRUD Surface Once

The reference API supports the following core collection operations:

- create collection: `POST /collections/{name}`
- inspect collection: `GET /collections/{name}/info`
- list collections: `GET /collections`
- delete collection: `DELETE /collections/{name}`
- add or upsert points: `POST /collections/{name}/points`
- delete points: `DELETE /collections/{name}/points`
- search: `POST /collections/{name}/search`

This add/upsert route is the OSS delta-ingestion story. The repo does not hide
collection updates behind a separate streaming-only API.

Create a dense collection:

```bash
curl -X POST http://127.0.0.1:8080/collections/cookbook-dense \
  -H "Content-Type: application/json" \
  -d '{"dimension": 4, "kind": "dense"}'
```

Inspect it:

```bash
curl http://127.0.0.1:8080/collections/cookbook-dense/info
```

List all collections:

```bash
curl http://127.0.0.1:8080/collections
```

Delete a point:

```bash
curl -X DELETE http://127.0.0.1:8080/collections/cookbook-dense/points \
  -H "Content-Type: application/json" \
  -d '{"ids": ["doc-2"]}'
```

Delete the collection:

```bash
curl -X DELETE http://127.0.0.1:8080/collections/cookbook-dense
```

Skip ahead if:

- you already understand the base CRUD shape and just want feature-specific examples

## Step 3. Dense Retrieval, BM25, And Hybrid Search

Create a dense collection:

```bash
curl -X POST http://127.0.0.1:8080/collections/dense-guide \
  -H "Content-Type: application/json" \
  -d '{"dimension": 4, "kind": "dense", "distance": "cosine"}'
```

Add points with both vectors and text payloads:

```bash
curl -X POST http://127.0.0.1:8080/collections/dense-guide/points \
  -H "Content-Type: application/json" \
  -d '{
    "points": [
      {
        "id": "doc-1",
        "vector": [1, 0, 0, 0],
        "payload": {"text": "invoice total due", "doc_type": "invoice", "tenant": "acme", "token_count": 64}
      },
      {
        "id": "doc-2",
        "vector": [0.9, 0.1, 0, 0],
        "payload": {"text": "invoice backup receipt", "doc_type": "invoice", "tenant": "acme", "token_count": 48}
      },
      {
        "id": "doc-3",
        "vector": [0, 1, 0, 0],
        "payload": {"text": "board report summary", "doc_type": "report", "tenant": "beta", "token_count": 90}
      }
    ]
  }'
```

### 3A. Dense vector-only search

```bash
curl -X POST http://127.0.0.1:8080/collections/dense-guide/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [1, 0, 0, 0], "top_k": 3}'
```

### 3B. BM25-only sparse search

`dense` collections support sparse text-only retrieval through `query_text`.

```bash
curl -X POST http://127.0.0.1:8080/collections/dense-guide/search \
  -H "Content-Type: application/json" \
  -d '{"query_text": "invoice", "top_k": 3}'
```

### 3C. Hybrid dense + BM25 search

Provide both `vector` and `query_text`:

```bash
curl -X POST http://127.0.0.1:8080/collections/dense-guide/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [1, 0, 0, 0],
    "query_text": "invoice",
    "top_k": 3
  }'
```

Choose the dense fusion mode explicitly when needed:

```bash
curl -X POST http://127.0.0.1:8080/collections/dense-guide/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [1, 0, 0, 0],
    "query_text": "invoice",
    "dense_hybrid_mode": "rrf",
    "top_k": 3
  }'
```

Use `dense_hybrid_mode: "tabu"` when `latence_solver` is installed and you want
solver refinement over the fused pool.

### 3D. Optional Latence graph-aware search

The premium graph lane stays behind the normal search contract. It does not add
another retrieval endpoint:

```bash
curl -X POST http://127.0.0.1:8080/collections/dense-guide/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [1, 0, 0, 0],
    "query_text": "invoice dependency",
    "graph_mode": "auto",
    "graph_local_budget": 4,
    "graph_community_budget": 4,
    "graph_evidence_budget": 8,
    "graph_explain": true,
    "top_k": 3
  }'
```

Important behavior:

- graph augmentation runs after the dense and BM25 first stage
- graph candidates are appended additively instead of replacing the base order
- provenance is returned in `metadata.graph.provenance` when `graph_explain=true`
- collection info exposes sidecar health and freshness metadata

### 3E. Payload filters

Filters are flat payload equality checks and work across supported collection kinds.

```bash
curl -X POST http://127.0.0.1:8080/collections/dense-guide/search \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "invoice",
    "filter": {"tenant": "acme"},
    "top_k": 3
  }'
```

### 3F. Return stored vectors

```bash
curl -X POST http://127.0.0.1:8080/collections/dense-guide/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [1, 0, 0, 0],
    "top_k": 2,
    "with_vector": true
  }'
```

### 3G. Prefer base64 transport for new clients

Float arrays remain valid, but the preferred/default transport for larger
vectors is the shared base64 payload contract used by the optimizer surface.

```python
import numpy as np

from voyager_index import encode_vector_payload

body = {
    "vector": encode_vector_payload(np.array([1, 0, 0, 0], dtype="float32"), dtype="float16"),
    "query_text": "invoice",
    "top_k": 3,
}
```

### 3H. Dot-product dense collections

The dense HTTP surface also supports `distance: "dot"`:

```bash
curl -X POST http://127.0.0.1:8080/collections/dense-dot \
  -H "Content-Type: application/json" \
  -d '{"dimension": 2, "kind": "dense", "distance": "dot"}'
```

### 3I. Shard collections for the max-performance path

When you want the LEMUR-routed production path with ColBANDIT and quantized
scoring controls, create a `shard` collection instead of a plain dense one:

```bash
curl -X POST http://127.0.0.1:8080/collections/shard-guide \
  -H "Content-Type: application/json" \
  -d '{
    "dimension": 128,
    "kind": "shard",
    "n_shards": 256,
    "compression": "rroq158",
    "rroq158_k": 8192,
    "rroq158_group_size": 128,
    "rroq158_seed": 42,
    "quantization_mode": "fp8",
    "transfer_mode": "pinned",
    "router_device": "cpu",
    "use_colbandit": true
  }'
```

For the no-degradation safe-fallback lane, swap `compression` to
`"rroq4_riem"` and the related knobs to `rroq4_riem_k` /
`rroq4_riem_group_size` / `rroq4_riem_seed`:

```bash
curl -X POST http://127.0.0.1:8080/collections/shard-safe-fallback \
  -H "Content-Type: application/json" \
  -d '{
    "dimension": 128,
    "kind": "shard",
    "n_shards": 256,
    "compression": "rroq4_riem",
    "rroq4_riem_k": 8192,
    "rroq4_riem_group_size": 32,
    "rroq4_riem_seed": 42,
    "quantization_mode": "rroq4_riem",
    "router_device": "cpu",
    "use_colbandit": true
  }'
```

The detailed CPU/GPU tuning guide for this path is
`docs/guides/max-performance-reference-api.md`.

Important scope note:

- shard HTTP search is vector-only
- BM25 hybrid over HTTP stays on `dense` collections
- shard + BM25 fusion is available through `HybridSearchManager` in programmatic flows

Skip ahead if:

- you only care about sparse/hybrid text flows and do not need multivector retrieval

## Step 4. Optional Dense Refinement With The Solver

This is the canonical OSS knapsack-solver path reused by both dense refinement and `/reference/optimize`.
It is not just a convenience feature. It is the repo's deliberately contrarian
answer to the mainstream `RRF -> large reranker` pattern: retrieve broadly,
then explicitly optimize the final packed context instead of hoping rank fusion
alone gives the best pack.

Requirements:

- `latence_solver` must be built and importable
- you must pass a dense `vector`
- `strategy` must be `"optimized"`

Example:

```bash
curl -X POST http://127.0.0.1:8080/collections/dense-guide/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [1, 0, 0, 0],
    "query_text": "invoice",
    "strategy": "optimized",
    "top_k": 3,
    "max_tokens": 220,
    "max_chunks": 2
  }'
```

When this lane is active, the response may include:

- `objective_score`
- `total_tokens`

Important limits:

- this path requires `latence_solver` to be installed locally
- `/reference/optimize` is the canonical public OSS solver endpoint
- `/collections/{name}/optimize` remains a compatibility placeholder; use `/reference/optimize`
- `strategy="optimized"` without a dense `vector` is invalid for this dense solver path

## Step 5. Stateless `/reference/optimize`

Probe the shared optimizer health endpoint:

```bash
curl http://127.0.0.1:8080/reference/optimize/health
```

The public solver endpoint accepts the same canonical request contract used internally:

- dense candidates with dense vectors
- dense candidates plus BM25-side scores in candidate metadata
- multivector candidates scored through the canonical Triton MaxSim facade
- multivector candidates plus BM25-side scores

This endpoint expects precomputed base64 vector payloads in the `OptimizerRequest` JSON body and requires `latence_solver` to be installed.

Framing:

- mainstream stacks often stop at fusion or send a large candidate pool into a heavyweight reranker
- this solver path is the more provocative alternative: treat final context assembly as an optimization problem
- that is why it makes the most sense once candidates come from heterogeneous sources and must compete for limited LLM context budget

## Step 6. Late-Interaction Retrieval

Create the collection:

```bash
curl -X POST http://127.0.0.1:8080/collections/li-guide \
  -H "Content-Type: application/json" \
  -d '{"dimension": 4, "kind": "late_interaction", "storage_mode": "sync"}'
```

Add multivector points:

```bash
curl -X POST http://127.0.0.1:8080/collections/li-guide/points \
  -H "Content-Type: application/json" \
  -d '{
    "points": [
      {
        "id": "li-1",
        "vectors": [[1, 0, 0, 0], [1, 0, 0, 0]],
        "payload": {"text": "invoice total due", "label": "invoice"}
      },
      {
        "id": "li-2",
        "vectors": [[0, 1, 0, 0], [0, 1, 0, 0]],
        "payload": {"text": "meeting notes", "label": "meeting"}
      }
    ]
  }'
```

Search:

```bash
curl -X POST http://127.0.0.1:8080/collections/li-guide/search \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": [[1, 0, 0, 0], [1, 0, 0, 0]],
    "filter": {"label": "invoice"},
    "with_vector": true,
    "top_k": 2
  }'
```

Important limits:

- `late_interaction` supports `vector` or `vectors`
- it does not support `query_text`
- filters work, but exact multivector filtering can hit the readiness scan ceiling if the filtered candidate set is too large

## Step 7. Multimodal Retrieval

Collection writes still expect embeddings, but the supported product flow can
start from source docs or existing page images.

Render source docs into PageBundle-like page assets:

```bash
curl -X POST http://127.0.0.1:8080/reference/preprocess/documents \
  -H "Content-Type: application/json" \
  -d '{
    "source_paths": ["/data/source/invoice.pdf", "/data/source/sheet.xlsx"],
    "output_dir": "/data/rendered-pages"
  }'
```

Then send the returned `image_path` assets through your embedding provider and
store the resulting vectors with `POST /collections/{name}/points`.

Create the collection:

```bash
curl -X POST http://127.0.0.1:8080/collections/mm-guide \
  -H "Content-Type: application/json" \
  -d '{"dimension": 4, "kind": "multimodal"}'
```

Insert precomputed patch embeddings:

```bash
curl -X POST http://127.0.0.1:8080/collections/mm-guide/points \
  -H "Content-Type: application/json" \
  -d '{
    "points": [
      {
        "id": "page-1",
        "vectors": [[1, 0, 0, 0], [1, 0, 0, 0]],
        "payload": {"doc_id": "invoice.pdf", "page_number": 1, "kind": "invoice"}
      },
      {
        "id": "page-2",
        "vectors": [[0, 1, 0, 0], [0, 1, 0, 0]],
        "payload": {"doc_id": "report.pdf", "page_number": 2, "kind": "report"}
      }
    ]
  }'
```

Search:

```bash
curl -X POST http://127.0.0.1:8080/collections/mm-guide/search \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": [[1, 0, 0, 0], [1, 0, 0, 0]],
    "filter": {"kind": "invoice"},
    "with_vector": true,
    "top_k": 2
  }'
```

Important limits:

- `multimodal` collection writes still expect precomputed embeddings, not raw image inference inside the API
- it supports `vector` or `vectors`, but not `query_text`

## Step 8. Multimodal Optimized Screening

The reference HTTP API supports `strategy="optimized"` for multimodal search.
That means:

- the default `auto` path keeps exact MaxSim as the final scoring contract
- the service may use a lightweight screening index first
- solver orderings are explicit opt-in experiments, not the default multimodal path

Example:

```bash
curl -X POST http://127.0.0.1:8080/collections/mm-guide/search \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": [[1, 0, 0, 0], [1, 0, 0, 0]],
    "strategy": "optimized",
    "multimodal_optimize_mode": "auto",
    "top_k": 2
  }'
```

Advanced explicit solver ordering example:

```bash
curl -X POST http://127.0.0.1:8080/collections/mm-guide/search \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": [[1, 0, 0, 0], [1, 0, 0, 0]],
    "strategy": "optimized",
    "multimodal_optimize_mode": "maxsim_then_solver",
    "multimodal_candidate_budget": 80,
    "multimodal_maxsim_frontier_k": 40,
    "top_k": 2
  }'
```

What is and is not configurable from the HTTP API today:

- yes: opting into `strategy="optimized"`
- yes: selecting `multimodal_optimize_mode`
- yes: setting `multimodal_candidate_budget`, `multimodal_prefilter_k`, and `multimodal_maxsim_frontier_k`
- no: selecting internal screening backend experiments through an HTTP field

Current measured guidance on the rendered `tmp_data` benchmark (`547` pages, `8`
real-model queries):

- `maxsim_only` won at about `997 ms` average latency and `0.699` `ndcg`
- `maxsim_then_solver` dropped to about `0.473` `ndcg` and rose to about `3989 ms`
- `solver_prefilter_maxsim` dropped to about `0.188` `ndcg` and rose to about `5732 ms`

So keep `multimodal_optimize_mode: "auto"` unless you are intentionally
experimenting. Backend-selection and deeper screening experiments still live in
the library and validation surface. See:

- `internal/contracts/MULTIMODAL_FOUNDATION.md`
- `internal/memos/SCREENING_PROMOTION_DECISION_MEMO.md`
- `internal/validation/README.md`

When the solver *does* make sense:

- use exact Triton MaxSim for pure multimodal retrieval over one coherent multimodal index
- use the solver as the final packing layer when your chunk pool already combines signals from multiple sources such as BM25, ontology/rules, metadata heuristics, dense retrieval, and multimodal retrieval
- in that mixed-source case, prefer the solver over simple rank fusion like RRF when you care about token budget, redundancy control, diversity, and pack quality

## Step 9. Precision Profiles And Quantization

Public OSS guidance currently exposes five multimodal precision profiles:

- `Default`: RROQ-1.58 (Riemannian 1.58-bit ternary, K=8192). Fused Triton
  on GPU + Rust SIMD on CPU. ~5.5× smaller storage than FP16 and strictly
  faster than FP16 on CPU at production K=8192 (5.8× p95 in 8-worker
  layout). New default for newly built indexes.
- `Exact`: FP16 Triton MaxSim
- `Fast`: INT8 Triton MaxSim where available
- `Experimental`: FP8
- `Memory Saver`: RoQ4

Important truth:

- these profiles are documented and benchmarked in OSS
- the reference HTTP API does not currently expose a public request field that selects `INT8`, `FP8`, or `RoQ`
- deeper quantization experiments currently live in the library / validation surface rather than as a stable HTTP configuration contract

Where to explore them:

- overview: `README.md`
- multimodal guidance: `internal/contracts/MULTIMODAL_FOUNDATION.md`
- benchmark posture: `docs/benchmarks.md`
- validation evidence: `internal/validation/README.md`

## Step 9A. Groundedness Tracker (sidecar)

Post-generation groundedness scoring is provided by the optional
Latence Trace sidecar from [latence.ai](https://latence.ai). The
sidecar runs next to `voyager-index` and exposes `POST /groundedness`,
accepting the same `chunk_ids` / `raw_context` contract you would
otherwise have called on `voyager-index`. Retrieval and groundedness
scoring run in separate processes so each can be scaled, restarted,
and operated independently. See the
[Groundedness sidecar guide](guides/groundedness-sidecar.md) for the
deployment story and the integration shape.

## Step 10. Persistence, Restart Safety, And Readiness

After creating and filling collections, inspect persistence:

```bash
curl http://127.0.0.1:8080/collections/dense-guide/info
curl http://127.0.0.1:8080/collections/li-guide/info
curl http://127.0.0.1:8080/collections/mm-guide/info
```

Collection storage roots:

- `hybrid/` for dense
- `colbert/` for late interaction
- `colpali/` for multimodal

What survives restart:

- collection metadata
- indexed points
- dense sparse-index state
- late-interaction embeddings and metadata
- multimodal manifest, chunk files, and screening state

What `/ready` can tell you:

- failed collection reloads
- degraded startup state
- filter scan ceiling issues for exact filtered multivector search

## Step 11. Optional vLLM Provider Flow

The public provider seam is:

- `voyager_index.SUPPORTED_MULTIMODAL_MODELS`
- `voyager_index.VllmPoolingProvider`

Example:

```bash
python examples/vllm_pooling_provider.py
```

Recommended flow:

1. serve a supported model on a user-operated pooling endpoint
2. request token or patch embeddings
3. send those embeddings to the reference API as `vectors`

This is an optional integration, not a managed service bundled into the OSS repo.

## Step 12. Optional Document-Processing Pipeline Input

The OSS repo documents the `PageBundle` seam and now ships a local preprocessing
helper plus reference endpoint for turning source docs into page-image assets.

Those surfaces are:

- `voyager_index.render_documents(...)`
- `POST /reference/preprocess/documents`
- the `PageBundle` contract in `internal/contracts/ADAPTER_CONTRACTS.md`

Typical flow:

1. either a user-side document pipeline or the built-in preprocessing helper renders page-native assets, text, and metadata
2. a provider produces embeddings for those pages or chunks
3. `voyager-index` ingests the resulting text and vectors into `dense`, `late_interaction`, or `multimodal` collections

Minimal `PageBundle` shape:

```json
{
  "bundle_version": "1",
  "doc_id": "invoice-001",
  "pages": [
    {
      "page_id": "invoice-001-p1",
      "page_number": 1,
      "image_path": "/path/to/page-1.png",
      "text": "invoice total due",
      "markdown": "# Invoice\n\nTotal due ..."
    }
  ]
}
```

How this connects to BM25:

- `text` or `markdown` can be converted into payload text fields
- those payload text fields support the dense collection's sparse/BM25 branch through `query_text`

Important truth:

- the OSS reference API now includes a preprocessing endpoint for the local doc-to-image stage
- embedding generation still stays outside the OSS runtime; the package does not embed the upstream pooling provider

## Step 13. Optional Ontology / Knowledge Features

The OSS repo also documents `OntologySidecar` in `internal/contracts/ADAPTER_CONTRACTS.md`.

Typical flow:

1. dataset intelligence produces ontology, entity, relation, or concept sidecars
2. those sidecars stay external to the OSS runtime
3. retrieval or downstream systems align on stable IDs and consume the sidecar intentionally

Minimal `OntologySidecar` shape:

```json
{
  "bundle_version": "1",
  "target_kind": "document",
  "targets": [
    {
      "target_id": "invoice-001",
      "entities": ["invoice", "payment"],
      "relations": ["invoice->payment"],
      "concepts": ["accounts_payable"],
      "scores": {"relevance": 0.92}
    }
  ]
}
```

Important truth:

- ontology generation and graph construction still come from external producer or commercial systems
- the reference HTTP API exposes graph-aware search knobs, but not a standalone graph CRUD or traversal API
- the Latence graph lane is additive and optional; the OSS retrieval path remains valid without it
- `Neo4jConfig` and other graph-adjacent research concepts are not part of the supported OSS HTTP contract

## Step 14. What Is Deliberately Not In The OSS HTTP API

These are important so users do not mistake documented seams for shipped HTTP features:

- `/collections/{name}/optimize` is only a compatibility placeholder; use `/reference/optimize`
- premium solver backends are not in OSS
- Voyager reasoning/research code is not in OSS; it now lives in the separate `latence-voyager` repo
- dedicated graph management or traversal endpoints are not part of the reference API
- document intelligence and embedding serving are external producer/provider roles
- multimodal screening-backend selection and quantization tuning are not yet public HTTP knobs

## Step 15. Recommended Reading Paths

If you want to keep going:

- shortest first-run path: `docs/reference_api_tutorial.md`
- runnable examples: `examples/README.md`
- multimodal and screening details: `internal/contracts/MULTIMODAL_FOUNDATION.md`
- benchmark and validation posture: `docs/benchmarks.md`, `internal/memos/SCREENING_PROMOTION_DECISION_MEMO.md`, `internal/validation/README.md`
- contracts and external seams: `internal/contracts/ADAPTER_CONTRACTS.md`
- public contract summary: `internal/contracts/OSS_FOUNDATION.md`
