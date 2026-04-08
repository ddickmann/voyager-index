# Reference API Tutorial

This is the primary OSS happy path for `voyager-index`.

It walks through the public HTTP API only. You do not need to read the full
repo first.

If you want the advanced, full-surface follow-up after this beginner path, use
`docs/full_feature_cookbook.md`.

If you prefer to run the advanced flow as a logged example with a JSON report,
use `examples/reference_api_feature_tour.py`.

## 1. Install

Install from PyPI (recommended):

```bash
pip install voyager-index[server]
```

That install includes the supported document-rendering stack for
`POST /reference/preprocess/documents` and `voyager_index.render_documents(...)`.

For the full install matrix and product overview, start with `README.md`.

Optional native packages add prebuilt Rust kernels:

```bash
pip install voyager-index[native,server]
```

What they add:

- `latence_hnsw`: optional native dense/HNSW acceleration
- `latence_solver`: canonical OSS solver package for optimized dense refinement and `/reference/optimize`
- `latence_gem_router`: GEM-inspired set-native multi-vector routing core

Neither native package is required for the default OSS tutorial path.

### Install from source (contributors)

```bash
git clone https://github.com/ddickmann/voyager-index.git
cd voyager-index
bash scripts/install_from_source.sh --cpu
```

## 2. Start The Server

```bash
voyager-index-server
```

The interactive API docs are available at:

```text
http://127.0.0.1:8080/docs
http://127.0.0.1:8080/redoc
```

By default the reference service:

- binds to `127.0.0.1`
- stores collections under `VOYAGER_INDEX_PATH` or `/data/voyager-index`
- runs with `WORKERS=1`
- exposes `/health`, `/ready`, and `/metrics`

## 3. Run The Happy-Path Example

```bash
python examples/reference_api_happy_path.py
```

That script exercises:

- one `dense` collection
- one `late_interaction` collection
- one `multimodal` collection
- payload filters
- collection listing
- truthful OSS notes about optional solver and provider integrations

## 4. Dense Collection

Create the collection:

```bash
curl -X POST http://127.0.0.1:8080/collections/tutorial-dense \
  -H "Content-Type: application/json" \
  -d '{"dimension": 4, "kind": "dense"}'
```

Insert points:

`POST /collections/{name}/points` is also the delta-ingestion path: sending the
same IDs again replaces or upserts those points.

```bash
curl -X POST http://127.0.0.1:8080/collections/tutorial-dense/points \
  -H "Content-Type: application/json" \
  -d '{
    "points": [
      {
        "id": "invoice",
        "vector": [1, 0, 0, 0],
        "payload": {"text": "invoice total due", "doc_type": "invoice"}
      },
      {
        "id": "report",
        "vector": [0, 1, 0, 0],
        "payload": {"text": "board report summary", "doc_type": "report"}
      }
    ]
  }'
```

Search with vector + text + filter:

```bash
curl -X POST http://127.0.0.1:8080/collections/tutorial-dense/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [1, 0, 0, 0],
    "query_text": "invoice",
    "filter": {"doc_type": "invoice"},
    "top_k": 2
  }'
```

## 5. Late-Interaction Collection

Create and query a multivector text collection with precomputed embeddings:

```bash
curl -X POST http://127.0.0.1:8080/collections/tutorial-li \
  -H "Content-Type: application/json" \
  -d '{"dimension": 4, "kind": "late_interaction"}'
```

```bash
curl -X POST http://127.0.0.1:8080/collections/tutorial-li/points \
  -H "Content-Type: application/json" \
  -d '{
    "points": [
      {
        "id": "doc-1",
        "vectors": [[1, 0, 0, 0], [1, 0, 0, 0]],
        "payload": {"text": "invoice total due", "label": "invoice"}
      }
    ]
  }'
```

```bash
curl -X POST http://127.0.0.1:8080/collections/tutorial-li/search \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": [[1, 0, 0, 0], [1, 0, 0, 0]],
    "filter": {"label": "invoice"},
    "with_vector": true,
    "top_k": 2
  }'
```

## 6. Multimodal Collection

The collection write API still expects precomputed embeddings, but the supported
ingestion flow can now start from source docs and page images.

Render source docs into PageBundle-like page assets:

```bash
curl -X POST http://127.0.0.1:8080/reference/preprocess/documents \
  -H "Content-Type: application/json" \
  -d '{
    "source_paths": ["/data/source/invoice.pdf"],
    "output_dir": "/data/rendered-pages"
  }'
```

That response gives you page-level `image_path` values plus optional extracted
text. Feed those images into your embedding provider, then store the resulting
patch/token embeddings with `POST /collections/{name}/points`.

Create the collection:

```bash
curl -X POST http://127.0.0.1:8080/collections/tutorial-mm \
  -H "Content-Type: application/json" \
  -d '{"dimension": 4, "kind": "multimodal"}'
```

Insert precomputed patch embeddings:

```bash
curl -X POST http://127.0.0.1:8080/collections/tutorial-mm/points \
  -H "Content-Type: application/json" \
  -d '{
    "points": [
      {
        "id": "page-1",
        "vectors": [[1, 0, 0, 0], [1, 0, 0, 0]],
        "payload": {"doc_id": "invoice.pdf", "page_number": 1, "kind": "invoice"}
      }
    ]
  }'
```

Search:

```bash
curl -X POST http://127.0.0.1:8080/collections/tutorial-mm/search \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": [[1, 0, 0, 0], [1, 0, 0, 0]],
    "filter": {"kind": "invoice"},
    "with_vector": true,
    "top_k": 2
  }'
```

Optional optimized multimodal search:

```bash
curl -X POST http://127.0.0.1:8080/collections/tutorial-mm/search \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": [[1, 0, 0, 0], [1, 0, 0, 0]],
    "strategy": "optimized",
    "multimodal_optimize_mode": "auto",
    "multimodal_candidate_budget": 80,
    "top_k": 2
  }'
```

Important guidance:

- `multimodal_optimize_mode: "auto"` currently resolves to exact MaxSim plus trust-aware lightweight screening
- explicit solver orderings are available as `solver_prefilter_maxsim` and `maxsim_then_solver`
- the current full rendered `tmp_data` benchmark (`547` pages, `8` real-model queries) kept `maxsim_only` as the winner; both solver orderings were slower and lower quality on that corpus
- use those explicit solver orderings only for intentional experiments; for mixed-source chunk pools, prefer `POST /reference/optimize` as the final packing layer instead

## 7. Persistence And Inspection

List collections:

```bash
curl http://127.0.0.1:8080/collections
```

Check readiness and health:

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/ready
curl http://127.0.0.1:8080/docs
```

Collections persist on disk under the configured root:

- `hybrid/` for `dense`
- `colbert/` for `late_interaction`
- `colpali/` for `multimodal`

## 8. Optional Integrations

### Optional Local Native Refinement

If `latence_solver` is built and importable, `dense` search can use
`strategy="optimized"` plus `max_tokens` and `max_chunks` for local refinement.

This is an optional enhancement, not the baseline OSS contract.

### Stateless Optimize Endpoint

The same solver contract is also exposed publicly at:

```bash
curl http://127.0.0.1:8080/reference/optimize/health
```

`POST /reference/optimize` accepts the canonical `OptimizerRequest` JSON shape:

- `query_text`
- `query_vectors` as base64 dense or multivector payloads
- `candidates`
- `constraints`
- `solver_config`
- `metadata`

This endpoint is the OSS reference surface for dense, dense+BM25, multivector, and multivector+BM25 solver requests. It requires `latence_solver` to be installed.

Practical guidance:

- pure multimodal retrieval should stay on exact Triton MaxSim
- if you already have chunks coming from BM25, ontology/rules, dense retrieval, multimodal retrieval, or a mixed union of those sources, `/reference/optimize` is the intended last-layer defense and is a stronger replacement for simple rank fusion like RRF
- this is intentionally a more innovative and somewhat provocative framing than the mainstream stack: retrieval stays truthful, but final LLM context assembly becomes an explicit optimization step rather than an afterthought

### Optional vLLM Pooling Provider

Use `voyager_index.VllmPoolingProvider` to build requests for a user-operated
pooling endpoint:

```bash
python examples/vllm_pooling_provider.py
```

This integration is optional and user-operated. `voyager-index` does not ship a
managed paid service inside the OSS repo.

## 9. Honest OSS Boundaries

The following are intentionally outside the OSS promise:

- ad hoc collection-specific optimize variants like `/collections/{name}/optimize`; use `/reference/optimize` instead
- Voyager research code and premium solver backends: live in the separate `latence-voyager` repo
- graph and Neo4j-adjacent concepts: library-side concepts, not part of the reference HTTP API contract
- built-in document intelligence or remote embedding serving: use external producers and providers, then send embeddings into the OSS API

## 10. Next Stops

- `docs/full_feature_cookbook.md`
- `examples/README.md`
- `notebooks/README.md`
- `MULTIMODAL_FOUNDATION.md`
- `BENCHMARKS.md`
- `docs/validation/README.md`
