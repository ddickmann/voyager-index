# Reference API Tutorial

This is the shortest HTTP-only path through the shipped `voyager-index`
reference API.

Read this first if you want:

- a single-machine retrieval service
- base64-ready request examples
- dense, late-interaction, multimodal, and shard collection basics

Then continue with:

- `docs/guides/max-performance-reference-api.md` for worker and CPU/GPU tuning
- `docs/full_feature_cookbook.md` for the broader surface

## 1. Install

```bash
pip install "voyager-index[full]"
pip install "voyager-index[full,gpu]"
pip install "voyager-index[server,shard]"
pip install "voyager-index[server,shard,solver]"  # adds Tabu Search solver
pip install "voyager-index[server,shard,native]"  # adds both public native wheels
pip install "voyager-index[server,shard,latence-graph]"  # adds the optional Latence graph lane
```

The `server` extra includes the supported document-rendering stack for
`POST /reference/preprocess/documents`.

The supported native extras are:

- `shard-native`: `latence_shard_engine` for the fused Rust shard CPU fast-path
- `solver`: `latence_solver` for `dense_hybrid_mode="tabu"` and `/reference/optimize`
- `native`: both public native wheels together

The optional `latence-graph` extra enables the premium Latence graph sidecar.
Without it, graph-aware search requests fall back to the OSS retrieval path.

## 2. Start The Server

Local development:

```bash
voyager-index-server
```

Single-host production-style start:

```bash
HOST=0.0.0.0 WORKERS=4 voyager-index-server
```

OpenAPI:

```text
http://127.0.0.1:8080/docs
http://127.0.0.1:8080/redoc
```

Useful probes:

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/ready
curl http://127.0.0.1:8080/metrics
```

## 3. Dense Collection

Create a collection:

```bash
curl -X POST http://127.0.0.1:8080/collections/tutorial-dense \
  -H "Content-Type: application/json" \
  -d '{"dimension": 4, "kind": "dense"}'
```

Insert points with the preferred base64 transport:

```python
import requests

from voyager_index import encode_vector_payload

body = {
    "points": [
        {
            "id": "invoice",
            "vector": encode_vector_payload([1, 0, 0, 0], dtype="float16"),
            "payload": {"text": "invoice total due", "doc_type": "invoice"},
        },
        {
            "id": "report",
            "vector": encode_vector_payload([0, 1, 0, 0], dtype="float16"),
            "payload": {"text": "board report summary", "doc_type": "report"},
        },
    ]
}

requests.post(
    "http://127.0.0.1:8080/collections/tutorial-dense/points",
    json=body,
    timeout=30,
).raise_for_status()
```

Search with dense + BM25 fusion:

```python
import requests

from voyager_index import encode_vector_payload

response = requests.post(
    "http://127.0.0.1:8080/collections/tutorial-dense/search",
    json={
        "vector": encode_vector_payload([1, 0, 0, 0], dtype="float16"),
        "query_text": "invoice",
        "filter": {"doc_type": "invoice"},
        "dense_hybrid_mode": "rrf",
        "top_k": 2,
    },
    timeout=30,
)
response.raise_for_status()
print(response.json()["results"][0])
```

Use solver refinement when `latence_solver` is installed:

```json
{
  "dense_hybrid_mode": "tabu"
}
```

Use the optional Latence graph lane when the sidecar is installed and the
collection payloads carry graph-aware metadata:

```json
{
  "graph_mode": "auto",
  "graph_local_budget": 4,
  "graph_community_budget": 4,
  "graph_evidence_budget": 8,
  "graph_explain": true
}
```

The graph lane runs after the dense and BM25 first stage and is merged
additively. Inspect the sidecar lifecycle with:

```bash
curl http://127.0.0.1:8080/collections/tutorial-dense/info
```

Look for `graph_health`, `graph_dataset_id`, `graph_sync_status`, and
`graph_last_successful_sync_at` in the response.

Public transparency note: the graph lane works on Latence graph data derived
from the indexed corpus and linked back to collection targets. The API exposes
sync, health, and provenance metadata without publishing proprietary extraction
heuristics.

## 4. Late-Interaction Collection

Create the collection:

```bash
curl -X POST http://127.0.0.1:8080/collections/tutorial-li \
  -H "Content-Type: application/json" \
  -d '{"dimension": 4, "kind": "late_interaction"}'
```

Insert and query multivectors:

```python
import requests

from voyager_index import encode_vector_payload

doc_vectors = [[1, 0, 0, 0], [1, 0, 0, 0]]

requests.post(
    "http://127.0.0.1:8080/collections/tutorial-li/points",
    json={
        "points": [
            {
                "id": "doc-1",
                "vectors": encode_vector_payload(doc_vectors, dtype="float16"),
                "payload": {"text": "invoice total due", "label": "invoice"},
            }
        ]
    },
    timeout=30,
).raise_for_status()

response = requests.post(
    "http://127.0.0.1:8080/collections/tutorial-li/search",
    json={
        "vectors": encode_vector_payload(doc_vectors, dtype="float16"),
        "filter": {"label": "invoice"},
        "with_vector": True,
        "top_k": 2,
    },
    timeout=30,
)
response.raise_for_status()
print(response.json()["results"][0])
```

Late-interaction collections can use the same `graph_mode`,
`graph_local_budget`, `graph_community_budget`, `graph_evidence_budget`, and
`graph_explain` knobs. The base late-interaction order is preserved and graph
rescues are appended additively.

## 5. Multimodal Collection

The multimodal collection API stores precomputed embeddings, but the reference
server also provides the preprocessing step.

Render source documents:

```bash
curl -X POST http://127.0.0.1:8080/reference/preprocess/documents \
  -H "Content-Type: application/json" \
  -d '{
    "source_paths": ["/data/source/invoice.pdf"],
    "output_dir": "/data/rendered-pages"
  }'
```

Create the collection:

```bash
curl -X POST http://127.0.0.1:8080/collections/tutorial-mm \
  -H "Content-Type: application/json" \
  -d '{"dimension": 4, "kind": "multimodal"}'
```

Insert and search patch embeddings:

```python
import requests

from voyager_index import encode_vector_payload

page_vectors = [[1, 0, 0, 0], [1, 0, 0, 0]]

requests.post(
    "http://127.0.0.1:8080/collections/tutorial-mm/points",
    json={
        "points": [
            {
                "id": "page-1",
                "vectors": encode_vector_payload(page_vectors, dtype="float16"),
                "payload": {"doc_id": "invoice.pdf", "page_number": 1, "kind": "invoice"},
            }
        ]
    },
    timeout=30,
).raise_for_status()

response = requests.post(
    "http://127.0.0.1:8080/collections/tutorial-mm/search",
    json={
        "vectors": encode_vector_payload(page_vectors, dtype="float16"),
        "filter": {"kind": "invoice"},
        "with_vector": True,
        "top_k": 2,
    },
    timeout=30,
)
response.raise_for_status()
print(response.json()["results"][0])
```

Practical guidance:

- `multimodal_optimize_mode="auto"` is the safe default
- explicit solver orderings are for targeted experiments
- pure multimodal retrieval usually wants exact MaxSim first, not solver-first packing
- the optional graph lane is available here too and follows the same additive merge contract

## 6. Shard Collection

Shard collections are the max-performance public retrieval path.

Create one:

```bash
curl -X POST http://127.0.0.1:8080/collections/tutorial-shard \
  -H "Content-Type: application/json" \
  -d '{
    "dimension": 128,
    "kind": "shard",
    "n_shards": 256,
    "compression": "fp16",
    "quantization_mode": "fp8",
    "transfer_mode": "pinned",
    "router_device": "cpu",
    "use_colbandit": true
  }'
```

Shard search is vector-only over HTTP:

```python
import numpy as np
import requests

from voyager_index import encode_vector_payload

query = np.random.default_rng(7).normal(size=(16, 128)).astype("float32")

response = requests.post(
    "http://127.0.0.1:8080/collections/tutorial-shard/search",
    json={
        "vectors": encode_vector_payload(query, dtype="float16"),
        "top_k": 10,
        "quantization_mode": "fp8",
        "transfer_mode": "pinned",
        "use_colbandit": True,
    },
    timeout=30,
)
response.raise_for_status()
print(response.json()["results"][0])
```

Enable the optional graph lane on shard collections with the same endpoint:

```python
response = requests.post(
    "http://127.0.0.1:8080/collections/tutorial-shard/search",
    json={
        "vectors": encode_vector_payload(query, dtype="float16"),
        "top_k": 10,
        "quantization_mode": "fp8",
        "graph_mode": "auto",
        "graph_local_budget": 4,
        "graph_community_budget": 4,
        "graph_evidence_budget": 8,
        "graph_explain": True,
        "query_payload": {
            "ontology_terms": ["Service C", "Export Control"],
            "workflow_type": "compliance",
        },
    },
    timeout=30,
)
response.raise_for_status()
print(response.json()["metadata"]["graph"])
```

Important truth-in-advertising note:

- shard HTTP search does not take `query_text`
- dense BM25 hybrid stays on `dense` collections over HTTP
- shard + BM25 fusion is an in-process `HybridSearchManager` workflow
- shard collections can still use the optional graph lane after first-stage retrieval
- on shard HTTP search, use `query_payload` rather than `query_text` to steer graph policy

## 7. Groundedness (Beta)

Use groundedness after generation, not as a replacement for retrieval:

- fast path: score a final answer against the exact `chunk_ids` passed to the LLM
- fallback path: score against `raw_context` when chunk IDs are unavailable,
  using sentence-aware packed windows by default
- output: scalar groundedness, response-token heatmaps, and top evidence links

For text collections, start the server with a groundedness-capable encoder:

```bash
VOYAGER_GROUNDEDNESS_MODEL=lightonai/GTE-ModernColBERT-v1 voyager-index-server
```

Chunk-ID mode:

```bash
curl -X POST http://127.0.0.1:8080/collections/tutorial-li/groundedness \
  -H "Content-Type: application/json" \
  -d '{
    "chunk_ids": ["doc-1"],
    "query_text": "invoice total due",
    "response_text": "The invoice total is due.",
    "evidence_limit": 3
  }'
```

Raw-context fallback:

```bash
curl -X POST http://127.0.0.1:8080/collections/tutorial-li/groundedness \
  -H "Content-Type: application/json" \
  -d '{
    "raw_context": "Invoice total due. Payment due on receipt.",
    "query_text": "invoice total due",
    "response_text": "The invoice total is due.",
    "segmentation_mode": "sentence_packed",
    "raw_context_chunk_tokens": 1024
  }'
```

The `raw_context` path now defaults to `segmentation_mode="sentence_packed"`
with a `raw_context_chunk_tokens` budget of `1024`, so you only need to pass
those fields when you want a non-default budget or another segmentation mode.
Keep the packed budget at or below the active encoder's real document-length
limit; the API returns a warning when the requested packed window is larger than
the encoder can reliably process.

Look for these response fields:

- `scores`
- `response_tokens`
- `support_units`
- `top_evidence`
- `eligibility`

Truth-in-advertising note:

- this is a **Beta** groundedness / hallucination detection signal
- it is useful for evidence views and QA
- it is **not** a final truth oracle, especially on negation, entity swaps,
  dates, or exact numeric claims
- very long raw-context packing is only as good as the encoder's effective
  document length

## 8. Persistence And Inspection

Collections persist under the configured storage root. Useful endpoints:

```bash
curl http://127.0.0.1:8080/collections
curl http://127.0.0.1:8080/collections/tutorial-shard/info
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/ready
```

When the graph lane is enabled, collection info also exposes sidecar health and
freshness metadata. Readiness will report degraded or failed graph sync states
without taking down the base retrieval service.

Shard-only admin endpoints:

- `POST /collections/{name}/compact`
- `POST /collections/{name}/checkpoint`
- `GET /collections/{name}/wal/status`
- `GET /collections/{name}/shards`
- `POST /collections/{name}/scroll`
- `POST /collections/{name}/retrieve`

## 9. Optional Solver Surface

Check availability:

```bash
curl http://127.0.0.1:8080/reference/optimize/health
```

`/reference/optimize` is the stateless solver endpoint for:

- dense
- dense + BM25
- late-interaction
- multimodal
- mixed candidate pools you want to pack or refine explicitly

Use it when you already have a candidate pool and want optimization, not when
you simply need standard exact multimodal retrieval.

## 10. Honest Boundaries

Outside the OSS HTTP contract:

- collection-specific ad hoc optimize endpoints
- built-in remote embedding hosting
- distributed control-plane features
- internal research backends that are not part of the shard-first public story

## 11. Next Stops

- `docs/full_feature_cookbook.md`
- `docs/guides/max-performance-reference-api.md`
- `docs/guides/shard-engine.md`
- `docs/benchmarks.md`
