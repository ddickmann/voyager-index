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
    "compression": "rroq158",
    "rroq158_k": 8192,
    "rroq158_group_size": 32,
    "rroq158_seed": 42,
    "quantization_mode": "fp8",
    "transfer_mode": "pinned",
    "router_device": "cpu",
    "use_colbandit": true
  }'
```

For the no-degradation safe-fallback lane (Riemannian 4-bit asymmetric),
swap to `compression="rroq4_riem"` and the related `rroq4_riem_*` knobs:

```bash
curl -X POST http://127.0.0.1:8080/collections/tutorial-shard-safe \
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

## 7. Groundedness Tracker (sidecar)

Post-generation hallucination scoring is provided by the optional
Latence Trace sidecar from [latence.ai](https://latence.ai), which
runs alongside the reference API. The sidecar exposes
`POST /groundedness` with the same `chunk_ids` / `raw_context` contract,
calibrated risk bands, and NLI / semantic-entropy / structured-source
peers. See the [Groundedness sidecar guide](guides/groundedness-sidecar.md)
for the deployment story.

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

## 10. API Endpoint Reference

The reference HTTP server exposes one consistent collection contract plus a
small set of cross-collection helpers. Full request and response schemas are
in the live OpenAPI docs at `http://127.0.0.1:8080/docs`; this table is the
quick map.

### Collections

| Endpoint                                          | Purpose                                                            |
|---------------------------------------------------|--------------------------------------------------------------------|
| `POST /collections/{name}`                        | Create collection (dense, late-interaction, multimodal, or shard)  |
| `GET /collections`                                | List collections                                                   |
| `GET /collections/{name}/info`                    | Inspect collection tuning, health, and graph sync state            |
| `DELETE /collections/{name}`                      | Drop a collection                                                  |

### Points

| Endpoint                                          | Purpose                                                            |
|---------------------------------------------------|--------------------------------------------------------------------|
| `POST /collections/{name}/points`                 | Add or upsert documents                                            |
| `DELETE /collections/{name}/points`               | Delete documents by ID                                             |
| `POST /collections/{name}/retrieve`               | Retrieve documents by ID                                           |
| `GET /collections/{name}/scroll`                  | Scroll through all stored documents                                |

### Search

| Endpoint                                          | Purpose                                                            |
|---------------------------------------------------|--------------------------------------------------------------------|
| `POST /collections/{name}/search`                 | Single-query search (dense, late-interaction, multimodal, shard)   |
| `POST /collections/{name}/search/batch`           | Batched multi-query search                                         |
| `POST /rerank`                                    | Rerank a candidate pool                                            |
| `POST /encode`                                    | Encode text or images to vectors via the active provider           |

Graph-aware search uses the same `POST /search` endpoint and adds the
optional `graph_mode`, `graph_local_budget`, `graph_community_budget`,
`graph_evidence_budget`, `graph_explain`, and `query_payload` fields.

### Durability and admin

| Endpoint                                          | Purpose                                                            |
|---------------------------------------------------|--------------------------------------------------------------------|
| `POST /collections/{name}/checkpoint`             | Force WAL checkpoint                                               |
| `GET /collections/{name}/wal/status`              | WAL health and replay status                                       |
| `GET /health`                                     | Liveness                                                           |
| `GET /ready`                                      | Readiness                                                          |

### Post-generation and document tooling

| Endpoint                                          | Purpose                                                            |
|---------------------------------------------------|--------------------------------------------------------------------|
| `POST /reference/preprocess/documents`            | PDF / DOCX / XLSX / image preprocessing                            |
| `POST /reference/optimize`                        | Tabu Search context packing on a supplied candidate pool           |
| `GET /reference/optimize/health`                  | Solver lane availability                                           |

## 11. Honest Boundaries

Outside the OSS HTTP contract:

- collection-specific ad hoc optimize endpoints
- built-in remote embedding hosting
- distributed control-plane features
- internal research backends that are not part of the shard-first public story

## 12. Next Stops

- `docs/full_feature_cookbook.md`
- `docs/guides/max-performance-reference-api.md`
- `docs/guides/shard-engine.md`
- `docs/benchmarks.md`
