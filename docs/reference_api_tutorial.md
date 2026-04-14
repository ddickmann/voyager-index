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
pip install "voyager-index[server,shard]"
pip install "voyager-index[server,shard,native]"  # adds Tabu Search solver
```

The `server` extra includes the supported document-rendering stack for
`POST /reference/preprocess/documents`.

The optional `native` extra currently adds one supported native wheel:

- `latence_solver` for `dense_hybrid_mode="tabu"` and `/reference/optimize`

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

Important truth-in-advertising note:

- shard HTTP search does not take `query_text`
- dense BM25 hybrid stays on `dense` collections over HTTP
- shard + BM25 fusion is an in-process `HybridSearchManager` workflow

## 7. Persistence And Inspection

Collections persist under the configured storage root. Useful endpoints:

```bash
curl http://127.0.0.1:8080/collections
curl http://127.0.0.1:8080/collections/tutorial-shard/info
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/ready
```

Shard-only admin endpoints:

- `POST /collections/{name}/compact`
- `POST /collections/{name}/checkpoint`
- `GET /collections/{name}/wal/status`
- `GET /collections/{name}/shards`
- `POST /collections/{name}/scroll`
- `POST /collections/{name}/retrieve`

## 8. Optional Solver Surface

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

## 9. Honest Boundaries

Outside the OSS HTTP contract:

- collection-specific ad hoc optimize endpoints
- built-in remote embedding hosting
- distributed control-plane features
- internal research backends that are not part of the shard-first public story

## 10. Next Stops

- `docs/full_feature_cookbook.md`
- `docs/guides/max-performance-reference-api.md`
- `docs/guides/shard-engine.md`
- `docs/benchmarks.md`
