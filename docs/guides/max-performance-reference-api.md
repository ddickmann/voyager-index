# Max-Performance Reference API Guide

This guide is the shortest path to the highest-throughput public `voyager-index`
deployment shape without giving up the production shard features that matter for
quality: LEMUR routing, ColBANDIT, dense BM25 hybrid fusion, and quantized scoring.

Use it together with `docs/reference_api_tutorial.md` for the basics and
`docs/guides/shard-engine.md` for deeper shard internals.

## Who this guide is for

Use this guide if you are deploying `voyager-index-server` on a single machine
and want to choose between:

- CPU exact serving
- streamed GPU exact or quantized serving
- GPU-corpus serving
- higher-QPS worker layouts

Skip this guide if you only need the first local index example; start with
`docs/reference_api_tutorial.md` instead.

## What "production path" means here

The reference server now supports:

- single-host multi-worker serving
- durable on-disk collections with CRUD, WAL-backed mutation handling, and recovery
- worker-visible collection revisions after mutations
- shared async task polling across workers
- dense, late-interaction, multimodal, and shard collections
- base64 vector transport as the preferred HTTP payload format
- shard search overrides for ColBANDIT, INT8, FP8, ROQ4, transfer mode, and exact-stage budgets
- dense BM25 hybrid search with selectable `rrf` or `tabu`

It is still your job to add auth, TLS termination, ingress policy, and
deployment-specific secret management around the service.

## 1. Start the server with multiple workers

The server now defaults to more than one worker on multi-core hosts, and the
container defaults are set to `WORKERS=4`.

```bash
export HOST=0.0.0.0
export PORT=8080
export WORKERS=4
export VOYAGER_INDEX_PATH=/data/voyager-index
voyager-index-server
```

For a larger single machine, `WORKERS=8` is also supported. Keep all workers on
the same storage root when you want shared collection visibility.

## 2. Prefer base64 vector transport

Raw JSON float arrays still work for compatibility, but base64 is the preferred
transport for larger dense or multivector payloads because it is smaller and
faster to serialize.

### Python helper

```python
import numpy as np

from voyager_index import encode_vector_payload

vector = np.random.rand(128).astype("float32")
payload = encode_vector_payload(vector, dtype="float16")

body = {
    "points": [
        {
            "id": "doc-1",
            "vector": payload,
            "payload": {"text": "invoice total due", "tenant": "acme"},
        }
    ]
}
```

For late-interaction or multimodal queries, pass a 2D matrix:

```python
query_vectors = np.random.rand(32, 128).astype("float32")
search_body = {
    "vectors": encode_vector_payload(query_vectors, dtype="float16"),
    "top_k": 10,
}
```

### Wire format

```json
{
  "vector": {
    "encoding": "float16",
    "shape": [1, 128],
    "dtype": "float16",
    "data_b64": "..."
  }
}
```

## 3. Dense collections: BM25 + RRF or Tabu

Dense collections now expose the hybrid mode directly on the public API.

Create a dense collection:

```bash
curl -X POST http://127.0.0.1:8080/collections/dense-prod \
  -H "Content-Type: application/json" \
  -d '{"dimension": 128, "kind": "dense", "distance": "cosine"}'
```

Use classic RRF fusion:

```bash
curl -X POST http://127.0.0.1:8080/collections/dense-prod/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": {"encoding": "float16", "shape": [1, 128], "dtype": "float16", "data_b64": "..."},
    "query_text": "invoice",
    "dense_hybrid_mode": "rrf",
    "top_k": 10
  }'
```

Use solver refinement when `latence_solver` is installed:

```bash
curl -X POST http://127.0.0.1:8080/collections/dense-prod/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": {"encoding": "float16", "shape": [1, 128], "dtype": "float16", "data_b64": "..."},
    "query_text": "invoice",
    "dense_hybrid_mode": "tabu",
    "top_k": 10
  }'
```

## 4. Shard collections: CPU and GPU high-performance paths

Shard collections are the max-performance public path for late-interaction style
retrieval. They keep LEMUR routing and let you select compression, scoring
precision, transfer strategy, and ColBANDIT from the API.

Create a shard collection:

```bash
curl -X POST http://127.0.0.1:8080/collections/shard-prod \
  -H "Content-Type: application/json" \
  -d '{
    "dimension": 128,
    "kind": "shard",
    "n_shards": 256,
    "k_candidates": 2000,
    "compression": "rroq158",
    "rroq158_k": 8192,
    "rroq158_group_size": 32,
    "rroq158_seed": 42,
    "quantization_mode": "fp8",
    "transfer_mode": "pinned",
    "router_device": "cpu",
    "lemur_search_k_cap": 2048,
    "max_docs_exact": 10000,
    "n_full_scores": 4096,
    "pinned_pool_buffers": 3,
    "pinned_buffer_max_tokens": 50000,
    "gpu_corpus_rerank_topn": 16,
    "use_colbandit": true,
    "variable_length_strategy": "bucketed"
  }'
```

### Recommended deployment shapes

- CPU-focused host: run without CUDA, keep `router_device="cpu"`, and use the
  default `rroq158` codec. The CPU lane is wired through the Rust SIMD kernel
  (`latence_shard_engine.rroq158_score_batch`, AVX2/BMI2/FMA + hardware
  `popcnt` + cached rayon thread pool) and is strictly faster than `fp16` on
  CPU at production K=8192 (5.8× p95 in 8-worker layout). When zero quality
  regression matters more than maximum throughput, switch to
  `compression="rroq4_riem"` — the Riemannian 4-bit asymmetric safe-fallback
  lane is wired on the same Rust SIMD path
  (`latence_shard_engine.rroq4_riem_score_batch`) with AVX2/FMA + cached
  rayon pool, ~3× smaller than fp16 on disk, ~0.5% NDCG@10 gap. Other
  Triton quantization modes (`int8`, `fp8`) still fall back to
  full-precision scoring on CPU; only `rroq158`, `rroq4_riem`, and `fp16`
  have native CPU paths today.
- GPU-scored host: keep `router_device="cpu"` or `"cuda"` depending on router
  contention, and use `transfer_mode="pinned"` or `"double_buffered"`.
- GPU full-corpus rerank path: raise `gpu_corpus_rerank_topn` when the corpus
  fits well in VRAM and you want a wider GPU exact rerank frontier.

### Request-time overrides

Every shard collection can still be tuned per request:

```bash
curl -X POST http://127.0.0.1:8080/collections/shard-prod/search \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {"encoding": "float16", "shape": [32, 128], "dtype": "float16", "data_b64": "..."},
    "top_k": 10,
    "quantization_mode": "roq4",
    "use_colbandit": true,
    "transfer_mode": "double_buffered",
    "lemur_search_k_cap": 1536,
    "max_docs_exact": 8000,
    "n_full_scores": 2048,
    "pinned_pool_buffers": 4,
    "pinned_buffer_max_tokens": 75000,
    "gpu_corpus_rerank_topn": 32,
    "n_centroid_approx": 0
  }'
```

### Mode selection summary

- `compression`: `rroq158` (default — Riemannian 1.58-bit, K=8192, GPU+CPU), `rroq4_riem` (Riemannian 4-bit asymmetric — safe-fallback for zero-regression workloads), `fp16`, `int8`, `roq4`
- `rroq158_k`: rroq158 spherical k-means centroid count (default `8192`)
- `rroq158_group_size`: rroq158 ternary group size (default `32`)
- `rroq158_seed`: rroq158 FWHT rotator + k-means initialisation seed (default `42`)
- `rroq4_riem_k`: rroq4_riem spherical k-means centroid count (default `8192`)
- `rroq4_riem_group_size`: rroq4_riem 4-bit asymmetric residual group size (default `32`)
- `rroq4_riem_seed`: rroq4_riem FWHT rotator + k-means initialisation seed (default `42`)
- `quantization_mode`: empty or `none` for exact, or `int8`, `fp8`, `roq4`, `rroq158`, `rroq4_riem`
- `transfer_mode`: `pageable`, `pinned`, `double_buffered`
- `dense_hybrid_mode`: `rrf`, `tabu`
- `n_full_scores`: proxy shortlist size before exact full scoring
- `pinned_pool_buffers` / `pinned_buffer_max_tokens`: advanced CPU->GPU fetch controls

ColBANDIT is now in the production shard serving path, not just an isolated
experiment. If `use_colbandit` is enabled, the request flows through the shard
reranker before the exact fallback path.

Important truth-in-advertising notes:

- shard HTTP search is vector-only and does not accept `query_text`
- BM25 hybrid over HTTP is exposed on dense collections; shard+dense BM25 fusion
  is currently a programmatic `HybridSearchManager` flow
- `quantization_mode` acceleration requires CUDA/Triton for the fast path
- on CPU, `int8` and `fp8` fall back to full-precision scoring, and `roq4`
  exact scoring remains CUDA-only

## 5. Late-interaction and multimodal clients

Late-interaction and multimodal collection APIs also accept the same base64
payload contract on both ingest and search:

- `vector` for a single token / single-vector request
- `vectors` for a multivector matrix

That lets you keep one client serializer for dense, late-interaction, multimodal,
and shard search.

## 6. Operational notes

- Keep all workers on the same `VOYAGER_INDEX_PATH`.
- Use `/ready` to catch degraded BM25 or collection-load states.
- Use `/health` for liveness only.
- Put a reverse proxy in front of the service for TLS, auth, and request policy.
- Keep JSON float-array clients if you need them, but prefer base64 in new code.

## 7. Minimal client pattern

```python
import numpy as np
import requests

from voyager_index import encode_vector_payload

query = np.random.rand(32, 128).astype("float32")
response = requests.post(
    "http://127.0.0.1:8080/collections/shard-prod/search",
    json={
        "vectors": encode_vector_payload(query, dtype="float16"),
        "top_k": 10,
        "quantization_mode": "fp8",
        "use_colbandit": True,
        "transfer_mode": "pinned",
    },
    timeout=30,
)
response.raise_for_status()
print(response.json())
```
