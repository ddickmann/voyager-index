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

## 7. Groundedness Tracker (Beta)

Use groundedness after generation, not as a replacement for retrieval:

- fast path: score a final answer against the exact `chunk_ids` passed to the LLM
- fallback path: score against `raw_context` when chunk IDs are unavailable,
  using sentence-aware packed windows by default
- output: headline `reverse_context`, calibrated headline
  `reverse_context_calibrated` (with per-token z-scores against an internal
  null bank), secondary `consensus_hardened`, secondary `literal_guarded` plus
  `literal_diagnostics` for unsupported dates/numbers/units/identifiers,
  response-token heatmaps, and top evidence links

For text collections, start the server with a groundedness-capable encoder:

```bash
VOYAGER_GROUNDEDNESS_MODEL=lightonai/GTE-ModernColBERT-v1 voyager-index-server
```

Or point groundedness at a remote `vllm-factory` ModernColBERT deployment:

```bash
VOYAGER_GROUNDEDNESS_VLLM_ENDPOINT=http://127.0.0.1:8000 \
VOYAGER_GROUNDEDNESS_VLLM_MODEL=VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT \
voyager-index-server
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
    "response_text": "The invoice total is due."
  }'
```

The `raw_context` path now defaults to `segmentation_mode="sentence_packed"`
with a `raw_context_chunk_tokens` budget of `256`, so you only need to pass
those fields when you want a different budget or another segmentation mode.
There is no explicit overlap field: if the next sentence would cross the budget,
it is carried into the next support unit intact. Keep the packed budget at or
below the active encoder's real document-length limit; the API returns a
warning when the requested packed window is larger than the encoder can
reliably process.

Look for these response fields:

- `scores`
- `response_tokens`
- `support_units`
- `top_evidence`
- `eligibility`

`scores.reverse_context` remains the raw product headline.
`scores.reverse_context_calibrated` is the same signal standardized against an
internal null bank of unrelated short documents and squashed into `(0, 1)`; it
is the recommended UI-facing aggregate when you need a wide, readable dynamic
range. `scores.null_bank_size` reports the calibration sample size (`0` when
calibration is disabled, in which case the calibrated value falls back to the
raw headline and a `calibration_disabled` warning is emitted).
`scores.consensus_hardened` is a conservative secondary score that should be
read alongside per-token `support_unit_hits_above_threshold`,
`support_unit_soft_breadth`, and `effective_support_units`. Per-token
calibration diagnostics (`reverse_context_calibrated`, `reverse_context_z`,
`null_mean`, `null_std`) appear inside each `response_tokens` row.

`scores.literal_guarded` discounts the calibrated headline by `(1 - rate)^k`
where `k` is the number of unsupported response literals (dates, numbers,
units, currency, percent, URLs, emails, identifiers). The corresponding
`literal_diagnostics.response_literals`, `literal_diagnostics.matches`, and
`literal_diagnostics.mismatches` arrays describe exactly which literals were
matched against the support text union and which were not. Use this when you
want a literal-aware aggregate that is much more conservative than the
embedding-only headline for facts that should be verifiable lexically.

`scores.groundedness_v2` is the optional fused score that combines
`reverse_context_calibrated`, `literal_guarded`, and `nli_aggregate` into a
single convex combination (default weights `(0.5, 0.2, 0.3)`). Channels with
missing values are dropped and the remaining weights are renormalized, so the
fused score stays well defined even when the NLI verifier is disabled.

Enable the **NLI / claim-level verifier** by starting the server with
`VOYAGER_GROUNDEDNESS_NLI_ENABLED=1` (default model
`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`). When enabled the response
includes `scores.nli_aggregate`, `scores.nli_claim_count`,
`scores.nli_skipped_count`, per-token `nli_score` inside `response_tokens`,
and a `nli_diagnostics.claims` array with per-claim text, character offsets,
entailment / neutral / contradiction probabilities, signed score, premise
count, and any `skip_reason` (`no_premises`, `latency_budget`,
`nli_provider_error`). The verifier is bounded by
`VOYAGER_GROUNDEDNESS_NLI_LATENCY_MS` (default `2000` ms); on budget exhaustion
remaining claims are skipped and an `nli_budget_exceeded` warning is added.

Truth-in-advertising note:

- this is a **Beta** groundedness tracker signal
- it is useful for evidence views and QA
- it is **not** a final truth oracle, especially on negation, entity swaps,
  dates, or exact numeric claims
- very long raw-context packing is only as good as the encoder's effective
  document length
- on the current long-context audit (`lightonai/GTE-ModernColBERT-v1`,
  `256`-token windows), quality separation stayed strong but score-only latency
  was still about `90.9 ms` p50 / `97.2 ms` p95

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
