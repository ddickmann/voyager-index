# Python API Reference

## Mainline Surface

The mainline local API is `Index(..., engine="shard")`.

`engine="auto"` is still the constructor default for compatibility, but if you
want the shard production path described in the docs, pass `engine="shard"`
explicitly.

The optional Latence graph lane is not exposed through `Index.search()` on local
shard collections. The programmatic graph-aware surface today is
`SearchPipeline`, and the HTTP surface is the reference API.

## `voyager_index.Index`

Primary local interface for creating, mutating, and querying indexes.

### Constructor

```python
Index(
    path: str,
    dim: int,
    *,
    engine: str = "auto",
    mode: str | None = None,
    embedding_fn: Any | None = None,
    n_fine: int = 256,
    n_coarse: int = 32,
    max_degree: int = 32,
    ef_construction: int = 200,
    n_probes: int = 4,
    enable_wal: bool = True,
    **kwargs,
)
```

### Shard kwargs

These keyword arguments are passed through when `engine="shard"`:

| Argument | Meaning |
|---|---|
| `n_shards` | Number of sealed shards |
| `k_candidates` | LEMUR candidate budget before exact scoring |
| `compression` | Stored representation. Default: `rroq158` (Riemannian 1.58-bit, K=8192, GPU+CPU). Other options: `fp16`, `int8`, `roq4`, `rroq4_riem` (Riemannian 4-bit asymmetric — the safe-fallback lane for zero-regression workloads). The `rroq158` and `rroq4_riem` codecs need `n_tokens >= K` to train the codebook — they auto-shrink K to the largest valid power-of-two when the corpus is too small, and silently downgrade to fp16 when even that does not fit |
| `rroq158_k` | rroq158 spherical k-means centroid count. Default: `8192`. Must be a power of two and ≥ `rroq158_group_size` |
| `rroq158_seed` | rroq158 FWHT rotator + k-means initialisation seed. Default: `42` |
| `rroq158_group_size` | rroq158 ternary group size. Default: `32`. Must be a positive multiple of 32 (so the ternary planes pack into int32 words) |
| `rroq4_riem_k` | rroq4_riem spherical k-means centroid count. Default: `8192`. Must be a power of two and ≥ `rroq4_riem_group_size` |
| `rroq4_riem_seed` | rroq4_riem FWHT rotator + k-means initialisation seed. Default: `42` |
| `rroq4_riem_group_size` | rroq4_riem 4-bit asymmetric residual group size. Default: `32`. Must be a positive even integer that divides `dim` |
| `quantization_mode` | Active scoring mode: `none`, `int8`, `fp8`, `roq4`, `rroq158`, `rroq4_riem` |
| `transfer_mode` | CPU->GPU transfer path: `pageable`, `pinned`, `double_buffered` |
| `router_device` | Device for the LEMUR router, usually `cpu` or `cuda` |
| `lemur_epochs` | Router training epochs |
| `lemur_search_k_cap` | Router search cap |
| `max_docs_exact` | Exact-stage doc budget |
| `n_full_scores` | Proxy shortlist size before full scoring |
| `pinned_pool_buffers` | Pinned-memory buffer pool size |
| `pinned_buffer_max_tokens` | Max tokens per pinned transfer buffer |
| `gpu_corpus_rerank_topn` | GPU-corpus rerank frontier |
| `n_centroid_approx` | Optional centroid-approx candidate stage |
| `variable_length_strategy` | Variable-length exact scheduling mode |
| `uniform_shard_tokens` | Optional shard packing knob |
| `seed` | Random seed |
| `device` | Scoring device for the manager, typically `cpu` or `cuda` |

### Core methods

| Method | Signature | Notes |
|---|---|---|
| `add` | `(vectors, *, ids=None, payloads=None)` | Add multivector documents |
| `add_batch` | `(vectors, *, ids=None, payloads=None)` | Alias for `add()` |
| `add_texts` | `(texts, *, ids=None, payloads=None)` | Uses `embedding_fn` |
| `upsert` | `(vectors, *, ids, payloads=None)` | Insert or replace by ID |
| `search` | `(query, k=10, *, ef=100, n_probes=4, filters=None, explain=False)` | Main query path |
| `search_text` | `(text, k=10, *, ef=100, filters=None, explain=False)` | Uses `embedding_fn` |
| `search_batch` | `(queries, k=10, *, ef=100, n_probes=4, filters=None)` | Batch query path |
| `delete` | `(ids)` | Tombstone documents by ID |
| `update_payload` | `(doc_id, payload)` | Payload-only update |
| `get` | `(ids)` | Retrieve stored payloads |
| `scroll` | `(limit=100, offset=0, *, filters=None)` | Pagination |
| `stats` | `()` | Returns `IndexStats` |
| `snapshot` | `(output_path)` | Tarball snapshot |
| `flush` | `()` | Force pending writes to disk |
| `close` | `()` | Release resources |
| `set_metrics_hook` | `(hook)` | Metrics callback |

### Properties

| Property | Type |
|---|---|
| `path` | `Path` |
| `dim` | `int` |
| `engine` | `str` |

### Example

```python
from voyager_index import Index

idx = Index(
    "my-index",
    dim=128,
    engine="shard",
    n_shards=64,
    k_candidates=512,
    # compression defaults to "rroq158" (K=8192). Override with "fp16" /
    # "int8" / "roq4" if required.
    quantization_mode="fp8",
)
```

## `voyager_index.IndexBuilder`

Fluent builder for the same surface.

```python
from voyager_index import IndexBuilder

idx = (
    IndexBuilder("my-index", dim=128)
    .with_shard(
        n_shards=64,
        k_candidates=512,
        # compression defaults to "rroq158" (K=8192). Override here if you
        # need the legacy "fp16" lane.
        quantization_mode="fp8",
        transfer_mode="pinned",
    )
    .with_wal(enabled=True)
    .build()
)
```

| Method | Meaning |
|---|---|
| `with_shard(**kwargs)` | Select shard engine; recommended path |
| `with_wal(enabled=True)` | Enable WAL-backed mutation safety |
| `with_quantization(n_fine=256, n_coarse=32)` | Codebook config helper |
| `with_gpu_rerank(device="cuda")` | Legacy compatibility helper |
| `with_roq(bits=4, device="cuda")` | Legacy compatibility helper |
| `with_gem(**kwargs)` | Compatibility backend, not the documented mainline |
| `with_hnsw(**kwargs)` | Compatibility backend, not the documented mainline |
| `build()` | Returns `Index` |

## Transport Helpers

Use these helpers for the preferred HTTP wire format:

```python
from voyager_index import VectorPayload, decode_payload, encode_roq_payload, encode_vector_payload
```

| Helper | Meaning |
|---|---|
| `encode_vector_payload(vectors, dtype="float32")` | Encode float vectors to JSON-ready base64 |
| `encode_roq_payload(vectors, num_bits=4, seed=42)` | Encode ROQ payloads |
| `decode_payload(payload)` | Decode a transport payload back to `numpy.ndarray` |
| `VectorPayload` | Public transport payload type |

## Data Classes

### `voyager_index.SearchResult`

```python
@dataclass
class SearchResult:
    doc_id: int
    score: float
    payload: Optional[Dict[str, Any]] = None
    token_scores: Optional[List[float]] = None
    matched_tokens: Optional[List[int]] = None
```

### `voyager_index.ScrollPage`

```python
@dataclass
class ScrollPage:
    results: List[SearchResult]
    next_offset: Optional[int] = None
```

### `voyager_index.IndexStats`

```python
@dataclass
class IndexStats:
    total_documents: int = 0
    sealed_segments: int = 0
    active_documents: int = 0
    dim: int = 0
    engine: str = ""
```

## Search And Config Exports

### `voyager_index.BM25Config`

```python
@dataclass
class BM25Config:
    k1: float = 1.5
    b: float = 0.75
    epsilon: float = 0.25
```

### `voyager_index.FusionConfig`

```python
@dataclass
class FusionConfig:
    strategy: str = "rrf"
    weights: Optional[Dict[str, float]] = None
    normalization: str = "minmax"
    top_k: int = 10
    min_score: float = 0.0
```

### `voyager_index.IndexConfig`

Higher-level configuration surface used by package helpers.

### `voyager_index.Neo4jConfig`

```python
@dataclass
class Neo4jConfig:
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = ""
    database: str = "neo4j"
    max_hop_distance: int = 2
    relationship_types: Optional[List[str]] = None
```

`Neo4jConfig` is a legacy graph-adjacent config surface. It is not the shipped
Latence graph sidecar product lane documented elsewhere in this repo.

### `voyager_index.SearchPipeline`

Programmatic dense + BM25 retrieval surface.

This is the main local Python entry point for graph-aware retrieval today. It
accepts:

- `query_payload` for ontology hints, workflow hints, or graph policy cues
- `graph_mode` with `off`, `auto`, or `force`
- `graph_options` such as `local_budget`, `community_budget`,
  `evidence_budget`, `max_hops`, and `explain`

Example:

```python
import numpy as np

from voyager_index import SearchPipeline

pipeline = SearchPipeline("graph-demo", dim=128, use_roq=False, on_disk=False)
query = np.random.default_rng(7).normal(size=(128,)).astype("float32")

response = pipeline.search(
    query,
    top_k_retrieval=16,
    query_text="service c lineage policy",
    query_payload={
        "ontology_terms": ["Service C", "Export Control"],
        "workflow_type": "compliance",
    },
    graph_mode="auto",
    graph_options={
        "local_budget": 4,
        "community_budget": 4,
        "evidence_budget": 8,
        "max_hops": 2,
        "explain": True,
    },
)
```

Behavioral notes:

- `SearchPipeline` is where dense + BM25 + optional graph retrieval comes together in-process
- shard HTTP search remains vector-only, so use `query_payload` rather than `query_text` to steer graph policy there
- graph candidates are merged additively after first-stage retrieval

### `voyager_index.ColbertIndex`

Higher-level late-interaction text helper exported by the package.

### `voyager_index.ColPaliEngine`

Multimodal retrieval engine for ColPali-family embeddings.

### `voyager_index.MultiModalEngine`

Combined multimodal retrieval surface.

## Multimodal And Preprocessing Exports

### `voyager_index.MultimodalModelSpec`

```python
@dataclass(frozen=True)
class MultimodalModelSpec:
    plugin_name: str
    model_id: str
    architecture: str
    embedding_style: str
    modalities: tuple[str, ...]
    pooling_task: str
    serve_command: str
```

### `voyager_index.VllmPoolingProvider`

Shared vLLM-compatible embedding provider for multimodal flows.

### `voyager_index.enumerate_renderable_documents`

Discovers supported source documents under a directory tree.

### `voyager_index.render_documents`

Renders those documents into page-level assets for embedding and indexing.

## Triton Kernel Exports

All GPU kernels require the `gpu` extra and are optional.

### `voyager_index.fast_colbert_scores`

Exact MaxSim late-interaction scoring.

### `voyager_index.roq_maxsim_1bit`
### `voyager_index.roq_maxsim_2bit`
### `voyager_index.roq_maxsim_4bit`
### `voyager_index.roq_maxsim_8bit`

ROQ scoring kernels exported through the public package.

### `voyager_index.TRITON_AVAILABLE`

Boolean flag indicating whether Triton is available.

## Server Export

The public server module is:

```python
from voyager_index.server import app, create_app, main
```

Use `voyager-index-server` for the packaged CLI entry point.
