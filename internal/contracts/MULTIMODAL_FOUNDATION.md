# Multimodal Foundation

This repository treats multimodal late-interaction retrieval as a primary OSS
path, not an afterthought.

## Supported Phase-1 Model Matrix

| Plugin | Model | Architecture | Output Style |
| --- | --- | --- | --- |
| `collfm2` | `VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1` | LFM2-VL + ColPali pooling | ColPali-style multivectors |
| `colqwen3` | `VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1` | Qwen3-VL + ColPali pooling | ColPali-style multivectors |
| `nemotron_colembed` | `nvidia/nemotron-colembed-vl-4b-v2` | bidirectional Qwen3-VL token encoder | ColBERT-style multivectors |

All three models can be surfaced through a vLLM pooling endpoint and are
represented in the public Python package via
`voyager_index.SUPPORTED_MULTIMODAL_MODELS`.

For the standard OSS vLLM-powered ColPali path, the default public choice is
`collfm2`, exposed as `voyager_index.DEFAULT_MULTIMODAL_MODEL` and
`voyager_index.DEFAULT_MULTIMODAL_MODEL_SPEC`.

## Public Multimodal Seam

The supported OSS provider seam is:

- `voyager_index.SUPPORTED_MULTIMODAL_MODELS` for the published model matrix
- `voyager_index.VllmPoolingProvider` for request construction and pooling calls

`vllm-factory` is the preferred accelerated provider implementation for these
models, but plugin internals are not part of the `voyager-index` package
contract.

## Collection Kinds

The reference API supports three collection kinds:

- `dense`: single-vector documents, backed by the hybrid dense+sparse pipeline
- `late_interaction`: text-oriented multivector documents, backed by `ColbertIndex`
- `multimodal`: visual or multimodal multivector documents, backed by `ColPaliEngine`

## Default Multimodal Production Path

For the OSS reference service, the recommended default is:

- disk-backed multimodal collections under the configured local root
- exact Triton MaxSim scoring in FP16
- `strategy="optimized"` defaulting to exact MaxSim plus lightweight screening, not to a different scoring contract
- trust-aware screening controls: bootstrap calibration, persisted health states, risky-query bypass, and exact fallback
- quantized multimodal storage or search only as an explicit opt-in profile

That keeps the default path truthful for local prototypes and small production
deployments, while preserving optional room for faster or smaller experimental
profiles later.

## Optimized Screening Backends

`ColPaliEngine.screen_candidates()` is the production seam for multimodal
lightweight screening. Today it supports two internal backends:

- `prototype_hnsw`: the current optimized default lightweight screening index
- `centroid`: a GPU-friendly centroid tensor lightweight screening index that remains experimental

Both feed candidate IDs into the same exact MaxSim rerank stage, so the API
contract stays truthful even when screening is enabled.

## GEM-Lite Screening Path

The `prototype_hnsw` screening backend now carries a GEM-lite path inspired by
[GEM: A Native Graph-based Index for Multi-Vector Retrieval](https://arxiv.org/abs/2603.20336).

Adopted concepts:

- a two-stage centroid codebook over stored document prototypes
- TF-IDF-style coarse cluster profiles for documents and queries
- qCH-style proxy scoring plus cluster-overlap reranking before exact MaxSim

Explicitly not implemented yet:

- a GEM-native graph
- EMD-based routing edges
- bridge-aware graph maintenance
- semantic shortcuts and the rest of GEM's dual-graph machinery

So this repo should not be described as exposing a native GEM index today. The
actual contract is narrower and safer: exact MaxSim still owns final ranking,
while the prototype sidecar uses GEM-inspired metadata to make candidate
screening more structured.

On the current larger local validation run over fully rendered `tmp_data`
(`547` page records, `4` benchmark queries in this environment), GEM-lite was
active (`512` fine centroids, `8` coarse clusters) and neutral relative to the
legacy prototype sidecar:

- legacy prototype sidecar: `recall_at_k=0.60`, `elapsed_ms≈2674.6`
- GEM-lite prototype sidecar: `recall_at_k=0.60`, `elapsed_ms≈2674.9`

That is enough evidence to keep the path in-tree as a maintained screening
enhancement, but not enough to claim that GEM-lite has already won promotion as
the next major multimodal default.

The public HTTP surface also exposes explicit multimodal ordering knobs for
advanced users:

- `multimodal_optimize_mode`: `auto`, `maxsim_only`, `solver_prefilter_maxsim`, `maxsim_then_solver`
- `multimodal_candidate_budget`
- `multimodal_prefilter_k`
- `multimodal_maxsim_frontier_k`

`auto` currently resolves to `maxsim_only` because that was the winner on the
current full rendered `tmp_data` benchmark.

Internal `sidecar` is still a reasonable nickname in code for these derived
indexes, but the public docs should prefer `lightweight screening index`.

Measured status on current `tmp_data` evidence:

- `internal/validation/validation-sidecar/quantization.json` (`corpus_points=539`, `query_count=8`) shows centroid screening at about `1.11x` end-to-end speedup vs full precision, but only `0.625` `recall_at_k`, with `fallback_rate=0.375`
- the same run shows pooled and prototype lightweight screening indexes around `1.10x` end-to-end speedup with about `0.60` `recall_at_k`
- `internal/validation/validation-sidecar-slice/quantization.json` shows all three screening backends at `0.975` `recall_at_k` on a smaller real-model slice, but still slower than exact end-to-end
- the synthetic scale smoke test also failed to show an exact-search latency win for centroid screening in this environment
- `internal/memos/SCREENING_PROMOTION_DECISION_MEMO.md` defines the current promotion policy and the exact thresholds a screening backend must beat before it graduates

Because of that, centroid screening is wired in but not promoted as the default
optimized multimodal backend.

Measured multimodal ordering result on the current full rendered `tmp_data`
benchmark (`547` pages, `8` real-model queries):

- `maxsim_only`: about `997 ms` average latency, `0.699` `ndcg`, winner
- `maxsim_then_solver`: about `3989 ms` average latency, `0.473` `ndcg`
- `solver_prefilter_maxsim`: about `5732 ms` average latency, `0.188` `ndcg`

So the OSS default stays exact MaxSim plus screening. The solver remains
available as:

- the standalone OSS optimizer via `POST /reference/optimize`
- dense optimized refinement
- an explicit multimodal experiment when users intentionally set
  `multimodal_optimize_mode`

That distinction matters:

- pure multimodal retrieval should stay on exact Triton MaxSim
- once you are assembling a candidate pool from mixed sources such as BM25,
  ontology/rules, metadata filters, dense retrieval, or multimodal retrieval,
  the solver becomes a sensible final packing layer
- in that mixed-source case, the solver is the better replacement for simple
  fusion heuristics like RRF because it can optimize token budget, diversity,
  quorum, and redundancy jointly rather than only blending ranks
- that split is intentionally opinionated: exact multivector relevance should
  remain exact, while final context assembly should be treated as an explicit
  optimization problem rather than delegated to mainstream fusion heuristics

This retrieval infrastructure is not `Voyager`; it is the OSS index/search layer
that can later feed the separate Voyager reasoning engine.

## Ingestion Contract

### Dense

- Request field: `vector`
- Shape: `(dim,)`
- Recommended payload keys: `text`, `title`, `doc_id`

### Late Interaction

- Request field: `vectors`
- Shape: `(tokens, dim)`
- Recommended payload keys: `text`, `doc_id`, `chunk_id`, `token_count`

### Multimodal

- Request field: `vectors`
- Shape: `(patches_or_tokens, dim)`
- Recommended payload keys:
  - `doc_id`
  - `page_number`
  - `image_path` or `source_uri`
  - `text` when OCR or transcript text is available

The collection write API intentionally expects precomputed embeddings. The
supported product flow can now start from source docs by rendering them into
page-image assets through `voyager_index.render_documents(...)` or
`POST /reference/preprocess/documents`, then handing those pages to an external
embedding provider. The index package still does not promise built-in embedding
inference or remote compute-side serving as part of the multimodal OSS
contract.

## Query Contract

- Dense collections accept `vector`, `vectors`, or `query_text`
- `late_interaction` and `multimodal` collections accept `vectors`
- `with_vector=true` returns stored vectors back as `vector` or `vectors` in the
  response payload

## Storage Layout

Each collection lives under the configured root directory:

- `collection.json`: collection metadata, external-to-internal ID mapping, payloads
- `hybrid/`: dense collection HNSW + BM25 state
- `colbert/`: late-interaction HDF5 storage and metadata
- `colpali/`: multimodal engine state, including `manifest.json`, chunk files, `screening_state.json`, and optional derived lightweight screening index directories

Within `colpali/`, the persisted source of truth is a manifest plus chunk files
on disk. In-memory tensors and VRAM caches are runtime accelerators only; they
are not separate storage backends.

## Precision Guidance

For multimodal retrieval, use these profiles intentionally:

- `Exact` (default): FP16 Triton MaxSim
- `Fast` (opt-in): INT8 Triton MaxSim where the fused path is already available
- `Experimental`: FP8
- `Memory Saver`: RoQ4 when storage pressure matters more than latency

## vLLM Provider Flow

Use `voyager_index.VllmPoolingProvider` for the provider side of the contract:

1. Optionally render source docs into page images with `voyager_index.render_documents(...)` or `POST /reference/preprocess/documents`
2. Serve one of the supported models with `pooling_task="token_embed"`
3. Request query or document embeddings from the pooling endpoint
4. Store those token or patch embeddings via the reference API

See `examples/README.md` for the runnable example order and
`docs/reference_api_tutorial.md` for the end-to-end HTTP walkthrough.

## Example

```python
from voyager_index import DEFAULT_MULTIMODAL_MODEL_SPEC, VllmPoolingProvider

spec = DEFAULT_MULTIMODAL_MODEL_SPEC
provider = VllmPoolingProvider("http://127.0.0.1:8200", spec.model_id)
response = provider.pool(
    [{"text": "invoice total", "image": "/path/to/page.png"}],
    extra_kwargs={"pooling_task": spec.pooling_task},
)
```
