# voyager-index

[![CI](https://github.com/ddickmann/voyager-index/actions/workflows/ci.yml/badge.svg)](https://github.com/ddickmann/voyager-index/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/voyager-index)](https://pypi.org/project/voyager-index/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

`voyager-index` is a shard-first, multi-vector native late-interaction
retrieval system for ColBERT, ColPali, ColQwen, and other token- or patch-level
embedding workloads.

It is built around one production path:

- a shard engine with LEMUR routing
- exact or quantized MaxSim scoring on CPU or GPU
- optional GPU-resident corpus mode
- durable CRUD, WAL, checkpoint, and recovery
- a reference HTTP API with base64 vector transport
- BM25 hybrid search plus `rrf` or `tabu` solver refinement

Legacy `gem` and `hnsw` backends still exist in the codebase for compatibility,
but the supported product story, docs, and CI are shard-first.

## Who This Is For

Use `voyager-index` if you are building:

- late-interaction retrieval with ColBERT-style multivectors
- multimodal retrieval with ColPali-style page or patch embeddings
- a single-host retrieval service that needs honest CPU and GPU paths
- a system that needs CRUD, WAL-backed recovery, and simple deployment

## When To Use Or Not Use It

Use it when:

- you already have token- or patch-level embeddings
- you want one stack for SDK, HTTP API, CPU fallback, and GPU acceleration
- you care about MaxSim quality and want compression knobs without changing the retrieval model

Do not use it when:

- you want a generic distributed vector database first and late interaction second
- you only have pooled dense vectors and do not need multivector retrieval
- you need a fully managed multi-tenant control plane out of the box

## Why It Exists

Most OSS retrieval stacks still treat late interaction as "ANN first, real
scoring later". `voyager-index` keeps late interaction as the main event:
routing narrows the work, but MaxSim remains the truth scorer, and the same
system also handles multimodal preprocessing, BM25 fusion, and durable updates.

## Quickstart

Install the shard path:

```bash
pip install "voyager-index[shard]"
pip install "voyager-index[server,shard]"        # reference API + preprocessing
pip install "voyager-index[server,shard,gpu]"    # Triton MaxSim on CUDA
pip install "voyager-index[server,shard,native]" # adds Tabu Search solver
```

First Python search:

```python
import numpy as np

from voyager_index import Index

rng = np.random.default_rng(7)
docs = [rng.normal(size=(16, 128)).astype("float32") for _ in range(32)]
query = rng.normal(size=(16, 128)).astype("float32")

idx = Index(
    "demo-index",
    dim=128,
    engine="shard",
    n_shards=32,
    k_candidates=256,
    compression="fp16",
)
idx.add(docs, ids=list(range(len(docs))))
results = idx.search(query, k=5)
print(results[0])
idx.close()
```

First HTTP search with the preferred base64 transport:

```python
import numpy as np
import requests

from voyager_index import encode_vector_payload

query = np.random.default_rng(7).normal(size=(16, 128)).astype("float32")

response = requests.post(
    "http://127.0.0.1:8080/collections/my-shard/search",
    json={
        "vectors": encode_vector_payload(query, dtype="float16"),
        "top_k": 5,
        "quantization_mode": "fp8",
        "use_colbandit": True,
    },
    timeout=30,
)
response.raise_for_status()
print(response.json()["results"][0])
```

Run the server:

```bash
HOST=0.0.0.0 WORKERS=4 voyager-index-server
# OpenAPI docs: http://127.0.0.1:8080/docs
```

## What You Get

| Capability | What it means |
|---|---|
| Shard-first retrieval | LEMUR routes queries into a manageable candidate pool before MaxSim |
| CPU and GPU modes | Same collection format, with exact CPU fallback and Triton GPU acceleration |
| Triton MaxSim | Exact late-interaction scoring on CUDA with fused kernels |
| Quantization | `int8`, `fp8`, and `roq4` scoring modes on the fast GPU path |
| ColBANDIT | Query-time pruning is wired into the production shard pipeline |
| GPU corpus mode | When the corpus fits in VRAM, scoring becomes gather + Triton MaxSim |
| Durable CRUD | Add, upsert, delete, checkpoint, compact, reload after restart |
| Multimodal preprocessing | PDF, DOCX, XLSX, and images into page assets for ColPali-family models |
| Hybrid search | BM25 plus dense retrieval with `rrf` or `tabu` |
| Base64 HTTP transport | Preferred/default wire format for large dense and multivector requests |

## CPU And GPU Modes

| Mode | What stays the same | What changes |
|---|---|---|
| CPU exact | LEMUR routing, CRUD, WAL, collection format | Exact scoring stays on CPU; quantized GPU kernels are not used |
| GPU streamed | LEMUR routing, shard storage, CRUD, WAL | Candidate docs are fetched from disk/CPU memory and scored on GPU |
| GPU corpus | LEMUR routing, API, query path | Corpus stays resident in VRAM for the lowest exact-stage latency |

Practical rule:

- start with CPU if you want the simplest deployment
- add `gpu` when exact MaxSim or quantized rerank latency matters
- use GPU corpus mode when the corpus fits comfortably in VRAM

## Hybrid, Solver, And Multimodal

- Dense collections expose BM25-only, vector-only, or fused BM25+dense search.
- Set `dense_hybrid_mode="rrf"` for simple fusion.
- Set `dense_hybrid_mode="tabu"` when `latence_solver` is installed and you want solver-backed packing/refinement.
- Late-interaction and multimodal collections share the same base64 vector contract.
- Multimodal ingestion starts with `enumerate_renderable_documents()` and `render_documents()`, then stores the resulting embeddings in the same retrieval stack.

## Documentation

- [Quickstart](docs/getting-started/quickstart.md)
- [Installation](docs/getting-started/installation.md)
- [Python API Reference](docs/api/python.md)
- [Reference API Tutorial](docs/reference_api_tutorial.md)
- [Shard Engine Guide](docs/guides/shard-engine.md)
- [Max-Performance Reference API Guide](docs/guides/max-performance-reference-api.md)
- [Scaling Guide](docs/guides/scaling.md)
- [Benchmarks And Methodology](docs/benchmarks.md)
- [Production Notes](PRODUCTION.md)

## Benchmark Methodology

The benchmark story is intentionally split in two:

- `benchmarks/oss_reference_benchmark.py` is the reproducible smoke benchmark for package and API sanity.
- the 100k comparison is the product benchmark and will be published with fixed methodology and raw reports.

Rules for published comparisons:

- same corpus and embeddings across all systems
- same `top_k` and recall target
- warmup runs are excluded from measured latency
- GPU corpus and non-GPU corpus modes are reported separately
- QPS claims are paired with recall, not shown alone

### 100k Comparison Placeholder

Pending fresh measurement on the same corpus and hardware:

| System | Mode | Corpus placement | Recall@10 | p50 | p95 | QPS | Status |
|---|---|---|---|---|---|---|---|
| Plaid | pending | pending | pending | pending | pending | pending | pending measurement |
| FastPlaid | GPU corpus | pending | pending | pending | pending | pending | pending measurement |
| Qdrant | pending | pending | pending | pending | pending | pending | pending measurement |
| Voyager | shard streamed | CPU/disk -> GPU | pending | pending | pending | pending | pending measurement |
| Voyager | shard GPU corpus | GPU resident | pending | pending | pending | pending | pending measurement |

## Install From Source

```bash
git clone https://github.com/ddickmann/voyager-index.git
cd voyager-index
bash scripts/install_from_source.sh --cpu
```

Today the supported native add-on story is simple:

- `latence_solver`: optional solver wheel for `tabu` refinement and `/reference/optimize`

## Docker

```bash
docker build -f deploy/reference-api/Dockerfile -t voyager-index .
docker run -p 8080:8080 -v "$(pwd)/data:/data" voyager-index
```

## Public Surface

- `Index` and `IndexBuilder` for local shard collections
- `SearchPipeline` for dense + sparse fusion in-process
- `ColbertIndex` for late-interaction text workflows
- `ColPaliEngine` and `MultiModalEngine` for multimodal retrieval
- `encode_vector_payload()` and `decode_payload()` for shared base64 transport
- `voyager-index-server` for the reference HTTP API

## License

Apache-2.0. See `LICENSE`.
