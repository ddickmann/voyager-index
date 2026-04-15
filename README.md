# voyager-index

[![CI](https://github.com/ddickmann/voyager-index/actions/workflows/ci.yml/badge.svg)](https://github.com/ddickmann/voyager-index/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/voyager-index)](https://pypi.org/project/voyager-index/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Late-interaction retrieval for on-prem AI systems.**
**Runs on a single machine, supports CPU or GPU execution, and keeps MaxSim as the truth scorer.**

voyager-index is built for teams that want a **multi-vector native**
**ColBERT / ColPali-style retrieval
quality** without adopting a distributed search stack. It combines proxy routing,
exact or quantized MaxSim, multimodal preprocessing, and database-grade
operations behind one API.

The OSS engine stays fast on its own. An optional **Latence graph sidecar**
adds graph-aware candidate rescue, provenance, and freshness-aware metadata as a
premium plane without becoming a hard dependency of the base retrieval path.

```bash
pip install "voyager-index[server,shard,gpu]"
voyager-index-server                          # OpenAPI at :8080/docs
```

**For developers:** one retrieval contract across CPU and GPU, with real APIs and
real failure recovery.
**For infrastructure leaders:** strong late-interaction search on modest
hardware, without taking on distributed search complexity.

## Start Here

Use the shard-first production lane first, then add optional planes only when
they earn their keep:

```bash
pip install "voyager-index[server,shard]"         # production lane on CPU
pip install "voyager-index[server,shard,gpu]"     # add Triton GPU scoring
pip install "voyager-index[server,shard,native]"  # add Tabu Search solver
pip install "voyager-index[server,shard,latence-graph]"  # add optional Latence graph lane
```

```bash
HOST=0.0.0.0 WORKERS=4 voyager-index-server
# OpenAPI docs at http://127.0.0.1:8080/docs
```

If you are evaluating quickly:

- run the [Quickstart](docs/getting-started/quickstart.md)
- use the [Reference API Tutorial](docs/reference_api_tutorial.md) for the HTTP path
- use the [Shard Engine Guide](docs/guides/shard-engine.md) for the high-performance lane
- use the [Latence Graph Sidecar Guide](docs/guides/latence-graph-sidecar.md) for the optional premium lane

## Who This Fits

`voyager-index` is a strong fit when:

- late-interaction retrieval quality matters
- on-prem deployment matters
- single-node operability matters
- CPU and GPU flexibility matters
- you want an API-facing retrieval service, not just an offline benchmark artifact

It is probably **not** the first choice if you need:

- a large distributed control plane across many nodes
- purely dense ANN retrieval at extreme scale
- a hosted multi-tenant SaaS search platform

## Why

Most retrieval systems optimize the shortlist and treat late interaction as an
add-on. voyager-index is built the other way around: **MaxSim is the final
scorer, and the rest of the system exists to make that practical in production.**

- **Proxy routing instead of mandatory graph dependency.** A learned proxy router
  collapses multi-vector candidate generation to ANN over compact routing
  representations, then hands off to exact or quantized MaxSim. The optional
  Latence graph lane augments after first-stage retrieval instead of replacing
  the router.
- **Fast CPU and GPU execution.** Rust fused scoring for CPU, Triton kernels for
  GPU, with the same retrieval contract across both modes.
- **Operational features included.** CRUD, WAL, checkpoint, recovery, metadata,
  and API serving are part of the system, not an afterthought.
- **Built for single-node deployment.** No distributed control plane required for
  the common on-prem use case.

## Current Public Proof

`voyager-index` has two public proof layers, and they should be read together:

- **Core production lane:** the shard-first route is proven by the BEIR shard
  benchmark in `benchmarks/beir_benchmark.py`. That harness measures
  search-only GPU-corpus Triton MaxSim and CPU multiworker fused Rust scoring on
  the same production lane the API serves, with current public results showing
  `2.6-5.0 ms` GPU P95, `164.8-346.8` GPU QPS, and `41.6-271.7` CPU QPS across
  the listed BEIR sets.
- **Optional Latence graph lane:** the graph lane is proven separately by
  `tools/benchmarks/benchmark_latence_graph_quality.py` plus the graph tests. In
  the current representative harness it delivers `+0.75` recall, `+0.333`
  NDCG, and `+0.75` support coverage on graph-shaped queries, `0.0` ordinary-
  query deltas, `57%` graph activation, `3.5` average added candidates on
  graph-shaped queries, and passing route-conformance checks.

The graph proof is intentionally scoped: it shows the shipped graph contract,
additive rescue semantics, provenance tagging, and retrieval uplift on
graph-shaped fixtures. It is not presented as a graph-on BEIR table.

The graph data itself comes from structured Latence graph data derived from the
indexed corpus and synchronized into the sidecar as target-linked graph
contracts. The public guide explains the architecture and provenance model
without exposing proprietary internals.

For full methodology and benchmark caveats, see [Benchmarks And Methodology](docs/benchmarks.md).

## BEIR Benchmark

Measured on **NVIDIA RTX A5000 (24 GB)** using `lightonai/GTE-ModernColBERT-v1`.
Numbers below are **search-only** and exclude query encoding. CPU results use
**8 native Rust workers**. These are full-query-set results, not a sampled
subset.

These results are meant to show three things:

1. **Retrieval quality** on standard BEIR datasets
2. **Search latency and throughput** under realistic conditions
3. **What is achievable on modest on-prem hardware**

| Dataset | Documents | MAP@100 | NDCG@10 | NDCG@100 | Recall@10 | Recall@100 | GPU QPS | GPU P95 (ms) | CPU QPS | CPU P95 (ms) |
|---------|----------:|--------:|--------:|---------:|----------:|-----------:|--------:|-------------:|--------:|-------------:|
| arguana | 8,674 | 0.2598 | 0.3679 | 0.4171 | 0.7402 | 0.9586 | 270.0 | 4.1 | 41.6 | 202.7 |
| fiqa | 57,638 | 0.3818 | 0.4436 | 0.5049 | 0.5059 | 0.7297 | 164.8 | 5.0 | 80.2 | 115.7 |
| nfcorpus | 3,633 | 0.1963 | 0.3833 | 0.3485 | 0.3404 | 0.3348 | 282.6 | 3.8 | 123.3 | 84.4 |
| quora | 15,675 | 0.9686 | 0.9766 | 0.9790 | 0.9930 | 0.9993 | 346.8 | 2.6 | 271.7 | 46.9 |
| scidocs | 25,657 | 0.1383 | 0.1977 | 0.2763 | 0.2070 | 0.4369 | 246.8 | 4.3 | 83.9 | 111.8 |
| scifact | 5,183 | 0.7141 | 0.7544 | 0.7730 | 0.8766 | 0.9567 | 263.4 | 4.0 | 69.1 | 138.4 |

### How to read these results

- **GPU P95 under 6 ms** across all listed datasets shows the fast path is
  practical on A5000-class hardware.
- **CPU mode remains viable** when GPU capacity is limited or reserved for model
  serving.
- Quality metrics are strong while using the same shard/Rust/Triton retrieval
  stack that powers the production API.

### Comparison note: next-plaid

[next-plaid](https://github.com/lightonai/next-plaid) is an important
open-source reference for ColBERT-style serving. Their published numbers are
measured on **NVIDIA H100 80 GB** with the same embedding model. Our numbers
above are measured on an **RTX A5000** and are **search-only**; their reported
QPS includes encoding. Quora is omitted below because their README uses a much
larger corpus for that dataset.

| Dataset | System | NDCG@10 | MAP@100 | Recall@100 | GPU QPS | GPU P95 (ms) | CPU QPS | CPU P95 (ms) |
|---------|--------|--------:|--------:|-----------:|--------:|-------------:|--------:|-------------:|
| arguana | voyager | **0.3679** | **0.2598** | **0.9586** | **270.0** | **4.1** | **41.6** | **202.7** |
| | next-plaid | 0.3499 | 0.2457 | 0.9337 | 13.6 | 170.1 | 17.4 | 454.7 |
| fiqa | voyager | 0.4436 | 0.3818 | 0.7297 | **164.8** | **5.0** | **80.2** | **115.7** |
| | next-plaid | **0.4506** | **0.3871** | **0.7459** | 18.2 | 170.6 | 17.6 | 259.1 |
| nfcorpus | voyager | **0.3833** | **0.1963** | **0.3348** | **282.6** | **3.8** | **123.3** | **84.4** |
| | next-plaid | 0.3828 | 0.1870 | 0.3228 | 6.6 | 262.1 | 16.9 | 219.4 |
| scidocs | voyager | **0.1977** | **0.1383** | 0.4369 | **246.8** | **4.3** | **83.9** | **111.8** |
| | next-plaid | 0.1914 | 0.1352 | **0.4418** | 17.5 | 139.3 | 16.5 | 281.7 |
| scifact | voyager | 0.7544 | 0.7141 | 0.9567 | **263.4** | **4.0** | **69.1** | **138.4** |
| | next-plaid | **0.7593** | **0.7186** | **0.9633** | 7.9 | 169.5 | 16.9 | 305.4 |

In our current benchmark setup, voyager-index is **competitive or better on
retrieval quality** across most listed datasets and shows **materially higher
search throughput with much lower P95 latency** on an RTX A5000. **This is not
a fully apples-to-apples comparison:** next-plaid reports H100 numbers and
includes encoding in QPS, while our numbers are search-only on a smaller GPU.
The table above uses full-query evaluation specifically to avoid publishing a
flattering slice.

## Architecture

voyager-index separates the problem into **routing, storage, exact scoring,
optimization, durability, and serving**. This keeps the retrieval contract stable
across CPU, GPU, and mixed deployment modes.

```text
query vectors (token / patch embeddings)
  → LEMUR routing MLP
  → FAISS ANN over routing representations
  → candidate document IDs
  → optional BM25 fusion when query_text is available
  → optional centroid-approx or doc-mean proxy pruning
  → optional ColBANDIT query-time pruning
  → exact or quantized MaxSim
       Rust fused exact (CPU, mmap, SIMD, GIL-free)
       Triton FP16 / INT8 / FP8 / ROQ-4 (GPU)
       GPU-corpus gather + rerank
  → optional Latence graph augmentation
  → optional solver/context packing
  → top-K results or packed context
```

| Layer | What it does | Why it matters |
|-------|-------------|----------------|
| **Routing** | LEMUR MLP, FAISS MIPS, candidate budgets | Makes late interaction tractable without graph construction |
| **Storage** | Safetensors shards, merged mmap, GPU-resident corpus | Honest CPU and GPU layouts for any corpus size |
| **Exact scoring** | Triton MaxSim, Rust fused MaxSim, quantized kernels | MaxSim stays the truth scorer across all deployment shapes |
| **Optimization** | ColBANDIT pruning, centroid approximation, ROQ-4 | Moves the latency/recall frontier without changing the retrieval contract |
| **Optional graph plane** | Latence graph sidecar, target-linked graph contracts, additive rescue, provenance | Keeps graph awareness premium and post-retrieval |
| **Durability** | WAL, memtable, checkpoint, crash recovery | A retrieval engine that behaves like a real database |
| **Serving** | FastAPI, base64 transport, multi-worker, OpenAPI | One `pip install`, one server, one API contract |

## What Makes It Different

### No mandatory graph dependency

voyager-index uses proxy routing plus exact MaxSim reranking without requiring a
graph build step in the OSS serving path. When installed, the optional Latence
graph sidecar is invoked after first-stage retrieval and merged additively. That
keeps the system simpler to operate while preserving a premium graph lane.

### Rust + Triton hot paths

The CPU path is a native Rust extension (`latence_shard_engine`) with
memory-mapped shards, fused MaxSim, SIMD acceleration, and GIL-free execution.
The GPU path uses Triton kernels for exact and quantized scoring with
variable-length document scheduling.

### Research-backed features in the serving path

LEMUR routing, ColBANDIT query-time pruning, ROQ rotational quantization, and
budget-aware context optimization are integrated into the shipped system rather
than isolated in research notebooks.

### Operational features, not just benchmarking

CRUD, WAL, checkpoint, crash recovery, payload metadata, scroll, and retrieve
are included because retrieval systems in production need operational discipline,
not just benchmark wins.

### Multimodal native

The same serving stack supports text token embeddings (ColBERT) and image patch
embeddings (ColPali/ColQwen), with preprocessing for PDF, DOCX, XLSX, and image
inputs.

## Quickstart

### Install

```bash
pip install "voyager-index[shard]"               # CPU only
pip install "voyager-index[server,shard]"         # + FastAPI server
pip install "voyager-index[server,shard,gpu]"     # + Triton GPU kernels
pip install "voyager-index[server,shard,native]"  # + Tabu Search solver
pip install "voyager-index[server,shard,latence-graph]"  # + optional Latence graph lane
```

### Python SDK

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

### HTTP API

```bash
HOST=0.0.0.0 WORKERS=4 voyager-index-server
# OpenAPI docs at http://127.0.0.1:8080/docs
```

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
print(response.json()["results"][0])
```

### Docker

```bash
docker build -f deploy/reference-api/Dockerfile -t voyager-index .
docker run -p 8080:8080 -v "$(pwd)/data:/data" voyager-index
```

## Execution Modes

| Mode | Corpus placement | Best for |
|------|-----------------|----------|
| **CPU exact** | Disk/mmap → Rust fused MaxSim | Simplest deployment, no GPU required |
| **GPU streamed** | Disk/CPU → GPU transfer → Triton MaxSim | Large corpora that don't fit in VRAM |
| **GPU corpus** | Fully resident in VRAM | Lowest latency when corpus fits |

All three modes share the same collection format, API contract, and retrieval
semantics. Start with CPU, add GPU when latency matters.

## Engineering Knobs

### Routing and candidate budgets

- `k_candidates` — LEMUR candidate budget before exact scoring
- `max_docs_exact` — hard ceiling for the exact-stage document set
- `n_full_scores` — proxy shortlist size before full MaxSim
- `use_colbandit` — enable query-time pruning

### Storage and transfer

- `n_shards` — number of storage shards
- `compression` — `fp16`, `int8`, or `roq4`
- `transfer_mode` — `pageable`, `pinned`, or `double_buffered`

### Scoring and hardware

- `quantization_mode` — exact, `int8`, `fp8`, or `roq4`
- `router_device` — where LEMUR executes (`cpu` or `cuda`)
- `gpu_corpus_rerank_topn` — GPU rerank frontier for corpus-resident mode

## Hybrid and Multimodal

- **BM25 + dense fusion** via `rrf` or Tabu Search refinement
- **Optional Latence graph augmentation** via `graph_mode`, independent graph budgets, and `graph_explain`
- **Multimodal collections** share the same base64 vector contract as text
- **Document preprocessing** handles PDF, DOCX, XLSX, and images via
  `render_documents()` and the `/reference/preprocess/documents` API endpoint

Production note:

- `dense` HTTP collections are where BM25 + `query_text` hybrid search lives today
- `shard` HTTP collections are the high-performance vector route and can still use the optional graph lane additively through `graph_mode` and `query_payload`

## API Surface

| Endpoint | Purpose |
|----------|---------|
| `POST /collections/{name}` | Create collection |
| `GET /collections/{name}/info` | Inspect collection tuning, health, and graph sync state |
| `POST /collections/{name}/points` | Add / upsert documents |
| `POST /collections/{name}/search` | Search |
| `POST /collections/{name}/search/batch` | Batch search |
| `GET /collections/{name}/scroll` | Scroll through results |
| `POST /collections/{name}/retrieve` | Retrieve by ID |
| `DELETE /collections/{name}/points` | Delete documents |
| `POST /collections/{name}/checkpoint` | Checkpoint WAL |
| `GET /collections/{name}/wal/status` | WAL status |
| `POST /encode` | Encode text/images to vectors |
| `POST /rerank` | Rerank results |
| `POST /reference/optimize` | Context packing (Tabu Search) |

Graph-aware search uses the same search endpoint and adds:

- `graph_mode`
- `graph_local_budget`
- `graph_community_budget`
- `graph_evidence_budget`
- `graph_explain`
- `query_payload` for ontology hints, workflow hints, and vector-only graph policy steering

## Public Python Surface

- `Index` and `IndexBuilder` — local shard collections
- `SearchPipeline` — dense + sparse fusion in-process
- `ColbertIndex` — late-interaction text workflows
- `ColPaliEngine` and `MultiModalEngine` — multimodal retrieval
- `encode_vector_payload()`, `decode_payload()` — base64 transport helpers
- `voyager-index-server` — reference HTTP server

## Documentation

- [Quickstart](docs/getting-started/quickstart.md)
- [Installation](docs/getting-started/installation.md)
- [Python API Reference](docs/api/python.md)
- [Reference API Tutorial](docs/reference_api_tutorial.md)
- [Shard Engine Guide](docs/guides/shard-engine.md)
- [Latence Graph Sidecar Guide](docs/guides/latence-graph-sidecar.md)
- [Enterprise Control Plane Boundary](docs/guides/control-plane.md)
- [Max-Performance Guide](docs/guides/max-performance-reference-api.md)
- [Scaling Guide](docs/guides/scaling.md)
- [Benchmarks And Methodology](docs/benchmarks.md)
- [Production Notes](PRODUCTION.md)

## Install From Source

```bash
git clone https://github.com/ddickmann/voyager-index.git
cd voyager-index
bash scripts/install_from_source.sh --cpu
```

## License

Apache-2.0. See [LICENSE](LICENSE).
