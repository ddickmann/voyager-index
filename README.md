# voyager-index

[![CI](https://github.com/ddickmann/voyager-index/actions/workflows/ci.yml/badge.svg)](https://github.com/ddickmann/voyager-index/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/voyager-index)](https://pypi.org/project/voyager-index/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Late-interaction retrieval for on-prem AI. One node. CPU or GPU. MaxSim is the truth scorer.**

## The pain

ColBERT-quality retrieval is table-stakes for serious RAG, and the production options
force a choice you should not have to make.

- Managed SaaS — fast to start, hard to control, your data leaves the box.
- Distributed clusters — strong recall, expensive to operate.
- Offline benchmarks — great numbers, no API, no WAL, no recovery.

Most "production" stacks treat MaxSim as an optional rerank stage and lose its
signal under aggressive shortlisting. Most engines that ship operationally drop
late interaction entirely.

## The solution

`voyager-index` is a multi-vector native retrieval engine built around MaxSim as
the **final scorer** — and engineered so a single machine can serve it.

- **One-node deployment.** No control plane, no orchestration tax.
- **One contract across CPU and GPU.** Rust SIMD on CPU, Triton on GPU.
- **Quantized fast paths.** FP16, INT8, FP8, ROQ-4, all reranked back to float truth.
- **Late-interaction native.** ColBERT, ColPali, ColQwen out of the box.
- **Database semantics.** WAL, checkpoint, crash recovery, scroll, retrieve.
- **Optional graph lane.** The Latence sidecar augments first-stage retrieval — never required.

## How

```bash
pip install "voyager-index[full,gpu]"   # drop ,gpu on CPU-only hosts
voyager-index-server                    # OpenAPI at http://127.0.0.1:8080/docs
```

Python:

```python
import numpy as np
from voyager_index import Index

rng = np.random.default_rng(7)
docs = [rng.normal(size=(16, 128)).astype("float32") for _ in range(32)]
query = rng.normal(size=(16, 128)).astype("float32")

idx = Index("demo", dim=128, engine="shard", n_shards=32,
            k_candidates=256, compression="fp16")
idx.add(docs, ids=list(range(len(docs))))
print(idx.search(query, k=5)[0])
```

HTTP (base64 vector payloads, fp8 GPU scoring, ColBANDIT pruning):

```python
import numpy as np, requests
from voyager_index import encode_vector_payload

q = np.random.default_rng(7).normal(size=(16, 128)).astype("float32")
r = requests.post(
    "http://127.0.0.1:8080/collections/demo/search",
    json={"vectors": encode_vector_payload(q, dtype="float16"), "top_k": 5,
          "quantization_mode": "fp8", "use_colbandit": True},
    timeout=30,
)
print(r.json()["results"][0])
```

Docker:

```bash
docker build -f deploy/reference-api/Dockerfile -t voyager-index .
docker run -p 8080:8080 -v "$(pwd)/data:/data" voyager-index
```

## Features

- **Routing** — LEMUR proxy router + FAISS MIPS shortlist, optional ColBANDIT query-time pruning.
- **Scoring** — Triton MaxSim and fused Rust MaxSim, INT8 / FP8 / ROQ-4 with float rerank.
- **Storage** — safetensors shards, memory-mapped CPU, GPU-resident corpus mode.
- **Hybrid** — BM25 + dense fusion via RRF or Tabu Search refinement.
- **Multimodal** — text (ColBERT), images (ColPali / ColQwen), preprocessing for PDF / DOCX / XLSX.
- **Operations** — WAL, checkpoint, crash recovery, scroll, retrieve, multi-worker FastAPI.
- **Optional graph lane** — Latence sidecar for graph-aware rescue and provenance, additive to the OSS path.
- **Optional groundedness lane** — `latence-trace` sidecar for post-generation hallucination scoring against retrieved `chunk_ids` or raw context. Commercial sidecar; not part of the OSS distribution. See the [Groundedness sidecar guide](docs/guides/groundedness-sidecar.md).

## Benchmarks

### BEIR retrieval — RTX A5000, search-only, full query set

Encoder: `lightonai/GTE-ModernColBERT-v1`. CPU lane uses 8 native Rust workers.

| Dataset  | Docs   | NDCG@10 | Recall@100 | GPU QPS | GPU P95 (ms) | CPU QPS | CPU P95 (ms) |
|----------|-------:|--------:|-----------:|--------:|-------------:|--------:|-------------:|
| arguana  | 8,674  | 0.3679  | 0.9586     | 270.0   | 4.1          | 41.6    | 202.7        |
| fiqa     | 57,638 | 0.4436  | 0.7297     | 164.8   | 5.0          | 80.2    | 115.7        |
| nfcorpus | 3,633  | 0.3833  | 0.3348     | 282.6   | 3.8          | 123.3   | 84.4         |
| quora    | 15,675 | 0.9766  | 0.9993     | 346.8   | 2.6          | 271.7   | 46.9         |
| scidocs  | 25,657 | 0.1977  | 0.4369     | 246.8   | 4.3          | 83.9    | 111.8        |
| scifact  | 5,183  | 0.7544  | 0.9567     | 263.4   | 4.0          | 69.1    | 138.4        |

GPU P95 stays under 6 ms across every dataset. The full per-dataset
[head-to-head against `next-plaid`](docs/benchmarks.md#comparison-vs-next-plaid)
(same model, H100, encoding included), methodology, and caveats live in
[docs/benchmarks.md](docs/benchmarks.md).

## Architecture

```text
query (token / patch embeddings)
  → LEMUR routing MLP → FAISS ANN → candidate IDs
  → optional BM25 fusion · centroid pruning · ColBANDIT
  → exact MaxSim   (Rust SIMD CPU  |  Triton FP16/INT8/FP8/ROQ-4 GPU)
  → optional Latence graph augmentation
  → top-K (or packed context)
```

| Layer        | What ships                                                       |
|--------------|------------------------------------------------------------------|
| Routing      | LEMUR MLP + FAISS MIPS, candidate budgets                        |
| Storage      | safetensors shards, mmap, GPU-resident corpus mode               |
| Scoring      | Triton + Rust fused MaxSim with INT8 / FP8 / ROQ-4 fast paths    |
| Optional graph | Latence sidecar, additive after first-stage retrieval          |
| Durability   | WAL, memtable, checkpoint, crash recovery                        |
| Serving      | FastAPI, base64 vector transport, multi-worker, OpenAPI          |

Three execution modes share the same collection format and API contract:
**CPU exact** (mmap → Rust fused), **GPU streamed** (CPU → GPU → Triton), and
**GPU corpus** (fully VRAM-resident). Start with CPU, add GPU when latency matters.

## Documentation

- [Quickstart](docs/getting-started/quickstart.md) · [Installation](docs/getting-started/installation.md) · [Install from source](docs/getting-started/installation.md#from-source)
- [Reference API Tutorial](docs/reference_api_tutorial.md) · [HTTP endpoint reference](docs/reference_api_tutorial.md#10-api-endpoint-reference) · [Python API](docs/api/python.md)
- [Shard Engine Guide](docs/guides/shard-engine.md) · [Max-Performance Guide](docs/guides/max-performance-reference-api.md) · [Scaling Guide](docs/guides/scaling.md)
- [Latence Graph Sidecar Guide](docs/guides/latence-graph-sidecar.md) · [Groundedness Sidecar Guide](docs/guides/groundedness-sidecar.md) · [Enterprise Control Plane](docs/guides/control-plane.md)
- [Benchmarks And Methodology](docs/benchmarks.md) · [Production Notes](PRODUCTION.md)
- [Full Feature Cookbook](docs/full_feature_cookbook.md)

## Community And Project Health

- File a bug: [bug report template](.github/ISSUE_TEMPLATE/bug_report.yml)
- Request a feature: [feature request template](.github/ISSUE_TEMPLATE/feature_request.yml)
- Open a PR: [pull request template](.github/pull_request_template.md)
- Contributing guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Security policy: [SECURITY.md](SECURITY.md)
- Release process: [RELEASING.md](RELEASING.md)
- Code of Conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

## License

Apache-2.0. See [LICENSE](LICENSE).
