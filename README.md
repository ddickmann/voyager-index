# colsearch

[![CI](https://github.com/ddickmann/colsearch/actions/workflows/ci.yml/badge.svg)](https://github.com/ddickmann/colsearch/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/colsearch)](https://pypi.org/project/colsearch/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)

**ColSearch — the production search category for late-interaction retrieval.** One node. CPU or GPU. MaxSim is the truth scorer. 1.58-bit ColBERT at 40 B/token. **3.12× FastPlaid on H100.**

> **Renamed from `voyager-index`.** `pip install colsearch` is the new home. The legacy `import voyager_index` path keeps working in `0.1.7` (with a one-line `DeprecationWarning`) and will be removed in `0.2.0`. See [CHANGELOG.md](CHANGELOG.md).

## The pain

ColBERT-quality retrieval is table-stakes for serious RAG, and the production options force a choice you should not have to make.

- Managed SaaS — fast to start, hard to control, your data leaves the box.
- Distributed clusters — strong recall, expensive to operate.
- Offline benchmarks — great numbers, no API, no WAL, no recovery.

Most "production" stacks treat MaxSim as an optional rerank stage and lose its signal under aggressive shortlisting. Most engines that ship operationally drop late interaction entirely.

## The solution

`colsearch` is a multi-vector native retrieval engine built around MaxSim as the **final scorer** — and engineered so a single machine can serve it.

- **One-node deployment.** No control plane, no orchestration tax.
- **One contract across CPU and GPU.** Rust SIMD on CPU, Triton + fused CUDA on GPU.
- **Production-ready 1.58-bit kernel.** ~40 B per token (~6.4× smaller than FP16), parity-tested on both lanes, NDCG@10 within ~1 pt of FP16 on BEIR. **3.12× geomean QPS over FastPlaid on BEIR-8 / H100** with the FP16 lane — see [`benchmarks/competitive_benchmark.md`](benchmarks/competitive_benchmark.md).
- **Late-interaction native.** ColBERT, ColPali, ColQwen out of the box.
- **Database semantics.** WAL, checkpoint, crash recovery, scroll, retrieve.
- **Optional graph lane.** The Latence sidecar augments first-stage retrieval — never required.

## How

```bash
pip install "colsearch[full]"        # CPU-only host
pip install "colsearch[full,gpu]"    # GPU host
colsearch-server                     # OpenAPI at http://127.0.0.1:8080/docs
```

Python:

```python
import numpy as np
from colsearch import Index

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
from colsearch import encode_vector_payload

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
docker build -f deploy/reference-api/Dockerfile -t colsearch .
docker run -p 8080:8080 -v "$(pwd)/data:/data" colsearch
```

## Features

- **Routing** — LEMUR proxy router + FAISS MIPS shortlist, optional ColBANDIT query-time pruning.
- **Scoring** — Triton MaxSim and fused Rust MaxSim, FP16 / INT8 / FP8 / ROQ-4 / RROQ-1.58 / RROQ-4 with float rerank.
- **Storage** — safetensors shards, memory-mapped CPU, GPU-resident corpus mode.
- **Hybrid** — BM25 + dense fusion via RRF or Tabu Search refinement.
- **Multimodal** — text (ColBERT), images (ColPali / ColQwen), preprocessing for PDF / DOCX / XLSX.
- **Operations** — WAL, checkpoint, crash recovery, scroll, retrieve, multi-worker FastAPI.
- **Optional graph lane** — Latence sidecar for graph-aware rescue and provenance, additive to the OSS path.
- **Optional groundedness lane** — **Latence Trace** premium sidecar for post-generation hallucination scoring against retrieved `chunk_ids` or raw context. See the [Groundedness sidecar guide](docs/guides/groundedness-sidecar.md) and [latence.ai](https://latence.ai) for access.

## Benchmarks

Headline competitive numbers (single-H100, BEIR-8, single-client sequential — same methodology FastPlaid publishes):

| Lane                                    | GPU geomean QPS | vs FastPlaid |
| --------------------------------------- | --------------: | -----------: |
| colsearch `fp16/gpu`                    |       **446.7** |   **3.12×**  |
| colsearch `rroq158_gs128/gpu` (40 B/tok)| **294.4**      |   **2.06×**  |
| FastPlaid GPU                           |          143.2  |        1.00× |

`colsearch fp16/gpu` beats FastPlaid GPU on **8 / 8** BEIR-8 datasets; the 1.58-bit lane wins **6 / 8** while shipping a **6.4×** smaller index. Per-dataset rows, NDCG@10 quality, CPU lane, hardware/env block, and reproduction command live in [`benchmarks/competitive_benchmark.md`](benchmarks/competitive_benchmark.md).

The full 4-codec × BEIR-6 × 2-mode internal sweep (FP16 / INT8 / RROQ-1.58 / RROQ-4-Riem) lives in [docs/benchmarks.md](docs/benchmarks.md); raw per-cell JSONL with provenance under [`reports/beir_2026q2/`](reports/beir_2026q2/).

### Reproducible head-to-head against FastPlaid

A single-command benchmark runs colsearch against [FastPlaid](https://github.com/lightonai/fast-plaid) on FastPlaid's published BEIR-8 matrix using **identical per-token embeddings** for both libraries (so the comparison varies only the indexing/scoring engine, not the encoder model). Tutorial, including the recommended single-H100 cloud setup: [`docs/benchmarks/fast-plaid-head-to-head.md`](docs/benchmarks/fast-plaid-head-to-head.md). Quick smoke (one small dataset, ~30 s on H100):

```bash
pip install fast-plaid pylate
python benchmarks/data/prepare_beir_datasets.py --datasets nfcorpus
python benchmarks/fast_plaid_head_to_head.py --datasets nfcorpus
```

## Codecs

| Codec           | Bytes/token (dim=128) | vs FP16 | NDCG@10 vs FP16 (BEIR avg) | Notes |
| --------------- | --------------------: | ------: | -------------------------: | ----- |
| `fp16`          |                  256  |   1.00× |                       0.0  | Truth lane. |
| `int8`          |                  128  |   0.50× |                  −0.06 pt  | GPU-only. |
| `rroq158`       |               **40**  | **0.16×** |                −1.37 pt  | Production default. Riemannian 1.58-bit ternary, K=8192, gs=128. GPU (Triton + fused CUDA b1 MMA on H100) and CPU (Rust SIMD + popcount) both wired and parity-tested. |
| `rroq4_riem`    |                   88  |   0.34× |                  +0.02 pt  | No-quality-loss lane (~3× smaller). Use when zero NDCG@10 regression is required. |

`Compression.RROQ158` (group_size=128) is the default codec for newly built indexes. For dims that aren't a multiple of 128 the encoder transparently steps down to gs=64 / gs=32 with a log warning. Pass `compression=Compression.FP16` (Python) or `--compression fp16` (CLI) to opt out. Decision matrix and per-dim recipe: [docs/guides/quantization-tuning.md](docs/guides/quantization-tuning.md). Math (RaBitQ extension + Riemannian log map + FWHT-rotated tangent ternary): [docs/guides/rroq-mathematics.md](docs/guides/rroq-mathematics.md).

Existing FP16 / RROQ158 / RROQ4_RIEM indexes load unchanged — the manifest carries the build-time codec.

## Architecture

```text
query (token / patch embeddings)
  → LEMUR routing MLP → FAISS ANN → candidate IDs
  → optional BM25 fusion · centroid pruning · ColBANDIT
  → exact MaxSim   (Rust SIMD CPU FP16/RROQ-1.58/RROQ-4-Riem  |  Triton + fused CUDA GPU FP16/INT8/FP8/ROQ-4/RROQ-1.58/RROQ-4-Riem)
  → optional FP16 rerank on top-N shortlist
  → optional Latence graph augmentation
  → top-K (or packed context)
```

| Layer          | What ships                                                       |
| -------------- | ---------------------------------------------------------------- |
| Routing        | LEMUR MLP + FAISS MIPS, candidate budgets                        |
| Storage        | safetensors shards, mmap, GPU-resident corpus mode               |
| Scoring        | Triton + Rust fused MaxSim, FP16 / INT8 / FP8 / ROQ-4 / RROQ-1.58 / RROQ-4-Riem |
| Optional graph | Latence sidecar, additive after first-stage retrieval            |
| Durability     | WAL, memtable, checkpoint, crash recovery                        |
| Serving        | FastAPI, base64 vector transport, multi-worker, OpenAPI          |

Three execution modes share the same collection format and API contract: **CPU exact** (mmap → Rust fused), **GPU streamed** (CPU → GPU → Triton), and **GPU corpus** (fully VRAM-resident). Start with CPU, add GPU when latency matters.

## Documentation

- [Quickstart](docs/getting-started/quickstart.md) · [Installation](docs/getting-started/installation.md) · [Install from source](docs/getting-started/installation.md#from-source)
- [Reference API Tutorial](docs/reference_api_tutorial.md) · [HTTP endpoint reference](docs/reference_api_tutorial.md#10-api-endpoint-reference) · [Python API](docs/api/python.md)
- [Shard Engine Guide](docs/guides/shard-engine.md) · [Max-Performance Guide](docs/guides/max-performance-reference-api.md) · [Scaling Guide](docs/guides/scaling.md)
- [Latence Graph Sidecar Guide](docs/guides/latence-graph-sidecar.md) · [Groundedness Sidecar Guide](docs/guides/groundedness-sidecar.md) · [Enterprise Control Plane](docs/guides/control-plane.md)
- [Benchmarks And Methodology](docs/benchmarks.md) · [Competitive Benchmark vs FastPlaid](benchmarks/competitive_benchmark.md) · [Production Notes](PRODUCTION.md)
- [Full Feature Cookbook](docs/full_feature_cookbook.md)

## Community And Project Health

- File a bug: [bug report template](.github/ISSUE_TEMPLATE/bug_report.yml)
- Request a feature: [feature request template](.github/ISSUE_TEMPLATE/feature_request.yml)
- Open a PR: [pull request template](.github/pull_request_template.md)
- Contributing guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Security policy: [SECURITY.md](SECURITY.md)
- Release process: [RELEASING.md](RELEASING.md)
- Code of Conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

## License & Commercial Use

**This project is licensed under [Creative Commons Attribution-NonCommercial 4.0 International](LICENSE) (`CC-BY-NC-4.0`). It is free for non-commercial use only.**

You may use, copy, modify, and redistribute `colsearch` for:

- research, evaluation, and benchmarking
- academic work and teaching
- personal projects and experimentation
- internal, non-revenue R&D inside an organization

You may **not** use `colsearch`, in whole or in part, for any commercial or revenue-generating purpose without a separate commercial license. This explicitly includes:

- selling, sublicensing, or relicensing the code
- offering it as a hosted, managed, or SaaS product
- embedding it in any product, service, or integration that you sell, license, or otherwise monetize
- using it to provide paid consulting deliverables built on top of it

For commercial licensing inquiries, contact **commercial@latence.ai**.

**Carveout — vendored Qdrant subtree.** The directory `src/kernels/vendor/qdrant/` is a vendored copy of upstream [qdrant/qdrant](https://github.com/qdrant/qdrant) and remains under its original **Apache-2.0** license; that license travels with those files into any derivative binaries. See [LICENSING.md](LICENSING.md) and [internal/contracts/QDRANT_VENDORING.md](internal/contracts/QDRANT_VENDORING.md) for the full carveout.

**Prior releases.** Releases previously distributed under the `voyager-index` name (versions `0.1.0` through `0.1.6`, Apache-2.0) remain available to recipients who already obtained them under their original license. The CC-BY-NC-4.0 license above and the `colsearch` package name apply to this source tree and to all `0.1.7+` releases made from it going forward.
